use crate::models::{AppSettings, RamDiskStatus};
use anyhow::{anyhow, Context};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::thread;
use std::time::Duration;

const CACHE_FOLDER: &str = "TEMP";
const MARKER_FILE: &str = ".duplicate-video-search-ram-cache";
const TASK_NAME: &str = "DuplicateVideoSearch-RamDisk";
const RELEASE_TASK_NAME: &str = "DuplicateVideoSearch-RamDisk-Release";
const DRIVER_DOWNLOAD_URL: &str =
    "https://sourceforge.net/projects/imdisk-toolkit/files/20250206/ImDiskTk-x64.zip/download";
const MIN_SIZE_MB: usize = 512;
const MAX_SIZE_MB: usize = 1_048_576;

pub fn status(settings: &AppSettings) -> RamDiskStatus {
    status_for_cache(
        &settings.local_preprocess_temp_dir,
        settings.ram_disk_size_mb,
        settings.ram_disk_setup_completed,
    )
}

pub fn configure_with_elevation(
    settings: &AppSettings,
    size_mb: usize,
) -> anyhow::Result<RamDiskStatus> {
    let size_mb = size_mb.clamp(MIN_SIZE_MB, MAX_SIZE_MB);
    let drive_letter = choose_drive_letter(&settings.local_preprocess_temp_dir, size_mb)?;
    run_elevated_configuration(size_mb, drive_letter)?;
    let cache_path = cache_path(drive_letter).display().to_string();
    let result = status_for_cache(&cache_path, size_mb, true);
    if !result.ready {
        return Err(anyhow!(result.message.clone()));
    }
    Ok(result)
}

pub fn try_ensure_from_scheduled_task(settings: &AppSettings) -> RamDiskStatus {
    let current = status(settings);
    if current.ready || !settings.ram_disk_enabled || !current.driver_installed {
        return current;
    }

    #[cfg(windows)]
    {
        let mut command = silent_command("schtasks.exe");
        if command
            .args(["/Run", "/TN", TASK_NAME])
            .status()
            .is_ok_and(|status| status.success())
        {
            for _ in 0..30 {
                thread::sleep(Duration::from_millis(500));
                let next = status(settings);
                if next.ready {
                    return next;
                }
            }
        }
    }

    status(settings)
}

pub struct TaskRamDisk {
    settings: AppSettings,
    active: bool,
    released: bool,
}

impl TaskRamDisk {
    pub fn activate(settings: &AppSettings) -> anyhow::Result<Self> {
        if !settings.ram_disk_enabled
            || !settings.local_preprocess_enabled
            || !settings.ai_frame_cache_enabled
        {
            return Ok(Self {
                settings: settings.clone(),
                active: false,
                released: false,
            });
        }
        if !settings.ram_disk_setup_completed {
            return Err(anyhow!(
                "内存缓存盘尚未完成配置，请先在设置中申请权限并创建"
            ));
        }

        let next = try_ensure_from_scheduled_task(settings);
        if !next.ready {
            return Err(anyhow!("内存缓存盘不可用：{}", next.message));
        }

        Ok(Self {
            settings: settings.clone(),
            active: true,
            released: false,
        })
    }

    pub fn release(mut self) -> anyhow::Result<RamDiskStatus> {
        self.released = true;
        if self.active {
            release_managed(&self.settings)
        } else {
            Ok(status(&self.settings))
        }
    }
}

impl Drop for TaskRamDisk {
    fn drop(&mut self) {
        if self.active && !self.released {
            let _ = release_managed(&self.settings);
        }
    }
}

pub fn release_managed(settings: &AppSettings) -> anyhow::Result<RamDiskStatus> {
    if !cfg!(windows) {
        return Ok(status(settings));
    }
    let Some(drive_letter) = parse_drive_letter(&settings.local_preprocess_temp_dir) else {
        return Ok(status(settings));
    };
    release_managed_drive(drive_letter)?;
    Ok(status(settings))
}

pub fn open_driver_download() -> anyhow::Result<()> {
    #[cfg(windows)]
    {
        use std::ffi::OsStr;
        use windows_sys::Win32::UI::Shell::ShellExecuteW;
        use windows_sys::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;

        let operation = wide_null(OsStr::new("open"));
        let target = wide_null(OsStr::new(DRIVER_DOWNLOAD_URL));
        let result = unsafe {
            ShellExecuteW(
                std::ptr::null_mut(),
                operation.as_ptr(),
                target.as_ptr(),
                std::ptr::null(),
                std::ptr::null(),
                SW_SHOWNORMAL,
            )
        };
        if result as isize <= 32 {
            return Err(anyhow!(
                "open ImDisk driver download failed ({})",
                result as isize
            ));
        }
        return Ok(());
    }

    #[cfg(not(windows))]
    Err(anyhow!(
        "automatic RAM disk setup is only supported on Windows"
    ))
}

pub fn handle_elevated_arguments() -> Option<i32> {
    let args = env::args().collect::<Vec<_>>();
    let command_index = args.iter().position(|value| {
        value == "--configure-ram-disk"
            || value == "--ensure-ram-disk"
            || value == "--release-ram-disk"
    })?;
    let result = if args[command_index] == "--release-ram-disk" {
        let drive_letter = args
            .get(command_index + 1)
            .and_then(|value| parse_drive_letter(value))?;
        release_managed_drive_inner(drive_letter, false)
    } else {
        let size_mb = args.get(command_index + 1)?.parse::<usize>().ok()?;
        let drive_letter = args
            .get(command_index + 2)
            .and_then(|value| parse_drive_letter(value))?;
        let register_task = args[command_index] == "--configure-ram-disk";
        configure_elevated(size_mb, drive_letter, register_task)
    };
    Some(match result {
        Ok(()) => 0,
        Err(error) => {
            let _ = write_setup_error(&error.to_string());
            1
        }
    })
}

fn status_for_cache(
    cache: &str,
    configured_size_mb: usize,
    setup_completed: bool,
) -> RamDiskStatus {
    let supported = cfg!(windows);
    let driver_installed = resolve_imdisk().is_some();
    let drive_letter = parse_drive_letter(cache);
    let cache_path = drive_letter
        .map(cache_path)
        .unwrap_or_else(|| PathBuf::from(cache));
    let mounted = drive_letter.is_some_and(|letter| is_imdisk_mount(letter));
    let ready = mounted && cache_path.is_dir();
    let actual_capacity_mb = drive_letter.and_then(drive_capacity_mb);
    let message = if !supported {
        "自动内存盘仅支持 Windows".to_string()
    } else if !driver_installed {
        "未检测到 ImDisk 驱动；安装驱动后可继续自动配置".to_string()
    } else if ready {
        format!("内存缓存已就绪：{}", cache_path.display())
    } else if drive_letter.is_none() {
        "一级缓存路径不是有效的盘符路径，需要重新配置".to_string()
    } else if drive_letter.is_some_and(|letter| drive_root(letter).exists()) {
        "目标盘符已被其他磁盘占用，将在配置时自动选择空闲盘符".to_string()
    } else if setup_completed {
        "内存盘当前未挂载；会在任务开始时自动挂载，任务结束后自动卸载".to_string()
    } else {
        "首次使用需要申请管理员权限并创建内存盘".to_string()
    };

    RamDiskStatus {
        supported,
        driver_installed,
        mounted,
        ready,
        setup_completed,
        configured_size_mb,
        actual_capacity_mb,
        drive_letter: drive_letter.map(|letter| letter.to_string()),
        cache_path: cache_path.display().to_string(),
        message,
    }
}

fn configure_elevated(
    size_mb: usize,
    drive_letter: char,
    register_task: bool,
) -> anyhow::Result<()> {
    if !cfg!(windows) {
        return Err(anyhow!(
            "automatic RAM disk setup is only supported on Windows"
        ));
    }
    let size_mb = size_mb.clamp(MIN_SIZE_MB, MAX_SIZE_MB);
    let imdisk = resolve_imdisk().context("ImDisk driver is not installed")?;
    let root = drive_root(drive_letter);
    let cache = cache_path(drive_letter);

    if root.exists() {
        let capacity_matches = drive_capacity_mb(drive_letter)
            .is_some_and(|capacity| capacity.abs_diff(size_mb as u64) <= 8);
        let marker_exists = root.join(MARKER_FILE).exists();
        if is_imdisk_mount(drive_letter) && capacity_matches {
            fs::create_dir_all(&cache)
                .with_context(|| format!("create RAM cache folder {}", cache.display()))?;
            fs::write(
                root.join(MARKER_FILE),
                b"Duplicate Video Search RAM cache\n",
            )?;
            if register_task {
                register_task_helpers(size_mb, drive_letter)?;
            }
            return Ok(());
        }
        if !marker_exists {
            return Err(anyhow!(
                "drive {}: is already in use and is not managed by Duplicate Video Search",
                drive_letter
            ));
        }
        let status = silent_command(&imdisk)
            .args(["-D", "-m", &format!("{drive_letter}:")])
            .status()
            .context("remove previous RAM disk")?;
        if !status.success() {
            return Err(anyhow!(
                "failed to remove previous RAM disk on {drive_letter}:"
            ));
        }
    }

    let status = silent_command(&imdisk)
        .args([
            "-a",
            "-t",
            "vm",
            "-s",
            &format!("{size_mb}M"),
            "-m",
            &format!("{drive_letter}:"),
            "-o",
            "rw,fix,hd",
            "-p",
            "/fs:ntfs /q /y /v:DVS_RAM_CACHE",
        ])
        .status()
        .context("create ImDisk RAM disk")?;
    if !status.success() {
        return Err(anyhow!("ImDisk failed to create the RAM disk"));
    }

    for _ in 0..40 {
        if root.exists() {
            fs::create_dir_all(&cache)
                .with_context(|| format!("create RAM cache folder {}", cache.display()))?;
            fs::write(
                root.join(MARKER_FILE),
                b"Duplicate Video Search RAM cache\n",
            )?;
            if register_task {
                register_task_helpers(size_mb, drive_letter)?;
            }
            let _ = fs::remove_file(setup_error_path());
            return Ok(());
        }
        thread::sleep(Duration::from_millis(250));
    }

    Err(anyhow!(
        "ImDisk reported success but the RAM disk did not become available"
    ))
}

fn release_managed_drive(drive_letter: char) -> anyhow::Result<()> {
    release_managed_drive_inner(drive_letter, true)
}

fn release_managed_drive_inner(
    drive_letter: char,
    allow_task_fallback: bool,
) -> anyhow::Result<()> {
    if !cfg!(windows) {
        return Ok(());
    }
    let root = drive_root(drive_letter);
    if !is_imdisk_mount(drive_letter) {
        return Ok(());
    }
    if !root.join(MARKER_FILE).exists() {
        return Err(anyhow!(
            "refusing to remove drive {drive_letter}: because it is not marked as a Duplicate Video Search RAM cache"
        ));
    }

    let mut direct_error = None;
    for _ in 0..8 {
        match remove_ram_disk(drive_letter) {
            Ok(()) => {
                if wait_until_unmounted(drive_letter) {
                    return Ok(());
                }
                direct_error = Some(anyhow!(
                    "drive {drive_letter}: is still mounted after ImDisk remove command"
                ));
            }
            Err(error) => direct_error = Some(error),
        }
        thread::sleep(Duration::from_millis(500));
        if !is_imdisk_mount(drive_letter) {
            return Ok(());
        }
    }

    if allow_task_fallback && run_release_task().is_ok() {
        for _ in 0..30 {
            if wait_until_unmounted(drive_letter) {
                return Ok(());
            }
            thread::sleep(Duration::from_millis(500));
        }
    }

    Err(direct_error
        .unwrap_or_else(|| anyhow!("failed to remove managed RAM disk on {drive_letter}:")))
}

fn remove_ram_disk(drive_letter: char) -> anyhow::Result<()> {
    let imdisk = resolve_imdisk().context("ImDisk driver is not installed")?;
    let status = silent_command(&imdisk)
        .args(["-D", "-m", &format!("{drive_letter}:")])
        .status()
        .context("remove RAM disk")?;
    if !status.success() {
        return Err(anyhow!(
            "failed to remove managed RAM disk on {drive_letter}:"
        ));
    }
    Ok(())
}

fn wait_until_unmounted(drive_letter: char) -> bool {
    for _ in 0..20 {
        if !is_imdisk_mount(drive_letter) {
            return true;
        }
        thread::sleep(Duration::from_millis(250));
    }
    !is_imdisk_mount(drive_letter)
}

fn run_release_task() -> anyhow::Result<()> {
    let status = silent_command("schtasks.exe")
        .args(["/Run", "/TN", RELEASE_TASK_NAME])
        .status()
        .context("run RAM disk release task")?;
    if !status.success() {
        return Err(anyhow!("failed to run RAM disk release task"));
    }
    Ok(())
}

fn choose_drive_letter(preferred_cache: &str, size_mb: usize) -> anyhow::Result<char> {
    let preferred = parse_drive_letter(preferred_cache);
    let candidates = preferred.into_iter().chain(
        ('R'..='Z')
            .rev()
            .filter(move |letter| Some(*letter) != preferred),
    );
    for letter in candidates {
        let root = drive_root(letter);
        if !root.exists() || root.join(MARKER_FILE).exists() {
            return Ok(letter);
        }
        if is_imdisk_mount(letter)
            && drive_capacity_mb(letter)
                .is_some_and(|capacity| capacity.abs_diff(size_mb as u64) <= 8)
        {
            return Ok(letter);
        }
    }
    Err(anyhow!(
        "no free drive letter is available between R: and Z:"
    ))
}

fn register_task_helpers(size_mb: usize, drive_letter: char) -> anyhow::Result<()> {
    register_ensure_task(size_mb, drive_letter)?;
    register_release_task(drive_letter)?;
    Ok(())
}

fn register_ensure_task(size_mb: usize, drive_letter: char) -> anyhow::Result<()> {
    let executable = env::current_exe().context("resolve current executable")?;
    register_on_demand_task(
        TASK_NAME,
        executable.display().to_string(),
        format!("--ensure-ram-disk {size_mb} {drive_letter}"),
    )
    .context("register RAM disk mount task")
}

fn register_release_task(drive_letter: char) -> anyhow::Result<()> {
    let executable = env::current_exe().context("resolve current executable")?;
    register_on_demand_task(
        RELEASE_TASK_NAME,
        executable.display().to_string(),
        format!("--release-ram-disk {drive_letter}"),
    )
    .context("register RAM disk release task")
}

fn register_on_demand_task(
    task_name: &str,
    executable: String,
    arguments: String,
) -> anyhow::Result<()> {
    let xml_path = env::temp_dir().join(format!(
        "duplicate-video-search-{}-{}.xml",
        task_name.replace('\\', "_"),
        std::process::id()
    ));
    let xml = format!(
        r#"<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Author>Duplicate Video Search</Author>
  </RegistrationInfo>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>HighestAvailable</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>false</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>true</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT1H</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{}</Command>
      <Arguments>{}</Arguments>
    </Exec>
  </Actions>
</Task>
"#,
        xml_escape(&executable),
        xml_escape(&arguments)
    );
    fs::write(&xml_path, encode_utf16le_with_bom(&xml))
        .with_context(|| format!("write scheduled task XML {}", xml_path.display()))?;
    let status = silent_command("schtasks.exe")
        .args(["/Create", "/TN", task_name, "/XML"])
        .arg(&xml_path)
        .args(["/F"])
        .status()
        .with_context(|| format!("register scheduled task {task_name}"))?;
    let _ = fs::remove_file(&xml_path);
    if !status.success() {
        return Err(anyhow!("failed to register scheduled task {task_name}"));
    }
    Ok(())
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn encode_utf16le_with_bom(value: &str) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(2 + value.len() * 2);
    bytes.extend_from_slice(&[0xFF, 0xFE]);
    for code_unit in value.encode_utf16() {
        bytes.extend_from_slice(&code_unit.to_le_bytes());
    }
    bytes
}

#[cfg(windows)]
fn run_elevated_configuration(size_mb: usize, drive_letter: char) -> anyhow::Result<()> {
    use std::ffi::OsStr;
    use windows_sys::Win32::Foundation::{CloseHandle, WAIT_OBJECT_0};
    use windows_sys::Win32::System::Threading::{
        GetExitCodeProcess, WaitForSingleObject, INFINITE,
    };
    use windows_sys::Win32::UI::Shell::{
        ShellExecuteExW, SEE_MASK_NOCLOSEPROCESS, SHELLEXECUTEINFOW,
    };
    use windows_sys::Win32::UI::WindowsAndMessaging::SW_HIDE;

    let executable = env::current_exe().context("resolve current executable")?;
    let verb = wide_null(OsStr::new("runas"));
    let file = wide_null(executable.as_os_str());
    let parameters = wide_null(OsStr::new(&format!(
        "--configure-ram-disk {size_mb} {drive_letter}"
    )));
    let mut info = SHELLEXECUTEINFOW {
        cbSize: std::mem::size_of::<SHELLEXECUTEINFOW>() as u32,
        fMask: SEE_MASK_NOCLOSEPROCESS,
        hwnd: std::ptr::null_mut(),
        lpVerb: verb.as_ptr(),
        lpFile: file.as_ptr(),
        lpParameters: parameters.as_ptr(),
        lpDirectory: std::ptr::null(),
        nShow: SW_HIDE,
        hInstApp: std::ptr::null_mut(),
        lpIDList: std::ptr::null_mut(),
        lpClass: std::ptr::null(),
        hkeyClass: std::ptr::null_mut(),
        dwHotKey: 0,
        Anonymous: Default::default(),
        hProcess: std::ptr::null_mut(),
    };
    let launched = unsafe { ShellExecuteExW(&mut info) };
    if launched == 0 || info.hProcess.is_null() {
        return Err(anyhow!(
            "administrator permission was cancelled or could not be requested"
        ));
    }
    let wait_result = unsafe { WaitForSingleObject(info.hProcess, INFINITE) };
    if wait_result != WAIT_OBJECT_0 {
        unsafe { CloseHandle(info.hProcess) };
        return Err(anyhow!("waiting for elevated RAM disk setup failed"));
    }
    let mut exit_code = 1u32;
    let got_exit_code = unsafe { GetExitCodeProcess(info.hProcess, &mut exit_code) };
    unsafe { CloseHandle(info.hProcess) };
    if got_exit_code == 0 || exit_code != 0 {
        let detail = fs::read_to_string(setup_error_path()).unwrap_or_default();
        return Err(anyhow!(
            "RAM disk setup failed{}",
            if detail.trim().is_empty() {
                String::new()
            } else {
                format!(": {}", detail.trim())
            }
        ));
    }
    Ok(())
}

#[cfg(not(windows))]
fn run_elevated_configuration(_size_mb: usize, _drive_letter: char) -> anyhow::Result<()> {
    Err(anyhow!(
        "automatic RAM disk setup is only supported on Windows"
    ))
}

fn resolve_imdisk() -> Option<PathBuf> {
    let candidates = [
        env::var_os("SystemRoot").map(|root| PathBuf::from(root).join(r"System32\imdisk.exe")),
        env::var_os("ProgramFiles").map(|root| PathBuf::from(root).join(r"ImDisk\imdisk.exe")),
        env::var_os("ProgramFiles(x86)").map(|root| PathBuf::from(root).join(r"ImDisk\imdisk.exe")),
    ];
    candidates
        .into_iter()
        .flatten()
        .find(|path| path.is_file())
        .or_else(|| {
            env::var_os("PATH").and_then(|path| {
                env::split_paths(&path)
                    .map(|directory| directory.join("imdisk.exe"))
                    .find(|candidate| candidate.is_file())
            })
        })
}

fn is_imdisk_mount(drive_letter: char) -> bool {
    let Some(imdisk) = resolve_imdisk() else {
        return false;
    };
    silent_command(imdisk)
        .args(["-l", "-m", &format!("{drive_letter}:")])
        .status()
        .is_ok_and(|status| status.success())
}

#[cfg(windows)]
fn drive_capacity_mb(drive_letter: char) -> Option<u64> {
    use std::ffi::OsStr;
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;

    let root = wide_null(OsStr::new(&format!("{drive_letter}:\\")));
    let mut available = 0u64;
    let mut total = 0u64;
    let mut free = 0u64;
    let success =
        unsafe { GetDiskFreeSpaceExW(root.as_ptr(), &mut available, &mut total, &mut free) };
    (success != 0).then_some(total / (1024 * 1024))
}

#[cfg(not(windows))]
fn drive_capacity_mb(_drive_letter: char) -> Option<u64> {
    None
}

fn parse_drive_letter(value: &str) -> Option<char> {
    let mut chars = value.trim().chars();
    let letter = chars.next()?.to_ascii_uppercase();
    if !letter.is_ascii_alphabetic() {
        return None;
    }
    match chars.next() {
        None | Some(':') => Some(letter),
        _ => None,
    }
}

fn drive_root(drive_letter: char) -> PathBuf {
    PathBuf::from(format!("{drive_letter}:\\"))
}

fn cache_path(drive_letter: char) -> PathBuf {
    drive_root(drive_letter).join(CACHE_FOLDER)
}

fn setup_error_path() -> PathBuf {
    crate::paths::data_dir().join("ram-disk-setup-error.txt")
}

fn write_setup_error(message: &str) -> anyhow::Result<()> {
    crate::paths::ensure_data_dirs()?;
    fs::write(setup_error_path(), message).context("write RAM disk setup error")
}

#[cfg(windows)]
fn silent_command<S: AsRef<std::ffi::OsStr>>(program: S) -> Command {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x08000000;
    let mut command = Command::new(program);
    command.creation_flags(CREATE_NO_WINDOW);
    command
}

#[cfg(not(windows))]
fn silent_command<S: AsRef<std::ffi::OsStr>>(program: S) -> Command {
    Command::new(program)
}

#[cfg(windows)]
fn wide_null(value: &std::ffi::OsStr) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    value.encode_wide().chain([0]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_drive_letter_from_cache_path() {
        assert_eq!(parse_drive_letter(r"z:\TEMP"), Some('Z'));
        assert_eq!(parse_drive_letter("z"), Some('Z'));
        assert_eq!(parse_drive_letter("not-a-drive"), None);
    }

    #[test]
    fn cache_path_uses_requested_drive() {
        assert_eq!(cache_path('Y'), PathBuf::from(r"Y:\TEMP"));
    }
}
