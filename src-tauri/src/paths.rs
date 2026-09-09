use std::env;
use std::fs;
use std::path::{Path, PathBuf};

pub const ALLOWED_SOURCE: &str = r"\\EXAMPLE-NAS\Test";

/// Compare direct parent directories without requiring the files to be online.
pub fn same_parent_folder(left: &str, right: &str) -> bool {
    fn parent_key(path: &str) -> Option<String> {
        let normalized = path.replace('/', "\\").to_lowercase();
        normalized.rsplit_once('\\').map(|(parent, _)| parent.to_string())
    }
    match (parent_key(left), parent_key(right)) {
        (Some(left), Some(right)) => left == right,
        _ => false,
    }
}

pub fn workspace_root() -> PathBuf {
    if let Ok(value) = env::var("DVS_HOME") {
        return PathBuf::from(value);
    }

    if let Ok(exe) = env::current_exe() {
        if let Some(parent) = exe.parent() {
            for ancestor in parent.ancestors() {
                if ancestor.join("package.json").exists() && ancestor.join("src-tauri").exists() {
                    return ancestor.to_path_buf();
                }
            }
            if parent
                .file_name()
                .and_then(|value| value.to_str())
                .is_some_and(|name| name.eq_ignore_ascii_case("quick-release"))
            {
                if let Some(project_root) = parent.parent() {
                    if project_root.join("package.json").exists()
                        && project_root.join("src-tauri").exists()
                    {
                        return project_root.to_path_buf();
                    }
                }
            }
            return parent.to_path_buf();
        }
    }

    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("src-tauri must live under the workspace root")
        .to_path_buf()
}

pub fn data_dir() -> PathBuf {
    workspace_root().join("data")
}

pub fn database_path() -> PathBuf {
    data_dir().join("index.sqlite")
}

pub fn ensure_data_dirs() -> anyhow::Result<()> {
    fs::create_dir_all(data_dir().join("operations"))?;
    fs::create_dir_all(data_dir().join("reports"))?;
    fs::create_dir_all(data_dir().join("thumbnails"))?;
    fs::create_dir_all(data_dir().join("ai-frame-cache"))?;
    fs::create_dir_all(data_dir().join("nas-frame-import"))?;
    fs::create_dir_all(data_dir().join("local-video-staging"))?;
    fs::create_dir_all(data_dir().join("backups"))?;
    fs::create_dir_all(data_dir().join("tools"))?;
    Ok(())
}

pub fn is_allowed_scan_path(path: &Path) -> bool {
    let normalized = path.to_string_lossy().replace('/', "\\").to_lowercase();
    let allowed = ALLOWED_SOURCE.to_lowercase();
    let samples = workspace_root()
        .join("samples")
        .to_string_lossy()
        .replace('/', "\\")
        .to_lowercase();

    normalized == allowed
        || normalized.starts_with(&(allowed + "\\"))
        || normalized == samples
        || normalized.starts_with(&(samples + "\\"))
}
