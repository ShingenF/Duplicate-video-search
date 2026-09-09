use crate::media;
use crate::models::{AppSettings, NasPreprocessSummary, VideoRecord};
use crate::paths;
use anyhow::{anyhow, Context};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::thread;

#[derive(Debug, Clone)]
pub struct NasPreprocessProgress {
    pub total_videos: usize,
    pub imported: usize,
    pub skipped: usize,
    pub failed: usize,
    pub current_path: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RemoteRequest {
    version: u32,
    frame_count: usize,
    workers: usize,
    videos: Vec<RemoteVideoRequest>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RemoteVideoRequest {
    id: i64,
    smb_path: String,
    remote_path: String,
    size_bytes: u64,
    modified_unix_ms: i64,
    duration_seconds: Option<f64>,
    positions: Vec<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RemoteManifest {
    version: u32,
    videos: Vec<RemoteVideoManifest>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RemoteVideoManifest {
    id: i64,
    smb_path: String,
    status: String,
    error: Option<String>,
    size_bytes: Option<u64>,
    duration_seconds: Option<f64>,
    frames: Vec<RemoteFrameManifest>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RemoteFrameManifest {
    index: usize,
    timestamp_seconds: f64,
    file_name: String,
}

pub fn enabled(settings: &AppSettings) -> bool {
    settings.nas_ssh_preprocess_enabled
        && !settings.nas_ssh_host.trim().is_empty()
        && !settings.nas_remote_root.trim().is_empty()
}

pub fn preprocess_ai_frame_cache<F>(
    videos: &[VideoRecord],
    settings: &AppSettings,
    mut on_progress: F,
) -> anyhow::Result<NasPreprocessSummary>
where
    F: FnMut(NasPreprocessProgress),
{
    if !enabled(settings) {
        return Ok(NasPreprocessSummary {
            total_videos: 0,
            imported: 0,
            skipped: 0,
            failed: 0,
            bundle_dir: None,
            recent_errors: Vec::new(),
        });
    }

    let frame_count = settings.ai_frame_count.clamp(8, 512);
    let mut skipped = 0usize;
    let mut requests = Vec::new();
    for video in videos {
        if media::load_ai_frame_cache_for_video(video, frame_count)
            .ok()
            .flatten()
            .is_some()
        {
            skipped += 1;
            continue;
        }
        let Some(video_id) = video.id else {
            skipped += 1;
            continue;
        };
        let remote_path = map_smb_path_to_remote(&video.path, settings)?;
        requests.push(RemoteVideoRequest {
            id: video_id,
            smb_path: video.path.clone(),
            remote_path,
            size_bytes: video.size_bytes,
            modified_unix_ms: video.modified_unix_ms,
            duration_seconds: video.duration_seconds,
            positions: media::sample_positions(video.duration_seconds.unwrap_or(0.0), frame_count),
        });
    }

    let total_videos = requests.len();
    on_progress(NasPreprocessProgress {
        total_videos,
        imported: 0,
        skipped,
        failed: 0,
        current_path: Some("starting NAS SSH preprocessing".to_string()),
    });

    if requests.is_empty() {
        return Ok(NasPreprocessSummary {
            total_videos: 0,
            imported: 0,
            skipped,
            failed: 0,
            bundle_dir: None,
            recent_errors: Vec::new(),
        });
    }

    let import_root = paths::data_dir().join("nas-frame-import");
    fs::create_dir_all(&import_root)?;
    let run_dir = import_root.join(format!("run-{}", media::now_ms()));
    let payload_dir = run_dir.join("payload");
    fs::create_dir_all(&payload_dir)?;
    let bundle_path = run_dir.join("nas-frames.tar.gz");

    let request = RemoteRequest {
        version: 1,
        frame_count,
        workers: settings.nas_ssh_ffmpeg_workers.clamp(1, 4),
        videos: requests,
    };
    let request_json = serde_json::to_string(&request)?;
    let script = remote_script(&request_json);

    on_progress(NasPreprocessProgress {
        total_videos,
        imported: 0,
        skipped,
        failed: 0,
        current_path: Some("running ffprobe/ffmpeg on NAS".to_string()),
    });
    run_ssh_tar(settings, &script, &bundle_path)?;
    extract_tar(&bundle_path, &payload_dir)?;
    let _ = fs::remove_file(&bundle_path);

    let manifest_path = payload_dir.join("manifest.json");
    let manifest_file =
        File::open(&manifest_path).with_context(|| format!("open {}", manifest_path.display()))?;
    let manifest = serde_json::from_reader::<_, RemoteManifest>(manifest_file)
        .with_context(|| format!("parse {}", manifest_path.display()))?;
    if manifest.version != 1 {
        return Err(anyhow!(
            "unsupported NAS frame manifest version {}",
            manifest.version
        ));
    }

    let video_by_id = videos
        .iter()
        .filter_map(|video| video.id.map(|id| (id, video)))
        .collect::<HashMap<_, _>>();
    let mut imported = 0usize;
    let mut failed = 0usize;
    let mut recent_errors = Vec::new();

    for remote_video in manifest.videos {
        on_progress(NasPreprocessProgress {
            total_videos,
            imported,
            skipped,
            failed,
            current_path: Some(remote_video.smb_path.clone()),
        });

        if remote_video.status != "ok" {
            failed += 1;
            push_recent_error(
                &mut recent_errors,
                format!(
                    "{}: {}",
                    remote_video.smb_path,
                    remote_video
                        .error
                        .unwrap_or_else(|| "NAS preprocessing failed".to_string())
                ),
            );
            continue;
        }
        let Some(video) = video_by_id.get(&remote_video.id).copied() else {
            failed += 1;
            push_recent_error(
                &mut recent_errors,
                format!(
                    "{}: local video record was not found",
                    remote_video.smb_path
                ),
            );
            continue;
        };

        match import_remote_frames(&payload_dir, &remote_video, video, frame_count) {
            Ok(true) => imported += 1,
            Ok(false) => {
                failed += 1;
                push_recent_error(
                    &mut recent_errors,
                    format!("{}: incomplete NAS frame set", remote_video.smb_path),
                );
            }
            Err(error) => {
                failed += 1;
                push_recent_error(
                    &mut recent_errors,
                    format!("{}: {error}", remote_video.smb_path),
                );
            }
        }
    }

    on_progress(NasPreprocessProgress {
        total_videos,
        imported,
        skipped,
        failed,
        current_path: None,
    });

    Ok(NasPreprocessSummary {
        total_videos,
        imported,
        skipped,
        failed,
        bundle_dir: Some(run_dir.display().to_string()),
        recent_errors,
    })
}

fn import_remote_frames(
    payload_dir: &Path,
    remote_video: &RemoteVideoManifest,
    video: &VideoRecord,
    frame_count: usize,
) -> anyhow::Result<bool> {
    let mut frames = remote_video.frames.iter().collect::<Vec<_>>();
    frames.sort_by_key(|frame| frame.index);
    let mut positions = Vec::new();
    let mut rgb_frames = Vec::new();
    if let Some(remote_size) = remote_video.size_bytes {
        if remote_size != video.size_bytes {
            return Err(anyhow!(
                "remote file size differs from SMB metadata: {} vs {}",
                remote_size,
                video.size_bytes
            ));
        }
    }
    if let (Some(remote_duration), Some(local_duration)) =
        (remote_video.duration_seconds, video.duration_seconds)
    {
        if (remote_duration - local_duration).abs() > 1.0 {
            return Err(anyhow!(
                "remote duration differs from SMB metadata: {:.3}s vs {:.3}s",
                remote_duration,
                local_duration
            ));
        }
    }
    for frame in frames {
        let image_path =
            payload_dir.join(frame.file_name.replace('/', std::path::MAIN_SEPARATOR_STR));
        if !image_path.exists() {
            return Err(anyhow!(
                "imported frame is missing: {}",
                image_path.display()
            ));
        }
        positions.push(frame.timestamp_seconds);
        rgb_frames.push(media::convert_image_to_ai_rgb_frame(&image_path)?);
    }
    Ok(media::write_ai_frame_cache_for_video(video, frame_count, positions, rgb_frames)?.is_some())
}

fn map_smb_path_to_remote(path: &str, settings: &AppSettings) -> anyhow::Result<String> {
    let smb_root = if settings.nas_smb_root.trim().is_empty() {
        paths::ALLOWED_SOURCE
    } else {
        settings.nas_smb_root.trim()
    };
    let path_key = normalize_windows_path(path);
    let root_key = normalize_windows_path(smb_root);
    let relative = if path_key == root_key {
        String::new()
    } else if let Some(rest) = path_key.strip_prefix(&(root_key.clone() + "\\")) {
        rest.to_string()
    } else {
        return Err(anyhow!(
            "SMB path is outside configured NAS SMB root: {} not under {}",
            path,
            smb_root
        ));
    };
    let remote_root = settings.nas_remote_root.trim().trim_end_matches('/');
    if remote_root.is_empty() {
        return Err(anyhow!("NAS remote root is empty"));
    }
    if relative.is_empty() {
        Ok(remote_root.to_string())
    } else {
        Ok(format!("{}/{}", remote_root, relative.replace('\\', "/")))
    }
}

fn normalize_windows_path(path: &str) -> String {
    path.replace('/', "\\").trim_end_matches('\\').to_string()
}

fn run_ssh_tar(settings: &AppSettings, script: &str, bundle_path: &Path) -> anyhow::Result<()> {
    if !settings.nas_ssh_password.is_empty() {
        return run_ssh_tar_with_plink(settings, script, bundle_path);
    }

    let target = if settings.nas_ssh_user.trim().is_empty() {
        settings.nas_ssh_host.trim().to_string()
    } else {
        format!(
            "{}@{}",
            settings.nas_ssh_user.trim(),
            settings.nas_ssh_host.trim()
        )
    };
    let mut child = media::silent_command("ssh")
        .args([
            "-p",
            &settings.nas_ssh_port.to_string(),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=12",
            &target,
            "bash",
            "-s",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("start ssh for NAS preprocessing")?;

    {
        let mut stdin = child.stdin.take().context("open ssh stdin")?;
        stdin
            .write_all(script.as_bytes())
            .context("write NAS preprocessing script to ssh")?;
    }

    let mut stdout = child.stdout.take().context("open ssh stdout")?;
    let out_path = bundle_path.to_path_buf();
    let stdout_handle = thread::spawn(move || -> anyhow::Result<()> {
        let mut file =
            File::create(&out_path).with_context(|| format!("create {}", out_path.display()))?;
        std::io::copy(&mut stdout, &mut file).context("copy NAS frame tar stream")?;
        Ok(())
    });

    let mut stderr = child.stderr.take().context("open ssh stderr")?;
    let stderr_handle = thread::spawn(move || {
        let mut text = String::new();
        let _ = stderr.read_to_string(&mut text);
        text
    });

    let status = child.wait().context("wait for NAS SSH preprocessing")?;
    stdout_handle
        .join()
        .map_err(|_| anyhow!("NAS frame tar copy thread panicked"))??;
    let stderr_text = stderr_handle
        .join()
        .map_err(|_| anyhow!("NAS stderr thread panicked"))?;

    if !status.success() {
        let _ = fs::remove_file(bundle_path);
        return Err(anyhow!(
            "NAS SSH preprocessing failed: {}",
            stderr_text.lines().take(6).collect::<Vec<_>>().join(" | ")
        ));
    }
    if !bundle_path.exists() || bundle_path.metadata()?.len() == 0 {
        return Err(anyhow!("NAS SSH preprocessing returned an empty bundle"));
    }
    Ok(())
}

fn run_ssh_tar_with_plink(
    settings: &AppSettings,
    script: &str,
    bundle_path: &Path,
) -> anyhow::Result<()> {
    let plink = resolve_plink().context("plink.exe was not found in tools or PATH")?;
    let target = plink_target(settings)?;
    if settings.nas_ssh_host_key.trim().is_empty() {
        ensure_plink_host_key(settings, &plink, &target)?;
    }

    let mut child = media::silent_command(&plink)
        .args(plink_args(settings, &target, true))
        .args(["bash", "-s"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("start plink for NAS preprocessing")?;

    {
        let mut stdin = child.stdin.take().context("open plink stdin")?;
        stdin
            .write_all(script.as_bytes())
            .context("write NAS preprocessing script to plink")?;
    }

    let mut stdout = child.stdout.take().context("open plink stdout")?;
    let out_path = bundle_path.to_path_buf();
    let stdout_handle = thread::spawn(move || -> anyhow::Result<()> {
        let mut file =
            File::create(&out_path).with_context(|| format!("create {}", out_path.display()))?;
        std::io::copy(&mut stdout, &mut file).context("copy NAS frame tar stream")?;
        Ok(())
    });

    let mut stderr = child.stderr.take().context("open plink stderr")?;
    let stderr_handle = thread::spawn(move || {
        let mut text = String::new();
        let _ = stderr.read_to_string(&mut text);
        text
    });

    let status = child.wait().context("wait for NAS plink preprocessing")?;
    stdout_handle
        .join()
        .map_err(|_| anyhow!("NAS frame tar copy thread panicked"))??;
    let stderr_text = stderr_handle
        .join()
        .map_err(|_| anyhow!("NAS stderr thread panicked"))?;

    if !status.success() {
        let _ = fs::remove_file(bundle_path);
        return Err(anyhow!(
            "NAS plink preprocessing failed: {}",
            stderr_text.lines().take(6).collect::<Vec<_>>().join(" | ")
        ));
    }
    if !bundle_path.exists() || bundle_path.metadata()?.len() == 0 {
        return Err(anyhow!("NAS plink preprocessing returned an empty bundle"));
    }
    Ok(())
}

fn ensure_plink_host_key(settings: &AppSettings, plink: &Path, target: &str) -> anyhow::Result<()> {
    let mut child = media::silent_command(plink)
        .args(plink_args(settings, target, false))
        .arg("exit")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("start plink host key check")?;
    {
        let mut stdin = child.stdin.take().context("open plink host key stdin")?;
        stdin.write_all(b"y\n").ok();
    }
    let mut stderr = String::new();
    if let Some(mut pipe) = child.stderr.take() {
        let _ = pipe.read_to_string(&mut stderr);
    }
    let status = child.wait().context("wait for plink host key check")?;
    if !status.success() {
        return Err(anyhow!(
            "plink host key check failed: {}",
            stderr.lines().take(6).collect::<Vec<_>>().join(" | ")
        ));
    }
    Ok(())
}

fn plink_args(settings: &AppSettings, target: &str, batch: bool) -> Vec<String> {
    let mut args = vec![
        "-ssh".to_string(),
        "-P".to_string(),
        settings.nas_ssh_port.to_string(),
        "-pw".to_string(),
        settings.nas_ssh_password.clone(),
    ];
    let host_key = settings.nas_ssh_host_key.trim();
    if !host_key.is_empty() {
        args.push("-hostkey".to_string());
        args.push(host_key.to_string());
    }
    if batch {
        args.push("-batch".to_string());
    }
    args.push(target.to_string());
    args
}

fn plink_target(settings: &AppSettings) -> anyhow::Result<String> {
    if settings.nas_ssh_user.trim().is_empty() {
        return Err(anyhow!("NAS SSH user is empty when password login is used"));
    }
    if settings.nas_ssh_host.trim().is_empty() {
        return Err(anyhow!("NAS SSH host is empty"));
    }
    Ok(format!(
        "{}@{}",
        settings.nas_ssh_user.trim(),
        settings.nas_ssh_host.trim()
    ))
}

fn resolve_plink() -> Option<PathBuf> {
    let local = paths::workspace_root().join("tools").join("plink.exe");
    if local.exists() {
        return Some(local);
    }
    if let Ok(output) = media::silent_command("where.exe").arg("plink").output() {
        if output.status.success() {
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                let path = PathBuf::from(line.trim());
                if path.exists() {
                    return Some(path);
                }
            }
        }
    }
    None
}

fn extract_tar(bundle_path: &Path, payload_dir: &Path) -> anyhow::Result<()> {
    let output = media::silent_command("tar")
        .arg("-xzf")
        .arg(bundle_path)
        .arg("-C")
        .arg(payload_dir)
        .output()
        .with_context(|| format!("extract {}", bundle_path.display()))?;
    if !output.status.success() {
        return Err(anyhow!(
            "extract NAS frame bundle failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}

fn push_recent_error(errors: &mut Vec<String>, message: String) {
    errors.push(message);
    if errors.len() > 8 {
        errors.remove(0);
    }
}

fn remote_script(request_json: &str) -> String {
    let mut script = String::new();
    script.push_str(
        r#"set -euo pipefail
tmp="$(mktemp -d /tmp/dvs-nas-preprocess.XXXXXX)"
worker_pid=""
cleanup() {
  trap - EXIT HUP INT TERM
  if [ -n "$worker_pid" ]; then
    pkill -TERM -P "$worker_pid" 2>/dev/null || true
    kill "$worker_pid" 2>/dev/null || true
    wait "$worker_pid" 2>/dev/null || true
  fi
  rm -rf "$tmp"
}
trap cleanup EXIT HUP INT TERM
mkdir -p "$tmp/out/frames"
cat > "$tmp/request.json" <<'DVS_JSON'
"#,
    );
    script.push_str(request_json);
    script.push_str(
        r#"
DVS_JSON
cat > "$tmp/worker.py" <<'DVS_PY'
import concurrent.futures
import json
import os
import subprocess
import sys

request_path = sys.argv[1]
out_dir = sys.argv[2]
with open(request_path, "r", encoding="utf-8") as f:
    request = json.load(f)

workers = max(1, min(int(request.get("workers", 1)), 4))

def parse_float(value):
    try:
        return float(value)
    except Exception:
        return None

def parse_int(value):
    try:
        return int(float(value))
    except Exception:
        return None

def probe(remote_path):
    output = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            remote_path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if output.returncode != 0:
        raise RuntimeError(output.stderr.strip() or "ffprobe failed")
    data = json.loads(output.stdout or "{}")
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    fmt = data.get("format") or {}
    return {
        "durationSeconds": parse_float(fmt.get("duration")) or parse_float(video.get("duration")),
        "width": video.get("width"),
        "height": video.get("height"),
        "codec": video.get("codec_name"),
        "bitrate": parse_int(video.get("bit_rate")) or parse_int(fmt.get("bit_rate")),
    }

def process_video(item):
    result = {
        "id": item.get("id"),
        "smbPath": item.get("smbPath"),
        "remotePath": item.get("remotePath"),
        "status": "failed",
        "error": None,
        "sizeBytes": None,
        "modifiedUnixMs": None,
        "durationSeconds": item.get("durationSeconds"),
        "width": None,
        "height": None,
        "codec": None,
        "bitrate": None,
        "frames": [],
    }
    remote_path = item.get("remotePath")
    try:
        stat = os.stat(remote_path)
        result["sizeBytes"] = stat.st_size
        result["modifiedUnixMs"] = int(stat.st_mtime * 1000)
        result.update(probe(remote_path))
        video_dir = os.path.join(out_dir, "frames", str(item.get("id")))
        os.makedirs(video_dir, exist_ok=True)
        for index, position in enumerate(item.get("positions") or []):
            rel_name = "frames/{}/frame_{:04d}.jpg".format(item.get("id"), index)
            abs_name = os.path.join(out_dir, rel_name)
            ffmpeg = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    "{:.3f}".format(float(position)),
                    "-i",
                    remote_path,
                    "-frames:v",
                    "1",
                    "-vf",
                    "scale=-2:720:flags=lanczos",
                    "-q:v",
                    "3",
                    abs_name,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if ffmpeg.returncode == 0 and os.path.exists(abs_name):
                result["frames"].append(
                    {
                        "index": index,
                        "timestampSeconds": float(position),
                        "fileName": rel_name,
                    }
                )
            else:
                raise RuntimeError(ffmpeg.stderr.strip() or "ffmpeg frame extraction failed")
        result["status"] = "ok" if result["frames"] else "failed"
        if result["status"] != "ok":
            result["error"] = "no frames were extracted"
    except Exception as exc:
        result["error"] = str(exc)
    return result

videos = request.get("videos") or []
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
    results = list(executor.map(process_video, videos))

manifest = {"version": 1, "videos": results}
with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False)
DVS_PY
python3 "$tmp/worker.py" "$tmp/request.json" "$tmp/out" &
worker_pid="$!"
wait "$worker_pid"
worker_pid=""
tar -czf - -C "$tmp/out" .
"#,
    );
    script
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> AppSettings {
        AppSettings {
            compare_within_same_folder: false,
            ui_language: "zh".to_string(),
            backup_dir: String::new(),
            naming_source_dirs: Vec::new(),
            keeper_size_priority_duration_seconds: 300.0,
            scan_worker_count: 1,
            sample_hash_count: 11,
            temporal_hash_threshold: 8,
            min_temporal_match_points: 3,
            allowed_unmatched_sample_frames: 14,
            allow_direct_delete: false,
            restrict_scan_to_test_path: true,
            ai_vision_enabled: true,
            ai_model_path: String::new(),
            ai_device: "auto".to_string(),
            ai_frame_count: 128,
            ai_batch_size: 32,
            ai_similarity_threshold: 0.86,
            ai_min_matched_frames: 8,
            ai_clip_matching_enabled: true,
            ai_index_after_scan: false,
            ai_extract_worker_count: 4,
            ai_gpu_worker_count: 4,
            ai_match_worker_count: 8,
            ai_frame_cache_enabled: true,
            delete_ai_frame_cache_after_index: false,
            ram_disk_enabled: true,
            ram_disk_size_mb: 16 * 1024,
            ram_disk_setup_completed: false,
            local_preprocess_enabled: false,
            local_preprocess_video_workers: 1,
            local_preprocess_overlap_start_percent: 95,
            local_preprocess_process_workers: 2,
            local_preprocess_frame_workers: 16,
            local_preprocess_temp_dir: String::new(),
            local_preprocess_secondary_temp_dir: String::new(),
            local_preprocess_secondary_threshold_mb: 16 * 1024,
            nas_ssh_preprocess_enabled: true,
            nas_ssh_host: "nas".to_string(),
            nas_ssh_user: "user".to_string(),
            nas_ssh_password: String::new(),
            nas_ssh_host_key: String::new(),
            nas_ssh_port: 22,
            nas_remote_root: "/mnt/pool/Test".to_string(),
            nas_smb_root: r"\\EXAMPLE-NAS\Test".to_string(),
            nas_ssh_ffmpeg_workers: 1,
        }
    }

    #[test]
    fn maps_smb_path_to_remote_root() {
        let settings = settings();
        assert_eq!(
            map_smb_path_to_remote(r"\\EXAMPLE-NAS\Test\Example\a.mp4", &settings).unwrap(),
            "/mnt/pool/Test/Example/a.mp4"
        );
    }

    #[test]
    fn rejects_paths_outside_smb_root() {
        let settings = settings();
        assert!(map_smb_path_to_remote(r"\\EXAMPLE-NAS\Other\a.mp4", &settings).is_err());
    }
}
