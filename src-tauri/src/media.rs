use crate::database;
use crate::models::{ScanProgress, ScanSummary, ToolStatus, VideoRecord};
use crate::paths;
use anyhow::{anyhow, Context};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{HashSet, VecDeque};
use std::env;
use std::ffi::OsStr;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{self, Command};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use walkdir::WalkDir;

const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "mov", "m4v", "avi", "mkv", "mts", "m2ts", "wmv", "mpg", "mpeg", "3gp", "webm",
];
pub(crate) const AI_FRAME_SIZE: usize = 224;
pub(crate) const AI_FRAME_BYTES: usize = AI_FRAME_SIZE * AI_FRAME_SIZE * 3;
const AI_FRAME_SEEK_FALLBACK_OFFSETS: &[f64] = &[
    -0.25, 0.25, -0.5, 0.5, -1.0, 1.0, -2.0, 2.0, -5.0, 5.0, -10.0, 10.0, -20.0, 20.0, -30.0, 30.0,
];

#[derive(Debug, Clone)]
pub(crate) struct AiFramePack {
    pub positions: Vec<f64>,
    pub frames: Vec<Vec<u8>>,
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct FileCleanupStats {
    pub files: usize,
    pub bytes: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct AiFrameCacheManifest {
    version: u32,
    source_path: String,
    frame_size: usize,
    pixel_format: String,
    configured_frame_count: usize,
    positions: Vec<f64>,
    data_file: String,
}

#[derive(Debug, Deserialize)]
struct ProbeOutput {
    streams: Option<Vec<ProbeStream>>,
    format: Option<ProbeFormat>,
}

#[derive(Debug, Deserialize)]
struct ProbeStream {
    codec_type: Option<String>,
    codec_name: Option<String>,
    width: Option<u32>,
    height: Option<u32>,
    avg_frame_rate: Option<String>,
    bit_rate: Option<String>,
    duration: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ProbeFormat {
    format_name: Option<String>,
    duration: Option<String>,
    bit_rate: Option<String>,
}

#[derive(Debug, Clone)]
struct MediaInfo {
    container_format: Option<String>,
    duration_seconds: Option<f64>,
    width: Option<u32>,
    height: Option<u32>,
    bitrate: Option<u64>,
    codec: Option<String>,
    frame_rate: Option<f64>,
    audio_codec: Option<String>,
}

pub fn tool_status() -> anyhow::Result<ToolStatus> {
    paths::ensure_data_dirs()?;
    let ffmpeg = resolve_tool("ffmpeg");
    let ffprobe = resolve_tool("ffprobe");

    Ok(ToolStatus {
        workspace_root: paths::workspace_root().display().to_string(),
        data_dir: paths::data_dir().display().to_string(),
        database_path: paths::database_path().display().to_string(),
        allowed_source: paths::ALLOWED_SOURCE.to_string(),
        ffmpeg_version: ffmpeg
            .as_ref()
            .and_then(|path| command_first_line(path, &["-version"])),
        ffprobe_version: ffprobe
            .as_ref()
            .and_then(|path| command_first_line(path, &["-version"])),
        ffmpeg_path: ffmpeg.map(|path| path.display().to_string()),
        ffprobe_path: ffprobe.map(|path| path.display().to_string()),
    })
}

pub fn scan_source(source: &str) -> anyhow::Result<ScanSummary> {
    scan_source_with_progress(source, |_| {})
}

pub fn scan_sources_with_progress<F>(
    sources: &[String],
    mut on_progress: F,
) -> anyhow::Result<Vec<ScanSummary>>
where
    F: FnMut(ScanProgress),
{
    let mut summaries = Vec::new();
    for source in sources {
        crate::cancel::bail_if_requested()?;
        summaries.push(scan_source_with_progress(source, |progress| {
            on_progress(progress);
        })?);
    }
    Ok(summaries)
}

pub fn scan_source_with_progress<F>(source: &str, mut on_progress: F) -> anyhow::Result<ScanSummary>
where
    F: FnMut(ScanProgress),
{
    let source_path = PathBuf::from(source);
    let source_input_path = filesystem_input_path(&source_path);
    let source_text = display_path_text(&source_path);
    let settings = crate::operations::get_app_settings()?;
    if !crate::operations::is_path_allowed_by_settings(&source_path, &settings) {
        return Err(anyhow!(
            "scan path is outside the allowed source: {}",
            paths::ALLOWED_SOURCE
        ));
    }
    if !source_input_path.exists() {
        return Err(anyhow!("source path does not exist: {}", source));
    }

    let started = Instant::now();
    let conn = database::open_database()?;
    let session_id = database::create_or_reset_scan_session(&conn, &source_text, now_ms())?;
    on_progress(ScanProgress {
        source: source_text.clone(),
        phase: "counting".to_string(),
        total_files: 0,
        scanned: 0,
        failed: 0,
        current_path: None,
    });

    let excluded_dirs = vec![PathBuf::from(&settings.backup_dir)];
    let files = collect_video_files(&source_input_path, &excluded_dirs);
    let total_files = files.len();
    let mut scanned = 0usize;
    let mut failed = 0usize;

    on_progress(ScanProgress {
        source: source_text.clone(),
        phase: "scanning".to_string(),
        total_files,
        scanned,
        failed,
        current_path: None,
    });

    let mut pending_scans = Vec::new();
    let scan_result = (|| -> anyhow::Result<()> {
        for path in files {
            crate::cancel::bail_if_requested()?;
            on_progress(ScanProgress {
                source: source_text.clone(),
                phase: "scanning".to_string(),
                total_files,
                scanned,
                failed,
                current_path: Some(display_path_text(&path)),
            });

            if settings.local_preprocess_enabled {
                match lightweight_video_record(&path) {
                    Ok(video) => {
                        let existing = database::find_video_by_path(&conn, &video.path)?;
                        let video_id = if let Some(existing) =
                            existing.filter(|existing| same_file_observation(existing, &video))
                        {
                            existing
                                .id
                                .ok_or_else(|| anyhow!("cached video is missing an id"))?
                        } else if database::find_video_by_path(&conn, &video.path)?.is_some() {
                            database::upsert_video(&conn, &video)?
                        } else if let Some(migrated_id) =
                            migrate_existing_content_identity(&conn, &path)?
                        {
                            migrated_id
                        } else {
                            database::upsert_video(&conn, &video)?
                        };
                        database::link_video_to_scan_session(&conn, session_id, video_id)?;
                        scanned += 1;
                    }
                    Err(error) => {
                        failed += 1;
                        let video = failed_record(&path, error.to_string());
                        let video_id = database::upsert_video(&conn, &video)?;
                        database::link_video_to_scan_session(&conn, session_id, video_id)?;
                    }
                }

                on_progress(ScanProgress {
                    source: source_text.clone(),
                    phase: "listing-local-staging".to_string(),
                    total_files,
                    scanned,
                    failed,
                    current_path: Some(display_path_text(&path)),
                });
                continue;
            }

            match current_indexed_video(&conn, &path, &settings) {
                Ok(Some(video)) => {
                    let video_id = video
                        .id
                        .ok_or_else(|| anyhow!("cached video is missing an id"))?;
                    database::link_video_to_scan_session(&conn, session_id, video_id)?;
                    scanned += 1;
                }
                Ok(None) => {
                    if database::find_video_by_path(&conn, &display_path_text(&path))?.is_some() {
                        pending_scans.push(path.clone());
                    } else if let Some(video_id) = migrate_existing_content_identity(&conn, &path)?
                    {
                        database::link_video_to_scan_session(&conn, session_id, video_id)?;
                        scanned += 1;
                    } else {
                        pending_scans.push(path.clone());
                    }
                }
                Err(error) => {
                    failed += 1;
                    let video = failed_record(&path, error.to_string());
                    let video_id = database::upsert_video(&conn, &video)?;
                    database::link_video_to_scan_session(&conn, session_id, video_id)?;
                }
            }

            on_progress(ScanProgress {
                source: source_text.clone(),
                phase: "scanning".to_string(),
                total_files,
                scanned,
                failed,
                current_path: Some(display_path_text(&path)),
            });
        }

        let worker_count = settings
            .scan_worker_count
            .clamp(1, 4)
            .min(pending_scans.len().max(1));
        if !pending_scans.is_empty() {
            let queue = Arc::new(Mutex::new(VecDeque::from(pending_scans)));
            let (sender, receiver) = mpsc::channel();
            let worker_settings = settings.clone();
            for _ in 0..worker_count {
                let queue = Arc::clone(&queue);
                let sender = sender.clone();
                let worker_settings = worker_settings.clone();
                thread::spawn(move || loop {
                    if crate::cancel::is_requested() {
                        break;
                    }
                    let path = {
                        let mut queue = queue.lock().expect("scan queue lock poisoned");
                        queue.pop_front()
                    };
                    let Some(path) = path else {
                        break;
                    };
                    let result = crate::cancel::bail_if_requested()
                        .and_then(|_| scan_video(&path, &worker_settings))
                        .map_err(|error| error.to_string());
                    if sender.send((path, result)).is_err() {
                        break;
                    }
                });
            }
            drop(sender);

            for (path, result) in receiver {
                crate::cancel::bail_if_requested()?;
                on_progress(ScanProgress {
                    source: source_text.clone(),
                    phase: format!("scanning x{worker_count}"),
                    total_files,
                    scanned,
                    failed,
                    current_path: Some(display_path_text(&path)),
                });
                match result {
                    Ok(video) => {
                        let video_id =
                            database::upsert_video_preserving_content_identity(&conn, &video)?;
                        database::link_video_to_scan_session(&conn, session_id, video_id)?;
                        scanned += 1;
                    }
                    Err(error) => {
                        failed += 1;
                        let video = failed_record(&path, error);
                        let video_id = database::upsert_video(&conn, &video)?;
                        database::link_video_to_scan_session(&conn, session_id, video_id)?;
                    }
                }
                on_progress(ScanProgress {
                    source: source_text.clone(),
                    phase: format!("scanning x{worker_count}"),
                    total_files,
                    scanned,
                    failed,
                    current_path: Some(display_path_text(&path)),
                });
            }
        }

        Ok(())
    })();

    match scan_result {
        Ok(()) => {}
        Err(error) if crate::cancel::is_requested() => {
            on_progress(ScanProgress {
                source: source_text.clone(),
                phase: "cancelled".to_string(),
                total_files,
                scanned,
                failed,
                current_path: None,
            });
            database::complete_scan_session(
                &conn,
                session_id,
                now_ms(),
                total_files,
                scanned,
                failed,
            )?;
            return Err(error);
        }
        Err(error) => return Err(error),
    }

    on_progress(ScanProgress {
        source: source_text.clone(),
        phase: "completed".to_string(),
        total_files,
        scanned,
        failed,
        current_path: None,
    });
    database::complete_scan_session(&conn, session_id, now_ms(), total_files, scanned, failed)?;
    let _ = database::delete_orphan_videos(&conn);

    Ok(ScanSummary {
        session_id: Some(session_id),
        source: source_text,
        database_path: paths::database_path().display().to_string(),
        total_files,
        scanned,
        failed,
        elapsed_ms: started.elapsed().as_millis(),
    })
}

fn current_indexed_video(
    conn: &rusqlite::Connection,
    path: &Path,
    settings: &crate::models::AppSettings,
) -> anyhow::Result<Option<VideoRecord>> {
    let metadata = path
        .metadata()
        .with_context(|| format!("read metadata for {}", path.display()))?;
    let modified_unix_ms = metadata
        .modified()
        .ok()
        .and_then(system_time_to_ms)
        .unwrap_or_else(now_ms);
    let path_text = display_path_text(path);
    let Some(video) = database::find_video_by_path(conn, &path_text)? else {
        return Ok(None);
    };

    if video.scan_status != "ok"
        || video.size_bytes != metadata.len()
        || video.modified_unix_ms != modified_unix_ms
        || video.sample_hashes.len()
            < expected_sample_hash_count(video.duration_seconds, settings.sample_hash_count)
        || video.preview_images.is_empty()
        || video
            .preview_images
            .iter()
            .any(|preview| !PathBuf::from(preview).exists())
    {
        return Ok(None);
    }

    if settings.ai_vision_enabled
        && settings.ai_frame_cache_enabled
        && !settings.nas_ssh_preprocess_enabled
        && !settings.local_preprocess_enabled
    {
        match load_ai_frame_cache_for_video(&video, settings.ai_frame_count) {
            Ok(Some(pack)) if !pack.frames.is_empty() => {}
            _ => return Ok(None),
        }
    }

    Ok(Some(video))
}

fn migrate_existing_content_identity(
    conn: &rusqlite::Connection,
    path: &Path,
) -> anyhow::Result<Option<i64>> {
    let identity = match content_identity_record(path) {
        Ok(identity) => identity,
        Err(_) => return Ok(None),
    };
    database::migrate_video_by_content_identity(conn, &identity)
}

fn collect_video_files(source_path: &Path, excluded_dirs: &[PathBuf]) -> Vec<PathBuf> {
    WalkDir::new(source_path)
        .follow_links(false)
        .into_iter()
        .filter_entry(|entry| {
            !entry.file_type().is_dir() || !is_inside_any_excluded_dir(entry.path(), excluded_dirs)
        })
        .filter_map(Result::ok)
        .filter(|entry| {
            entry.file_type().is_file()
                && is_video_path(entry.path())
                && !is_inside_any_excluded_dir(entry.path(), excluded_dirs)
        })
        .map(|entry| entry.path().to_path_buf())
        .collect()
}

fn is_inside_any_excluded_dir(path: &Path, excluded_dirs: &[PathBuf]) -> bool {
    let path_key = normalized_path_key(path);
    excluded_dirs
        .iter()
        .filter(|dir| !dir.as_os_str().is_empty())
        .map(|dir| normalized_path_key(dir))
        .any(|dir_key| path_key == dir_key || path_key.starts_with(&(dir_key + "\\")))
}

fn normalized_path_key(path: &Path) -> String {
    display_path_text(path)
        .replace('/', "\\")
        .trim_end_matches('\\')
        .to_lowercase()
}

fn scan_video(path: &Path, settings: &crate::models::AppSettings) -> anyhow::Result<VideoRecord> {
    let metadata = path
        .metadata()
        .with_context(|| format!("read metadata for {}", path.display()))?;
    let modified_unix_ms = metadata
        .modified()
        .ok()
        .and_then(system_time_to_ms)
        .unwrap_or_else(now_ms);
    let size_bytes = metadata.len();
    let media_info = probe_media(path)?;
    let partial_hash = partial_file_hash(path, size_bytes).ok();
    let sample_hashes = sample_hashes(
        path,
        media_info.duration_seconds,
        settings.sample_hash_count,
    )
    .unwrap_or_default();
    let preview_images = preview_images(path, media_info.duration_seconds, partial_hash.as_deref())
        .unwrap_or_default();
    if settings.ai_vision_enabled
        && settings.ai_frame_cache_enabled
        && !settings.nas_ssh_preprocess_enabled
        && !settings.local_preprocess_enabled
    {
        let _ = ensure_ai_frame_cache(
            path,
            media_info.duration_seconds,
            partial_hash.as_deref(),
            size_bytes,
            modified_unix_ms,
            settings.ai_frame_count,
            settings.ai_extract_worker_count,
        );
    }
    let quality_score = quality_score(&media_info, size_bytes);

    Ok(VideoRecord {
        id: None,
        path: display_path_text(path),
        file_name: path
            .file_name()
            .map(|value| value.to_string_lossy().to_string())
            .unwrap_or_default(),
        parent_path: path.parent().map(display_path_text).unwrap_or_default(),
        size_bytes,
        modified_unix_ms,
        extension: path
            .extension()
            .map(|value| value.to_string_lossy().to_lowercase())
            .unwrap_or_default(),
        container_format: media_info.container_format,
        duration_seconds: media_info.duration_seconds,
        width: media_info.width,
        height: media_info.height,
        bitrate: media_info.bitrate,
        codec: media_info.codec,
        frame_rate: media_info.frame_rate,
        audio_codec: media_info.audio_codec,
        partial_hash,
        sample_hashes,
        preview_images,
        scan_status: "ok".to_string(),
        error: None,
        quality_score,
        scanned_at_unix_ms: now_ms(),
    })
}

fn content_identity_record(path: &Path) -> anyhow::Result<VideoRecord> {
    let metadata = path
        .metadata()
        .with_context(|| format!("read metadata for {}", path.display()))?;
    let modified_unix_ms = metadata
        .modified()
        .ok()
        .and_then(system_time_to_ms)
        .unwrap_or_else(now_ms);
    let size_bytes = metadata.len();
    let media_info = probe_media(path)?;
    let partial_hash = partial_file_hash(path, size_bytes).ok();
    let quality_score = quality_score(&media_info, size_bytes);

    Ok(VideoRecord {
        id: None,
        path: display_path_text(path),
        file_name: path
            .file_name()
            .map(|value| value.to_string_lossy().to_string())
            .unwrap_or_default(),
        parent_path: path.parent().map(display_path_text).unwrap_or_default(),
        size_bytes,
        modified_unix_ms,
        extension: path
            .extension()
            .map(|value| value.to_string_lossy().to_lowercase())
            .unwrap_or_default(),
        container_format: media_info.container_format,
        duration_seconds: media_info.duration_seconds,
        width: media_info.width,
        height: media_info.height,
        bitrate: media_info.bitrate,
        codec: media_info.codec,
        frame_rate: media_info.frame_rate,
        audio_codec: media_info.audio_codec,
        partial_hash,
        sample_hashes: Vec::new(),
        preview_images: Vec::new(),
        scan_status: "ok".to_string(),
        error: None,
        quality_score,
        scanned_at_unix_ms: now_ms(),
    })
}

pub(crate) fn scan_video_from_local_copy(
    original_path: &Path,
    local_path: &Path,
    original_modified_unix_ms: i64,
    settings: &crate::models::AppSettings,
) -> anyhow::Result<VideoRecord> {
    let metadata = local_path
        .metadata()
        .with_context(|| format!("read metadata for {}", local_path.display()))?;
    let size_bytes = metadata.len();
    let media_info = probe_media(local_path)?;
    let partial_hash = partial_file_hash(local_path, size_bytes).ok();
    let sample_hashes = sample_hashes(
        local_path,
        media_info.duration_seconds,
        settings.sample_hash_count,
    )
    .unwrap_or_default();
    let preview_images = preview_images(
        local_path,
        media_info.duration_seconds,
        partial_hash.as_deref(),
    )
    .unwrap_or_default();
    let quality_score = quality_score(&media_info, size_bytes);

    Ok(VideoRecord {
        id: None,
        path: display_path_text(original_path),
        file_name: original_path
            .file_name()
            .map(|value| value.to_string_lossy().to_string())
            .unwrap_or_default(),
        parent_path: original_path
            .parent()
            .map(display_path_text)
            .unwrap_or_default(),
        size_bytes,
        modified_unix_ms: original_modified_unix_ms,
        extension: original_path
            .extension()
            .map(|value| value.to_string_lossy().to_lowercase())
            .unwrap_or_default(),
        container_format: media_info.container_format,
        duration_seconds: media_info.duration_seconds,
        width: media_info.width,
        height: media_info.height,
        bitrate: media_info.bitrate,
        codec: media_info.codec,
        frame_rate: media_info.frame_rate,
        audio_codec: media_info.audio_codec,
        partial_hash,
        sample_hashes,
        preview_images,
        scan_status: "ok".to_string(),
        error: None,
        quality_score,
        scanned_at_unix_ms: now_ms(),
    })
}

pub(crate) fn video_has_full_scan(
    video: &VideoRecord,
    settings: &crate::models::AppSettings,
) -> bool {
    video.scan_status == "ok"
        && video.size_bytes > 0
        && video.duration_seconds.is_some()
        && video.partial_hash.is_some()
        && video.sample_hashes.len()
            >= expected_sample_hash_count(video.duration_seconds, settings.sample_hash_count)
        && !video.preview_images.is_empty()
        && video
            .preview_images
            .iter()
            .all(|preview| PathBuf::from(preview).exists())
}

fn same_file_observation(existing: &VideoRecord, observed: &VideoRecord) -> bool {
    existing.size_bytes == observed.size_bytes
        && (existing.modified_unix_ms - observed.modified_unix_ms).abs() <= 2_000
}

fn lightweight_video_record(path: &Path) -> anyhow::Result<VideoRecord> {
    let metadata = path
        .metadata()
        .with_context(|| format!("read metadata for {}", path.display()))?;
    if !metadata.is_file() {
        return Err(anyhow!("source is not a file: {}", path.display()));
    }
    let modified_unix_ms = metadata
        .modified()
        .ok()
        .and_then(system_time_to_ms)
        .unwrap_or_else(now_ms);

    Ok(VideoRecord {
        id: None,
        path: display_path_text(path),
        file_name: path
            .file_name()
            .map(|value| value.to_string_lossy().to_string())
            .unwrap_or_default(),
        parent_path: path.parent().map(display_path_text).unwrap_or_default(),
        size_bytes: metadata.len(),
        modified_unix_ms,
        extension: path
            .extension()
            .map(|value| value.to_string_lossy().to_lowercase())
            .unwrap_or_default(),
        container_format: None,
        duration_seconds: None,
        width: None,
        height: None,
        bitrate: None,
        codec: None,
        frame_rate: None,
        audio_codec: None,
        partial_hash: None,
        sample_hashes: Vec::new(),
        preview_images: Vec::new(),
        scan_status: "ok".to_string(),
        error: None,
        quality_score: 0.0,
        scanned_at_unix_ms: now_ms(),
    })
}

fn failed_record(path: &Path, error: String) -> VideoRecord {
    let metadata = path.metadata().ok();
    let size_bytes = metadata.as_ref().map(|value| value.len()).unwrap_or(0);
    let modified_unix_ms = metadata
        .and_then(|value| value.modified().ok())
        .and_then(system_time_to_ms)
        .unwrap_or_else(now_ms);

    VideoRecord {
        id: None,
        path: display_path_text(path),
        file_name: path
            .file_name()
            .map(|value| value.to_string_lossy().to_string())
            .unwrap_or_default(),
        parent_path: path.parent().map(display_path_text).unwrap_or_default(),
        size_bytes,
        modified_unix_ms,
        extension: path
            .extension()
            .map(|value| value.to_string_lossy().to_lowercase())
            .unwrap_or_default(),
        container_format: None,
        duration_seconds: None,
        width: None,
        height: None,
        bitrate: None,
        codec: None,
        frame_rate: None,
        audio_codec: None,
        partial_hash: None,
        sample_hashes: Vec::new(),
        preview_images: Vec::new(),
        scan_status: "failed".to_string(),
        error: Some(error),
        quality_score: 0.0,
        scanned_at_unix_ms: now_ms(),
    }
}

fn probe_media(path: &Path) -> anyhow::Result<MediaInfo> {
    let ffprobe = resolve_tool("ffprobe").context("ffprobe was not found")?;
    let input_path = command_input_path(path);
    let output = silent_command(ffprobe)
        .args([
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
        ])
        .arg(input_path)
        .output()
        .with_context(|| format!("run ffprobe for {}", path.display()))?;

    if !output.status.success() {
        return Err(anyhow!(
            "ffprobe failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    parse_probe_output(&output.stdout, path.display())
}

fn parse_probe_output(stdout: &[u8], label: impl std::fmt::Display) -> anyhow::Result<MediaInfo> {
    let parsed: ProbeOutput = serde_json::from_slice(stdout)
        .with_context(|| format!("parse ffprobe output for {label}"))?;
    let streams = parsed.streams.unwrap_or_default();
    let video = streams
        .iter()
        .find(|stream| stream.codec_type.as_deref() == Some("video"));
    let audio = streams
        .iter()
        .find(|stream| stream.codec_type.as_deref() == Some("audio"));

    let duration_seconds = parsed
        .format
        .as_ref()
        .and_then(|format| parse_f64(format.duration.as_deref()))
        .or_else(|| video.and_then(|stream| parse_f64(stream.duration.as_deref())));
    let bitrate = video
        .and_then(|stream| parse_u64(stream.bit_rate.as_deref()))
        .or_else(|| {
            parsed
                .format
                .as_ref()
                .and_then(|format| parse_u64(format.bit_rate.as_deref()))
        });

    Ok(MediaInfo {
        container_format: parsed.format.and_then(|format| format.format_name),
        duration_seconds,
        width: video.and_then(|stream| stream.width),
        height: video.and_then(|stream| stream.height),
        bitrate,
        codec: video.and_then(|stream| stream.codec_name.clone()),
        frame_rate: video.and_then(|stream| parse_frame_rate(stream.avg_frame_rate.as_deref())),
        audio_codec: audio.and_then(|stream| stream.codec_name.clone()),
    })
}

fn sample_hashes(
    path: &Path,
    duration_seconds: Option<f64>,
    sample_hash_count: usize,
) -> anyhow::Result<Vec<String>> {
    let ffmpeg = resolve_tool("ffmpeg").context("ffmpeg was not found")?;
    let duration = duration_seconds.unwrap_or(0.0);
    let positions = sample_positions(duration, sample_hash_count);

    let mut hashes = Vec::new();
    for position in positions {
        let input_path = command_input_path(path);
        let output = silent_command(&ffmpeg)
            .args([
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-threads",
                "1",
                "-filter_threads",
                "1",
                "-ss",
                &format!("{position:.3}"),
                "-i",
            ])
            .arg(input_path)
            .args([
                "-frames:v",
                "1",
                "-vf",
                "scale=9:8,format=gray",
                "-f",
                "rawvideo",
                "pipe:1",
            ])
            .output();

        let Ok(output) = output else {
            continue;
        };
        if !output.status.success() {
            continue;
        }
        if let Some(hash) = dhash_9x8_gray(&output.stdout) {
            hashes.push(hash);
        }
    }

    Ok(hashes)
}

pub(crate) fn sample_positions(duration: f64, sample_hash_count: usize) -> Vec<f64> {
    if duration <= 6.0 {
        return vec![0.2];
    }

    let sample_hash_count = sample_hash_count.clamp(1, 512);
    let fractions = if sample_hash_count == 11 {
        vec![
            0.06, 0.12, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.88, 0.94,
        ]
    } else if sample_hash_count == 1 {
        vec![0.50]
    } else {
        let start = 0.06;
        let end = 0.94;
        let step = (end - start) / (sample_hash_count - 1) as f64;
        (0..sample_hash_count)
            .map(|index| start + step * index as f64)
            .collect::<Vec<_>>()
    };

    fractions
        .iter()
        .map(|fraction| (duration * fraction).min(duration - 0.25).max(0.2))
        .collect()
}

fn expected_sample_hash_count(duration_seconds: Option<f64>, sample_hash_count: usize) -> usize {
    if duration_seconds.unwrap_or(0.0) > 6.0 {
        sample_hash_count.clamp(3, 20)
    } else {
        1
    }
}

fn preview_images(
    path: &Path,
    duration_seconds: Option<f64>,
    partial_hash: Option<&str>,
) -> anyhow::Result<Vec<String>> {
    let ffmpeg = resolve_tool("ffmpeg").context("ffmpeg was not found")?;
    let duration = duration_seconds.unwrap_or(0.0);
    let positions = if duration > 8.0 {
        vec![
            (duration * 0.18).min(duration - 0.25).max(0.2),
            (duration * 0.50).min(duration - 0.25).max(0.2),
            (duration * 0.82).min(duration - 0.25).max(0.2),
        ]
    } else {
        vec![0.2]
    };

    let stem = partial_hash
        .map(|value| value.chars().take(16).collect::<String>())
        .unwrap_or_else(|| {
            let mut hasher = Sha256::new();
            hasher.update(path.to_string_lossy().as_bytes());
            hex::encode(hasher.finalize())
                .chars()
                .take(16)
                .collect::<String>()
        });

    let thumb_dir = paths::data_dir().join("thumbnails");
    fs::create_dir_all(&thumb_dir)?;
    let mut images = Vec::new();

    for (index, position) in positions.iter().enumerate() {
        let output_path = thumb_dir.join(format!("{stem}-{index}.jpg"));
        let input_path = command_input_path(path);
        let output = silent_command(&ffmpeg)
            .args([
                "-y",
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-threads",
                "1",
                "-filter_threads",
                "1",
                "-ss",
                &format!("{position:.3}"),
                "-i",
            ])
            .arg(input_path)
            .args([
                "-frames:v",
                "1",
                "-vf",
                "scale=320:-1:flags=lanczos",
                "-q:v",
                "3",
            ])
            .arg(&output_path)
            .output();

        let Ok(output) = output else {
            continue;
        };
        if output.status.success() && output_path.exists() {
            images.push(output_path.display().to_string());
        }
    }

    Ok(images)
}

pub(crate) fn load_ai_frame_cache_for_video(
    video: &VideoRecord,
    configured_frame_count: usize,
) -> anyhow::Result<Option<AiFramePack>> {
    let key = ai_frame_cache_key(
        &video.path,
        video.partial_hash.as_deref(),
        video.size_bytes,
        video.modified_unix_ms,
        video.duration_seconds,
        configured_frame_count,
    );
    let (manifest_path, data_path) = ai_frame_cache_paths(&key);
    let Some(pack) = read_ai_frame_pack(&manifest_path, &data_path)? else {
        return Ok(None);
    };
    let expected = sample_positions(
        video.duration_seconds.unwrap_or(0.0),
        configured_frame_count.clamp(8, 512),
    )
    .len();
    if pack.frames.len() < expected {
        if pack.frames.len() >= minimum_usable_ai_frame_count(expected) {
            return Ok(Some(pack));
        }
        return Ok(None);
    }
    Ok(Some(pack))
}

pub(crate) fn ensure_ai_frame_cache_for_video(
    video: &VideoRecord,
    configured_frame_count: usize,
    extract_workers: usize,
) -> anyhow::Result<Option<AiFramePack>> {
    if let Some(pack) = load_ai_frame_cache_for_video(video, configured_frame_count)? {
        return Ok(Some(pack));
    }
    ensure_ai_frame_cache(
        Path::new(&video.path),
        video.duration_seconds,
        video.partial_hash.as_deref(),
        video.size_bytes,
        video.modified_unix_ms,
        configured_frame_count,
        extract_workers,
    )
}

pub(crate) fn write_ai_frame_cache_for_video(
    video: &VideoRecord,
    configured_frame_count: usize,
    positions: Vec<f64>,
    frames: Vec<Vec<u8>>,
) -> anyhow::Result<Option<AiFramePack>> {
    if positions.is_empty() || frames.is_empty() || positions.len() != frames.len() {
        return Ok(None);
    }
    if frames.iter().any(|frame| frame.len() < AI_FRAME_BYTES) {
        return Err(anyhow!("imported AI frame has an invalid byte length"));
    }

    let frame_count = configured_frame_count.clamp(8, 512);
    let expected = sample_positions(video.duration_seconds.unwrap_or(0.0), frame_count).len();
    if frames.len() < expected {
        if frames.len() < minimum_usable_ai_frame_count(expected) {
            return Ok(None);
        }
    }

    let key = ai_frame_cache_key(
        &video.path,
        video.partial_hash.as_deref(),
        video.size_bytes,
        video.modified_unix_ms,
        video.duration_seconds,
        frame_count,
    );
    let (manifest_path, data_path) = ai_frame_cache_paths(&key);
    write_ai_frame_cache(
        &manifest_path,
        &data_path,
        &video.path,
        frame_count,
        positions,
        frames,
    )
}

fn ensure_ai_frame_cache(
    path: &Path,
    duration_seconds: Option<f64>,
    partial_hash: Option<&str>,
    size_bytes: u64,
    modified_unix_ms: i64,
    configured_frame_count: usize,
    extract_workers: usize,
) -> anyhow::Result<Option<AiFramePack>> {
    let frame_count = configured_frame_count.clamp(8, 512);
    let key = ai_frame_cache_key(
        &path.display().to_string(),
        partial_hash,
        size_bytes,
        modified_unix_ms,
        duration_seconds,
        frame_count,
    );
    let (manifest_path, data_path) = ai_frame_cache_paths(&key);
    let positions = sample_positions(duration_seconds.unwrap_or(0.0), frame_count);
    let minimum_frames = minimum_usable_ai_frame_count(positions.len());
    if let Some(pack) = read_ai_frame_pack(&manifest_path, &data_path)? {
        if pack.frames.len() >= minimum_frames {
            return Ok(Some(pack));
        }
    }

    let mut extracted = Vec::new();
    let indexed_positions = positions
        .iter()
        .enumerate()
        .map(|(index, position)| (index, *position))
        .collect::<Vec<_>>();
    for work in indexed_positions.chunks(extract_workers.clamp(1, 64)) {
        let mut frames = thread::scope(|scope| {
            work.iter()
                .map(|(frame_index, position)| {
                    scope.spawn(move || {
                        extract_ai_rgb_frame(path, *position)
                            .map(|frame| (*frame_index, *position, frame))
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .filter_map(|handle| handle.join().ok()?.ok())
                .collect::<Vec<_>>()
        });
        extracted.append(&mut frames);
    }
    extracted.sort_by_key(|(frame_index, _, _)| *frame_index);

    if extracted.len() < minimum_frames {
        return Ok(None);
    }

    write_ai_frame_cache(
        &manifest_path,
        &data_path,
        &path.display().to_string(),
        frame_count,
        extracted.iter().map(|(_, position, _)| *position).collect(),
        extracted.into_iter().map(|(_, _, frame)| frame).collect(),
    )
}

fn write_ai_frame_cache(
    manifest_path: &Path,
    data_path: &Path,
    source_path: &str,
    frame_count: usize,
    positions: Vec<f64>,
    frames: Vec<Vec<u8>>,
) -> anyhow::Result<Option<AiFramePack>> {
    if positions.is_empty() || frames.is_empty() || positions.len() != frames.len() {
        return Ok(None);
    }
    if frames.iter().any(|frame| frame.len() < AI_FRAME_BYTES) {
        return Err(anyhow!("AI frame has an invalid byte length"));
    }

    let cache_dir = paths::data_dir().join("ai-frame-cache");
    fs::create_dir_all(&cache_dir)?;
    let tmp_data_path = data_path.with_extension("rgb.tmp");
    let tmp_manifest_path = manifest_path.with_extension("json.tmp");
    {
        let mut file = File::create(&tmp_data_path)
            .with_context(|| format!("create {}", tmp_data_path.display()))?;
        for frame in &frames {
            file.write_all(frame)?;
        }
    }
    let manifest = AiFrameCacheManifest {
        version: 1,
        source_path: source_path.to_string(),
        frame_size: AI_FRAME_SIZE,
        pixel_format: "rgb24".to_string(),
        configured_frame_count: frame_count,
        positions: positions.clone(),
        data_file: data_path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_string(),
    };
    {
        let file = File::create(&tmp_manifest_path)
            .with_context(|| format!("create {}", tmp_manifest_path.display()))?;
        serde_json::to_writer_pretty(file, &manifest)?;
    }
    let _ = fs::remove_file(&data_path);
    let _ = fs::remove_file(&manifest_path);
    fs::rename(&tmp_data_path, &data_path)
        .with_context(|| format!("replace {}", data_path.display()))?;
    fs::rename(&tmp_manifest_path, &manifest_path)
        .with_context(|| format!("replace {}", manifest_path.display()))?;

    Ok(Some(AiFramePack { positions, frames }))
}

fn read_ai_frame_pack(
    manifest_path: &Path,
    data_path: &Path,
) -> anyhow::Result<Option<AiFramePack>> {
    if !manifest_path.exists() || !data_path.exists() {
        return Ok(None);
    }
    let manifest_file =
        File::open(manifest_path).with_context(|| format!("open {}", manifest_path.display()))?;
    let manifest = serde_json::from_reader::<_, AiFrameCacheManifest>(manifest_file)
        .with_context(|| format!("parse {}", manifest_path.display()))?;
    if manifest.version != 1
        || manifest.frame_size != AI_FRAME_SIZE
        || manifest.pixel_format != "rgb24"
        || manifest.positions.is_empty()
    {
        return Ok(None);
    }
    let bytes = fs::read(data_path).with_context(|| format!("read {}", data_path.display()))?;
    if bytes.len() < manifest.positions.len() * AI_FRAME_BYTES {
        return Ok(None);
    }
    let frames = bytes
        .chunks_exact(AI_FRAME_BYTES)
        .take(manifest.positions.len())
        .map(|chunk| chunk.to_vec())
        .collect::<Vec<_>>();
    Ok(Some(AiFramePack {
        positions: manifest.positions,
        frames,
    }))
}

fn ai_frame_cache_paths(key: &str) -> (PathBuf, PathBuf) {
    let cache_dir = paths::data_dir().join("ai-frame-cache");
    (
        cache_dir.join(format!("{key}.json")),
        cache_dir.join(format!("{key}.rgb")),
    )
}

fn ai_frame_cache_key(
    path_text: &str,
    partial_hash: Option<&str>,
    size_bytes: u64,
    modified_unix_ms: i64,
    duration_seconds: Option<f64>,
    configured_frame_count: usize,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"ai-frame-cache-v1");
    hasher.update(partial_hash.unwrap_or(path_text).as_bytes());
    hasher.update(size_bytes.to_le_bytes());
    hasher.update(modified_unix_ms.to_le_bytes());
    hasher.update(format!("{:.3}", duration_seconds.unwrap_or(0.0)).as_bytes());
    hasher.update(configured_frame_count.clamp(8, 512).to_le_bytes());
    hex::encode(hasher.finalize())
        .chars()
        .take(32)
        .collect::<String>()
}

pub(crate) fn prune_stale_ai_frame_cache(
    videos: &[VideoRecord],
    configured_frame_count: usize,
) -> anyhow::Result<FileCleanupStats> {
    let cache_dir = paths::data_dir().join("ai-frame-cache");
    if !cache_dir.exists() {
        return Ok(FileCleanupStats::default());
    }

    let frame_count = configured_frame_count.clamp(8, 512);
    let valid_keys = videos
        .iter()
        .filter(|video| video.scan_status == "ok")
        .map(|video| {
            ai_frame_cache_key(
                &video.path,
                video.partial_hash.as_deref(),
                video.size_bytes,
                video.modified_unix_ms,
                video.duration_seconds,
                frame_count,
            )
        })
        .collect::<HashSet<_>>();

    let mut stats = FileCleanupStats::default();
    for entry in fs::read_dir(&cache_dir)
        .with_context(|| format!("read {}", cache_dir.display()))?
        .filter_map(Result::ok)
    {
        let path = entry.path();
        if !entry
            .file_type()
            .map(|kind| kind.is_file())
            .unwrap_or(false)
        {
            continue;
        }
        let keep = path
            .file_stem()
            .and_then(|value| value.to_str())
            .is_some_and(|stem| valid_keys.contains(stem));
        if !keep {
            remove_file_counting(&path, &mut stats);
        }
    }

    Ok(stats)
}

pub(crate) fn delete_ai_frame_cache_for_video(
    video: &VideoRecord,
    configured_frame_count: usize,
) -> anyhow::Result<FileCleanupStats> {
    let key = ai_frame_cache_key(
        &video.path,
        video.partial_hash.as_deref(),
        video.size_bytes,
        video.modified_unix_ms,
        video.duration_seconds,
        configured_frame_count,
    );
    let (manifest_path, data_path) = ai_frame_cache_paths(&key);
    let mut stats = FileCleanupStats::default();
    remove_file_counting(&manifest_path, &mut stats);
    remove_file_counting(&data_path, &mut stats);
    Ok(stats)
}

pub(crate) fn prune_unreferenced_thumbnails(
    videos: &[VideoRecord],
) -> anyhow::Result<FileCleanupStats> {
    let thumb_dir = paths::data_dir().join("thumbnails");
    if !thumb_dir.exists() {
        return Ok(FileCleanupStats::default());
    }

    let referenced = videos
        .iter()
        .flat_map(|video| video.preview_images.iter())
        .map(|path| normalized_cleanup_path_key(Path::new(path)))
        .collect::<HashSet<_>>();

    let mut stats = FileCleanupStats::default();
    for entry in fs::read_dir(&thumb_dir)
        .with_context(|| format!("read {}", thumb_dir.display()))?
        .filter_map(Result::ok)
    {
        let path = entry.path();
        if !entry
            .file_type()
            .map(|kind| kind.is_file())
            .unwrap_or(false)
        {
            continue;
        }
        if !referenced.contains(&normalized_cleanup_path_key(&path)) {
            remove_file_counting(&path, &mut stats);
        }
    }
    Ok(stats)
}

fn remove_file_counting(path: &Path, stats: &mut FileCleanupStats) {
    let bytes = fs::metadata(path)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    if fs::remove_file(path).is_ok() {
        stats.files = stats.files.saturating_add(1);
        stats.bytes = stats.bytes.saturating_add(bytes);
    }
}

fn normalized_cleanup_path_key(path: &Path) -> String {
    display_path_text(path)
        .replace('/', "\\")
        .trim_end_matches('\\')
        .to_lowercase()
}

pub(crate) fn extract_ai_rgb_frame(path: &Path, position: f64) -> anyhow::Result<Vec<u8>> {
    match extract_ai_rgb_frame_at(path, position) {
        Ok(frame) => return Ok(frame),
        Err(first_error) => {
            for offset in AI_FRAME_SEEK_FALLBACK_OFFSETS {
                let fallback_position = (position + offset).max(0.2);
                if (fallback_position - position).abs() < 0.001 {
                    continue;
                }
                if let Ok(frame) = extract_ai_rgb_frame_at(path, fallback_position) {
                    return Ok(frame);
                }
            }
            return Err(anyhow!(
                "ffmpeg frame extraction failed for {} at {:.3} and fallback offsets: {first_error}",
                path.display(),
                position
            ));
        }
    }
}

fn extract_ai_rgb_frame_at(path: &Path, position: f64) -> anyhow::Result<Vec<u8>> {
    let ffmpeg = resolve_tool("ffmpeg").context("ffmpeg was not found")?;
    let input_path = command_input_path(path);
    let output = silent_command(ffmpeg)
        .args([
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-filter_threads",
            "1",
            "-ss",
            &format!("{position:.3}"),
            "-i",
        ])
        .arg(input_path)
        .args([
            "-frames:v",
            "1",
            "-vf",
            "scale=224:224:force_original_aspect_ratio=increase,crop=224:224,format=rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ])
        .output()
        .with_context(|| format!("extract AI frame from {}", path.display()))?;
    if !output.status.success() {
        return Err(anyhow!(
            "ffmpeg frame extraction failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    if output.stdout.len() < AI_FRAME_BYTES {
        return Err(anyhow!(
            "ffmpeg returned {} bytes, expected at least {}",
            output.stdout.len(),
            AI_FRAME_BYTES
        ));
    }
    Ok(output.stdout[..AI_FRAME_BYTES].to_vec())
}

pub(crate) fn minimum_usable_ai_frame_count(expected: usize) -> usize {
    if expected <= 1 {
        return expected;
    }
    ((expected as f64) * 0.75).ceil() as usize
}

pub(crate) fn convert_image_to_ai_rgb_frame(path: &Path) -> anyhow::Result<Vec<u8>> {
    let ffmpeg = resolve_tool("ffmpeg").context("ffmpeg was not found")?;
    let input_path = command_input_path(path);
    let output = silent_command(ffmpeg)
        .args([
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-filter_threads",
            "1",
            "-i",
        ])
        .arg(input_path)
        .args([
            "-frames:v",
            "1",
            "-vf",
            "scale=224:224:force_original_aspect_ratio=increase,crop=224:224,format=rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ])
        .output()
        .with_context(|| format!("convert imported frame {}", path.display()))?;
    if !output.status.success() {
        return Err(anyhow!(
            "ffmpeg imported frame conversion failed for {}: {}",
            path.display(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    if output.stdout.len() < AI_FRAME_BYTES {
        return Err(anyhow!(
            "ffmpeg returned {} bytes, expected at least {}",
            output.stdout.len(),
            AI_FRAME_BYTES
        ));
    }
    Ok(output.stdout[..AI_FRAME_BYTES].to_vec())
}

fn dhash_9x8_gray(bytes: &[u8]) -> Option<String> {
    if bytes.len() < 72 {
        return None;
    }

    let mut value = 0u64;
    for y in 0..8 {
        for x in 0..8 {
            let left = bytes[y * 9 + x];
            let right = bytes[y * 9 + x + 1];
            if left > right {
                value |= 1u64 << (y * 8 + x);
            }
        }
    }
    Some(format!("{value:016x}"))
}

fn partial_file_hash(path: &Path, size_bytes: u64) -> anyhow::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let chunk_size = 1024 * 1024usize;
    let mut positions = vec![0u64];
    if size_bytes > chunk_size as u64 {
        positions.push(size_bytes / 2);
        positions.push(size_bytes.saturating_sub(chunk_size as u64));
    }

    let mut seen = HashSet::new();
    let mut buffer = vec![0u8; chunk_size];
    for position in positions {
        if !seen.insert(position) {
            continue;
        }
        file.seek(SeekFrom::Start(position))?;
        let bytes_read = file.read(&mut buffer)?;
        hasher.update(position.to_le_bytes());
        hasher.update(&buffer[..bytes_read]);
    }

    Ok(hex::encode(hasher.finalize()))
}

fn quality_score(info: &MediaInfo, size_bytes: u64) -> f64 {
    let duration = info.duration_seconds.unwrap_or(0.0);
    let pixels = info.width.unwrap_or(0) as f64 * info.height.unwrap_or(0) as f64;
    let bitrate = info.bitrate.unwrap_or(0) as f64;
    duration * 1_000_000.0 + pixels * 100.0 + bitrate / 128.0 + size_bytes as f64 / 1_000_000.0
}

fn is_video_path(path: &Path) -> bool {
    path.extension()
        .and_then(|value| value.to_str())
        .map(|extension| VIDEO_EXTENSIONS.contains(&extension.to_lowercase().as_str()))
        .unwrap_or(false)
}

pub(crate) fn resolve_tool(name: &str) -> Option<PathBuf> {
    let env_key = format!("DVS_{}", name.to_uppercase());
    if let Ok(value) = env::var(env_key) {
        let path = PathBuf::from(value);
        if path.exists() {
            return Some(path);
        }
    }

    let local_tool = local_cached_tool_path(name);
    if usable_cached_tool(&local_tool) {
        return Some(local_tool);
    }

    let discovered = discover_tool(name)?;
    if let Some(cached) = cache_tool_locally(name, &discovered) {
        return Some(cached);
    }

    Some(discovered)
}

fn discover_tool(name: &str) -> Option<PathBuf> {
    if let Ok(output) = silent_command("where.exe").arg(name).output() {
        if output.status.success() {
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                let path = PathBuf::from(line.trim());
                if path.exists() {
                    return Some(path);
                }
            }
        }
    }

    if let Ok(local_app_data) = env::var("LOCALAPPDATA") {
        let winget_link = PathBuf::from(local_app_data)
            .join("Microsoft")
            .join("WinGet")
            .join("Links")
            .join(format!("{name}.exe"));
        if winget_link.exists() {
            return Some(winget_link);
        }
    }

    None
}

fn local_cached_tool_path(name: &str) -> PathBuf {
    paths::data_dir().join("tools").join(format!("{name}.exe"))
}

fn usable_cached_tool(path: &Path) -> bool {
    path.metadata()
        .is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0)
}

fn cache_tool_locally(name: &str, source: &Path) -> Option<PathBuf> {
    let destination = local_cached_tool_path(name);
    let source = fs::canonicalize(source).unwrap_or_else(|_| source.to_path_buf());
    if normalize_tool_path(&source) == normalize_tool_path(&destination) {
        return Some(destination);
    }
    if usable_cached_tool(&destination) {
        return Some(destination);
    }

    let parent = destination.parent()?;
    fs::create_dir_all(parent).ok()?;
    let temp = parent.join(format!("{name}.exe.tmp-{}", process::id()));
    let copied = fs::copy(&source, &temp).ok()?;
    if copied == 0 {
        let _ = fs::remove_file(&temp);
        return None;
    }
    let _ = fs::remove_file(&destination);
    match fs::rename(&temp, &destination) {
        Ok(()) => Some(destination),
        Err(_) if usable_cached_tool(&destination) => {
            let _ = fs::remove_file(&temp);
            Some(destination)
        }
        Err(_) => {
            let _ = fs::remove_file(&temp);
            None
        }
    }
}

fn normalize_tool_path(path: &Path) -> String {
    path.to_string_lossy().replace('/', "\\").to_lowercase()
}

fn command_first_line(path: &Path, args: &[&str]) -> Option<String> {
    let output = silent_command(path).args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()
        .map(|line| line.to_string())
}

pub(crate) fn silent_command<S: AsRef<OsStr>>(program: S) -> Command {
    let mut command = Command::new(program);
    apply_no_window(&mut command);
    command
}

#[cfg(windows)]
pub(crate) fn command_input_path(path: &Path) -> PathBuf {
    let value = path.as_os_str().to_string_lossy();
    if value.starts_with(r"\\?\") {
        return path.to_path_buf();
    }
    if let Some(rest) = value.strip_prefix(r"\\") {
        return PathBuf::from(format!(r"\\?\UNC\{rest}"));
    }
    if path.is_absolute() {
        return PathBuf::from(format!(r"\\?\{value}"));
    }
    path.to_path_buf()
}

#[cfg(not(windows))]
pub(crate) fn command_input_path(path: &Path) -> PathBuf {
    path.to_path_buf()
}

pub(crate) fn filesystem_input_path(path: &Path) -> PathBuf {
    let input_path = command_input_path(path);
    if input_path.exists() || !path.exists() {
        input_path
    } else {
        path.to_path_buf()
    }
}

#[cfg(windows)]
pub(crate) fn display_path_text(path: &Path) -> String {
    let value = path.as_os_str().to_string_lossy().replace('/', "\\");
    if let Some(rest) = value.strip_prefix(r"\\?\UNC\") {
        return format!(r"\\{rest}");
    }
    if let Some(rest) = value.strip_prefix(r"\\?\") {
        return rest.to_string();
    }
    value
}

#[cfg(not(windows))]
pub(crate) fn display_path_text(path: &Path) -> String {
    path.display().to_string()
}

#[cfg(windows)]
fn apply_no_window(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x08000000;
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn apply_no_window(_command: &mut Command) {}

fn parse_f64(value: Option<&str>) -> Option<f64> {
    value?.parse::<f64>().ok()
}

fn parse_u64(value: Option<&str>) -> Option<u64> {
    value?.parse::<u64>().ok()
}

fn parse_frame_rate(value: Option<&str>) -> Option<f64> {
    let value = value?;
    if let Some((numerator, denominator)) = value.split_once('/') {
        let numerator = numerator.parse::<f64>().ok()?;
        let denominator = denominator.parse::<f64>().ok()?;
        if denominator == 0.0 {
            return None;
        }
        Some(numerator / denominator)
    } else {
        value.parse::<f64>().ok()
    }
}

fn system_time_to_ms(value: SystemTime) -> Option<i64> {
    value
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_millis() as i64)
}

pub(crate) fn now_ms() -> i64 {
    system_time_to_ms(SystemTime::now()).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dhash_generates_stable_hex() {
        let bytes = vec![0u8; 72];
        assert_eq!(dhash_9x8_gray(&bytes).unwrap(), "0000000000000000");
    }

    #[test]
    fn parses_fractional_frame_rate() {
        assert_eq!(parse_frame_rate(Some("30000/1001")).unwrap().round(), 30.0);
        assert_eq!(parse_frame_rate(Some("0/0")), None);
    }

    #[test]
    fn detects_paths_inside_excluded_backup_dir() {
        let excluded = vec![PathBuf::from(r"\\EXAMPLE-NAS\Test\recycle")];
        assert!(is_inside_any_excluded_dir(
            Path::new(r"\\EXAMPLE-NAS\Test\recycle\old.mp4"),
            &excluded
        ));
        assert!(!is_inside_any_excluded_dir(
            Path::new(r"\\EXAMPLE-NAS\Test\recordings\old.mp4"),
            &excluded
        ));
    }

    #[test]
    fn deletes_exact_ai_frame_cache_files_for_video() {
        let unique = format!("test-{}", now_ms());
        let video = VideoRecord {
            id: Some(1),
            path: format!(r"\\EXAMPLE-NAS\Test\{unique}.mp4"),
            file_name: format!("{unique}.mp4"),
            parent_path: r"\\EXAMPLE-NAS\Test".to_string(),
            size_bytes: 1234,
            modified_unix_ms: 5678,
            extension: "mp4".to_string(),
            container_format: Some("mov,mp4".to_string()),
            duration_seconds: Some(60.0),
            width: Some(1920),
            height: Some(1080),
            bitrate: Some(1_000_000),
            codec: Some("h264".to_string()),
            frame_rate: Some(30.0),
            audio_codec: Some("aac".to_string()),
            partial_hash: Some(unique.clone()),
            sample_hashes: Vec::new(),
            preview_images: Vec::new(),
            scan_status: "ok".to_string(),
            error: None,
            quality_score: 0.0,
            scanned_at_unix_ms: 0,
        };
        let key = ai_frame_cache_key(
            &video.path,
            video.partial_hash.as_deref(),
            video.size_bytes,
            video.modified_unix_ms,
            video.duration_seconds,
            128,
        );
        let (manifest_path, data_path) = ai_frame_cache_paths(&key);
        fs::create_dir_all(manifest_path.parent().unwrap()).unwrap();
        fs::write(&manifest_path, b"{}").unwrap();
        fs::write(&data_path, b"rgb-bytes").unwrap();

        let stats = delete_ai_frame_cache_for_video(&video, 128).unwrap();
        assert_eq!(stats.files, 2);
        assert_eq!(stats.bytes, 11);
        assert!(!manifest_path.exists());
        assert!(!data_path.exists());
    }

    #[test]
    fn same_file_observation_ignores_missing_full_scan_assets() {
        let mut existing = VideoRecord {
            id: Some(1),
            path: r"\\EXAMPLE-NAS\Test\video.mp4".to_string(),
            file_name: "video.mp4".to_string(),
            parent_path: r"\\EXAMPLE-NAS\Test".to_string(),
            size_bytes: 1234,
            modified_unix_ms: 5678,
            extension: "mp4".to_string(),
            container_format: Some("mov,mp4".to_string()),
            duration_seconds: Some(60.0),
            width: Some(1920),
            height: Some(1080),
            bitrate: Some(1_000_000),
            codec: Some("h264".to_string()),
            frame_rate: Some(30.0),
            audio_codec: Some("aac".to_string()),
            partial_hash: Some("hash".to_string()),
            sample_hashes: vec!["a".to_string()],
            preview_images: vec![r"D:\old-data\missing-preview.jpg".to_string()],
            scan_status: "ok".to_string(),
            error: None,
            quality_score: 1.0,
            scanned_at_unix_ms: 0,
        };
        let mut observed = existing.clone();
        observed.id = None;
        observed.container_format = None;
        observed.duration_seconds = None;
        observed.width = None;
        observed.height = None;
        observed.bitrate = None;
        observed.codec = None;
        observed.frame_rate = None;
        observed.audio_codec = None;
        observed.partial_hash = None;
        observed.sample_hashes.clear();
        observed.preview_images.clear();

        assert!(same_file_observation(&existing, &observed));

        existing.modified_unix_ms += 3_000;
        assert!(!same_file_observation(&existing, &observed));
    }

    #[test]
    #[cfg(windows)]
    fn display_path_text_strips_extended_unc_prefix() {
        assert_eq!(
            display_path_text(Path::new(r"\\?\UNC\EXAMPLE-NAS\Videos\folder\video.mp4")),
            r"\\EXAMPLE-NAS\Videos\folder\video.mp4"
        );
    }
}
