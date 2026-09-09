pub mod ai;
pub mod cancel;
pub mod database;
pub mod local_preprocess;
pub mod matching;
pub mod media;
pub mod models;
pub mod nas_preprocess;
pub mod operations;
pub mod paths;
pub mod ram_disk;

use models::{
    AiIndexProgress, AiIndexSummary, AiModelStatus, AppSettings, BatchMergeTask,
    CompletedAiFrameCacheCleanupSummary, DeleteIndexProgress, DeleteIndexSummary,
    IndexRefreshSummary, MatchGroup, MatchRefreshProgress, OperationHistoryEntry, OperationResult,
    RamDiskStatus, ReplacementPlan, ScanProgress, ScanSession, ScanSummary, StaleVideoPruneSummary,
    StorageCleanupSummary, StorageUsageSummary, ToolStatus, VacuumSummary, VideoRecord,
};
use tauri::{Emitter, Manager};

#[tauri::command]
async fn get_tool_status() -> Result<ToolStatus, String> {
    tauri::async_runtime::spawn_blocking(media::tool_status)
        .await
        .map_err(|error| format!("tool status worker failed: {error}"))?
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn scan_source(app: tauri::AppHandle, source: String) -> Result<ScanSummary, String> {
    let scan_source = source.clone();
    cancel::reset();
    tauri::async_runtime::spawn_blocking(move || {
        media::scan_source_with_progress(&scan_source, |progress: ScanProgress| {
            let _ = app.emit("scan-progress", progress);
        })
    })
    .await
    .map_err(|error| format!("scan worker failed: {error}"))?
    .map_err(|error| error.to_string())
}

#[tauri::command]
async fn scan_sources(
    app: tauri::AppHandle,
    sources: Vec<String>,
) -> Result<Vec<ScanSummary>, String> {
    let scan_sources = sources.clone();
    cancel::reset();
    tauri::async_runtime::spawn_blocking(move || {
        media::scan_sources_with_progress(&scan_sources, |progress: ScanProgress| {
            let _ = app.emit("scan-progress", progress);
        })
    })
    .await
    .map_err(|error| format!("scan worker failed: {error}"))?
    .map_err(|error| error.to_string())
}

#[tauri::command]
async fn list_videos(session_ids: Option<Vec<i64>>) -> Result<Vec<VideoRecord>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let conn = database::open_database()?;
        let videos = match session_ids {
            Some(ids) => database::list_videos_for_sessions(&conn, &ids)?,
            None => database::list_videos(&conn)?,
        };
        Ok::<Vec<VideoRecord>, anyhow::Error>(videos)
    })
    .await
    .map_err(|error| format!("video list worker failed: {error}"))?
    .map_err(|error| error.to_string())
}

#[tauri::command]
fn list_scan_sessions() -> Result<Vec<ScanSession>, String> {
    let conn = database::open_database().map_err(|error| error.to_string())?;
    database::list_scan_sessions(&conn).map_err(|error| error.to_string())
}

#[tauri::command]
async fn delete_scan_session(session_id: i64) -> Result<(), String> {
    tauri::async_runtime::spawn_blocking(move || {
        let conn = database::open_database()?;
        database::delete_scan_session(&conn, session_id)
    })
    .await
    .map_err(|error| format!("delete scan session worker failed: {error}"))?
    .map_err(|error| error.to_string())
}

#[tauri::command]
async fn delete_scan_sessions(
    app: tauri::AppHandle,
    session_ids: Vec<i64>,
) -> Result<DeleteIndexSummary, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let conn = database::open_database()?;
        database::delete_scan_sessions_with_progress(
            &conn,
            &session_ids,
            |progress: DeleteIndexProgress| {
                let _ = app.emit("delete-index-progress", progress);
            },
        )
    })
    .await
    .map_err(|error| format!("delete scan sessions worker failed: {error}"))?
    .map_err(|error| error.to_string())
}

#[tauri::command]
async fn refresh_index_sources() -> Result<IndexRefreshSummary, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let conn = database::open_database()?;
        database::refresh_index_sources(&conn)
    })
    .await
    .map_err(|error| format!("refresh index sources worker failed: {error}"))?
    .map_err(|error| error.to_string())
}

#[tauri::command]
async fn prune_stale_videos(
    session_ids: Option<Vec<i64>>,
) -> Result<StaleVideoPruneSummary, String> {
    tauri::async_runtime::spawn_blocking(move || operations::prune_stale_videos(session_ids))
        .await
        .map_err(|error| format!("prune stale videos worker failed: {error}"))?
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn get_storage_usage() -> Result<StorageUsageSummary, String> {
    tauri::async_runtime::spawn_blocking(operations::storage_usage)
        .await
        .map_err(|error| format!("storage usage worker failed: {error}"))?
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn cleanup_storage() -> Result<StorageCleanupSummary, String> {
    tauri::async_runtime::spawn_blocking(operations::cleanup_storage)
        .await
        .map_err(|error| format!("storage cleanup worker failed: {error}"))?
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn cleanup_completed_ai_frame_cache() -> Result<CompletedAiFrameCacheCleanupSummary, String> {
    tauri::async_runtime::spawn_blocking(operations::cleanup_completed_ai_frame_cache)
        .await
        .map_err(|error| format!("completed AI frame cache cleanup worker failed: {error}"))?
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn vacuum_database() -> Result<VacuumSummary, String> {
    tauri::async_runtime::spawn_blocking(operations::vacuum_database)
        .await
        .map_err(|error| format!("database vacuum worker failed: {error}"))?
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn list_match_groups(
    app: tauri::AppHandle,
    session_ids: Option<Vec<i64>>,
    min_confidence: Option<f64>,
    max_confidence: Option<f64>,
) -> Result<Vec<MatchGroup>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let mut conn = database::open_database()?;
        let videos = match session_ids {
            Some(ids) => database::list_videos_for_sessions(&conn, &ids)?,
            None => database::list_videos(&conn)?,
        };
        let videos = operations::prune_missing_videos(&conn, videos)?;
        let settings = operations::get_app_settings()?;
        if !settings.ai_vision_enabled {
            return Ok::<Vec<MatchGroup>, anyhow::Error>(Vec::new());
        }
        let progress_app = app.clone();
        let min_confidence = min_confidence.unwrap_or(0.90).clamp(0.0, 1.0);
        let max_confidence = max_confidence.unwrap_or(1.0).clamp(0.0, 1.0);
        let (range_min, range_max) = if min_confidence <= max_confidence {
            (min_confidence, max_confidence)
        } else {
            (max_confidence, min_confidence)
        };
        let mut groups = ai::build_ai_match_groups_with_progress(
            &mut conn,
            &videos,
            &settings,
            range_min,
            |progress: MatchRefreshProgress| {
                let _ = progress_app.emit("match-refresh-progress", progress);
            },
        )?;
        groups.retain(|group| group.confidence <= range_max);
        groups.sort_by(|a, b| {
            b.confidence
                .total_cmp(&a.confidence)
                .then_with(|| b.reclaimable_bytes.cmp(&a.reclaimable_bytes))
        });
        Ok::<Vec<MatchGroup>, anyhow::Error>(groups)
    })
    .await
    .map_err(|error| format!("match group worker failed: {error}"))?
    .map_err(|error| error.to_string())
}

#[tauri::command]
async fn get_ai_model_status() -> Result<AiModelStatus, String> {
    tauri::async_runtime::spawn_blocking(ai::ai_model_status)
        .await
        .map_err(|error| format!("AI model status worker failed: {error}"))?
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn build_ai_index(
    app: tauri::AppHandle,
    session_ids: Option<Vec<i64>>,
    force_rebuild: Option<bool>,
) -> Result<AiIndexSummary, String> {
    cancel::reset();
    tauri::async_runtime::spawn_blocking(move || {
        let settings = operations::get_app_settings()?;
        let _ram_disk = ram_disk::TaskRamDisk::activate(&settings)?;
        ai::build_ai_index_with_progress(
            session_ids,
            force_rebuild.unwrap_or(false),
            |progress: AiIndexProgress| {
                let _ = app.emit("ai-index-progress", progress);
            },
        )
    })
    .await
    .map_err(|error| format!("AI index worker failed: {error}"))?
    .map_err(|error| error.to_string())
}

#[tauri::command]
fn cancel_current_work() -> Result<(), String> {
    cancel::request();
    Ok(())
}

#[tauri::command]
fn create_replacement_plan(
    high_quality_id: i64,
    naming_source_id: i64,
) -> Result<ReplacementPlan, String> {
    operations::create_replacement_plan(high_quality_id, naming_source_id)
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn execute_replacement(
    high_quality_id: i64,
    naming_source_id: i64,
    confirmation: String,
) -> Result<OperationResult, String> {
    operations::execute_replacement(high_quality_id, naming_source_id, &confirmation)
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn execute_merge_selection(
    high_quality_id: i64,
    naming_source_id: i64,
    extra_video_ids: Vec<i64>,
    disposal: String,
    confirmation: String,
    filename_source_id: Option<i64>,
) -> Result<OperationResult, String> {
    operations::execute_merge_selection(
        high_quality_id,
        naming_source_id,
        extra_video_ids,
        &disposal,
        &confirmation,
        filename_source_id,
    )
    .map_err(|error| error.to_string())
}

#[tauri::command]
async fn execute_batch_merge_selection(
    tasks: Vec<BatchMergeTask>,
    disposal: String,
    confirmation: String,
) -> Result<OperationResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        operations::execute_batch_merge_selection(tasks, &disposal, &confirmation)
    })
    .await
    .map_err(|error| format!("batch merge worker failed: {error}"))?
    .map_err(|error| error.to_string())
}

#[tauri::command]
fn open_video(path: String) -> Result<(), String> {
    operations::open_video(&path).map_err(|error| error.to_string())
}

#[tauri::command]
fn get_app_settings() -> Result<AppSettings, String> {
    operations::get_app_settings().map_err(|error| error.to_string())
}

#[tauri::command]
fn save_app_settings(settings: AppSettings) -> Result<AppSettings, String> {
    operations::save_app_settings(&settings).map_err(|error| error.to_string())
}

#[tauri::command]
async fn get_ram_disk_status() -> Result<RamDiskStatus, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let settings = operations::get_app_settings()?;
        Ok::<RamDiskStatus, anyhow::Error>(ram_disk::status(&settings))
    })
    .await
    .map_err(|error| format!("RAM disk status worker failed: {error}"))?
    .map_err(|error| error.to_string())
}

#[tauri::command]
async fn ensure_ram_disk() -> Result<RamDiskStatus, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let settings = operations::get_app_settings()?;
        Ok::<RamDiskStatus, anyhow::Error>(ram_disk::try_ensure_from_scheduled_task(&settings))
    })
    .await
    .map_err(|error| format!("RAM disk startup worker failed: {error}"))?
    .map_err(|error| error.to_string())
}

#[tauri::command]
async fn release_ram_disk() -> Result<RamDiskStatus, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let settings = operations::get_app_settings()?;
        ram_disk::release_managed(&settings)
    })
    .await
    .map_err(|error| format!("RAM disk release worker failed: {error}"))?
    .map_err(|error| error.to_string())
}

#[tauri::command]
async fn configure_ram_disk(size_mb: usize) -> Result<RamDiskStatus, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let settings = operations::get_app_settings()?;
        let configured = ram_disk::configure_with_elevation(&settings, size_mb)?;
        let mut updated = settings;
        updated.ram_disk_enabled = true;
        updated.ram_disk_size_mb = size_mb.clamp(512, 1_048_576);
        updated.ram_disk_setup_completed = true;
        updated.local_preprocess_temp_dir = configured.cache_path.clone();
        operations::save_app_settings(&updated)?;
        Ok::<RamDiskStatus, anyhow::Error>(configured)
    })
    .await
    .map_err(|error| format!("RAM disk configuration worker failed: {error}"))?
    .map_err(|error| error.to_string())
}

#[tauri::command]
fn open_ram_disk_driver_download() -> Result<(), String> {
    ram_disk::open_driver_download().map_err(|error| error.to_string())
}

#[tauri::command]
fn execute_file_action(
    video_ids: Vec<i64>,
    action: String,
    confirmation: String,
    defer_index_update: Option<bool>,
) -> Result<OperationResult, String> {
    operations::execute_file_action_with_options(
        video_ids,
        &action,
        &confirmation,
        defer_index_update.unwrap_or(false),
    )
    .map_err(|error| error.to_string())
}

#[tauri::command]
async fn pick_folder(initial_path: Option<String>) -> Result<Option<String>, String> {
    tauri::async_runtime::spawn_blocking(move || operations::pick_folder(initial_path))
        .await
        .map_err(|error| format!("folder picker failed: {error}"))?
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn pick_folders(initial_path: Option<String>) -> Result<Vec<String>, String> {
    tauri::async_runtime::spawn_blocking(move || operations::pick_folders(initial_path))
        .await
        .map_err(|error| format!("folder picker failed: {error}"))?
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn list_operation_history() -> Result<Vec<OperationHistoryEntry>, String> {
    operations::list_operation_history().map_err(|error| error.to_string())
}

#[tauri::command]
fn rollback_operation(operation_id: String) -> Result<OperationResult, String> {
    operations::rollback_operation(&operation_id).map_err(|error| error.to_string())
}

pub fn run() {
    tauri::Builder::default()
        .setup(|app| {
            paths::ensure_data_dirs()?;
            app.asset_protocol_scope()
                .allow_directory(paths::data_dir(), true)?;
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_tool_status,
            scan_source,
            scan_sources,
            list_videos,
            list_scan_sessions,
            delete_scan_session,
            delete_scan_sessions,
            refresh_index_sources,
            prune_stale_videos,
            get_storage_usage,
            cleanup_storage,
            cleanup_completed_ai_frame_cache,
            vacuum_database,
            list_match_groups,
            get_ai_model_status,
            build_ai_index,
            cancel_current_work,
            create_replacement_plan,
            execute_replacement,
            execute_merge_selection,
            execute_batch_merge_selection,
            open_video,
            get_app_settings,
            save_app_settings,
            get_ram_disk_status,
            ensure_ram_disk,
            release_ram_disk,
            configure_ram_disk,
            open_ram_disk_driver_download,
            execute_file_action,
            pick_folder,
            pick_folders,
            list_operation_history,
            rollback_operation
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
