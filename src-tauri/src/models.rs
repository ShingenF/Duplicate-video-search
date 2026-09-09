use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolStatus {
    pub workspace_root: String,
    pub data_dir: String,
    pub database_path: String,
    pub allowed_source: String,
    pub ffmpeg_path: Option<String>,
    pub ffprobe_path: Option<String>,
    pub ffmpeg_version: Option<String>,
    pub ffprobe_version: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScanSummary {
    pub session_id: Option<i64>,
    pub source: String,
    pub database_path: String,
    pub total_files: usize,
    pub scanned: usize,
    pub failed: usize,
    pub elapsed_ms: u128,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScanSession {
    pub id: i64,
    pub source: String,
    pub started_unix_ms: i64,
    pub completed_unix_ms: Option<i64>,
    pub total_files: usize,
    pub scanned: usize,
    pub failed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScanProgress {
    pub source: String,
    pub phase: String,
    pub total_files: usize,
    pub scanned: usize,
    pub failed: usize,
    pub current_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IndexRefreshSummary {
    pub removed_missing_sources: usize,
    pub skipped_unavailable_roots: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeleteIndexSummary {
    pub requested_sessions: usize,
    pub deleted_sessions: usize,
    pub affected_videos: usize,
    pub deleted_videos: usize,
    pub deleted_pair_scores: usize,
    pub deleted_match_edges: usize,
    pub deleted_frame_embeddings: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StaleVideoPruneSummary {
    pub checked_videos: usize,
    pub removed_missing_videos: usize,
    pub removed_orphan_videos: usize,
    pub deleted_pair_scores: usize,
    pub deleted_match_edges: usize,
    pub deleted_frame_embeddings: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StorageUsageItem {
    pub key: String,
    pub path: String,
    pub bytes: u64,
    pub file_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StorageDatabaseStats {
    pub videos: usize,
    pub scan_sessions: usize,
    pub frame_embeddings: usize,
    pub ai_pair_scores: usize,
    pub ai_match_edges: usize,
    pub embedding_models: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StorageUsageSummary {
    pub data_dir: String,
    pub total_bytes: u64,
    pub items: Vec<StorageUsageItem>,
    pub database_stats: StorageDatabaseStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StorageCleanupSummary {
    pub deleted_ai_frame_cache_files: usize,
    pub deleted_ai_frame_cache_bytes: u64,
    pub deleted_thumbnail_files: usize,
    pub deleted_thumbnail_bytes: u64,
    pub removed_orphan_videos: usize,
    pub deleted_pair_scores: usize,
    pub deleted_match_edges: usize,
    pub deleted_frame_embeddings: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CompletedAiFrameCacheCleanupSummary {
    pub checked_videos: usize,
    pub eligible_videos: usize,
    pub deleted_files: usize,
    pub deleted_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VacuumSummary {
    pub before_bytes: u64,
    pub after_bytes: u64,
    pub reclaimed_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeleteIndexProgress {
    pub phase: String,
    pub phase_processed: usize,
    pub phase_total: usize,
    pub requested_sessions: usize,
    pub deleted_sessions: usize,
    pub affected_videos: usize,
    pub deleted_videos: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AiIndexProgress {
    pub phase: String,
    pub total_videos: usize,
    pub processed: usize,
    pub skipped: usize,
    #[serde(default)]
    pub insufficient_frames: usize,
    pub failed: usize,
    #[serde(default)]
    pub started: usize,
    #[serde(default)]
    pub prepared: usize,
    pub current_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RamDiskStatus {
    pub supported: bool,
    pub driver_installed: bool,
    pub mounted: bool,
    pub ready: bool,
    pub setup_completed: bool,
    pub configured_size_mb: usize,
    pub actual_capacity_mb: Option<u64>,
    pub drive_letter: Option<String>,
    pub cache_path: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MatchRefreshProgress {
    pub phase: String,
    pub total_videos: usize,
    pub total_pairs: usize,
    pub processed_pairs: usize,
    pub cached_pairs: usize,
    pub computed_pairs: usize,
    pub groups: usize,
    pub pairs_per_second: f64,
    pub phase_processed: usize,
    pub phase_total: usize,
    pub phase_pairs_per_second: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AiIndexSummary {
    pub model_id: String,
    pub model_path: String,
    pub total_videos: usize,
    pub processed: usize,
    pub skipped: usize,
    #[serde(default)]
    pub insufficient_frames: usize,
    pub failed: usize,
    pub elapsed_ms: u128,
    pub recent_errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct NasPreprocessSummary {
    pub total_videos: usize,
    pub imported: usize,
    pub skipped: usize,
    pub failed: usize,
    pub bundle_dir: Option<String>,
    pub recent_errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LocalPreprocessSummary {
    pub total_videos: usize,
    pub imported: usize,
    pub skipped: usize,
    #[serde(default)]
    pub insufficient_frames: usize,
    pub failed: usize,
    pub staging_dir: Option<String>,
    pub recent_errors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AiModelStatus {
    pub enabled: bool,
    pub ready: bool,
    pub model_id: Option<String>,
    pub model_path: String,
    pub device: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoRecord {
    pub id: Option<i64>,
    pub path: String,
    pub file_name: String,
    pub parent_path: String,
    pub size_bytes: u64,
    pub modified_unix_ms: i64,
    pub extension: String,
    pub container_format: Option<String>,
    pub duration_seconds: Option<f64>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub bitrate: Option<u64>,
    pub codec: Option<String>,
    pub frame_rate: Option<f64>,
    pub audio_codec: Option<String>,
    pub partial_hash: Option<String>,
    pub sample_hashes: Vec<String>,
    pub preview_images: Vec<String>,
    pub scan_status: String,
    pub error: Option<String>,
    pub quality_score: f64,
    pub scanned_at_unix_ms: i64,
}

#[derive(Debug, Clone)]
pub struct FrameEmbeddingRecord {
    pub video_id: i64,
    pub model_id: String,
    pub frame_index: usize,
    pub timestamp_seconds: f64,
    pub embedding: Vec<f32>,
    pub norm: f64,
}

#[derive(Debug, Clone)]
pub struct AiPairScoreRecord {
    pub video_a_id: i64,
    pub video_b_id: i64,
    pub model_id: String,
    pub cache_key: String,
    pub confidence: f64,
    pub matched: usize,
    pub compared: usize,
    pub average_similarity: f64,
    pub coverage: f64,
    pub created_unix_ms: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MatchRange {
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub start_fraction: f64,
    pub end_fraction: f64,
    pub matched_frames: usize,
    pub average_similarity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MatchHitPoint {
    pub seconds: f64,
    pub fraction: f64,
    pub similarity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MatchDetail {
    pub peer_video_id: Option<i64>,
    pub peer_file_name: String,
    pub points: Vec<MatchHitPoint>,
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub hit_count: usize,
    pub displayed_hit_count: usize,
    pub average_similarity: f64,
    pub confidence: f64,
    pub short_coverage: f64,
    pub long_compactness: f64,
    pub long_span_seconds: f64,
    pub long_span_fraction: f64,
    pub relation_type: String,
    pub relation_label: String,
    pub relation_explanation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MatchItem {
    pub video: VideoRecord,
    pub role: String,
    pub quality_rank: usize,
    #[serde(default)]
    pub match_ranges: Vec<MatchRange>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub match_detail: Option<MatchDetail>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MatchGroup {
    pub id: String,
    pub title: String,
    pub kind: String,
    pub confidence: f64,
    pub recommended_video_id: Option<i64>,
    pub reclaimable_bytes: u64,
    pub item_count: usize,
    pub evidence: Vec<String>,
    pub report: String,
    pub items: Vec<MatchItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ReplacementPlan {
    pub plan_id: String,
    pub high_quality_id: i64,
    pub naming_source_id: i64,
    pub high_quality_path: String,
    pub target_path: String,
    pub backup_path: String,
    pub plan_file: String,
    pub steps: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OperationResult {
    pub operation_id: String,
    pub status: String,
    pub log_file: String,
    pub messages: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchMergeTask {
    pub keeper_id: i64,
    pub naming_source_id: i64,
    #[serde(default)]
    pub filename_source_id: Option<i64>,
    #[serde(default)]
    pub extra_video_ids: Vec<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OperationHistoryEntry {
    pub operation_id: String,
    pub action: String,
    pub summary: String,
    pub created_at_unix_ms: i64,
    pub reversible: bool,
    pub rolled_back: bool,
    pub messages: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppSettings {
    #[serde(default = "default_ui_language")]
    pub ui_language: String,
    pub backup_dir: String,
    #[serde(default)]
    pub naming_source_dirs: Vec<String>,
    #[serde(default = "default_keeper_size_priority_duration_seconds")]
    pub keeper_size_priority_duration_seconds: f64,
    #[serde(default = "default_scan_worker_count")]
    pub scan_worker_count: usize,
    #[serde(default = "default_sample_hash_count")]
    pub sample_hash_count: usize,
    #[serde(default = "default_temporal_hash_threshold")]
    pub temporal_hash_threshold: u32,
    #[serde(default = "default_min_temporal_match_points")]
    pub min_temporal_match_points: usize,
    #[serde(default = "default_allowed_unmatched_sample_frames")]
    pub allowed_unmatched_sample_frames: usize,
    pub allow_direct_delete: bool,
    #[serde(default = "default_restrict_scan_to_test_path")]
    pub restrict_scan_to_test_path: bool,
    #[serde(default)]
    pub compare_within_same_folder: bool,
    #[serde(default)]
    pub ai_vision_enabled: bool,
    #[serde(default)]
    pub ai_model_path: String,
    #[serde(default = "default_ai_device")]
    pub ai_device: String,
    #[serde(default = "default_ai_frame_count")]
    pub ai_frame_count: usize,
    #[serde(default = "default_ai_batch_size")]
    pub ai_batch_size: usize,
    #[serde(default = "default_ai_similarity_threshold")]
    pub ai_similarity_threshold: f64,
    #[serde(default = "default_ai_min_matched_frames")]
    pub ai_min_matched_frames: usize,
    #[serde(default = "default_ai_clip_matching_enabled")]
    pub ai_clip_matching_enabled: bool,
    #[serde(default)]
    pub ai_index_after_scan: bool,
    #[serde(default = "default_ai_extract_worker_count")]
    pub ai_extract_worker_count: usize,
    #[serde(default = "default_ai_gpu_worker_count")]
    pub ai_gpu_worker_count: usize,
    #[serde(default = "default_ai_match_worker_count")]
    pub ai_match_worker_count: usize,
    #[serde(default = "default_ai_frame_cache_enabled")]
    pub ai_frame_cache_enabled: bool,
    #[serde(default)]
    pub delete_ai_frame_cache_after_index: bool,
    #[serde(default = "default_ram_disk_enabled")]
    pub ram_disk_enabled: bool,
    #[serde(default = "default_ram_disk_size_mb")]
    pub ram_disk_size_mb: usize,
    #[serde(default)]
    pub ram_disk_setup_completed: bool,
    #[serde(default)]
    pub local_preprocess_enabled: bool,
    #[serde(default = "default_local_preprocess_video_workers")]
    pub local_preprocess_video_workers: usize,
    #[serde(default = "default_local_preprocess_overlap_start_percent")]
    pub local_preprocess_overlap_start_percent: usize,
    #[serde(default = "default_local_preprocess_process_workers")]
    pub local_preprocess_process_workers: usize,
    #[serde(default = "default_local_preprocess_frame_workers")]
    pub local_preprocess_frame_workers: usize,
    #[serde(default)]
    pub local_preprocess_temp_dir: String,
    #[serde(default = "default_local_preprocess_secondary_temp_dir")]
    pub local_preprocess_secondary_temp_dir: String,
    #[serde(default = "default_local_preprocess_secondary_threshold_mb")]
    pub local_preprocess_secondary_threshold_mb: usize,
    #[serde(default)]
    pub nas_ssh_preprocess_enabled: bool,
    #[serde(default)]
    pub nas_ssh_host: String,
    #[serde(default)]
    pub nas_ssh_user: String,
    #[serde(default)]
    pub nas_ssh_password: String,
    #[serde(default)]
    pub nas_ssh_host_key: String,
    #[serde(default = "default_nas_ssh_port")]
    pub nas_ssh_port: u16,
    #[serde(default)]
    pub nas_remote_root: String,
    #[serde(default = "default_nas_smb_root")]
    pub nas_smb_root: String,
    #[serde(default = "default_nas_ssh_ffmpeg_workers")]
    pub nas_ssh_ffmpeg_workers: usize,
}

fn default_scan_worker_count() -> usize {
    1
}

fn default_keeper_size_priority_duration_seconds() -> f64 {
    300.0
}

fn default_ui_language() -> String {
    "zh".to_string()
}

fn default_sample_hash_count() -> usize {
    11
}

fn default_temporal_hash_threshold() -> u32 {
    8
}

fn default_min_temporal_match_points() -> usize {
    3
}

fn default_allowed_unmatched_sample_frames() -> usize {
    14
}

fn default_restrict_scan_to_test_path() -> bool {
    true
}

fn default_ai_device() -> String {
    "auto".to_string()
}

fn default_ai_frame_count() -> usize {
    128
}

fn default_ai_batch_size() -> usize {
    32
}

fn default_ai_similarity_threshold() -> f64 {
    0.86
}

fn default_ai_min_matched_frames() -> usize {
    8
}

fn default_ai_clip_matching_enabled() -> bool {
    true
}

fn default_ai_extract_worker_count() -> usize {
    4
}

fn default_ai_gpu_worker_count() -> usize {
    4
}

fn default_ai_match_worker_count() -> usize {
    8
}

fn default_ai_frame_cache_enabled() -> bool {
    true
}

fn default_ram_disk_enabled() -> bool {
    true
}

fn default_ram_disk_size_mb() -> usize {
    16 * 1024
}

fn default_local_preprocess_video_workers() -> usize {
    1
}

fn default_local_preprocess_overlap_start_percent() -> usize {
    95
}

fn default_local_preprocess_process_workers() -> usize {
    2
}

fn default_local_preprocess_frame_workers() -> usize {
    16
}

fn default_local_preprocess_secondary_temp_dir() -> String {
    r"D:\TEMP".to_string()
}

fn default_local_preprocess_secondary_threshold_mb() -> usize {
    16 * 1024
}

fn default_nas_ssh_port() -> u16 {
    22
}

fn default_nas_smb_root() -> String {
    r"\\EXAMPLE-NAS\Test".to_string()
}

fn default_nas_ssh_ffmpeg_workers() -> usize {
    1
}
