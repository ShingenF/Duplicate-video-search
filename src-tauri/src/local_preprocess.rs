use crate::database;
use crate::media;
use crate::models::{AppSettings, LocalPreprocessSummary, VideoRecord};
use crate::operations;
use crate::paths;
use anyhow::{anyhow, Context};
use std::collections::{HashSet, VecDeque};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Condvar, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const STAGING_FREE_RESERVE_BYTES: u64 = 0;
const COPY_BUFFER_BYTES: usize = 8 * 1024 * 1024;

#[derive(Debug, Clone)]
pub struct LocalPreprocessOptions {
    pub video_workers: usize,
    pub overlap_start_percent: usize,
    pub process_workers: usize,
    pub frame_workers: usize,
    pub temp_dir: Option<PathBuf>,
    pub secondary_temp_dir: Option<PathBuf>,
    pub secondary_threshold_mb: usize,
}

impl Default for LocalPreprocessOptions {
    fn default() -> Self {
        Self {
            video_workers: 1,
            overlap_start_percent: 95,
            process_workers: 2,
            frame_workers: 0,
            temp_dir: None,
            secondary_temp_dir: None,
            secondary_threshold_mb: 16 * 1024,
        }
    }
}

pub fn enabled(settings: &AppSettings) -> bool {
    settings.local_preprocess_enabled
}

pub fn options_from_settings(settings: &AppSettings) -> LocalPreprocessOptions {
    LocalPreprocessOptions {
        video_workers: settings.local_preprocess_video_workers.clamp(1, 8),
        overlap_start_percent: settings
            .local_preprocess_overlap_start_percent
            .clamp(50, 99),
        process_workers: settings.local_preprocess_process_workers.clamp(1, 4),
        frame_workers: settings.local_preprocess_frame_workers.clamp(1, 64),
        secondary_threshold_mb: settings
            .local_preprocess_secondary_threshold_mb
            .clamp(512, 1_048_576),
        temp_dir: if settings.local_preprocess_temp_dir.trim().is_empty() {
            None
        } else {
            Some(PathBuf::from(settings.local_preprocess_temp_dir.trim()))
        },
        secondary_temp_dir: if settings
            .local_preprocess_secondary_temp_dir
            .trim()
            .is_empty()
        {
            None
        } else {
            Some(PathBuf::from(
                settings.local_preprocess_secondary_temp_dir.trim(),
            ))
        },
    }
}

#[derive(Debug, Clone)]
pub struct LocalPreprocessProgress {
    pub total_videos: usize,
    pub started: usize,
    pub imported: usize,
    pub skipped: usize,
    pub insufficient_frames: usize,
    pub failed: usize,
    pub phase: String,
    pub current_path: Option<String>,
}

#[derive(Debug, Clone)]
struct LocalVideoJob {
    video: VideoRecord,
    frame_count: usize,
}

enum PipelineMessage {
    CopyStarted(String),
    CopyWaiting {
        message: String,
    },
    CopyFinished {
        path: String,
        result: anyhow::Result<CopyOutcome>,
    },
    ProcessFinished {
        path: String,
        result: anyhow::Result<VideoOutcome>,
    },
}

struct StagedVideoJob {
    video: VideoRecord,
    source_path: PathBuf,
    source_modified_unix_ms: i64,
    staged_path: PathBuf,
    video_dir: PathBuf,
}

enum CopyOutcome {
    Staged(StagedVideoJob),
    Skipped(VideoRecord),
}

enum VideoOutcome {
    Imported(VideoRecord),
    InsufficientFrames {
        video: VideoRecord,
        extracted: usize,
        requested: usize,
        minimum: usize,
    },
}

pub fn preprocess_ai_frame_cache<F>(
    videos: &[VideoRecord],
    settings: &AppSettings,
    options: LocalPreprocessOptions,
    on_progress: F,
) -> anyhow::Result<LocalPreprocessSummary>
where
    F: FnMut(LocalPreprocessProgress),
{
    preprocess_ai_frame_cache_with_ready(videos, settings, options, on_progress, |_| Ok(()))
}

pub fn preprocess_ai_frame_cache_with_ready<F, G>(
    videos: &[VideoRecord],
    settings: &AppSettings,
    options: LocalPreprocessOptions,
    mut on_progress: F,
    mut on_ready: G,
) -> anyhow::Result<LocalPreprocessSummary>
where
    F: FnMut(LocalPreprocessProgress),
    G: FnMut(VideoRecord) -> anyhow::Result<()>,
{
    paths::ensure_data_dirs()?;

    if !settings.ai_vision_enabled || !settings.ai_frame_cache_enabled {
        return Ok(LocalPreprocessSummary {
            total_videos: 0,
            imported: 0,
            skipped: videos.len(),
            insufficient_frames: 0,
            failed: 0,
            staging_dir: None,
            recent_errors: Vec::new(),
        });
    }

    let mut skipped = 0usize;
    let mut jobs = VecDeque::new();
    let mut recent_errors = Vec::new();
    let total_videos = videos.len();

    for video in videos {
        if video.scan_status != "ok" || video.id.is_none() {
            skipped += 1;
            continue;
        }
        let frame_count = expected_ai_frame_count(video, settings.ai_frame_count);
        if media::video_has_full_scan(video, settings)
            && media::load_ai_frame_cache_for_video(video, frame_count)
                .ok()
                .flatten()
                .is_some()
        {
            skipped += 1;
            on_ready(video.clone())?;
            continue;
        }
        jobs.push_back(LocalVideoJob {
            video: video.clone(),
            frame_count,
        });
    }

    on_progress(LocalPreprocessProgress {
        total_videos,
        started: skipped,
        imported: 0,
        skipped,
        insufficient_frames: 0,
        failed: 0,
        phase: "starting".to_string(),
        current_path: Some("starting local staged preprocessing".to_string()),
    });

    let job_total = jobs.len();
    if jobs.is_empty() {
        return Ok(LocalPreprocessSummary {
            total_videos,
            imported: 0,
            skipped,
            insufficient_frames: 0,
            failed: 0,
            staging_dir: None,
            recent_errors,
        });
    }

    let primary_staging_base = options
        .temp_dir
        .unwrap_or_else(|| paths::data_dir().join("local-video-staging"));
    let staging_plan = Arc::new(StagingPlan::new(
        primary_staging_base,
        options.secondary_temp_dir,
        (options.secondary_threshold_mb as u64).saturating_mul(1024 * 1024),
        &mut recent_errors,
    )?);

    let queue = Arc::new(Mutex::new(jobs));
    let (message_sender, message_receiver) = mpsc::channel();
    let (staged_sender, staged_receiver) = mpsc::channel::<StagedVideoJob>();
    let staged_receiver = Arc::new(Mutex::new(staged_receiver));
    let video_workers = options.video_workers.clamp(1, 8).min(job_total.max(1));
    let overlap_start_percent = options.overlap_start_percent.clamp(50, 99);
    let copy_pacer = Arc::new(CopyPacer::new(video_workers));
    let process_workers = options.process_workers.clamp(1, 4).min(job_total.max(1));
    let frame_workers = if options.frame_workers == 0 {
        settings.ai_extract_worker_count
    } else {
        options.frame_workers
    }
    .clamp(1, 64);

    for worker_index in 0..video_workers {
        let queue = Arc::clone(&queue);
        let sender = message_sender.clone();
        let settings = settings.clone();
        let staging_plan = Arc::clone(&staging_plan);
        let copy_pacer = Arc::clone(&copy_pacer);
        thread::spawn(move || loop {
            if crate::cancel::is_requested() {
                break;
            }
            let Ok(mut copy_permit) = copy_pacer.acquire() else {
                break;
            };
            let job = {
                let mut queue = queue.lock().expect("local preprocess queue lock poisoned");
                queue.pop_front()
            };
            let Some(job) = job else {
                break;
            };

            let path = job.video.path.clone();
            if sender
                .send(PipelineMessage::CopyStarted(path.clone()))
                .is_err()
            {
                break;
            }
            let wait_sender = sender.clone();
            let mut on_wait = move |message: String| {
                let _ = wait_sender.send(PipelineMessage::CopyWaiting { message });
            };
            let result = stage_video_job(
                &job.video,
                &settings,
                job.frame_count,
                worker_index,
                &staging_plan,
                &mut copy_permit,
                overlap_start_percent,
                &mut on_wait,
            );
            drop(copy_permit);
            if sender
                .send(PipelineMessage::CopyFinished { path, result })
                .is_err()
            {
                break;
            }
        });
    }

    for _ in 0..process_workers {
        let receiver = Arc::clone(&staged_receiver);
        let sender = message_sender.clone();
        let settings = settings.clone();
        thread::spawn(move || loop {
            if crate::cancel::is_requested() {
                break;
            }
            let staged = {
                let receiver = receiver
                    .lock()
                    .expect("local staging processor queue lock poisoned");
                receiver.recv()
            };
            let Ok(staged) = staged else {
                break;
            };
            let path = staged.video.path.clone();
            let result = process_staged_video_job(staged, &settings, frame_workers);
            if sender
                .send(PipelineMessage::ProcessFinished { path, result })
                .is_err()
            {
                break;
            }
        });
    }
    drop(message_sender);

    let mut imported = 0usize;
    let mut insufficient_frames = 0usize;
    let mut failed = 0usize;
    let mut completed_jobs = 0usize;
    let mut copied_jobs = 0usize;
    let mut started = skipped;
    let mut staged_sender = Some(staged_sender);

    while completed_jobs < job_total {
        if crate::cancel::is_requested() {
            break;
        }
        let Ok(message) = message_receiver.recv() else {
            break;
        };
        match message {
            PipelineMessage::CopyStarted(path) => {
                started = started.saturating_add(1).min(total_videos);
                on_progress(LocalPreprocessProgress {
                    total_videos,
                    started,
                    imported,
                    skipped,
                    insufficient_frames,
                    failed,
                    phase: format!("copying x{video_workers} / processing x{process_workers}"),
                    current_path: Some(path),
                })
            }
            PipelineMessage::CopyWaiting { message } => on_progress(LocalPreprocessProgress {
                total_videos,
                started,
                imported,
                skipped,
                insufficient_frames,
                failed,
                phase: "waiting-cache-space".to_string(),
                current_path: Some(message),
            }),
            PipelineMessage::CopyFinished { path, result } => {
                copied_jobs += 1;
                match result {
                    Ok(CopyOutcome::Staged(staged)) => {
                        let send_result = staged_sender
                            .as_ref()
                            .ok_or_else(|| anyhow!("staging processor queue is closed"))
                            .and_then(|sender| {
                                sender
                                    .send(staged)
                                    .map_err(|error| anyhow!("queue staged video failed: {error}"))
                            });
                        if let Err(error) = send_result {
                            failed += 1;
                            completed_jobs += 1;
                            push_recent_error(&mut recent_errors, format!("{path}: {error}"));
                        }
                    }
                    Ok(CopyOutcome::Skipped(video)) => {
                        skipped += 1;
                        completed_jobs += 1;
                        on_ready(video)?;
                    }
                    Err(error) => {
                        failed += 1;
                        completed_jobs += 1;
                        push_recent_error(&mut recent_errors, format!("{path}: {error}"));
                    }
                }
                if copied_jobs >= job_total {
                    staged_sender.take();
                }
                on_progress(LocalPreprocessProgress {
                    total_videos,
                    started,
                    imported,
                    skipped,
                    insufficient_frames,
                    failed,
                    phase: format!("copying x{video_workers} / processing x{process_workers}"),
                    current_path: Some(path),
                });
            }
            PipelineMessage::ProcessFinished { path, result } => {
                completed_jobs += 1;
                match result {
                    Ok(VideoOutcome::Imported(video)) => {
                        imported += 1;
                        on_ready(video)?;
                    }
                    Ok(VideoOutcome::InsufficientFrames {
                        video: _video,
                        extracted: _extracted,
                        requested: _requested,
                        minimum: _minimum,
                    }) => {
                        insufficient_frames += 1;
                    }
                    Err(error) => {
                        failed += 1;
                        push_recent_error(&mut recent_errors, format!("{path}: {error}"));
                    }
                }
                on_progress(LocalPreprocessProgress {
                    total_videos,
                    started,
                    imported,
                    skipped,
                    insufficient_frames,
                    failed,
                    phase: format!("copying x{video_workers} / processing x{process_workers}"),
                    current_path: Some(path),
                });
            }
        }
    }
    drop(staged_sender);

    staging_plan.cleanup_current_runs(&mut recent_errors);
    staging_plan.cleanup_stale_runs(&mut recent_errors);

    if crate::cancel::is_requested() {
        on_progress(LocalPreprocessProgress {
            total_videos,
            started,
            imported,
            skipped,
            insufficient_frames,
            failed,
            phase: "cancelled".to_string(),
            current_path: None,
        });
        return Err(anyhow!("operation cancelled"));
    }

    on_progress(LocalPreprocessProgress {
        total_videos,
        started,
        imported,
        skipped,
        insufficient_frames,
        failed,
        phase: "completed".to_string(),
        current_path: None,
    });

    Ok(LocalPreprocessSummary {
        total_videos,
        imported,
        skipped,
        insufficient_frames,
        failed,
        staging_dir: if recent_errors
            .iter()
            .any(|error| error.contains("failed to remove staging directory"))
        {
            Some(staging_plan.current_roots_label())
        } else {
            None
        },
        recent_errors,
    })
}

fn stage_video_job(
    video: &VideoRecord,
    settings: &AppSettings,
    requested_frame_count: usize,
    worker_index: usize,
    staging_plan: &Arc<StagingPlan>,
    copy_permit: &mut CopyPermit,
    overlap_start_percent: usize,
    on_wait: &mut impl FnMut(String),
) -> anyhow::Result<CopyOutcome> {
    crate::cancel::bail_if_requested()?;
    if media::video_has_full_scan(video, settings)
        && media::load_ai_frame_cache_for_video(video, requested_frame_count)?.is_some()
    {
        return Ok(CopyOutcome::Skipped(video.clone()));
    }

    let source_display_path = PathBuf::from(&video.path);
    if !operations::is_path_allowed_by_settings(&source_display_path, settings) {
        return Err(anyhow!(
            "video path is outside the allowed scan scope: {}",
            source_display_path.display()
        ));
    }
    let source_path = media::filesystem_input_path(&source_display_path);
    let source_modified_unix_ms = validate_source_metadata(video, &source_path)?;

    stage_video_job_disk(
        video,
        worker_index,
        staging_plan,
        copy_permit,
        overlap_start_percent,
        on_wait,
        source_path,
        source_modified_unix_ms,
    )
}

fn stage_video_job_disk(
    video: &VideoRecord,
    worker_index: usize,
    staging_plan: &Arc<StagingPlan>,
    copy_permit: &mut CopyPermit,
    overlap_start_percent: usize,
    on_wait: &mut impl FnMut(String),
    source_path: PathBuf,
    source_modified_unix_ms: i64,
) -> anyhow::Result<CopyOutcome> {
    let video_id = video
        .id
        .ok_or_else(|| anyhow!("video record is missing an id: {}", video.path))?;
    let staging_reservation = staging_plan.reserve(video.size_bytes, on_wait)?;
    let worker_dir = staging_reservation
        .tier
        .root
        .join(format!("worker-{worker_index}"));
    let video_dir = worker_dir.join(format!("video-{video_id}"));
    let staged_path = video_dir.join(stage_file_name(video));
    fs::create_dir_all(&video_dir)
        .with_context(|| format!("create staging directory {}", video_dir.display()))?;
    let staging_gate = Arc::clone(&staging_reservation.tier.gate);

    let copy_result = (|| {
        copy_with_cancel(
            &source_path,
            &staged_path,
            Some(&staging_gate),
            on_wait,
            |copied| {
                if copy_overlap_reached(copied, video.size_bytes, overlap_start_percent) {
                    copy_permit.release_next();
                }
            },
        )
        .with_context(|| {
            format!(
                "copy {} to local staging {}",
                source_path.display(),
                staged_path.display()
            )
        })?;
        let copied_size = staged_path
            .metadata()
            .with_context(|| format!("read metadata for {}", staged_path.display()))?
            .len();
        if video.size_bytes > 0 && copied_size != video.size_bytes {
            return Err(anyhow!(
                "copied file size differs from index: {} vs {}",
                copied_size,
                video.size_bytes
            ));
        }
        Ok(())
    })();

    match copy_result {
        Ok(()) => Ok(CopyOutcome::Staged(StagedVideoJob {
            video: video.clone(),
            source_path,
            source_modified_unix_ms,
            staged_path,
            video_dir,
        })),
        Err(error) => {
            let _ = fs::remove_dir_all(&video_dir);
            Err(error)
        }
    }
}

fn process_staged_video_job(
    staged: StagedVideoJob,
    settings: &AppSettings,
    frame_workers: usize,
) -> anyhow::Result<VideoOutcome> {
    let StagedVideoJob {
        video,
        source_path,
        source_modified_unix_ms,
        staged_path,
        video_dir,
    } = staged;

    let result = (|| {
        let mut indexed_video = media::scan_video_from_local_copy(
            &source_path,
            &staged_path,
            source_modified_unix_ms,
            settings,
        )?;
        indexed_video.id = video.id;
        let conn = database::open_database()?;
        let indexed_id = database::upsert_video_preserving_content_identity(&conn, &indexed_video)?;
        indexed_video.id = Some(indexed_id);

        let frame_count = expected_ai_frame_count(&indexed_video, settings.ai_frame_count);
        if media::load_ai_frame_cache_for_video(&indexed_video, frame_count)?.is_some() {
            return Ok(VideoOutcome::Imported(indexed_video));
        }

        let positions =
            media::sample_positions(indexed_video.duration_seconds.unwrap_or(0.0), frame_count);
        let extracted = extract_frames_from_staged(&staged_path, &positions, frame_workers)?;
        let minimum_frames = minimum_usable_frame_count(positions.len());
        if extracted.len() < minimum_frames {
            return Ok(VideoOutcome::InsufficientFrames {
                video: indexed_video,
                extracted: extracted.len(),
                requested: positions.len(),
                minimum: minimum_frames,
            });
        }

        let ordered_positions = extracted
            .iter()
            .map(|(_, position, _)| *position)
            .collect::<Vec<_>>();
        let frames = extracted
            .into_iter()
            .map(|(_, _, frame)| frame)
            .collect::<Vec<_>>();
        media::write_ai_frame_cache_for_video(
            &indexed_video,
            frame_count,
            ordered_positions,
            frames,
        )?
        .ok_or_else(|| anyhow!("local staged frame cache was incomplete"))?;

        Ok(VideoOutcome::Imported(indexed_video))
    })();

    let cleanup_result = fs::remove_dir_all(&video_dir);
    match (result, cleanup_result) {
        (Ok(outcome), Ok(())) => Ok(outcome),
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(anyhow!(
            "local frame cache was written, but removing staged video failed: {error}"
        )),
    }
}

fn minimum_usable_frame_count(expected: usize) -> usize {
    if expected <= 1 {
        return expected;
    }
    ((expected as f64) * 0.75).ceil() as usize
}

fn extract_frames_from_staged(
    staged_path: &Path,
    positions: &[f64],
    frame_workers: usize,
) -> anyhow::Result<Vec<(usize, f64, Vec<u8>)>> {
    let indexed_positions = positions
        .iter()
        .enumerate()
        .map(|(index, position)| (index, *position))
        .collect::<Vec<_>>();
    let mut frames = Vec::new();

    for work in indexed_positions.chunks(frame_workers.clamp(1, 64)) {
        crate::cancel::bail_if_requested()?;
        let mut extracted = thread::scope(|scope| {
            work.iter()
                .map(|(frame_index, position)| {
                    scope.spawn(move || {
                        media::extract_ai_rgb_frame(staged_path, *position)
                            .map(|frame| (*frame_index, *position, frame))
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .filter_map(|handle| handle.join().ok()?.ok())
                .collect::<Vec<_>>()
        });
        frames.append(&mut extracted);
    }

    frames.sort_by_key(|(frame_index, _, _)| *frame_index);
    Ok(frames)
}

struct StagingPlan {
    primary: Arc<StagingTier>,
    secondary: Option<Arc<StagingTier>>,
    secondary_threshold_bytes: u64,
}

struct StagingTier {
    base: PathBuf,
    root: PathBuf,
    gate: Arc<DiskCapacityGate>,
}

struct StagingReservation {
    tier: Arc<StagingTier>,
    _reservation: DiskReservation,
}

impl StagingPlan {
    fn new(
        primary_base: PathBuf,
        secondary_base: Option<PathBuf>,
        secondary_threshold_bytes: u64,
        recent_errors: &mut Vec<String>,
    ) -> anyhow::Result<Self> {
        cleanup_stale_staging_runs(&primary_base, None, recent_errors);
        let run_id = media::now_ms();
        let primary = Arc::new(StagingTier::new("primary", primary_base, run_id)?);

        let secondary = secondary_base
            .filter(|path| !same_staging_base(path, &primary.base))
            .map(|base| {
                cleanup_stale_staging_runs(&base, None, recent_errors);
                StagingTier::new("secondary", base, run_id).map(Arc::new)
            })
            .transpose()?;

        Ok(Self {
            primary,
            secondary,
            secondary_threshold_bytes,
        })
    }

    fn reserve(
        self: &Arc<Self>,
        bytes: u64,
        on_wait: &mut impl FnMut(String),
    ) -> anyhow::Result<StagingReservation> {
        if let Some(secondary) = &self.secondary {
            if bytes >= self.secondary_threshold_bytes.max(1) {
                let reservation = secondary.gate.acquire(bytes, on_wait).with_context(|| {
                    format!(
                        "secondary cache {} cannot reserve {}",
                        secondary.base.display(),
                        human_bytes(bytes)
                    )
                })?;
                return Ok(StagingReservation {
                    tier: Arc::clone(secondary),
                    _reservation: reservation,
                });
            }
        }

        match self.primary.gate.acquire(bytes, on_wait) {
            Ok(reservation) => {
                return Ok(StagingReservation {
                    tier: Arc::clone(&self.primary),
                    _reservation: reservation,
                });
            }
            Err(_primary_error)
                if self.secondary.is_some()
                    && self
                        .primary
                        .gate
                        .can_fit_single_request(bytes)
                        .is_some_and(|fits| !fits) =>
            {
                let secondary = self
                    .secondary
                    .as_ref()
                    .expect("secondary exists when primary cannot fit single request");
                let reservation = secondary.gate.acquire(bytes, on_wait).with_context(|| {
                    format!(
                        "primary cache {} cannot fit {}; secondary cache {} also failed",
                        self.primary.base.display(),
                        human_bytes(bytes),
                        secondary.base.display()
                    )
                })?;
                return Ok(StagingReservation {
                    tier: Arc::clone(secondary),
                    _reservation: reservation,
                });
            }
            Err(error) => return Err(error),
        }
    }

    fn cleanup_current_runs(&self, recent_errors: &mut Vec<String>) {
        cleanup_staging_dir(&self.primary.root, recent_errors);
        if let Some(secondary) = &self.secondary {
            cleanup_staging_dir(&secondary.root, recent_errors);
        }
    }

    fn cleanup_stale_runs(&self, recent_errors: &mut Vec<String>) {
        cleanup_stale_staging_runs(&self.primary.base, Some(&self.primary.root), recent_errors);
        if let Some(secondary) = &self.secondary {
            cleanup_stale_staging_runs(&secondary.base, Some(&secondary.root), recent_errors);
        }
    }

    fn current_roots_label(&self) -> String {
        match &self.secondary {
            Some(secondary) => format!(
                "{}; {}",
                self.primary.root.display(),
                secondary.root.display()
            ),
            None => self.primary.root.display().to_string(),
        }
    }
}

impl StagingTier {
    fn new(name: &'static str, base: PathBuf, run_id: i64) -> anyhow::Result<Self> {
        let root = base.join(format!("run-{run_id}"));
        fs::create_dir_all(&root)
            .with_context(|| format!("create staging directory {}", root.display()))?;
        Ok(Self {
            base: base.clone(),
            root,
            gate: Arc::new(DiskCapacityGate::new(name, base)),
        })
    }
}

fn same_staging_base(left: &Path, right: &Path) -> bool {
    normalize_path_text(left) == normalize_path_text(right)
}

fn normalize_path_text(path: &Path) -> String {
    path.to_string_lossy()
        .replace('/', "\\")
        .trim_end_matches('\\')
        .to_lowercase()
}

struct DiskCapacityGate {
    name: &'static str,
    staging_base: PathBuf,
    state: Mutex<DiskCapacityState>,
    changed: Condvar,
}

#[derive(Debug, Default)]
struct DiskCapacityState {
    reserved_bytes: u64,
}

struct DiskReservation {
    gate: Arc<DiskCapacityGate>,
    bytes: u64,
}

struct CopyPacer {
    max_active: usize,
    state: Mutex<CopyPacerState>,
    changed: Condvar,
}

#[derive(Debug)]
struct CopyPacerState {
    active: usize,
    available_starts: usize,
    next_start_index: usize,
    threshold_reached: HashSet<usize>,
    completed: HashSet<usize>,
    released_next_for: HashSet<usize>,
}

struct CopyPermit {
    pacer: Arc<CopyPacer>,
    start_index: usize,
}

impl CopyPacer {
    fn new(max_active: usize) -> Self {
        Self {
            max_active: max_active.clamp(1, 8),
            state: Mutex::new(CopyPacerState {
                active: 0,
                available_starts: 1,
                next_start_index: 1,
                threshold_reached: HashSet::new(),
                completed: HashSet::new(),
                released_next_for: HashSet::new(),
            }),
            changed: Condvar::new(),
        }
    }

    fn acquire(self: &Arc<Self>) -> anyhow::Result<CopyPermit> {
        loop {
            crate::cancel::bail_if_requested()?;
            let mut state = self.state.lock().expect("copy pacer lock poisoned");
            if state.available_starts > 0 {
                state.available_starts -= 1;
                state.active += 1;
                let start_index = state.next_start_index;
                state.next_start_index = state.next_start_index.saturating_add(1);
                return Ok(CopyPermit {
                    pacer: Arc::clone(self),
                    start_index,
                });
            }
            let _ = self
                .changed
                .wait_timeout(state, Duration::from_millis(250))
                .expect("copy pacer wait poisoned");
        }
    }

    fn release_next_if_ready(&self, start_index: usize) {
        let mut state = self.state.lock().expect("copy pacer lock poisoned");
        state.threshold_reached.insert(start_index);
        self.release_next_locked(&mut state, start_index);
    }

    fn finish_copy(&self, start_index: usize) {
        let mut state = self.state.lock().expect("copy pacer lock poisoned");
        state.active = state.active.saturating_sub(1);
        state.completed.insert(start_index);
        state.threshold_reached.insert(start_index);
        self.release_next_locked(&mut state, start_index);
        self.release_next_locked(&mut state, start_index.saturating_add(1));
        self.changed.notify_one();
    }

    fn release_next_locked(&self, state: &mut CopyPacerState, start_index: usize) {
        if !state.threshold_reached.contains(&start_index)
            || state.released_next_for.contains(&start_index)
            || state.active.saturating_add(state.available_starts) >= self.max_active
        {
            return;
        }

        if start_index > 1 && !state.completed.contains(&start_index.saturating_sub(1)) {
            return;
        }

        state.released_next_for.insert(start_index);
        state.available_starts = state.available_starts.saturating_add(1);
        self.changed.notify_one();
    }

    #[cfg(test)]
    fn available_starts_for_test(&self) -> usize {
        self.state
            .lock()
            .expect("copy pacer lock poisoned")
            .available_starts
    }
}

impl CopyPermit {
    fn release_next(&mut self) {
        self.pacer.release_next_if_ready(self.start_index);
    }
}

impl Drop for CopyPermit {
    fn drop(&mut self) {
        self.pacer.finish_copy(self.start_index);
    }
}

impl DiskCapacityGate {
    fn new(name: &'static str, staging_base: PathBuf) -> Self {
        Self {
            name,
            staging_base,
            state: Mutex::new(DiskCapacityState::default()),
            changed: Condvar::new(),
        }
    }

    fn can_fit_single_request(&self, bytes: u64) -> Option<bool> {
        let bytes = bytes.max(1);
        disk_capacity_bytes(&self.staging_base).map(|capacity| {
            capacity
                .total_bytes
                .map(|total| total >= bytes.saturating_add(STAGING_FREE_RESERVE_BYTES))
                .unwrap_or(true)
        })
    }

    fn acquire(
        self: &Arc<Self>,
        bytes: u64,
        on_wait: &mut impl FnMut(String),
    ) -> anyhow::Result<DiskReservation> {
        let bytes = bytes.max(1);
        let required_bytes = bytes.saturating_add(STAGING_FREE_RESERVE_BYTES);
        let mut last_wait_notice = Instant::now()
            .checked_sub(Duration::from_secs(30))
            .unwrap_or_else(Instant::now);
        loop {
            crate::cancel::bail_if_requested()?;
            let mut state = self.state.lock().expect("disk capacity gate lock poisoned");
            let capacity = disk_capacity_bytes(&self.staging_base);
            match capacity {
                Some(capacity) if capacity.free_bytes >= required_bytes => {
                    state.reserved_bytes = state.reserved_bytes.saturating_add(bytes);
                    return Ok(DiskReservation {
                        gate: Arc::clone(self),
                        bytes,
                    });
                }
                Some(capacity)
                    if capacity
                        .total_bytes
                        .is_some_and(|total| total < required_bytes) =>
                {
                    return Err(anyhow!(
                        "{} cache disk cannot fit {}: total capacity {}, needs at least {}",
                        self.name,
                        human_bytes(bytes),
                        human_bytes(capacity.total_bytes.unwrap_or(0)),
                        human_bytes(required_bytes)
                    ));
                }
                None => {
                    state.reserved_bytes = state.reserved_bytes.saturating_add(bytes);
                    return Ok(DiskReservation {
                        gate: Arc::clone(self),
                        bytes,
                    });
                }
                Some(_capacity) => {
                    if last_wait_notice.elapsed() >= Duration::from_secs(5) {
                        last_wait_notice = Instant::now();
                        on_wait(format!(
                            "{} cache full; paused. Raise overlap % or lower secondary threshold MB.",
                            self.name,
                        ));
                    }
                    let _ = self
                        .changed
                        .wait_timeout(state, Duration::from_millis(750))
                        .expect("disk capacity gate wait poisoned");
                }
            }
        }
    }

    fn wait_for_write_room(
        &self,
        bytes: u64,
        on_wait: &mut impl FnMut(String),
    ) -> anyhow::Result<()> {
        let bytes = bytes.max(1);
        let required_bytes = bytes.saturating_add(STAGING_FREE_RESERVE_BYTES);
        let mut last_wait_notice = Instant::now()
            .checked_sub(Duration::from_secs(30))
            .unwrap_or_else(Instant::now);
        loop {
            crate::cancel::bail_if_requested()?;
            let state = self.state.lock().expect("disk capacity gate lock poisoned");
            match disk_capacity_bytes(&self.staging_base) {
                Some(capacity) if capacity.free_bytes >= required_bytes => return Ok(()),
                None => return Ok(()),
                Some(_) => {
                    if last_wait_notice.elapsed() >= Duration::from_secs(5) {
                        last_wait_notice = Instant::now();
                        on_wait(format!(
                            "{} cache full; paused. Raise overlap % or lower secondary threshold MB.",
                            self.name,
                        ));
                    }
                    let _ = self
                        .changed
                        .wait_timeout(state, Duration::from_millis(750))
                        .expect("disk capacity gate wait poisoned");
                }
            }
        }
    }

    fn release(&self, bytes: u64) {
        let mut state = self.state.lock().expect("disk capacity gate lock poisoned");
        state.reserved_bytes = state.reserved_bytes.saturating_sub(bytes);
        self.changed.notify_all();
    }
}

impl Drop for DiskReservation {
    fn drop(&mut self) {
        self.gate.release(self.bytes);
    }
}

fn copy_with_cancel<F, W>(
    source: &Path,
    destination: &Path,
    capacity_gate: Option<&Arc<DiskCapacityGate>>,
    on_wait: &mut W,
    mut on_progress: F,
) -> anyhow::Result<u64>
where
    F: FnMut(u64),
    W: FnMut(String),
{
    let mut input = File::open(source).with_context(|| format!("open {}", source.display()))?;
    let mut output =
        File::create(destination).with_context(|| format!("create {}", destination.display()))?;
    let mut buffer = vec![0u8; COPY_BUFFER_BYTES];
    let mut copied = 0u64;
    loop {
        crate::cancel::bail_if_requested()?;
        let read = input
            .read(&mut buffer)
            .with_context(|| format!("read {}", source.display()))?;
        if read == 0 {
            break;
        }
        if let Some(gate) = capacity_gate {
            gate.wait_for_write_room(read as u64, on_wait)?;
        }
        output
            .write_all(&buffer[..read])
            .with_context(|| format!("write {}", destination.display()))?;
        copied = copied.saturating_add(read as u64);
        on_progress(copied);
    }
    output
        .flush()
        .with_context(|| format!("flush {}", destination.display()))?;
    Ok(copied)
}

fn copy_overlap_reached(copied: u64, total: u64, percent: usize) -> bool {
    total > 0
        && (copied as u128).saturating_mul(100) >= (total as u128).saturating_mul(percent as u128)
}

struct DiskCapacity {
    free_bytes: u64,
    total_bytes: Option<u64>,
}

#[cfg(windows)]
fn disk_capacity_bytes(path: &Path) -> Option<DiskCapacity> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Storage::FileSystem::GetDiskFreeSpaceExW;

    let mut free_available = 0u64;
    let mut total_bytes = 0u64;
    let wide = path
        .as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect::<Vec<_>>();
    let ok = unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut free_available,
            &mut total_bytes,
            std::ptr::null_mut(),
        )
    };
    if ok == 0 {
        None
    } else {
        Some(DiskCapacity {
            free_bytes: free_available,
            total_bytes: Some(total_bytes),
        })
    }
}

#[cfg(not(windows))]
fn disk_capacity_bytes(_path: &Path) -> Option<DiskCapacity> {
    None
}

fn human_bytes(value: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut size = value as f64;
    let mut unit = 0usize;
    while size >= 1024.0 && unit < units.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", value, units[unit])
    } else {
        format!("{size:.1} {}", units[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_pacer_waits_for_previous_copy_before_third_start() {
        let pacer = Arc::new(CopyPacer::new(3));

        let mut first = pacer.acquire().expect("first copy can start");
        first.release_next();
        assert_eq!(pacer.available_starts_for_test(), 1);

        let mut second = pacer.acquire().expect("second copy can start");
        assert_eq!(pacer.available_starts_for_test(), 0);

        second.release_next();
        assert_eq!(
            pacer.available_starts_for_test(),
            0,
            "third copy waits until the first copy is complete"
        );

        drop(first);
        assert_eq!(
            pacer.available_starts_for_test(),
            1,
            "third copy can start after first copy is complete and second reached threshold"
        );

        let _third = pacer.acquire().expect("third copy can start");
        assert_eq!(pacer.available_starts_for_test(), 0);
    }

    #[test]
    fn copy_pacer_stays_serial_when_max_active_is_one() {
        let pacer = Arc::new(CopyPacer::new(1));

        let mut first = pacer.acquire().expect("first copy can start");
        first.release_next();
        assert_eq!(pacer.available_starts_for_test(), 0);

        drop(first);
        assert_eq!(pacer.available_starts_for_test(), 1);
    }

    #[test]
    fn copy_pacer_accepts_eight_as_upper_bound() {
        assert_eq!(CopyPacer::new(8).max_active, 8);
        assert_eq!(CopyPacer::new(9).max_active, 8);
    }
}

fn validate_source_metadata(video: &VideoRecord, source_path: &Path) -> anyhow::Result<i64> {
    let metadata = source_path
        .metadata()
        .with_context(|| format!("read metadata for {}", source_path.display()))?;
    if !metadata.is_file() {
        return Err(anyhow!("source is not a file: {}", source_path.display()));
    }
    if video.size_bytes > 0 && metadata.len() != video.size_bytes {
        return Err(anyhow!(
            "source file size changed since scan: {} vs {}",
            metadata.len(),
            video.size_bytes
        ));
    }
    let modified_unix_ms = metadata
        .modified()
        .ok()
        .and_then(system_time_to_ms)
        .unwrap_or_else(media::now_ms);
    if video.modified_unix_ms > 0 {
        if (modified_unix_ms - video.modified_unix_ms).abs() > 2_000 {
            return Err(anyhow!(
                "source modified time changed since scan: {} vs {}",
                modified_unix_ms,
                video.modified_unix_ms
            ));
        }
    }
    Ok(modified_unix_ms)
}

fn cleanup_staging_dir(path: &Path, errors: &mut Vec<String>) {
    if !path.exists() {
        return;
    }
    if let Err(error) = fs::remove_dir_all(path) {
        push_recent_error(
            errors,
            format!(
                "failed to remove staging directory {}: {error}",
                path.display()
            ),
        );
    }
}

fn cleanup_stale_staging_runs(base_dir: &Path, keep_dir: Option<&Path>, errors: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(base_dir) else {
        return;
    };
    let keep_dir = keep_dir.and_then(|path| path.canonicalize().ok());
    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_dir() || !is_run_staging_dir(&path) {
            continue;
        }
        if keep_dir
            .as_ref()
            .is_some_and(|keep| path.canonicalize().ok().as_ref() == Some(keep))
        {
            continue;
        }
        cleanup_staging_dir(&path, errors);
    }
}

fn is_run_staging_dir(path: &Path) -> bool {
    path.file_name()
        .and_then(|value| value.to_str())
        .is_some_and(|name| {
            name.strip_prefix("run-")
                .is_some_and(|rest| rest.len() >= 10 && rest.chars().all(|ch| ch.is_ascii_digit()))
        })
}

fn stage_file_name(video: &VideoRecord) -> String {
    let extension = video.extension.trim().trim_start_matches('.');
    let video_id = video.id.unwrap_or(0);
    if extension.is_empty() {
        format!("video-{video_id}.bin")
    } else {
        format!("video-{video_id}.{extension}")
    }
}

fn expected_ai_frame_count(video: &VideoRecord, configured: usize) -> usize {
    if video.duration_seconds.unwrap_or(0.0) > 6.0 {
        configured.clamp(8, 512)
    } else {
        1
    }
}

fn push_recent_error(errors: &mut Vec<String>, message: String) {
    errors.push(message);
    if errors.len() > 8 {
        errors.remove(0);
    }
}

fn system_time_to_ms(value: SystemTime) -> Option<i64> {
    value
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_millis() as i64)
}
