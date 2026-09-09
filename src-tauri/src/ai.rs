use crate::database;
use crate::media;
use crate::models::{
    AiIndexProgress, AiIndexSummary, AiModelStatus, AiPairScoreRecord, AppSettings,
    FrameEmbeddingRecord, MatchDetail, MatchGroup, MatchHitPoint, MatchItem, MatchRange,
    MatchRefreshProgress, VideoRecord,
};
use crate::operations;
use anyhow::{anyhow, Context};
use ndarray::{Array2, Array4};
use ort::{
    ep,
    session::{builder::GraphOptimizationLevel, Session},
    value::{Tensor, TensorRef},
};
use rusqlite::params;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const AI_MODEL_NAME: &str = "local-onnx-vision";
const AI_RUNTIME_NAME: &str = "onnxruntime";
const AI_MATCH_CACHE_VERSION: &str = "ai-match-v4";
const AI_PAIR_SCORE_READ_CHUNK: usize = 512;
const AI_PAIR_SCORE_WRITE_CHUNK: usize = 4096;
const AI_MATCH_DETAIL_MAX_PAIRS_PER_GROUP: usize = 3;
const AI_MATCH_DETAIL_MAX_POINTS: usize = 64;
const FRAME_SIMILARITY_MATMUL_ONNX: &[u8] = &[
    8, 7, 18, 22, 100, 117, 112, 108, 105, 99, 97, 116, 101, 45, 118, 105, 100, 101, 111, 45, 115,
    101, 97, 114, 99, 104, 58, 145, 1, 10, 22, 10, 1, 97, 10, 1, 98, 18, 6, 115, 99, 111, 114, 101,
    115, 34, 6, 77, 97, 116, 77, 117, 108, 18, 23, 102, 114, 97, 109, 101, 95, 115, 105, 109, 105,
    108, 97, 114, 105, 116, 121, 95, 109, 97, 116, 109, 117, 108, 90, 26, 10, 1, 97, 18, 21, 10,
    19, 8, 1, 18, 15, 10, 6, 18, 4, 114, 111, 119, 115, 10, 5, 18, 3, 100, 105, 109, 90, 29, 10, 1,
    98, 18, 24, 10, 22, 8, 1, 18, 18, 10, 5, 18, 3, 100, 105, 109, 10, 9, 18, 7, 99, 111, 108, 117,
    109, 110, 115, 98, 35, 10, 6, 115, 99, 111, 114, 101, 115, 18, 25, 10, 23, 8, 1, 18, 19, 10, 6,
    18, 4, 114, 111, 119, 115, 10, 9, 18, 7, 99, 111, 108, 117, 109, 110, 115, 66, 4, 10, 0, 16,
    13,
];

fn ort_error<E: std::fmt::Display>(error: E) -> anyhow::Error {
    anyhow!("{error}")
}

#[derive(Debug, Clone)]
struct PreparedModel {
    model_id: String,
    model_hash: String,
    model_path: PathBuf,
}

#[derive(Debug)]
enum IndexVideoOutcome {
    Processed,
    Skipped,
    InsufficientFrames(String),
    Failed(String),
}

#[derive(Debug)]
enum AiWorkerOutcome {
    Embedded(Vec<FrameEmbeddingRecord>),
    Failed(String),
}

#[derive(Debug)]
struct AiWorkerResult {
    video_id: i64,
    path: String,
    video: VideoRecord,
    outcome: AiWorkerOutcome,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AiPairDebugReport {
    pub left: AiPairDebugVideo,
    pub right: AiPairDebugVideo,
    pub settings: AiPairDebugSettings,
    pub model_id: Option<String>,
    pub current_cache_key: Option<String>,
    pub current_cached_score: Option<AiPairDebugCachedScore>,
    pub cached_scores: Vec<AiPairDebugCachedScore>,
    pub embedding_summary: Option<AiPairDebugEmbeddingSummary>,
    pub threshold_scores: Vec<AiPairDebugThresholdScore>,
    pub diagnosis: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AiPairDebugVideo {
    pub requested_path: String,
    pub found: bool,
    pub id: Option<i64>,
    pub indexed_path: Option<String>,
    pub file_name: Option<String>,
    pub duration_seconds: Option<f64>,
    pub size_bytes: Option<u64>,
    pub scan_status: Option<String>,
    pub embedding_count: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AiPairDebugSettings {
    pub ai_vision_enabled: bool,
    pub ai_similarity_threshold: f64,
    pub ai_min_matched_frames: usize,
    pub ai_frame_count: usize,
    pub ai_clip_matching_enabled: bool,
    pub ai_match_worker_count: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AiPairDebugCachedScore {
    pub cache_key: String,
    pub confidence: f64,
    pub matched: usize,
    pub compared: usize,
    pub average_similarity: f64,
    pub coverage: f64,
    pub created_unix_ms: i64,
    pub is_current_cache_key: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AiPairDebugEmbeddingSummary {
    pub max_frame_similarity: f64,
    pub average_best_short_frame_similarity: f64,
    pub p50_best_short_frame_similarity: f64,
    pub p75_best_short_frame_similarity: f64,
    pub p90_best_short_frame_similarity: f64,
    pub p95_best_short_frame_similarity: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AiPairDebugThresholdScore {
    pub frame_similarity_threshold: f64,
    pub confidence: f64,
    pub matched: usize,
    pub compared: usize,
    pub average_similarity: f64,
    pub coverage: f64,
    pub passes_min_matched_frames: bool,
}

pub fn ai_model_status() -> anyhow::Result<AiModelStatus> {
    let settings = operations::get_app_settings()?;
    if !settings.ai_vision_enabled {
        return Ok(AiModelStatus {
            enabled: false,
            ready: false,
            model_id: None,
            model_path: settings.ai_model_path,
            device: settings.ai_device,
            message: "AI vision matching is disabled".to_string(),
        });
    }

    let model_path = resolve_model_path(&settings.ai_model_path);
    if settings.ai_model_path.trim().is_empty() {
        return Ok(AiModelStatus {
            enabled: true,
            ready: false,
            model_id: None,
            model_path: String::new(),
            device: settings.ai_device,
            message: "AI model path is empty".to_string(),
        });
    }
    if !model_path.exists() {
        return Ok(AiModelStatus {
            enabled: true,
            ready: false,
            model_id: None,
            model_path: model_path.display().to_string(),
            device: settings.ai_device,
            message: "AI model file was not found".to_string(),
        });
    }

    let prepared = prepare_model(&settings)?;
    Ok(AiModelStatus {
        enabled: true,
        ready: true,
        model_id: Some(prepared.model_id),
        model_path: prepared.model_path.display().to_string(),
        device: settings.ai_device,
        message: "AI model file is available".to_string(),
    })
}

pub fn build_ai_index_with_progress<F>(
    session_ids: Option<Vec<i64>>,
    force_rebuild: bool,
    mut on_progress: F,
) -> anyhow::Result<AiIndexSummary>
where
    F: FnMut(AiIndexProgress),
{
    let settings = operations::get_app_settings()?;
    if !settings.ai_vision_enabled {
        return Err(anyhow!("AI vision matching is disabled in Settings"));
    }
    let prepared = prepare_model(&settings)?;
    let started = Instant::now();
    let mut conn = database::open_database()?;
    let requested_session_ids = session_ids.clone();
    let mut videos = match session_ids {
        Some(ids) => database::list_videos_for_sessions(&conn, &ids)?,
        None => database::list_videos(&conn)?,
    }
    .into_iter()
    .filter(|video| video.scan_status == "ok" && video.id.is_some())
    .collect::<Vec<_>>();

    let total_videos = videos.len();
    let mut processed = 0usize;
    let mut skipped = 0usize;
    let mut insufficient_frames = 0usize;
    let mut failed = 0usize;
    let mut recent_errors = Vec::new();
    let error_log = crate::paths::data_dir().join("ai-index-errors.log");
    let _ = std::fs::remove_file(&error_log);

    on_progress(AiIndexProgress {
        phase: "loading-model".to_string(),
        total_videos,
        processed,
        skipped,
        insufficient_frames,
        failed,
        started: 0,
        prepared: 0,
        current_path: None,
    });

    let mut indexed_video_ids = HashSet::new();

    if settings.ai_frame_cache_enabled && crate::local_preprocess::enabled(&settings) {
        let mut preprocess_videos = Vec::new();
        for video in &videos {
            crate::cancel::bail_if_requested()?;
            let Some(video_id) = video.id else {
                continue;
            };
            if try_reuse_existing_ai_index(&mut conn, video, &settings, &prepared, force_rebuild)? {
                indexed_video_ids.insert(video_id);
                skipped += 1;
                on_progress(AiIndexProgress {
                    phase: "checking-existing-index".to_string(),
                    total_videos,
                    processed,
                    skipped,
                    insufficient_frames,
                    failed,
                    started: processed + skipped + insufficient_frames + failed,
                    prepared: processed + skipped + insufficient_frames + failed,
                    current_path: Some(video.path.clone()),
                });
            } else {
                preprocess_videos.push(video.clone());
            }
        }

        if !preprocess_videos.is_empty() {
            let ai_workers = ai_worker_count(&settings, preprocess_videos.len());
            let (ai_sender, ai_results) = start_ai_workers(ai_workers, &settings, &prepared);
            let mut queued_ai = 0usize;
            let progress_sink = RefCell::new(&mut on_progress);
            let processed_count = Cell::new(processed);
            let skipped_count = Cell::new(skipped);
            let failed_count = Cell::new(failed);
            let insufficient_count = Cell::new(insufficient_frames);
            let started_count = Cell::new(processed + skipped + insufficient_frames + failed);
            let prepared_count = Cell::new(processed + skipped + insufficient_frames + failed);
            let base_skipped = skipped;
            let base_failed = failed;
            let base_insufficient = insufficient_frames;
            let summary = crate::local_preprocess::preprocess_ai_frame_cache_with_ready(
                &preprocess_videos,
                &settings,
                crate::local_preprocess::options_from_settings(&settings),
                |progress| {
                    let current_skipped = skipped_count.get();
                    let current_failed = failed_count
                        .get()
                        .max(base_failed.saturating_add(progress.failed));
                    let current_insufficient = insufficient_count
                        .get()
                        .max(base_insufficient.saturating_add(progress.insufficient_frames));
                    skipped_count.set(current_skipped);
                    failed_count.set(current_failed);
                    insufficient_count.set(current_insufficient);
                    let terminal = processed_count
                        .get()
                        .saturating_add(current_skipped)
                        .saturating_add(current_insufficient)
                        .saturating_add(current_failed);
                    let current_started = started_count
                        .get()
                        .max(base_skipped.saturating_add(progress.started))
                        .max(terminal)
                        .min(total_videos);
                    let current_prepared = prepared_count
                        .get()
                        .max(
                            base_skipped
                                .saturating_add(progress.imported)
                                .saturating_add(progress.skipped)
                                .saturating_add(progress.insufficient_frames)
                                .saturating_add(progress.failed),
                        )
                        .max(terminal)
                        .min(total_videos);
                    started_count.set(current_started);
                    prepared_count.set(current_prepared);
                    emit_progress(
                        &progress_sink,
                        AiIndexProgress {
                            phase: "local-preprocess".to_string(),
                            total_videos,
                            processed: processed_count.get(),
                            skipped: current_skipped,
                            insufficient_frames: current_insufficient,
                            failed: current_failed,
                            started: current_started,
                            prepared: current_prepared,
                            current_path: progress.current_path,
                        },
                    );
                },
                |video| {
                    let Some(video_id) = video.id else {
                        return Ok(());
                    };
                    if !indexed_video_ids.insert(video_id) {
                        return Ok(());
                    }
                    let next_prepared = prepared_count.get().saturating_add(1).min(total_videos);
                    prepared_count.set(next_prepared);
                    started_count.set(started_count.get().max(next_prepared));
                    if try_reuse_existing_ai_index(
                        &mut conn,
                        &video,
                        &settings,
                        &prepared,
                        force_rebuild,
                    )? {
                        let next_skipped = skipped_count.get().saturating_add(1);
                        skipped_count.set(next_skipped);
                        emit_progress(
                            &progress_sink,
                            AiIndexProgress {
                                phase: "indexing".to_string(),
                                total_videos,
                                processed: processed_count.get(),
                                skipped: next_skipped,
                                insufficient_frames: insufficient_count.get(),
                                failed: failed_count.get(),
                                started: started_count.get(),
                                prepared: prepared_count.get(),
                                current_path: Some(video.path),
                            },
                        );
                        return Ok(());
                    }
                    emit_progress(
                        &progress_sink,
                        AiIndexProgress {
                            phase: format!("queued-ai x{ai_workers}"),
                            total_videos,
                            processed: processed_count.get(),
                            skipped: skipped_count.get(),
                            insufficient_frames: insufficient_count.get(),
                            failed: failed_count.get(),
                            started: started_count.get(),
                            prepared: prepared_count.get(),
                            current_path: Some(video.path.clone()),
                        },
                    );

                    ai_sender
                        .send(video)
                        .map_err(|error| anyhow!("queue AI worker job failed: {error}"))?;
                    queued_ai += 1;
                    Ok(())
                },
            )
            .map_err(|error| {
                let message = format!("Local staged preprocess failed: {error}");
                append_error_log(&error_log, &message);
                anyhow!(message)
            })?;

            for error in summary.recent_errors {
                append_error_log(&error_log, &error);
                push_recent_error(&mut recent_errors, error);
            }
            skipped = skipped_count.get();
            failed = failed_count
                .get()
                .max(base_failed.saturating_add(summary.failed));
            insufficient_frames = insufficient_count
                .get()
                .max(base_insufficient.saturating_add(summary.insufficient_frames));
            drop(ai_sender);

            let mut completed_ai = 0usize;
            while completed_ai < queued_ai {
                crate::cancel::bail_if_requested()?;
                match ai_results.recv_timeout(Duration::from_millis(500)) {
                    Ok(result) => {
                        completed_ai += 1;
                        match persist_ai_worker_result(&mut conn, &prepared, &settings, result)? {
                            IndexVideoOutcome::Processed => processed += 1,
                            IndexVideoOutcome::Skipped => skipped += 1,
                            IndexVideoOutcome::InsufficientFrames(_message) => {
                                insufficient_frames += 1;
                            }
                            IndexVideoOutcome::Failed(message) => {
                                failed += 1;
                                append_error_log(&error_log, &message);
                                push_recent_error(&mut recent_errors, message);
                            }
                        }
                        emit_progress(
                            &progress_sink,
                            AiIndexProgress {
                                phase: format!("indexing x{ai_workers}"),
                                total_videos,
                                processed,
                                skipped,
                                insufficient_frames,
                                failed,
                                started: started_count.get(),
                                prepared: prepared_count.get(),
                                current_path: None,
                            },
                        );
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => continue,
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
        }

        videos = match requested_session_ids.clone() {
            Some(ids) => database::list_videos_for_sessions(&conn, &ids)?,
            None => database::list_videos(&conn)?,
        }
        .into_iter()
        .filter(|video| video.scan_status == "ok" && video.id.is_some())
        .collect::<Vec<_>>();
    } else if settings.ai_frame_cache_enabled && crate::nas_preprocess::enabled(&settings) {
        match crate::nas_preprocess::preprocess_ai_frame_cache(&videos, &settings, |progress| {
            on_progress(AiIndexProgress {
                phase: "nas-preprocess".to_string(),
                total_videos: progress.total_videos,
                processed,
                skipped,
                insufficient_frames: 0,
                failed: progress.failed,
                started: progress
                    .imported
                    .saturating_add(progress.skipped)
                    .saturating_add(progress.failed)
                    .min(progress.total_videos),
                prepared: progress
                    .imported
                    .saturating_add(progress.skipped)
                    .saturating_add(progress.failed)
                    .min(progress.total_videos),
                current_path: progress.current_path,
            });
        }) {
            Ok(summary) => {
                for error in summary.recent_errors {
                    append_error_log(&error_log, &error);
                    push_recent_error(&mut recent_errors, error);
                }
            }
            Err(error) => {
                let message =
                    format!("NAS SSH preprocess failed, falling back to SMB frames: {error}");
                append_error_log(&error_log, &message);
                push_recent_error(&mut recent_errors, message);
            }
        }
    }

    if videos
        .iter()
        .filter_map(|video| video.id)
        .any(|video_id| !indexed_video_ids.contains(&video_id))
    {
        let mut session = create_session(&settings, &prepared.model_path)
            .with_context(|| format!("load ONNX model {}", prepared.model_path.display()))?;
        on_progress(AiIndexProgress {
            phase: "indexing".to_string(),
            total_videos,
            processed,
            skipped,
            insufficient_frames,
            failed,
            started: processed + skipped + insufficient_frames + failed,
            prepared: processed + skipped + insufficient_frames + failed,
            current_path: None,
        });

        for video in videos {
            crate::cancel::bail_if_requested()?;
            let video_id = video.id.expect("filtered videos have ids");
            if !indexed_video_ids.insert(video_id) {
                continue;
            }
            on_progress(AiIndexProgress {
                phase: "indexing".to_string(),
                total_videos,
                processed,
                skipped,
                insufficient_frames,
                failed,
                started: (processed + skipped + insufficient_frames + failed + 1).min(total_videos),
                prepared: processed + skipped + insufficient_frames + failed,
                current_path: Some(video.path.clone()),
            });

            match index_video(
                &mut conn,
                &mut session,
                &video,
                &settings,
                &prepared,
                force_rebuild,
            )? {
                IndexVideoOutcome::Processed => processed += 1,
                IndexVideoOutcome::Skipped => skipped += 1,
                IndexVideoOutcome::InsufficientFrames(_message) => {
                    insufficient_frames += 1;
                }
                IndexVideoOutcome::Failed(message) => {
                    failed += 1;
                    append_error_log(&error_log, &message);
                    push_recent_error(&mut recent_errors, message);
                }
            }

            on_progress(AiIndexProgress {
                phase: "indexing".to_string(),
                total_videos,
                processed,
                skipped,
                insufficient_frames,
                failed,
                started: processed + skipped + insufficient_frames + failed,
                prepared: processed + skipped + insufficient_frames + failed,
                current_path: Some(video.path.clone()),
            });
        }
    }

    on_progress(AiIndexProgress {
        phase: "completed".to_string(),
        total_videos,
        processed,
        skipped,
        insufficient_frames,
        failed,
        started: total_videos,
        prepared: total_videos,
        current_path: None,
    });

    Ok(AiIndexSummary {
        model_id: prepared.model_id,
        model_path: prepared.model_path.display().to_string(),
        total_videos,
        processed,
        skipped,
        insufficient_frames,
        failed,
        elapsed_ms: started.elapsed().as_millis(),
        recent_errors,
    })
}

fn ai_worker_count(settings: &AppSettings, total_videos: usize) -> usize {
    let requested = if settings.ai_device == "cpu" {
        settings.ai_extract_worker_count.clamp(1, 4)
    } else {
        settings.ai_gpu_worker_count.clamp(1, 64)
    };
    requested.min(total_videos.max(1))
}

fn start_ai_workers(
    worker_count: usize,
    settings: &AppSettings,
    prepared: &PreparedModel,
) -> (mpsc::Sender<VideoRecord>, mpsc::Receiver<AiWorkerResult>) {
    let (work_sender, work_receiver) = mpsc::channel::<VideoRecord>();
    let (result_sender, result_receiver) = mpsc::channel::<AiWorkerResult>();
    let work_receiver = Arc::new(Mutex::new(work_receiver));

    for _ in 0..worker_count.max(1) {
        let work_receiver = Arc::clone(&work_receiver);
        let result_sender = result_sender.clone();
        let settings = settings.clone();
        let prepared = prepared.clone();
        thread::spawn(move || {
            let mut session = match create_session(&settings, &prepared.model_path) {
                Ok(session) => session,
                Err(error) => {
                    while let Ok(video) = work_receiver
                        .lock()
                        .expect("AI worker queue lock poisoned")
                        .recv()
                    {
                        let _ = result_sender.send(AiWorkerResult {
                            video_id: video.id.unwrap_or_default(),
                            path: video.path.clone(),
                            video,
                            outcome: AiWorkerOutcome::Failed(format!(
                                "load ONNX model failed: {error}"
                            )),
                        });
                    }
                    return;
                }
            };

            loop {
                if crate::cancel::is_requested() {
                    break;
                }
                let video = match work_receiver
                    .lock()
                    .expect("AI worker queue lock poisoned")
                    .recv()
                {
                    Ok(video) => video,
                    Err(_) => break,
                };
                let video_id = video.id.unwrap_or_default();
                let path = video.path.clone();
                let outcome = match crate::cancel::bail_if_requested()
                    .and_then(|_| embed_video(&mut session, &video, &settings, &prepared.model_id))
                {
                    Ok(embeddings) if embeddings.is_empty() => {
                        AiWorkerOutcome::Failed(format!("{path}: no AI frames were extracted"))
                    }
                    Ok(embeddings) => AiWorkerOutcome::Embedded(embeddings),
                    Err(error) => AiWorkerOutcome::Failed(format!("{path}: {error}")),
                };
                if result_sender
                    .send(AiWorkerResult {
                        video_id,
                        path,
                        video,
                        outcome,
                    })
                    .is_err()
                {
                    break;
                }
            }
        });
    }
    drop(result_sender);
    (work_sender, result_receiver)
}

fn try_reuse_existing_ai_index(
    conn: &mut rusqlite::Connection,
    video: &VideoRecord,
    settings: &AppSettings,
    prepared: &PreparedModel,
    force_rebuild: bool,
) -> anyhow::Result<bool> {
    if force_rebuild {
        return Ok(false);
    }
    let Some(video_id) = video.id else {
        return Ok(false);
    };
    if !media::video_has_full_scan(video, settings) {
        return Ok(false);
    }
    let expected_frames = expected_ai_frame_count(video, settings.ai_frame_count);
    if database::count_frame_embeddings(conn, video_id, &prepared.model_id)? >= expected_frames {
        return Ok(true);
    }
    let Some(source_video_id) = database::find_embedding_source_by_video_identity(
        conn,
        video,
        &prepared.model_id,
        expected_frames,
    )?
    else {
        return Ok(false);
    };
    let copied = database::copy_frame_embeddings_from_video(
        conn,
        source_video_id,
        video_id,
        &prepared.model_id,
    )?;
    Ok(copied >= expected_frames)
}

fn persist_ai_worker_result(
    conn: &mut rusqlite::Connection,
    prepared: &PreparedModel,
    settings: &AppSettings,
    result: AiWorkerResult,
) -> anyhow::Result<IndexVideoOutcome> {
    match result.outcome {
        AiWorkerOutcome::Embedded(embeddings) if !embeddings.is_empty() => {
            let dimension = embeddings[0].embedding.len();
            database::upsert_embedding_model(
                conn,
                &prepared.model_id,
                AI_MODEL_NAME,
                dimension,
                AI_RUNTIME_NAME,
                &prepared.model_path.display().to_string(),
                &prepared.model_hash,
                media::now_ms(),
            )?;
            database::replace_frame_embeddings(
                conn,
                result.video_id,
                &prepared.model_id,
                &embeddings,
            )?;
            maybe_delete_ai_frame_cache_after_index(&result.video, settings);
            Ok(IndexVideoOutcome::Processed)
        }
        AiWorkerOutcome::Embedded(_) => Ok(IndexVideoOutcome::InsufficientFrames(format!(
            "{}: no AI frames were extracted",
            result.path
        ))),
        AiWorkerOutcome::Failed(message) => Ok(IndexVideoOutcome::Failed(message)),
    }
}

fn index_video(
    conn: &mut rusqlite::Connection,
    session: &mut Session,
    video: &VideoRecord,
    settings: &AppSettings,
    prepared: &PreparedModel,
    force_rebuild: bool,
) -> anyhow::Result<IndexVideoOutcome> {
    let video_id = video
        .id
        .ok_or_else(|| anyhow!("filtered video is missing an id: {}", video.path))?;
    if try_reuse_existing_ai_index(conn, video, settings, prepared, force_rebuild)? {
        return Ok(IndexVideoOutcome::Skipped);
    }

    match embed_video(session, video, settings, &prepared.model_id) {
        Ok(embeddings) if !embeddings.is_empty() => {
            let dimension = embeddings[0].embedding.len();
            database::upsert_embedding_model(
                conn,
                &prepared.model_id,
                AI_MODEL_NAME,
                dimension,
                AI_RUNTIME_NAME,
                &prepared.model_path.display().to_string(),
                &prepared.model_hash,
                media::now_ms(),
            )?;
            database::replace_frame_embeddings(conn, video_id, &prepared.model_id, &embeddings)?;
            maybe_delete_ai_frame_cache_after_index(video, settings);
            Ok(IndexVideoOutcome::Processed)
        }
        Ok(_) => Ok(IndexVideoOutcome::InsufficientFrames(format!(
            "{}: no AI frames were extracted",
            video.path
        ))),
        Err(error) => Ok(IndexVideoOutcome::Failed(format!(
            "{}: {error}",
            video.path
        ))),
    }
}

fn maybe_delete_ai_frame_cache_after_index(video: &VideoRecord, settings: &AppSettings) {
    if !settings.delete_ai_frame_cache_after_index {
        return;
    }
    let frame_count = expected_ai_frame_count(video, settings.ai_frame_count);
    if let Err(error) = media::delete_ai_frame_cache_for_video(video, frame_count) {
        eprintln!(
            "failed to delete AI frame cache after indexing {}: {error}",
            video.path
        );
    }
}

fn push_recent_error(recent_errors: &mut Vec<String>, message: String) {
    recent_errors.push(message);
    if recent_errors.len() > 8 {
        recent_errors.remove(0);
    }
}

fn emit_progress<F>(sink: &RefCell<&mut F>, progress: AiIndexProgress)
where
    F: FnMut(AiIndexProgress),
{
    let mut emit = sink.borrow_mut();
    (**emit)(progress);
}

pub(crate) fn current_ai_match_cache_key(settings: &AppSettings) -> String {
    ai_match_cache_key(settings, settings.ai_similarity_threshold.clamp(0.0, 0.99))
}

fn ai_match_cache_key(settings: &AppSettings, frame_similarity_threshold: f64) -> String {
    format!(
        "{}|frames={}|frame_threshold={:.4}|min_frames={}|clip={}",
        AI_MATCH_CACHE_VERSION,
        settings.ai_frame_count.clamp(8, 512),
        frame_similarity_threshold,
        settings.ai_min_matched_frames.clamp(1, 128),
        settings.ai_clip_matching_enabled
    )
}

pub fn build_ai_match_groups(
    conn: &mut rusqlite::Connection,
    videos: &[VideoRecord],
    settings: &AppSettings,
    min_confidence: f64,
) -> anyhow::Result<Vec<MatchGroup>> {
    build_ai_match_groups_with_progress(conn, videos, settings, min_confidence, |_| {})
}

pub fn build_ai_match_groups_with_progress<F>(
    conn: &mut rusqlite::Connection,
    videos: &[VideoRecord],
    settings: &AppSettings,
    min_confidence: f64,
    mut on_progress: F,
) -> anyhow::Result<Vec<MatchGroup>>
where
    F: FnMut(MatchRefreshProgress),
{
    if !settings.ai_vision_enabled {
        return Ok(Vec::new());
    }
    let started = Instant::now();
    let model_path = resolve_model_path(&settings.ai_model_path);
    if !model_path.exists() {
        return Ok(Vec::new());
    }
    let model_path_text = model_path.display().to_string();
    let model_id = match database::latest_embedding_model_id_for_path(conn, &model_path_text)? {
        Some(model_id) => model_id,
        None => {
            let Ok(prepared) = prepare_model(settings) else {
                return Ok(Vec::new());
            };
            prepared.model_id
        }
    };
    let mut video_ids = videos
        .iter()
        .filter_map(|video| video.id)
        .collect::<Vec<_>>();
    video_ids.sort_unstable();
    on_progress(match_refresh_progress(
        "loading-embeddings",
        video_ids.len(),
        0,
        0,
        0,
        0,
        0,
        started,
    ));
    let embeddings = database::list_frame_embeddings(conn, &model_id, &video_ids)?;
    if embeddings.is_empty() {
        return Ok(Vec::new());
    }

    let mut by_video: HashMap<i64, Vec<FrameEmbeddingRecord>> = HashMap::new();
    for embedding in embeddings {
        by_video
            .entry(embedding.video_id)
            .or_default()
            .push(embedding);
    }
    for items in by_video.values_mut() {
        items.sort_by_key(|item| item.frame_index);
    }

    let video_by_id = videos
        .iter()
        .filter_map(|video| video.id.map(|id| (id, video)))
        .collect::<HashMap<_, _>>();
    let mut edges = Vec::new();
    let frame_similarity_threshold = settings.ai_similarity_threshold.clamp(0.0, 0.99);
    let min_confidence = min_confidence.clamp(0.0, 1.0);
    let cache_key = ai_match_cache_key(settings, frame_similarity_threshold);
    let mut ids = by_video.keys().copied().collect::<Vec<_>>();
    ids.sort_unstable();
    let pair_allowed = |left: i64, right: i64| {
        !settings.compare_within_same_folder
            || video_by_id.get(&left).zip(video_by_id.get(&right)).is_some_and(|(left, right)| {
                crate::paths::same_parent_folder(&left.path, &right.path)
            })
    };
    let total_pairs = if settings.compare_within_same_folder {
        ids.iter().enumerate().map(|(index, left)| {
            ids.iter().skip(index + 1).filter(|right| pair_allowed(*left, **right)).count()
        }).sum()
    } else {
        pair_count(ids.len())
    };
    on_progress(match_refresh_progress(
        "loading-cache",
        ids.len(),
        total_pairs,
        0,
        0,
        0,
        0,
        started,
    ));
    let cache_started = Instant::now();
    let cached_scores = database::list_ai_pair_scores_chunked(
        conn,
        &model_id,
        &cache_key,
        &ids,
        AI_PAIR_SCORE_READ_CHUNK,
        |left_done, loaded_pairs| {
            let phase_processed = pair_count_for_left_count(ids.len(), left_done);
            let phase_speed =
                phase_processed as f64 / cache_started.elapsed().as_secs_f64().max(0.001);
            on_progress(match_refresh_progress_with_phase_speed(
                "loading-cache",
                ids.len(),
                total_pairs,
                loaded_pairs,
                loaded_pairs,
                0,
                0,
                phase_speed,
                phase_processed,
                total_pairs,
                phase_speed,
            ));
        },
    )?;
    let mut cached_pairs = HashSet::new();
    for score in cached_scores {
        if !pair_allowed(score.video_a_id, score.video_b_id) {
            continue;
        }
        cached_pairs.insert(edge_key(score.video_a_id, score.video_b_id));
        let edge = edge_from_pair_score(&score);
        if edge.confidence >= min_confidence {
            edges.push(edge);
        }
    }
    let cached_count = cached_pairs.len();
    let cache_speed = total_pairs as f64 / cache_started.elapsed().as_secs_f64().max(0.001);
    on_progress(match_refresh_progress_with_phase_speed(
        "cache-loaded",
        ids.len(),
        total_pairs,
        cached_count,
        cached_count,
        0,
        0,
        cache_speed,
        total_pairs,
        total_pairs,
        cache_speed,
    ));

    let mut missing_pairs = Vec::new();
    for (left_index, left_id) in ids.iter().enumerate() {
        for right_id in ids.iter().skip(left_index + 1) {
            if pair_allowed(*left_id, *right_id)
                && !cached_pairs.contains(&edge_key(*left_id, *right_id)) {
                missing_pairs.push((*left_id, *right_id));
            }
        }
    }

    if !missing_pairs.is_empty() {
        let computed_edges = score_missing_pairs_parallel(
            &missing_pairs,
            &video_by_id,
            &by_video,
            settings,
            frame_similarity_threshold,
            |computed_pairs, pairs_per_second| {
                on_progress(match_refresh_progress_with_phase_speed(
                    "scoring-pairs",
                    ids.len(),
                    total_pairs,
                    cached_count.saturating_add(computed_pairs),
                    cached_count,
                    computed_pairs,
                    0,
                    pairs_per_second,
                    computed_pairs,
                    missing_pairs.len(),
                    pairs_per_second,
                ));
            },
        );
        let now = media::now_ms();
        let records = computed_edges
            .iter()
            .map(|edge| pair_score_from_edge(edge, &model_id, &cache_key, now))
            .collect::<Vec<_>>();
        let write_started = Instant::now();
        let mut written = 0usize;
        for chunk in records.chunks(AI_PAIR_SCORE_WRITE_CHUNK) {
            database::upsert_ai_pair_scores(conn, chunk)?;
            written = written.saturating_add(chunk.len());
            let write_speed = written as f64 / write_started.elapsed().as_secs_f64().max(0.001);
            on_progress(match_refresh_progress_with_phase_speed(
                "writing-cache",
                ids.len(),
                total_pairs,
                cached_count.saturating_add(missing_pairs.len()),
                cached_count,
                missing_pairs.len(),
                0,
                write_speed,
                written,
                records.len(),
                write_speed,
            ));
        }
        edges.extend(
            computed_edges
                .into_iter()
                .filter(|edge| edge.confidence >= min_confidence),
        );
    }

    on_progress(match_refresh_progress(
        "grouping",
        ids.len(),
        total_pairs,
        total_pairs,
        cached_count,
        missing_pairs.len(),
        0,
        started,
    ));
    let mut groups = connected_ai_groups(
        videos,
        &edges,
        min_confidence,
        settings.keeper_size_priority_duration_seconds,
    );
    let detail_total = ai_group_detail_pair_count(&groups, &edges);
    if detail_total > 0 {
        let detail_started = Instant::now();
        let group_count = groups.len();
        attach_ai_group_match_details(
            &mut groups,
            &edges,
            &video_by_id,
            &by_video,
            settings,
            frame_similarity_threshold,
            |done, total| {
                let speed = done as f64 / detail_started.elapsed().as_secs_f64().max(0.001);
                on_progress(match_refresh_progress_with_phase_speed(
                    "match-details",
                    ids.len(),
                    total_pairs,
                    total_pairs,
                    cached_count,
                    missing_pairs.len(),
                    group_count,
                    speed,
                    done,
                    total,
                    speed,
                ));
            },
        );
    }
    on_progress(match_refresh_progress(
        "pruning-cache",
        ids.len(),
        total_pairs,
        total_pairs,
        cached_count,
        missing_pairs.len(),
        groups.len(),
        started,
    ));
    database::prune_ai_pair_score_cache_keys(conn, &model_id, &cache_key)?;
    on_progress(match_refresh_progress(
        "completed",
        ids.len(),
        total_pairs,
        total_pairs,
        cached_count,
        missing_pairs.len(),
        groups.len(),
        started,
    ));
    Ok(groups)
}

pub fn debug_ai_pair(
    conn: &mut rusqlite::Connection,
    left_path: &str,
    right_path: &str,
    settings: &AppSettings,
) -> anyhow::Result<AiPairDebugReport> {
    let videos = database::list_videos(conn)?;
    let left_record = find_debug_video(&videos, left_path).cloned();
    let right_record = find_debug_video(&videos, right_path).cloned();
    let frame_similarity_threshold = settings.ai_similarity_threshold.clamp(0.0, 0.99);
    let model_path = resolve_model_path(&settings.ai_model_path);
    let model_path_text = model_path.display().to_string();
    let model_id = database::latest_embedding_model_id_for_path(conn, &model_path_text)?;
    let current_cache_key = model_id
        .as_ref()
        .map(|_| ai_match_cache_key(settings, frame_similarity_threshold));

    let mut report = AiPairDebugReport {
        left: debug_video(left_path, left_record.as_ref(), 0),
        right: debug_video(right_path, right_record.as_ref(), 0),
        settings: AiPairDebugSettings {
            ai_vision_enabled: settings.ai_vision_enabled,
            ai_similarity_threshold: settings.ai_similarity_threshold,
            ai_min_matched_frames: settings.ai_min_matched_frames,
            ai_frame_count: settings.ai_frame_count,
            ai_clip_matching_enabled: settings.ai_clip_matching_enabled,
            ai_match_worker_count: settings.ai_match_worker_count,
        },
        model_id: model_id.clone(),
        current_cache_key: current_cache_key.clone(),
        current_cached_score: None,
        cached_scores: Vec::new(),
        embedding_summary: None,
        threshold_scores: Vec::new(),
        diagnosis: Vec::new(),
    };

    if !settings.ai_vision_enabled {
        report
            .diagnosis
            .push("AI vision matching is disabled in settings.".to_string());
    }
    if left_record.is_none() {
        report
            .diagnosis
            .push("Left video was not found in the normal index.".to_string());
    }
    if right_record.is_none() {
        report
            .diagnosis
            .push("Right video was not found in the normal index.".to_string());
    }
    let Some(model_id) = model_id else {
        report
            .diagnosis
            .push("No embedding model record matches the configured model path.".to_string());
        return Ok(report);
    };
    let left_id = left_record.as_ref().and_then(|video| video.id);
    let right_id = right_record.as_ref().and_then(|video| video.id);
    let available_ids = [left_id, right_id]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let mut left_embeddings = Vec::new();
    let mut right_embeddings = Vec::new();
    if !available_ids.is_empty() {
        let embeddings = database::list_frame_embeddings(conn, &model_id, &available_ids)?;
        for embedding in embeddings {
            if Some(embedding.video_id) == left_id {
                left_embeddings.push(embedding);
            } else if Some(embedding.video_id) == right_id {
                right_embeddings.push(embedding);
            }
        }
        left_embeddings.sort_by_key(|item| item.frame_index);
        right_embeddings.sort_by_key(|item| item.frame_index);
        report.left.embedding_count = left_embeddings.len();
        report.right.embedding_count = right_embeddings.len();
    }
    let (Some(left_video), Some(right_video)) = (left_record.as_ref(), right_record.as_ref())
    else {
        return Ok(report);
    };
    let (Some(left_id), Some(right_id)) = (left_id, right_id) else {
        report
            .diagnosis
            .push("One of the videos has no database id.".to_string());
        return Ok(report);
    };

    report.cached_scores = list_pair_scores_for_debug(
        conn,
        &model_id,
        current_cache_key.as_deref(),
        left_id,
        right_id,
    )?;
    report.current_cached_score = report
        .cached_scores
        .iter()
        .find(|score| score.is_current_cache_key)
        .cloned();

    if left_embeddings.is_empty() {
        report
            .diagnosis
            .push("Left video has no AI frame embeddings for the current model.".to_string());
    }
    if right_embeddings.is_empty() {
        report
            .diagnosis
            .push("Right video has no AI frame embeddings for the current model.".to_string());
    }
    if left_embeddings.is_empty() || right_embeddings.is_empty() {
        return Ok(report);
    }

    let mut similarity_engine = FrameSimilarityEngine::from_settings(settings);
    let embedding_summary = pair_embedding_summary(
        left_video,
        right_video,
        &left_embeddings,
        &right_embeddings,
        &mut similarity_engine,
    );
    if embedding_summary.max_frame_similarity < frame_similarity_threshold {
        report.diagnosis.push(format!(
            "The best individual frame cosine ({:.4}) is below the current frame-level threshold ({:.4}). Lowering the result filter cannot fix that.",
            embedding_summary.max_frame_similarity, frame_similarity_threshold
        ));
    }
    report.embedding_summary = Some(embedding_summary);

    let mut thresholds = vec![
        frame_similarity_threshold,
        0.86,
        0.80,
        0.75,
        0.70,
        0.65,
        0.60,
        0.50,
        0.40,
    ];
    thresholds.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    thresholds.dedup_by(|a, b| (*a - *b).abs() < 0.0005);
    for threshold in thresholds {
        let edge = score_pair(
            left_video,
            right_video,
            &left_embeddings,
            &right_embeddings,
            settings,
            threshold,
            &mut similarity_engine,
        );
        report.threshold_scores.push(AiPairDebugThresholdScore {
            frame_similarity_threshold: threshold,
            confidence: edge.confidence,
            matched: edge.matched,
            compared: edge.compared,
            average_similarity: edge.average_similarity,
            coverage: edge.coverage,
            passes_min_matched_frames: edge.matched >= settings.ai_min_matched_frames.clamp(1, 128),
        });
    }

    if let Some(current_score) = report.threshold_scores.first() {
        if current_score.matched < settings.ai_min_matched_frames.clamp(1, 128) {
            report.diagnosis.push(format!(
                "At the current frame-level threshold, only {} frames match; the configured minimum is {}.",
                current_score.matched,
                settings.ai_min_matched_frames.clamp(1, 128)
            ));
        }
    }
    if report.current_cached_score.is_none() {
        report
            .diagnosis
            .push("No pair-score row exists for the current cache key yet; a manual refresh still needs to compute or write it.".to_string());
    }
    report.diagnosis.push(
        "The result-page minimum similarity filter only filters final group confidence; it does not lower the AI frame-level matching threshold used to create pair scores."
            .to_string(),
    );

    Ok(report)
}

fn find_debug_video<'a>(videos: &'a [VideoRecord], path: &str) -> Option<&'a VideoRecord> {
    let normalized = normalize_debug_path(path);
    videos
        .iter()
        .find(|video| normalize_debug_path(&video.path) == normalized)
        .or_else(|| {
            videos
                .iter()
                .find(|video| normalize_debug_path(&video.path).ends_with(&normalized))
        })
        .or_else(|| {
            let file_name = Path::new(path)
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or(path)
                .to_lowercase();
            videos
                .iter()
                .find(|video| video.file_name.to_lowercase() == file_name)
        })
}

fn normalize_debug_path(path: &str) -> String {
    path.replace('/', "\\")
        .trim()
        .trim_matches('"')
        .to_lowercase()
}

fn debug_video(
    path: &str,
    video: Option<&VideoRecord>,
    embedding_count: usize,
) -> AiPairDebugVideo {
    AiPairDebugVideo {
        requested_path: path.to_string(),
        found: video.is_some(),
        id: video.and_then(|item| item.id),
        indexed_path: video.map(|item| item.path.clone()),
        file_name: video.map(|item| item.file_name.clone()),
        duration_seconds: video.and_then(|item| item.duration_seconds),
        size_bytes: video.map(|item| item.size_bytes),
        scan_status: video.map(|item| item.scan_status.clone()),
        embedding_count,
    }
}

fn list_pair_scores_for_debug(
    conn: &rusqlite::Connection,
    model_id: &str,
    current_cache_key: Option<&str>,
    left_id: i64,
    right_id: i64,
) -> anyhow::Result<Vec<AiPairDebugCachedScore>> {
    let (video_a_id, video_b_id) = edge_key(left_id, right_id);
    let mut stmt = conn.prepare(
        r#"
        SELECT cache_key, confidence, matched_frame_count, compared_frame_count,
               average_similarity, coverage, created_unix_ms
        FROM ai_pair_scores
        WHERE video_a_id = ?1 AND video_b_id = ?2 AND model_id = ?3
        ORDER BY created_unix_ms DESC
        LIMIT 20
        "#,
    )?;
    let rows = stmt.query_map(params![video_a_id, video_b_id, model_id], |row| {
        let cache_key: String = row.get(0)?;
        Ok(AiPairDebugCachedScore {
            is_current_cache_key: current_cache_key.is_some_and(|current| current == cache_key),
            cache_key,
            confidence: row.get(1)?,
            matched: row.get::<_, i64>(2)? as usize,
            compared: row.get::<_, i64>(3)? as usize,
            average_similarity: row.get(4)?,
            coverage: row.get(5)?,
            created_unix_ms: row.get(6)?,
        })
    })?;
    let mut scores = Vec::new();
    for row in rows {
        scores.push(row?);
    }
    Ok(scores)
}

fn pair_embedding_summary(
    left_video: &VideoRecord,
    right_video: &VideoRecord,
    left_embeddings: &[FrameEmbeddingRecord],
    right_embeddings: &[FrameEmbeddingRecord],
    similarity_engine: &mut FrameSimilarityEngine,
) -> AiPairDebugEmbeddingSummary {
    let (short, long) = if left_video.duration_seconds.unwrap_or(0.0)
        <= right_video.duration_seconds.unwrap_or(0.0)
    {
        (left_embeddings, right_embeddings)
    } else {
        (right_embeddings, left_embeddings)
    };
    let similarities = similarity_engine.similarities(short, long);
    let mut best_similarities = (0..short.len())
        .map(|row| {
            (0..long.len())
                .map(|column| similarities.get(row, column))
                .fold(0.0f64, f64::max)
        })
        .collect::<Vec<_>>();
    best_similarities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let max_frame_similarity = best_similarities.last().copied().unwrap_or(0.0);
    let average_best_short_frame_similarity = if best_similarities.is_empty() {
        0.0
    } else {
        best_similarities.iter().sum::<f64>() / best_similarities.len() as f64
    };

    AiPairDebugEmbeddingSummary {
        max_frame_similarity,
        average_best_short_frame_similarity,
        p50_best_short_frame_similarity: quantile_sorted(&best_similarities, 0.50),
        p75_best_short_frame_similarity: quantile_sorted(&best_similarities, 0.75),
        p90_best_short_frame_similarity: quantile_sorted(&best_similarities, 0.90),
        p95_best_short_frame_similarity: quantile_sorted(&best_similarities, 0.95),
    }
}

fn quantile_sorted(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let index = ((values.len() - 1) as f64 * percentile.clamp(0.0, 1.0)).round() as usize;
    values[index.min(values.len() - 1)]
}

fn pair_count(items: usize) -> usize {
    items.saturating_mul(items.saturating_sub(1)) / 2
}

fn pair_count_for_left_count(total: usize, left_count: usize) -> usize {
    let left_count = left_count.min(total);
    left_count
        .saturating_mul(total)
        .saturating_sub(left_count.saturating_mul(left_count.saturating_add(1)) / 2)
}

fn match_refresh_progress(
    phase: &str,
    total_videos: usize,
    total_pairs: usize,
    processed_pairs: usize,
    cached_pairs: usize,
    computed_pairs: usize,
    groups: usize,
    started: Instant,
) -> MatchRefreshProgress {
    let elapsed = started.elapsed().as_secs_f64().max(0.001);
    match_refresh_progress_with_speed(
        phase,
        total_videos,
        total_pairs,
        processed_pairs,
        cached_pairs,
        computed_pairs,
        groups,
        processed_pairs as f64 / elapsed,
    )
}

fn match_refresh_progress_with_speed(
    phase: &str,
    total_videos: usize,
    total_pairs: usize,
    processed_pairs: usize,
    cached_pairs: usize,
    computed_pairs: usize,
    groups: usize,
    pairs_per_second: f64,
) -> MatchRefreshProgress {
    match_refresh_progress_with_phase_speed(
        phase,
        total_videos,
        total_pairs,
        processed_pairs,
        cached_pairs,
        computed_pairs,
        groups,
        pairs_per_second,
        processed_pairs,
        total_pairs,
        pairs_per_second,
    )
}

fn match_refresh_progress_with_phase_speed(
    phase: &str,
    total_videos: usize,
    total_pairs: usize,
    processed_pairs: usize,
    cached_pairs: usize,
    computed_pairs: usize,
    groups: usize,
    pairs_per_second: f64,
    phase_processed: usize,
    phase_total: usize,
    phase_pairs_per_second: f64,
) -> MatchRefreshProgress {
    MatchRefreshProgress {
        phase: phase.to_string(),
        total_videos,
        total_pairs,
        processed_pairs: processed_pairs.min(total_pairs),
        cached_pairs,
        computed_pairs,
        groups,
        pairs_per_second,
        phase_processed: phase_processed.min(phase_total),
        phase_total,
        phase_pairs_per_second,
    }
}

fn prepare_model(settings: &AppSettings) -> anyhow::Result<PreparedModel> {
    let model_path = resolve_model_path(&settings.ai_model_path);
    if !model_path.exists() {
        return Err(anyhow!(
            "AI model file was not found: {}",
            model_path.display()
        ));
    }
    let model_hash = file_sha256(&model_path)?;
    let model_id = format!("onnx-{}", &model_hash[..16]);
    Ok(PreparedModel {
        model_id,
        model_hash,
        model_path,
    })
}

pub(crate) fn resolve_model_path(path: &str) -> PathBuf {
    let trimmed = path.trim();
    if trimmed.is_empty() {
        return PathBuf::new();
    }
    let path = PathBuf::from(trimmed);
    if path.is_absolute() {
        path
    } else {
        crate::paths::workspace_root().join(path)
    }
}

fn create_session(settings: &AppSettings, model_path: &Path) -> anyhow::Result<Session> {
    let mut builder = Session::builder().map_err(ort_error)?;
    builder = builder
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(ort_error)?;
    builder = builder.with_intra_threads(1).map_err(ort_error)?;
    if settings.ai_device == "gpu" || settings.ai_device == "auto" {
        let directml = ep::DirectML::default().build();
        builder = builder
            .with_execution_providers([directml])
            .unwrap_or_else(|error| error.recover());
    }
    builder.commit_from_file(model_path).map_err(ort_error)
}

fn create_frame_similarity_session(settings: &AppSettings) -> anyhow::Result<Session> {
    let mut builder = Session::builder().map_err(ort_error)?;
    builder = builder
        .with_optimization_level(GraphOptimizationLevel::All)
        .map_err(ort_error)?;
    builder = builder.with_intra_threads(1).map_err(ort_error)?;
    if settings.ai_device == "gpu" || settings.ai_device == "auto" {
        let directml = ep::DirectML::default().build();
        builder = builder
            .with_execution_providers([directml])
            .unwrap_or_else(|error| error.recover());
    }
    builder
        .commit_from_memory(FRAME_SIMILARITY_MATMUL_ONNX)
        .map_err(ort_error)
}

fn embed_video(
    session: &mut Session,
    video: &VideoRecord,
    settings: &AppSettings,
    model_id: &str,
) -> anyhow::Result<Vec<FrameEmbeddingRecord>> {
    let duration = video.duration_seconds.unwrap_or(0.0);
    let frame_count = expected_ai_frame_count(video, settings.ai_frame_count);
    let batch_size = settings.ai_batch_size.clamp(1, 64);
    let mut records = Vec::new();

    let cached_frames = if settings.ai_frame_cache_enabled {
        let pack = if settings.local_preprocess_enabled {
            media::load_ai_frame_cache_for_video(video, frame_count)?
        } else {
            media::ensure_ai_frame_cache_for_video(
                video,
                frame_count,
                settings.ai_extract_worker_count,
            )?
        };
        pack.map(|pack| {
            pack.positions
                .into_iter()
                .zip(pack.frames)
                .enumerate()
                .map(|(index, (position, frame))| (index, position, frame))
                .collect::<Vec<_>>()
        })
    } else {
        None
    };
    let frames = match cached_frames {
        Some(frames) => frames,
        None if settings.local_preprocess_enabled => {
            return Err(anyhow!(
                "local staged frame cache is unavailable; direct SMB frame extraction is disabled"
            ));
        }
        None => extract_video_frames(
            video,
            duration,
            frame_count,
            settings.ai_extract_worker_count,
        ),
    };
    let expected_frames = media::sample_positions(duration, frame_count).len();
    if frames.len() < media::minimum_usable_ai_frame_count(expected_frames) {
        return Ok(Vec::new());
    }

    for chunk in frames.chunks(batch_size) {
        crate::cancel::bail_if_requested()?;
        if chunk.is_empty() {
            continue;
        }

        let mut input =
            Array4::<f32>::zeros((chunk.len(), 3, media::AI_FRAME_SIZE, media::AI_FRAME_SIZE));
        for (batch_index, (_, _, frame)) in chunk.iter().enumerate() {
            fill_normalized_nchw(&mut input, batch_index, frame);
        }

        let tensor = Tensor::from_array(input).map_err(ort_error)?;
        let outputs = session.run(ort::inputs![tensor]).map_err(ort_error)?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(ort_error)?;
        let vectors = output_vectors(shape, data, chunk.len())?;
        for (local_index, vector) in vectors.into_iter().enumerate() {
            let Some((frame_index, position, _)) = chunk.get(local_index) else {
                continue;
            };
            let (embedding, norm) = normalize(vector);
            if embedding.is_empty() {
                continue;
            }
            records.push(FrameEmbeddingRecord {
                video_id: video.id.expect("indexed video has id"),
                model_id: model_id.to_string(),
                frame_index: *frame_index,
                timestamp_seconds: *position,
                embedding,
                norm,
            });
        }
    }

    Ok(records)
}

fn extract_video_frames(
    video: &VideoRecord,
    duration: f64,
    frame_count: usize,
    extract_workers: usize,
) -> Vec<(usize, f64, Vec<u8>)> {
    let positions = media::sample_positions(duration, frame_count);
    let mut frames = Vec::new();
    let indexed_positions = positions
        .iter()
        .enumerate()
        .map(|(index, position)| (index, *position))
        .collect::<Vec<_>>();
    for work in indexed_positions.chunks(extract_workers.clamp(1, 64)) {
        let mut extracted = thread::scope(|scope| {
            work.iter()
                .map(|(frame_index, position)| {
                    scope.spawn(move || {
                        media::extract_ai_rgb_frame(Path::new(&video.path), *position)
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
    frames
}

fn append_error_log(path: &Path, message: &str) {
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{message}");
    }
}

fn fill_normalized_nchw(input: &mut Array4<f32>, batch_index: usize, rgb: &[u8]) {
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    for y in 0..media::AI_FRAME_SIZE {
        for x in 0..media::AI_FRAME_SIZE {
            let pixel = (y * media::AI_FRAME_SIZE + x) * 3;
            for channel in 0..3 {
                let value = rgb[pixel + channel] as f32 / 255.0;
                input[[batch_index, channel, y, x]] = (value - MEAN[channel]) / STD[channel];
            }
        }
    }
}

fn output_vectors(
    shape: &ort::value::Shape,
    data: &[f32],
    batch: usize,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let dims = shape
        .iter()
        .map(|dim| usize::try_from(*dim).unwrap_or(0))
        .collect::<Vec<_>>();
    if dims.is_empty() || dims[0] != batch {
        return Ok(vec![data.to_vec()]);
    }

    match dims.as_slice() {
        [batch_dim, dim] if *batch_dim == batch => Ok((0..batch)
            .map(|index| {
                let start = index * dim;
                data[start..start + dim].to_vec()
            })
            .collect()),
        [batch_dim, tokens, dim] if *batch_dim == batch && *tokens > 0 => Ok((0..batch)
            .map(|index| {
                let start = index * tokens * dim;
                data[start..start + dim].to_vec()
            })
            .collect()),
        [batch_dim, channels, height, width]
            if *batch_dim == batch && *height > 0 && *width > 0 =>
        {
            let spatial = height * width;
            Ok((0..batch)
                .map(|index| {
                    let sample_start = index * channels * spatial;
                    (0..*channels)
                        .map(|channel| {
                            let channel_start = sample_start + channel * spatial;
                            let values = &data[channel_start..channel_start + spatial];
                            values.iter().sum::<f32>() / spatial as f32
                        })
                        .collect()
                })
                .collect())
        }
        _ => {
            let per_sample = data.len() / batch.max(1);
            Ok((0..batch)
                .map(|index| {
                    let start = index * per_sample;
                    data[start..start + per_sample].to_vec()
                })
                .collect())
        }
    }
}

fn normalize(vector: Vec<f32>) -> (Vec<f32>, f64) {
    let norm = vector
        .iter()
        .map(|value| (*value as f64) * (*value as f64))
        .sum::<f64>()
        .sqrt();
    if norm <= f64::EPSILON {
        return (Vec::new(), 0.0);
    }
    (
        vector
            .into_iter()
            .map(|value| (value as f64 / norm) as f32)
            .collect(),
        norm,
    )
}

pub(crate) fn expected_ai_frame_count(video: &VideoRecord, configured: usize) -> usize {
    if video.duration_seconds.unwrap_or(0.0) > 6.0 {
        configured.clamp(8, 512)
    } else {
        1
    }
}

#[derive(Debug, Clone)]
struct AiEdge {
    a: i64,
    b: i64,
    confidence: f64,
    matched: usize,
    compared: usize,
    average_similarity: f64,
    coverage: f64,
    match_ranges: Vec<AiEdgeMatchRange>,
}

#[derive(Debug, Clone)]
struct AiEdgeMatchRange {
    video_id: i64,
    range: MatchRange,
}

#[derive(Debug, Clone)]
struct AiPairDisplayDetail {
    left_video_id: i64,
    right_video_id: i64,
    left_detail: MatchDetail,
    right_detail: MatchDetail,
}

#[derive(Debug, Clone)]
struct AiPairRelationMetrics {
    relation_type: &'static str,
    relation_label: &'static str,
    relation_explanation: String,
    short_coverage: f64,
    long_compactness: f64,
    long_span_seconds: f64,
    long_span_fraction: f64,
}

fn edge_from_pair_score(record: &AiPairScoreRecord) -> AiEdge {
    AiEdge {
        a: record.video_a_id,
        b: record.video_b_id,
        confidence: record.confidence,
        matched: record.matched,
        compared: record.compared,
        average_similarity: record.average_similarity,
        coverage: record.coverage,
        match_ranges: Vec::new(),
    }
}

fn pair_score_from_edge(
    edge: &AiEdge,
    model_id: &str,
    cache_key: &str,
    created_unix_ms: i64,
) -> AiPairScoreRecord {
    let (video_a_id, video_b_id) = edge_key(edge.a, edge.b);
    AiPairScoreRecord {
        video_a_id,
        video_b_id,
        model_id: model_id.to_string(),
        cache_key: cache_key.to_string(),
        confidence: edge.confidence,
        matched: edge.matched,
        compared: edge.compared,
        average_similarity: edge.average_similarity,
        coverage: edge.coverage,
        created_unix_ms,
    }
}

fn score_missing_pairs_parallel(
    pairs: &[(i64, i64)],
    video_by_id: &HashMap<i64, &VideoRecord>,
    by_video: &HashMap<i64, Vec<FrameEmbeddingRecord>>,
    settings: &AppSettings,
    frame_similarity_threshold: f64,
    mut on_progress: impl FnMut(usize, f64),
) -> Vec<AiEdge> {
    if pairs.is_empty() {
        return Vec::new();
    }

    let started = Instant::now();
    let workers = settings
        .ai_match_worker_count
        .clamp(1, 128)
        .min(pairs.len());
    let chunk_size = pairs.len().div_ceil(workers).max(1);

    thread::scope(|scope| {
        let (edge_sender, edge_receiver) = mpsc::channel::<Vec<AiEdge>>();
        let (progress_sender, progress_receiver) = mpsc::channel::<usize>();
        let mut spawned = 0usize;
        for chunk in pairs.chunks(chunk_size) {
            spawned = spawned.saturating_add(1);
            let edge_sender = edge_sender.clone();
            let progress_sender = progress_sender.clone();
            scope.spawn(move || {
                let mut edges = Vec::with_capacity(chunk.len());
                let mut pending_progress = 0usize;
                let mut similarity_engine = FrameSimilarityEngine::from_settings(settings);
                for (left_id, right_id) in chunk {
                    let Some(left_video) = video_by_id.get(left_id).copied() else {
                        pending_progress = pending_progress.saturating_add(1);
                        if pending_progress >= 256 {
                            let _ = progress_sender.send(pending_progress);
                            pending_progress = 0;
                        }
                        continue;
                    };
                    let Some(right_video) = video_by_id.get(right_id).copied() else {
                        pending_progress = pending_progress.saturating_add(1);
                        if pending_progress >= 256 {
                            let _ = progress_sender.send(pending_progress);
                            pending_progress = 0;
                        }
                        continue;
                    };
                    let Some(left_embeddings) = by_video.get(left_id) else {
                        pending_progress = pending_progress.saturating_add(1);
                        if pending_progress >= 256 {
                            let _ = progress_sender.send(pending_progress);
                            pending_progress = 0;
                        }
                        continue;
                    };
                    let Some(right_embeddings) = by_video.get(right_id) else {
                        pending_progress = pending_progress.saturating_add(1);
                        if pending_progress >= 256 {
                            let _ = progress_sender.send(pending_progress);
                            pending_progress = 0;
                        }
                        continue;
                    };

                    edges.push(score_pair(
                        left_video,
                        right_video,
                        left_embeddings,
                        right_embeddings,
                        settings,
                        frame_similarity_threshold,
                        &mut similarity_engine,
                    ));

                    pending_progress = pending_progress.saturating_add(1);
                    if pending_progress >= 256 {
                        let _ = progress_sender.send(pending_progress);
                        pending_progress = 0;
                    }
                }
                if pending_progress > 0 {
                    let _ = progress_sender.send(pending_progress);
                }
                let _ = edge_sender.send(edges);
            });
        }
        drop(edge_sender);
        drop(progress_sender);

        let mut edges = Vec::with_capacity(pairs.len());
        let mut finished = 0usize;
        let mut processed = 0usize;
        while finished < spawned {
            match edge_receiver.recv_timeout(Duration::from_millis(250)) {
                Ok(mut chunk_edges) => {
                    finished = finished.saturating_add(1);
                    edges.append(&mut chunk_edges);
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }

            let mut changed = false;
            while let Ok(delta) = progress_receiver.try_recv() {
                processed = processed.saturating_add(delta);
                changed = true;
            }
            if changed {
                let speed = processed as f64 / started.elapsed().as_secs_f64().max(0.001);
                on_progress(processed.min(pairs.len()), speed);
            }
        }
        while let Ok(delta) = progress_receiver.try_recv() {
            processed = processed.saturating_add(delta);
        }
        let speed = processed as f64 / started.elapsed().as_secs_f64().max(0.001);
        on_progress(processed.min(pairs.len()), speed);
        edges
    })
}

struct FrameSimilarityEngine {
    session: Option<Session>,
}

impl FrameSimilarityEngine {
    fn from_settings(settings: &AppSettings) -> Self {
        if settings.ai_device == "cpu" {
            return Self { session: None };
        }
        Self {
            session: create_frame_similarity_session(settings).ok(),
        }
    }

    #[cfg(test)]
    fn disabled() -> Self {
        Self { session: None }
    }

    fn similarities(
        &mut self,
        short: &[FrameEmbeddingRecord],
        long: &[FrameEmbeddingRecord],
    ) -> FrameSimilarityMatrix {
        if self.session.is_some() {
            match self.gpu_similarities(short, long) {
                Ok(matrix) => return matrix,
                Err(error) => {
                    eprintln!("GPU frame similarity matrix failed; falling back to CPU: {error}");
                    self.session = None;
                }
            }
        }
        cpu_similarities(short, long)
    }

    fn gpu_similarities(
        &mut self,
        short: &[FrameEmbeddingRecord],
        long: &[FrameEmbeddingRecord],
    ) -> anyhow::Result<FrameSimilarityMatrix> {
        let rows = short.len();
        let columns = long.len();
        if rows == 0 || columns == 0 {
            return Ok(FrameSimilarityMatrix::empty(rows, columns));
        }
        let dimension =
            common_embedding_dimension(short, long).context("frame embedding dimensions differ")?;
        let mut left_matrix = Array2::<f32>::zeros((rows, dimension));
        let mut right_matrix = Array2::<f32>::zeros((dimension, columns));
        for (row, embedding) in short.iter().enumerate() {
            for (column, value) in embedding.embedding.iter().enumerate().take(dimension) {
                left_matrix[[row, column]] = *value;
            }
        }
        for (column, embedding) in long.iter().enumerate() {
            for (row, value) in embedding.embedding.iter().enumerate().take(dimension) {
                right_matrix[[row, column]] = *value;
            }
        }

        let session = self
            .session
            .as_mut()
            .context("frame similarity GPU session is unavailable")?;
        let left_tensor = TensorRef::from_array_view(left_matrix.view()).map_err(ort_error)?;
        let right_tensor = TensorRef::from_array_view(right_matrix.view()).map_err(ort_error)?;
        let outputs = session
            .run(ort::inputs! {
                "a" => left_tensor,
                "b" => right_tensor,
            })
            .map_err(ort_error)?;
        let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(ort_error)?;
        let dims = shape
            .iter()
            .map(|dim| usize::try_from(*dim).unwrap_or(0))
            .collect::<Vec<_>>();
        if dims.as_slice() != [rows, columns] || data.len() != rows * columns {
            return Err(anyhow!(
                "unexpected frame similarity matrix shape {:?}, expected [{rows}, {columns}]",
                dims
            ));
        }

        Ok(FrameSimilarityMatrix {
            rows,
            columns,
            values: data.to_vec(),
        })
    }
}

#[derive(Debug, Clone)]
struct FrameSimilarityMatrix {
    rows: usize,
    columns: usize,
    values: Vec<f32>,
}

impl FrameSimilarityMatrix {
    fn empty(rows: usize, columns: usize) -> Self {
        Self {
            rows,
            columns,
            values: Vec::new(),
        }
    }

    fn get(&self, row: usize, column: usize) -> f64 {
        if row >= self.rows || column >= self.columns {
            return 0.0;
        }
        self.values
            .get(row * self.columns + column)
            .copied()
            .unwrap_or(0.0) as f64
    }
}

fn common_embedding_dimension(
    short: &[FrameEmbeddingRecord],
    long: &[FrameEmbeddingRecord],
) -> Option<usize> {
    let dimension = short
        .first()
        .or_else(|| long.first())
        .map(|embedding| embedding.embedding.len())?;
    if dimension == 0 {
        return None;
    }
    if short
        .iter()
        .chain(long.iter())
        .all(|embedding| embedding.embedding.len() == dimension)
    {
        Some(dimension)
    } else {
        None
    }
}

fn cpu_similarities(
    short: &[FrameEmbeddingRecord],
    long: &[FrameEmbeddingRecord],
) -> FrameSimilarityMatrix {
    let rows = short.len();
    let columns = long.len();
    let mut values = Vec::with_capacity(rows.saturating_mul(columns));
    for short_embedding in short {
        for long_embedding in long {
            values.push(cosine(&short_embedding.embedding, &long_embedding.embedding) as f32);
        }
    }
    FrameSimilarityMatrix {
        rows,
        columns,
        values,
    }
}

fn score_pair(
    left_video: &VideoRecord,
    right_video: &VideoRecord,
    left: &[FrameEmbeddingRecord],
    right: &[FrameEmbeddingRecord],
    settings: &AppSettings,
    frame_similarity_threshold: f64,
    similarity_engine: &mut FrameSimilarityEngine,
) -> AiEdge {
    let left_duration = left_video.duration_seconds.unwrap_or(0.0);
    let right_duration = right_video.duration_seconds.unwrap_or(0.0);
    let (short_video_id, long_video_id, short_video, long_video, short, long) =
        if left_duration <= right_duration {
            (
                left_video.id.unwrap_or(0),
                right_video.id.unwrap_or(0),
                left_video,
                right_video,
                left,
                right,
            )
        } else {
            (
                right_video.id.unwrap_or(0),
                left_video.id.unwrap_or(0),
                right_video,
                left_video,
                right,
                left,
            )
        };

    let summary = monotonic_match_summary(
        short,
        long,
        frame_similarity_threshold,
        settings.ai_clip_matching_enabled,
        similarity_engine,
    );
    let matched = summary.matched;
    let average_similarity = summary.average_similarity;
    let compared = short.len().min(long.len()).max(1);
    let coverage = matched as f64 / compared as f64;
    let min_frames = settings.ai_min_matched_frames.clamp(1, 128);
    let confidence = confidence_from_match(
        matched,
        average_similarity,
        coverage,
        frame_similarity_threshold,
        min_frames,
    );
    let mut match_ranges = Vec::new();
    if let Some((start, end)) = summary.short_range {
        if let Some(range) =
            match_range_from_indices(short, short_video, start, end, matched, average_similarity)
        {
            match_ranges.push(AiEdgeMatchRange {
                video_id: short_video_id,
                range,
            });
        }
    }
    if let Some((start, end)) = summary.long_range {
        if let Some(range) =
            match_range_from_indices(long, long_video, start, end, matched, average_similarity)
        {
            match_ranges.push(AiEdgeMatchRange {
                video_id: long_video_id,
                range,
            });
        }
    }

    AiEdge {
        a: short_video_id.min(long_video_id),
        b: short_video_id.max(long_video_id),
        confidence,
        matched,
        compared,
        average_similarity,
        coverage,
        match_ranges,
    }
}

fn score_pair_detail(
    left_video: &VideoRecord,
    right_video: &VideoRecord,
    left: &[FrameEmbeddingRecord],
    right: &[FrameEmbeddingRecord],
    settings: &AppSettings,
    frame_similarity_threshold: f64,
    confidence: f64,
    similarity_engine: &mut FrameSimilarityEngine,
) -> Option<AiPairDisplayDetail> {
    let left_id = left_video.id?;
    let right_id = right_video.id?;
    let left_duration = left_video.duration_seconds.unwrap_or(0.0);
    let right_duration = right_video.duration_seconds.unwrap_or(0.0);
    let (short_video, long_video, short_embeddings, long_embeddings, left_is_short) =
        if left_duration <= right_duration {
            (left_video, right_video, left, right, true)
        } else {
            (right_video, left_video, right, left, false)
        };

    let detail = monotonic_match_detail(
        short_embeddings,
        long_embeddings,
        frame_similarity_threshold,
        settings.ai_clip_matching_enabled,
        similarity_engine,
    );
    if detail.summary.matched == 0 || detail.points.is_empty() {
        return None;
    }

    let relation = relation_metrics(
        &detail,
        short_embeddings,
        long_embeddings,
        short_video,
        long_video,
    );
    let short_detail = build_match_detail_for_video(
        long_video,
        short_video,
        short_embeddings,
        &detail,
        &relation,
        true,
        confidence,
    )?;
    let long_detail = build_match_detail_for_video(
        short_video,
        long_video,
        long_embeddings,
        &detail,
        &relation,
        false,
        confidence,
    )?;

    let (left_detail, right_detail) = if left_is_short {
        (short_detail, long_detail)
    } else {
        (long_detail, short_detail)
    };

    Some(AiPairDisplayDetail {
        left_video_id: left_id,
        right_video_id: right_id,
        left_detail,
        right_detail,
    })
}

fn build_match_detail_for_video(
    peer_video: &VideoRecord,
    video: &VideoRecord,
    embeddings: &[FrameEmbeddingRecord],
    detail: &MonotonicMatchDetail,
    relation: &AiPairRelationMetrics,
    use_short_points: bool,
    confidence: f64,
) -> Option<MatchDetail> {
    let range = if use_short_points {
        let (start, end) = detail.summary.short_range?;
        match_range_from_indices(
            embeddings,
            video,
            start,
            end,
            detail.summary.matched,
            detail.summary.average_similarity,
        )?
    } else {
        let (start, end) = detail.summary.long_range?;
        match_range_from_indices(
            embeddings,
            video,
            start,
            end,
            detail.summary.matched,
            detail.summary.average_similarity,
        )?
    };
    let points = sampled_hit_points(embeddings, video, &detail.points, use_short_points);

    Some(MatchDetail {
        peer_video_id: peer_video.id,
        peer_file_name: peer_video.file_name.clone(),
        displayed_hit_count: points.len(),
        points,
        start_seconds: range.start_seconds,
        end_seconds: range.end_seconds,
        hit_count: detail.summary.matched,
        average_similarity: detail.summary.average_similarity,
        confidence,
        short_coverage: relation.short_coverage,
        long_compactness: relation.long_compactness,
        long_span_seconds: relation.long_span_seconds,
        long_span_fraction: relation.long_span_fraction,
        relation_type: relation.relation_type.to_string(),
        relation_label: relation.relation_label.to_string(),
        relation_explanation: relation.relation_explanation.clone(),
    })
}

fn sampled_hit_points(
    embeddings: &[FrameEmbeddingRecord],
    video: &VideoRecord,
    points: &[MonotonicMatchPoint],
    use_short_points: bool,
) -> Vec<MatchHitPoint> {
    if points.is_empty() || embeddings.is_empty() {
        return Vec::new();
    }
    let sample_indices = if points.len() <= AI_MATCH_DETAIL_MAX_POINTS {
        (0..points.len()).collect::<Vec<_>>()
    } else {
        (0..AI_MATCH_DETAIL_MAX_POINTS)
            .map(|index| index * (points.len() - 1) / (AI_MATCH_DETAIL_MAX_POINTS - 1))
            .collect::<Vec<_>>()
    };
    let duration = video
        .duration_seconds
        .filter(|value| value.is_finite() && *value > 0.0);
    sample_indices
        .into_iter()
        .filter_map(|point_index| {
            let point = points.get(point_index)?;
            let embedding_index = if use_short_points {
                point.short_index
            } else {
                point.long_index
            };
            let embedding = embeddings.get(embedding_index)?;
            let seconds = finite_non_negative(embedding.timestamp_seconds);
            let fraction = if let Some(duration) = duration {
                fraction_from_seconds(seconds, duration)
            } else {
                (embedding_index as f64 / embeddings.len().max(1) as f64).clamp(0.0, 1.0)
            };
            Some(MatchHitPoint {
                seconds,
                fraction,
                similarity: point.similarity.clamp(0.0, 1.0),
            })
        })
        .collect()
}

fn relation_metrics(
    detail: &MonotonicMatchDetail,
    short: &[FrameEmbeddingRecord],
    long: &[FrameEmbeddingRecord],
    short_video: &VideoRecord,
    long_video: &VideoRecord,
) -> AiPairRelationMetrics {
    let matched = detail.summary.matched;
    let short_coverage = matched as f64 / short.len().max(1) as f64;
    let (long_start, long_end) = detail.summary.long_range.unwrap_or((0, 0));
    let long_range = match_range_from_indices(
        long,
        long_video,
        long_start.min(long.len().saturating_sub(1)),
        long_end.min(long.len().saturating_sub(1)),
        matched,
        detail.summary.average_similarity,
    );
    let long_span_seconds = long_range
        .as_ref()
        .map(|range| (range.end_seconds - range.start_seconds).max(0.0))
        .unwrap_or(0.0);
    let long_span_fraction = long_range
        .as_ref()
        .map(|range| (range.end_fraction - range.start_fraction).clamp(0.0, 1.0))
        .unwrap_or(0.0);
    let long_span_frames = long_end.saturating_sub(long_start).saturating_add(1).max(1);
    let long_compactness = (matched as f64 / long_span_frames as f64).clamp(0.0, 1.0);
    let short_span_seconds = detail
        .summary
        .short_range
        .and_then(|(start, end)| {
            match_range_from_indices(
                short,
                short_video,
                start,
                end,
                matched,
                detail.summary.average_similarity,
            )
        })
        .map(|range| (range.end_seconds - range.start_seconds).max(0.0))
        .or_else(|| {
            short_video
                .duration_seconds
                .filter(|value| value.is_finite() && *value > 0.0)
        })
        .unwrap_or(0.0);
    let span_ratio = if short_span_seconds > 0.0 {
        long_span_seconds / short_span_seconds
    } else {
        999.0
    };
    let long_start_text = long_range
        .as_ref()
        .map(|range| format_seconds_compact(range.start_seconds))
        .unwrap_or_else(|| "0:00".to_string());
    let long_end_text = long_range
        .as_ref()
        .map(|range| format_seconds_compact(range.end_seconds))
        .unwrap_or_else(|| "0:00".to_string());

    let (relation_type, relation_label, relation_explanation) = if short_coverage >= 0.82
        && long_compactness >= 0.55
        && (0.60..=1.55).contains(&span_ratio)
    {
        (
            "complete_containment",
            "完整包含",
            format!("完整包含：短片大部分内容集中映射到长片 {long_start_text}-{long_end_text}"),
        )
    } else if short_coverage >= 0.55 && long_compactness >= 0.45 && span_ratio <= 2.0 {
        (
            "partial_containment",
            "不完整包含",
            "不完整包含：短片部分内容能映射到长片，但覆盖不足".to_string(),
        )
    } else if long_compactness >= 0.55 && long_span_fraction <= 0.25 {
        (
            "compact_edit",
            "紧凑剪辑",
            "紧凑剪辑：相似帧集中在长片单一片段".to_string(),
        )
    } else {
        (
            "not_contained",
            "互不包含",
            "互不包含：命中点分散，更像同场景相似，不像完整/短版关系".to_string(),
        )
    };

    AiPairRelationMetrics {
        relation_type,
        relation_label,
        relation_explanation,
        short_coverage: short_coverage.clamp(0.0, 1.0),
        long_compactness,
        long_span_seconds,
        long_span_fraction,
    }
}

fn format_seconds_compact(seconds: f64) -> String {
    if !seconds.is_finite() || seconds <= 0.0 {
        return "0:00".to_string();
    }
    let total = seconds.round() as u64;
    let hours = total / 3600;
    let minutes = (total % 3600) / 60;
    let seconds = total % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

fn match_range_from_indices(
    embeddings: &[FrameEmbeddingRecord],
    video: &VideoRecord,
    start_index: usize,
    end_index: usize,
    matched_frames: usize,
    average_similarity: f64,
) -> Option<MatchRange> {
    if embeddings.is_empty()
        || start_index >= embeddings.len()
        || end_index >= embeddings.len()
        || start_index > end_index
    {
        return None;
    }

    let duration = video
        .duration_seconds
        .filter(|value| value.is_finite() && *value > 0.0);
    let frame_step = estimated_frame_step_seconds(embeddings, duration);
    let mut start_seconds = finite_non_negative(embeddings[start_index].timestamp_seconds);
    let mut end_seconds = finite_non_negative(embeddings[end_index].timestamp_seconds + frame_step)
        .max(start_seconds);

    let (mut start_fraction, mut end_fraction) = if let Some(duration) = duration {
        start_seconds = start_seconds.min(duration);
        end_seconds = end_seconds.min(duration).max(start_seconds);
        if end_seconds <= start_seconds {
            end_seconds = (start_seconds + frame_step.max(0.001)).min(duration);
        }
        (
            fraction_from_seconds(start_seconds, duration),
            fraction_from_seconds(end_seconds, duration),
        )
    } else {
        let span = embeddings.len().max(1) as f64;
        (
            (start_index as f64 / span).clamp(0.0, 1.0),
            ((end_index + 1) as f64 / span).clamp(0.0, 1.0),
        )
    };

    if end_fraction <= start_fraction {
        let minimum = (1.0 / embeddings.len().max(1) as f64).min(0.05);
        end_fraction = (start_fraction + minimum).min(1.0);
        if end_fraction <= start_fraction {
            start_fraction = (end_fraction - minimum).max(0.0);
        }
    }

    Some(MatchRange {
        start_seconds,
        end_seconds,
        start_fraction,
        end_fraction,
        matched_frames,
        average_similarity,
    })
}

fn estimated_frame_step_seconds(embeddings: &[FrameEmbeddingRecord], duration: Option<f64>) -> f64 {
    let mut previous: Option<f64> = None;
    let mut total = 0.0;
    let mut count = 0usize;
    for embedding in embeddings {
        let timestamp = embedding.timestamp_seconds;
        if !timestamp.is_finite() {
            continue;
        }
        if let Some(previous_timestamp) = previous {
            let delta = timestamp - previous_timestamp;
            if delta.is_finite() && delta > 0.0 {
                total += delta;
                count = count.saturating_add(1);
            }
        }
        previous = Some(timestamp);
    }
    if count > 0 {
        return (total / count as f64).clamp(0.001, 600.0);
    }
    if let Some(duration) = duration {
        return (duration / embeddings.len().max(1) as f64).clamp(0.001, 600.0);
    }
    0.0
}

fn fraction_from_seconds(seconds: f64, duration: f64) -> f64 {
    if !seconds.is_finite() || !duration.is_finite() || duration <= 0.0 {
        return 0.0;
    }
    (seconds / duration).clamp(0.0, 1.0)
}

fn finite_non_negative(value: f64) -> f64 {
    if value.is_finite() {
        value.max(0.0)
    } else {
        0.0
    }
}

fn confidence_from_match(
    matched: usize,
    average_similarity: f64,
    coverage: f64,
    frame_similarity_threshold: f64,
    min_frames: usize,
) -> f64 {
    if matched >= min_frames {
        let denominator = (1.0 - frame_similarity_threshold).max(0.001);
        let quality_margin =
            ((average_similarity - frame_similarity_threshold) / denominator).clamp(0.0, 1.0);
        let evidence_strength = (matched as f64 / (min_frames as f64 * 2.0)).clamp(0.0, 1.0);
        (average_similarity + 0.08 * quality_margin * evidence_strength + 0.02 * coverage.min(1.0))
            .clamp(0.0, 1.0)
    } else {
        (average_similarity * coverage * 0.65).clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone, Copy)]
struct MonotonicMatchSummary {
    matched: usize,
    average_similarity: f64,
    short_range: Option<(usize, usize)>,
    long_range: Option<(usize, usize)>,
}

#[derive(Debug, Clone)]
struct MonotonicMatchDetail {
    summary: MonotonicMatchSummary,
    points: Vec<MonotonicMatchPoint>,
}

#[derive(Debug, Clone, Copy)]
struct MonotonicMatchPoint {
    short_index: usize,
    long_index: usize,
    similarity: f64,
}

#[derive(Debug, Clone, Copy)]
struct MonotonicMatchParent {
    previous: usize,
    matched: bool,
    similarity: f64,
}

impl MonotonicMatchParent {
    const UNSET: usize = usize::MAX;

    fn empty() -> Self {
        Self {
            previous: Self::UNSET,
            matched: false,
            similarity: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MonotonicMatchState {
    count: usize,
    sum: f64,
    short_start: usize,
    short_end: usize,
    long_start: usize,
    long_end: usize,
}

impl MonotonicMatchState {
    const UNSET: usize = usize::MAX;

    fn empty() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            short_start: Self::UNSET,
            short_end: Self::UNSET,
            long_start: Self::UNSET,
            long_end: Self::UNSET,
        }
    }

    fn extend(self, short_index: usize, long_index: usize, similarity: f64) -> Self {
        if self.count == 0 {
            return Self {
                count: 1,
                sum: similarity,
                short_start: short_index,
                short_end: short_index,
                long_start: long_index,
                long_end: long_index,
            };
        }

        Self {
            count: self.count + 1,
            sum: self.sum + similarity,
            short_start: self.short_start,
            short_end: short_index,
            long_start: self.long_start,
            long_end: long_index,
        }
    }
}

fn monotonic_match_summary(
    short: &[FrameEmbeddingRecord],
    long: &[FrameEmbeddingRecord],
    threshold: f64,
    clip_matching: bool,
    similarity_engine: &mut FrameSimilarityEngine,
) -> MonotonicMatchSummary {
    let rows = short.len();
    let columns = long.len();
    if rows == 0 || columns == 0 {
        return MonotonicMatchSummary {
            matched: 0,
            average_similarity: 0.0,
            short_range: None,
            long_range: None,
        };
    }

    let similarities = similarity_engine.similarities(short, long);
    let stride = columns + 1;
    let mut states = vec![MonotonicMatchState::empty(); (rows + 1) * stride];
    for row in 1..=rows {
        for column in 1..=columns {
            let index = row * stride + column;
            let up = (row - 1) * stride + column;
            let left = row * stride + column - 1;
            let mut best = states[up];
            if is_better_match(states[left].count, states[left].sum, best.count, best.sum) {
                best = states[left];
            }

            if clip_matching || row.abs_diff(column) <= 2 {
                let similarity = similarities.get(row - 1, column - 1);
                if similarity >= threshold {
                    let diagonal = (row - 1) * stride + column - 1;
                    let candidate = states[diagonal].extend(row - 1, column - 1, similarity);
                    if is_better_match(candidate.count, candidate.sum, best.count, best.sum) {
                        best = candidate;
                    }
                }
            }

            states[index] = best;
        }
    }
    let best = states[rows * stride + columns];
    if best.count == 0 {
        return MonotonicMatchSummary {
            matched: 0,
            average_similarity: 0.0,
            short_range: None,
            long_range: None,
        };
    }

    let short_range = if best.short_start == MonotonicMatchState::UNSET {
        None
    } else {
        Some((best.short_start, best.short_end))
    };
    let long_range = if best.long_start == MonotonicMatchState::UNSET {
        None
    } else {
        Some((best.long_start, best.long_end))
    };

    MonotonicMatchSummary {
        matched: best.count,
        average_similarity: best.sum / best.count as f64,
        short_range,
        long_range,
    }
}

fn monotonic_match_detail(
    short: &[FrameEmbeddingRecord],
    long: &[FrameEmbeddingRecord],
    threshold: f64,
    clip_matching: bool,
    similarity_engine: &mut FrameSimilarityEngine,
) -> MonotonicMatchDetail {
    let rows = short.len();
    let columns = long.len();
    if rows == 0 || columns == 0 {
        return MonotonicMatchDetail {
            summary: MonotonicMatchSummary {
                matched: 0,
                average_similarity: 0.0,
                short_range: None,
                long_range: None,
            },
            points: Vec::new(),
        };
    }

    let similarities = similarity_engine.similarities(short, long);
    let stride = columns + 1;
    let mut states = vec![MonotonicMatchState::empty(); (rows + 1) * stride];
    let mut parents = vec![MonotonicMatchParent::empty(); (rows + 1) * stride];
    for row in 1..=rows {
        for column in 1..=columns {
            let index = row * stride + column;
            let up = (row - 1) * stride + column;
            let left = row * stride + column - 1;
            let mut best = states[up];
            let mut parent = MonotonicMatchParent {
                previous: up,
                matched: false,
                similarity: 0.0,
            };
            if is_better_match(states[left].count, states[left].sum, best.count, best.sum) {
                best = states[left];
                parent = MonotonicMatchParent {
                    previous: left,
                    matched: false,
                    similarity: 0.0,
                };
            }

            if clip_matching || row.abs_diff(column) <= 2 {
                let similarity = similarities.get(row - 1, column - 1);
                if similarity >= threshold {
                    let diagonal = (row - 1) * stride + column - 1;
                    let candidate = states[diagonal].extend(row - 1, column - 1, similarity);
                    if is_better_match(candidate.count, candidate.sum, best.count, best.sum) {
                        best = candidate;
                        parent = MonotonicMatchParent {
                            previous: diagonal,
                            matched: true,
                            similarity,
                        };
                    }
                }
            }

            states[index] = best;
            if best.count > 0 {
                parents[index] = parent;
            }
        }
    }

    let best = states[rows * stride + columns];
    if best.count == 0 {
        return MonotonicMatchDetail {
            summary: MonotonicMatchSummary {
                matched: 0,
                average_similarity: 0.0,
                short_range: None,
                long_range: None,
            },
            points: Vec::new(),
        };
    }

    let mut points = Vec::with_capacity(best.count);
    let mut cursor = rows * stride + columns;
    while cursor < parents.len() {
        let parent = parents[cursor];
        if parent.previous == MonotonicMatchParent::UNSET {
            break;
        }
        if parent.matched {
            let row = cursor / stride;
            let column = cursor % stride;
            if row > 0 && column > 0 {
                points.push(MonotonicMatchPoint {
                    short_index: row - 1,
                    long_index: column - 1,
                    similarity: parent.similarity,
                });
            }
        }
        cursor = parent.previous;
    }
    points.reverse();

    let short_range = if best.short_start == MonotonicMatchState::UNSET {
        None
    } else {
        Some((best.short_start, best.short_end))
    };
    let long_range = if best.long_start == MonotonicMatchState::UNSET {
        None
    } else {
        Some((best.long_start, best.long_end))
    };

    MonotonicMatchDetail {
        summary: MonotonicMatchSummary {
            matched: best.count,
            average_similarity: best.sum / best.count as f64,
            short_range,
            long_range,
        },
        points,
    }
}

fn is_better_match(
    candidate_count: usize,
    candidate_sum: f64,
    best_count: usize,
    best_sum: f64,
) -> bool {
    candidate_count > best_count
        || (candidate_count == best_count && candidate_sum > best_sum + f64::EPSILON)
}

fn cosine(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (*a as f64) * (*b as f64))
        .sum::<f64>()
}

fn connected_ai_groups(
    videos: &[VideoRecord],
    edges: &[AiEdge],
    threshold: f64,
    size_priority_seconds: f64,
) -> Vec<MatchGroup> {
    let video_by_id = videos
        .iter()
        .filter_map(|video| video.id.map(|id| (id, video.clone())))
        .collect::<HashMap<_, _>>();
    let mut ordered_edges = edges.to_vec();
    ordered_edges.sort_by(|a, b| {
        b.confidence
            .total_cmp(&a.confidence)
            .then_with(|| b.matched.cmp(&a.matched))
            .then_with(|| a.compared.cmp(&b.compared))
    });
    let edge_by_pair = ordered_edges
        .iter()
        .cloned()
        .map(|edge| (edge_key(edge.a, edge.b), edge))
        .collect::<HashMap<_, _>>();
    let video_ids = video_by_id.keys().copied().collect::<Vec<_>>();
    let expansion_threshold = threshold.max(0.68);
    let mut assigned = HashSet::new();
    let mut groups = Vec::new();

    for seed in &ordered_edges {
        if assigned.contains(&seed.a) || assigned.contains(&seed.b) {
            continue;
        }
        let mut ids = vec![seed.a, seed.b];
        while let Some(next_id) = best_complete_link_candidate(
            &video_ids,
            &ids,
            &assigned,
            &edge_by_pair,
            expansion_threshold,
        ) {
            ids.push(next_id);
        }
        ids.sort_unstable();
        ids.dedup();
        if ids.len() < 2 {
            continue;
        }

        let group_edges = group_edges(&ids, &edge_by_pair);
        if group_edges.is_empty() {
            continue;
        }

        for id in &ids {
            assigned.insert(*id);
        }
        if let Some(group) = build_ai_group(
            &video_by_id,
            ids,
            &group_edges,
            threshold,
            size_priority_seconds,
        ) {
            groups.push(group);
        }
    }

    groups.sort_by(|a, b| {
        b.confidence
            .total_cmp(&a.confidence)
            .then_with(|| b.reclaimable_bytes.cmp(&a.reclaimable_bytes))
    });
    groups
}

fn best_complete_link_candidate(
    video_ids: &[i64],
    members: &[i64],
    assigned: &HashSet<i64>,
    edge_by_pair: &HashMap<(i64, i64), AiEdge>,
    threshold: f64,
) -> Option<i64> {
    let mut best: Option<(i64, f64)> = None;
    for candidate in video_ids {
        if assigned.contains(candidate) || members.contains(candidate) {
            continue;
        }
        let mut total = 0.0;
        let mut valid = true;
        for member in members {
            let Some(edge) = edge_by_pair.get(&edge_key(*candidate, *member)) else {
                valid = false;
                break;
            };
            if edge.confidence < threshold {
                valid = false;
                break;
            }
            total += edge.confidence;
        }
        if valid {
            let average = total / members.len().max(1) as f64;
            if best.is_none_or(|(_, score)| average > score) {
                best = Some((*candidate, average));
            }
        }
    }
    best.map(|(id, _)| id)
}

fn group_edges<'a>(ids: &[i64], edge_by_pair: &'a HashMap<(i64, i64), AiEdge>) -> Vec<&'a AiEdge> {
    let mut edges = Vec::new();
    for (left_index, left) in ids.iter().enumerate() {
        for right in ids.iter().skip(left_index + 1) {
            if let Some(edge) = edge_by_pair.get(&edge_key(*left, *right)) {
                edges.push(edge);
            }
        }
    }
    edges
}

fn merged_match_ranges(video_id: i64, edges: &[&AiEdge]) -> Vec<MatchRange> {
    let mut ranges = edges
        .iter()
        .flat_map(|edge| {
            edge.match_ranges
                .iter()
                .filter(move |range| range.video_id == video_id)
                .map(|range| range.range.clone())
        })
        .filter(|range| {
            range.start_fraction.is_finite()
                && range.end_fraction.is_finite()
                && range.end_fraction > range.start_fraction
        })
        .collect::<Vec<_>>();
    ranges.sort_by(|a, b| {
        a.start_fraction
            .total_cmp(&b.start_fraction)
            .then_with(|| a.end_fraction.total_cmp(&b.end_fraction))
    });

    let mut merged: Vec<MatchRange> = Vec::new();
    for range in ranges {
        if let Some(last) = merged.last_mut() {
            if range.start_fraction <= last.end_fraction + 0.01 {
                merge_match_range(last, range);
                continue;
            }
        }
        merged.push(range);
    }
    merged
}

fn merge_match_range(target: &mut MatchRange, next: MatchRange) {
    let target_frames = target.matched_frames.max(1);
    let next_frames = next.matched_frames.max(1);
    let total_frames = target_frames.saturating_add(next_frames);
    target.average_similarity = if total_frames == 0 {
        target.average_similarity.max(next.average_similarity)
    } else {
        ((target.average_similarity * target_frames as f64)
            + (next.average_similarity * next_frames as f64))
            / total_frames as f64
    };
    target.matched_frames = target.matched_frames.saturating_add(next.matched_frames);
    target.start_seconds = target.start_seconds.min(next.start_seconds);
    target.end_seconds = target.end_seconds.max(next.end_seconds);
    target.start_fraction = target
        .start_fraction
        .min(next.start_fraction)
        .clamp(0.0, 1.0);
    target.end_fraction = target.end_fraction.max(next.end_fraction).clamp(0.0, 1.0);
}

fn ai_group_detail_pair_count(groups: &[MatchGroup], edges: &[AiEdge]) -> usize {
    let edge_by_pair = edges
        .iter()
        .map(|edge| (edge_key(edge.a, edge.b), edge))
        .collect::<HashMap<_, _>>();
    groups
        .iter()
        .map(|group| detail_edges_for_group(group, &edge_by_pair).len())
        .sum()
}

fn attach_ai_group_match_details(
    groups: &mut [MatchGroup],
    edges: &[AiEdge],
    video_by_id: &HashMap<i64, &VideoRecord>,
    by_video: &HashMap<i64, Vec<FrameEmbeddingRecord>>,
    settings: &AppSettings,
    frame_similarity_threshold: f64,
    mut on_progress: impl FnMut(usize, usize),
) {
    let edge_by_pair = edges
        .iter()
        .map(|edge| (edge_key(edge.a, edge.b), edge))
        .collect::<HashMap<_, _>>();
    let total = groups
        .iter()
        .map(|group| detail_edges_for_group(group, &edge_by_pair).len())
        .sum::<usize>();
    if total == 0 {
        return;
    }

    let mut processed = 0usize;
    let mut similarity_engine = FrameSimilarityEngine::from_settings(settings);
    for group in groups {
        let detail_edges = detail_edges_for_group(group, &edge_by_pair);
        for edge in detail_edges {
            processed = processed.saturating_add(1);
            if let (
                Some(left_video),
                Some(right_video),
                Some(left_embeddings),
                Some(right_embeddings),
            ) = (
                video_by_id.get(&edge.a).copied(),
                video_by_id.get(&edge.b).copied(),
                by_video.get(&edge.a),
                by_video.get(&edge.b),
            ) {
                if let Some(detail) = score_pair_detail(
                    left_video,
                    right_video,
                    left_embeddings,
                    right_embeddings,
                    settings,
                    frame_similarity_threshold,
                    edge.confidence,
                    &mut similarity_engine,
                ) {
                    apply_pair_display_detail(group, detail);
                }
            }
            on_progress(processed, total);
        }
    }
}

fn detail_edges_for_group<'a>(
    group: &MatchGroup,
    edge_by_pair: &HashMap<(i64, i64), &'a AiEdge>,
) -> Vec<&'a AiEdge> {
    let ids = group
        .items
        .iter()
        .filter_map(|item| item.video.id)
        .collect::<Vec<_>>();
    if ids.len() < 2 {
        return Vec::new();
    }

    let recommended_id = group
        .recommended_video_id
        .or_else(|| group.items.first().and_then(|item| item.video.id));
    let mut candidates: Vec<(u8, &'a AiEdge)> = Vec::new();
    if let Some(recommended_id) = recommended_id {
        for id in &ids {
            if *id == recommended_id {
                continue;
            }
            if let Some(edge) = edge_by_pair.get(&edge_key(recommended_id, *id)) {
                candidates.push((0, *edge));
            }
        }
    }

    for (left_index, left) in ids.iter().enumerate() {
        for right in ids.iter().skip(left_index + 1) {
            if let Some(edge) = edge_by_pair.get(&edge_key(*left, *right)) {
                candidates.push((1, *edge));
            }
        }
    }

    candidates.sort_by(|(left_priority, left_edge), (right_priority, right_edge)| {
        left_priority
            .cmp(right_priority)
            .then_with(|| right_edge.confidence.total_cmp(&left_edge.confidence))
            .then_with(|| right_edge.matched.cmp(&left_edge.matched))
    });
    let mut seen = HashSet::new();
    let mut selected = Vec::new();
    for (_, edge) in candidates {
        if !seen.insert(edge_key(edge.a, edge.b)) {
            continue;
        }
        selected.push(edge);
        if selected.len() >= AI_MATCH_DETAIL_MAX_PAIRS_PER_GROUP {
            break;
        }
    }
    selected
}

fn apply_pair_display_detail(group: &mut MatchGroup, detail: AiPairDisplayDetail) {
    for item in &mut group.items {
        if item.video.id == Some(detail.left_video_id) {
            set_better_match_detail(item, detail.left_detail.clone());
        } else if item.video.id == Some(detail.right_video_id) {
            set_better_match_detail(item, detail.right_detail.clone());
        }
    }
}

fn set_better_match_detail(item: &mut MatchItem, detail: MatchDetail) {
    if item
        .match_detail
        .as_ref()
        .is_none_or(|current| detail.confidence > current.confidence)
    {
        item.match_detail = Some(detail);
    }
}

fn build_ai_group(
    video_by_id: &HashMap<i64, VideoRecord>,
    ids: Vec<i64>,
    edges: &[&AiEdge],
    fallback_confidence: f64,
    size_priority_seconds: f64,
) -> Option<MatchGroup> {
    let mut items = ids
        .iter()
        .filter_map(|id| video_by_id.get(id).cloned())
        .collect::<Vec<_>>();
    if items.len() < 2 {
        return None;
    }
    let group_key = ids.iter().min().copied().unwrap_or(0);
    items.sort_by(|a, b| compare_ai_keeper(b, a, size_priority_seconds));
    let recommended_id = items.first().and_then(|video| video.id);
    let confidence = if edges.is_empty() {
        fallback_confidence
    } else {
        (edges.iter().map(|edge| edge.confidence).sum::<f64>() / edges.len() as f64).clamp(0.0, 1.0)
    };
    let average_frame_similarity = if edges.is_empty() {
        confidence
    } else {
        (edges
            .iter()
            .map(|edge| edge.average_similarity)
            .sum::<f64>()
            / edges.len() as f64)
            .clamp(0.0, 1.0)
    };
    let average_coverage = if edges.is_empty() {
        0.0
    } else {
        (edges.iter().map(|edge| edge.coverage).sum::<f64>() / edges.len() as f64).clamp(0.0, 1.0)
    };
    let min_confidence = edges
        .iter()
        .map(|edge| edge.confidence)
        .fold(confidence, f64::min);
    let matched_evidence = edges.iter().map(|edge| edge.matched).max().unwrap_or(0);
    let compared_evidence = edges.iter().map(|edge| edge.compared).max().unwrap_or(0);
    let reclaimable_bytes = items
        .iter()
        .skip(1)
        .map(|video| video.size_bytes)
        .sum::<u64>();
    let title = items
        .first()
        .map(|video| video.file_name.clone())
        .unwrap_or_else(|| "AI visual match".to_string());
    let match_items = items
        .into_iter()
        .enumerate()
        .map(|(index, video)| {
            let match_ranges = video
                .id
                .map(|id| merged_match_ranges(id, edges))
                .unwrap_or_default();
            MatchItem {
                video,
                role: if index == 0 {
                    "推荐保留".to_string()
                } else {
                    "AI视觉相似".to_string()
                },
                quality_rank: index + 1,
                match_ranges,
                match_detail: None,
            }
        })
        .collect::<Vec<_>>();
    let evidence = vec![
        format!(
            "AI帧平均相似度 {}%",
            (average_frame_similarity * 100.0).round() as u32
        ),
        format!(
            "组内最低相似度 {}%",
            (min_confidence * 100.0).round() as u32
        ),
        format!("有序相似帧最多 {matched_evidence}/{compared_evidence} 个"),
        format!(
            "平均片段覆盖率 {}%",
            (average_coverage * 100.0).round() as u32
        ),
        "帧级阈值与列表过滤阈值分离，避免低阈值弱相似误配".to_string(),
        "完整连接分组，避免弱相似链式串联".to_string(),
    ];
    let report = ai_group_report(
        &format!("group-{}", 900_000 + group_key),
        confidence,
        reclaimable_bytes,
        &evidence,
        &match_items,
    );

    Some(MatchGroup {
        id: format!("group-{}", 900_000 + group_key),
        title,
        kind: "AI视觉匹配".to_string(),
        confidence,
        recommended_video_id: recommended_id,
        reclaimable_bytes,
        item_count: match_items.len(),
        evidence,
        report,
        items: match_items,
    })
}

fn compare_ai_keeper(
    left: &VideoRecord,
    right: &VideoRecord,
    size_priority_seconds: f64,
) -> std::cmp::Ordering {
    if durations_within(left, right, size_priority_seconds) {
        return left
            .size_bytes
            .cmp(&right.size_bytes)
            .then_with(|| right.path.cmp(&left.path));
    }
    left.quality_score
        .total_cmp(&right.quality_score)
        .then_with(|| right.path.cmp(&left.path))
}

fn durations_within(left: &VideoRecord, right: &VideoRecord, seconds: f64) -> bool {
    if seconds <= 0.0 || !seconds.is_finite() {
        return false;
    }
    let Some(left_duration) = left
        .duration_seconds
        .filter(|value| value.is_finite() && *value > 0.0)
    else {
        return false;
    };
    let Some(right_duration) = right
        .duration_seconds
        .filter(|value| value.is_finite() && *value > 0.0)
    else {
        return false;
    };
    (left_duration - right_duration).abs() <= seconds.clamp(0.0, 86_400.0)
}

fn edge_key(left: i64, right: i64) -> (i64, i64) {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

fn ai_group_report(
    group_id: &str,
    confidence: f64,
    reclaimable_bytes: u64,
    evidence: &[String],
    items: &[MatchItem],
) -> String {
    let mut lines = Vec::new();
    lines.push("Duplicate Video Search AI-only 报告".to_string());
    lines.push(format!("组别: {group_id}"));
    lines.push("类型: AI视觉匹配".to_string());
    lines.push(format!("置信度: {}%", (confidence * 100.0).round() as u32));
    lines.push(format!("估算可释放: {}", human_bytes(reclaimable_bytes)));
    lines.push(String::new());
    lines.push("文件:".to_string());
    for item in items {
        let video = &item.video;
        lines.push(format!("- {}: {}", item.role, video.path));
        lines.push(format!(
            "  规格: {}x{}, {:?}, {:.1}s, {}, {:.1} Mbps",
            video.width.unwrap_or(0),
            video.height.unwrap_or(0),
            video.codec.as_deref().unwrap_or("unknown"),
            video.duration_seconds.unwrap_or(0.0),
            human_bytes(video.size_bytes),
            video.bitrate.unwrap_or(0) as f64 / 1_000_000.0
        ));
    }
    lines.push(String::new());
    lines.push("证据:".to_string());
    for item in evidence {
        lines.push(format!("- {item}"));
    }
    lines.join("\n")
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", bytes, UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

fn file_sha256(path: &Path) -> anyhow::Result<String> {
    let mut file = File::open(path).with_context(|| format!("open model {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    fn test_settings() -> AppSettings {
        AppSettings {
            ui_language: "zh".to_string(),
            backup_dir: String::new(),
            naming_source_dirs: Vec::new(),
            keeper_size_priority_duration_seconds: 300.0,
            scan_worker_count: 1,
            sample_hash_count: 3,
            temporal_hash_threshold: 8,
            min_temporal_match_points: 3,
            allowed_unmatched_sample_frames: 14,
            allow_direct_delete: false,
            restrict_scan_to_test_path: true,
            compare_within_same_folder: false,
            ai_vision_enabled: true,
            ai_model_path: String::new(),
            ai_device: "auto".to_string(),
            ai_frame_count: 8,
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
            local_preprocess_enabled: true,
            local_preprocess_video_workers: 1,
            local_preprocess_overlap_start_percent: 95,
            local_preprocess_process_workers: 2,
            local_preprocess_frame_workers: 16,
            local_preprocess_temp_dir: String::new(),
            local_preprocess_secondary_temp_dir: String::new(),
            local_preprocess_secondary_threshold_mb: 16 * 1024,
            nas_ssh_preprocess_enabled: false,
            nas_ssh_host: String::new(),
            nas_ssh_user: String::new(),
            nas_ssh_password: String::new(),
            nas_ssh_host_key: String::new(),
            nas_ssh_port: 22,
            nas_remote_root: String::new(),
            nas_smb_root: String::new(),
            nas_ssh_ffmpeg_workers: 1,
        }
    }

    fn prepared_model() -> PreparedModel {
        PreparedModel {
            model_id: "model".to_string(),
            model_hash: "hash".to_string(),
            model_path: PathBuf::from("model.onnx"),
        }
    }

    fn full_video(path: &str, preview_path: &Path) -> VideoRecord {
        VideoRecord {
            id: None,
            path: path.to_string(),
            file_name: path.rsplit('\\').next().unwrap_or(path).to_string(),
            parent_path: r"\\EXAMPLE-NAS\Test".to_string(),
            size_bytes: 1024,
            modified_unix_ms: 1,
            extension: "mp4".to_string(),
            container_format: Some("mov,mp4".to_string()),
            duration_seconds: Some(60.0),
            width: Some(1920),
            height: Some(1080),
            bitrate: Some(1_000_000),
            codec: Some("h264".to_string()),
            frame_rate: Some(30.0),
            audio_codec: Some("aac".to_string()),
            partial_hash: Some("partial-hash".to_string()),
            sample_hashes: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            preview_images: vec![preview_path.display().to_string()],
            scan_status: "ok".to_string(),
            error: None,
            quality_score: 1.0,
            scanned_at_unix_ms: 1,
        }
    }

    fn lightweight_video(path: &str) -> VideoRecord {
        VideoRecord {
            id: None,
            path: path.to_string(),
            file_name: path.rsplit('\\').next().unwrap_or(path).to_string(),
            parent_path: r"\\EXAMPLE-NAS\Test".to_string(),
            size_bytes: 1024,
            modified_unix_ms: 1,
            extension: "mp4".to_string(),
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
            scanned_at_unix_ms: 1,
        }
    }

    fn open_test_database() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        database::migrate(&conn).unwrap();
        database::upsert_embedding_model(
            &conn,
            "model",
            "test",
            1,
            "test-runtime",
            "model.onnx",
            "hash",
            media::now_ms(),
        )
        .unwrap();
        conn
    }

    fn insert_embeddings(conn: &mut Connection, video_id: i64, count: usize) {
        let embeddings = (0..count)
            .map(|index| FrameEmbeddingRecord {
                video_id,
                model_id: "model".to_string(),
                frame_index: index,
                timestamp_seconds: index as f64,
                embedding: vec![1.0],
                norm: 1.0,
            })
            .collect::<Vec<_>>();
        database::replace_frame_embeddings(conn, video_id, "model", &embeddings).unwrap();
    }

    fn test_embedding(index: usize, values: &[f32]) -> FrameEmbeddingRecord {
        FrameEmbeddingRecord {
            video_id: 1,
            model_id: "model".to_string(),
            frame_index: index,
            timestamp_seconds: index as f64,
            embedding: values.to_vec(),
            norm: 1.0,
        }
    }

    #[test]
    fn folder_scope_filters_new_and_cached_ai_pairs() {
        let mut conn = open_test_database();
        let mut settings = test_settings();
        // Only existence and the registered model ID are needed for CPU scoring.
        settings.ai_model_path = std::env::current_exe().unwrap().display().to_string();
        settings.ai_device = "cpu".into();
        database::upsert_embedding_model(&conn, "model", "test", 1, "test-runtime",
            &settings.ai_model_path, "hash", 1).unwrap();
        let mut videos = Vec::new();
        for path in [r"C:\Videos\a.mp4", r"C:\Videos\b.mp4", r"C:\Videos\Child\c.mp4"] {
            let mut video = full_video(path, Path::new("unused.jpg"));
            let id = database::upsert_video(&conn, &video).unwrap();
            video.id = Some(id);
            insert_embeddings(&mut conn, id, 8);
            videos.push(video);
        }
        settings.compare_within_same_folder = true;
        let restricted = build_ai_match_groups(&mut conn, &videos, &settings, 0.5).unwrap();
        assert_eq!(restricted.len(), 1);
        assert_eq!(restricted[0].item_count, 2);
        let count = |conn: &Connection| conn.query_row("SELECT COUNT(*) FROM ai_pair_scores", [],
            |row| row.get::<_, i64>(0)).unwrap();
        assert_eq!(count(&conn), 1);
        settings.compare_within_same_folder = false;
        let global = build_ai_match_groups(&mut conn, &videos, &settings, 0.5).unwrap();
        assert_eq!(global[0].item_count, 3);
        assert_eq!(count(&conn), 3);
        settings.compare_within_same_folder = true;
        let cached = build_ai_match_groups(&mut conn, &videos, &settings, 0.5).unwrap();
        assert_eq!(cached.len(), 1);
        assert_eq!(cached[0].item_count, 2);
        assert_eq!(count(&conn), 3);
    }

    #[test]
    fn old_settings_default_to_global_comparison() {
        let mut value = serde_json::to_value(test_settings()).unwrap();
        value.as_object_mut().unwrap().remove("compareWithinSameFolder");
        let settings: AppSettings = serde_json::from_value(value).unwrap();
        assert!(!settings.compare_within_same_folder);
    }

    #[test]
    fn cpu_frame_similarity_matrix_matches_cosine_grid() {
        let short = vec![
            test_embedding(0, &[1.0, 0.0]),
            test_embedding(1, &[0.0, 1.0]),
        ];
        let long = vec![
            test_embedding(0, &[1.0, 0.0]),
            test_embedding(1, &[0.0, 1.0]),
            test_embedding(2, &[0.70710677, 0.70710677]),
        ];
        let mut engine = FrameSimilarityEngine::disabled();
        let matrix = engine.similarities(&short, &long);

        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.columns, 3);
        assert!((matrix.get(0, 0) - 1.0).abs() < 0.0001);
        assert!((matrix.get(0, 1) - 0.0).abs() < 0.0001);
        assert!((matrix.get(0, 2) - 0.70710677).abs() < 0.0001);
        assert!((matrix.get(1, 0) - 0.0).abs() < 0.0001);
        assert!((matrix.get(1, 1) - 1.0).abs() < 0.0001);
        assert!((matrix.get(1, 2) - 0.70710677).abs() < 0.0001);
    }

    #[test]
    fn frame_similarity_onnx_session_runs_matmul_on_cpu() {
        let mut settings = test_settings();
        settings.ai_device = "cpu".to_string();
        let session = create_frame_similarity_session(&settings).unwrap();
        let short = vec![
            test_embedding(0, &[1.0, 0.0]),
            test_embedding(1, &[0.0, 1.0]),
        ];
        let long = vec![
            test_embedding(0, &[1.0, 0.0]),
            test_embedding(1, &[0.70710677, 0.70710677]),
        ];
        let mut engine = FrameSimilarityEngine {
            session: Some(session),
        };
        let matrix = engine.similarities(&short, &long);

        assert_eq!(matrix.rows, 2);
        assert_eq!(matrix.columns, 2);
        assert!((matrix.get(0, 0) - 1.0).abs() < 0.0001);
        assert!((matrix.get(0, 1) - 0.70710677).abs() < 0.0001);
        assert!((matrix.get(1, 0) - 0.0).abs() < 0.0001);
        assert!((matrix.get(1, 1) - 0.70710677).abs() < 0.0001);
    }

    #[test]
    fn monotonic_match_summary_uses_precomputed_similarity_matrix() {
        let short = vec![
            test_embedding(0, &[1.0, 0.0, 0.0]),
            test_embedding(1, &[0.0, 1.0, 0.0]),
            test_embedding(2, &[0.0, 0.0, 1.0]),
        ];
        let long = vec![
            test_embedding(0, &[1.0, 0.0, 0.0]),
            test_embedding(1, &[0.0, 1.0, 0.0]),
            test_embedding(2, &[0.0, 0.0, 1.0]),
        ];
        let mut engine = FrameSimilarityEngine::disabled();
        let summary = monotonic_match_summary(&short, &long, 0.99, true, &mut engine);

        assert_eq!(summary.matched, 3);
        assert!((summary.average_similarity - 1.0).abs() < 0.0001);
        assert_eq!(summary.short_range, Some((0, 2)));
        assert_eq!(summary.long_range, Some((0, 2)));
    }

    #[test]
    fn reuses_existing_ai_index_for_full_scan_video() {
        let settings = test_settings();
        let prepared = prepared_model();
        let temp_dir = std::env::temp_dir().join(format!(
            "dvs-ai-reuse-full-{}-{}",
            std::process::id(),
            media::now_ms()
        ));
        std::fs::create_dir_all(&temp_dir).unwrap();
        let preview_path = temp_dir.join("preview.jpg");
        std::fs::write(&preview_path, b"preview").unwrap();

        let mut conn = open_test_database();
        let mut video = full_video(r"\\EXAMPLE-NAS\Test\full.mp4", &preview_path);
        let video_id = database::upsert_video(&conn, &video).unwrap();
        video.id = Some(video_id);
        insert_embeddings(&mut conn, video_id, 8);

        assert!(
            try_reuse_existing_ai_index(&mut conn, &video, &settings, &prepared, false).unwrap()
        );

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn does_not_reuse_existing_ai_index_for_lightweight_video() {
        let settings = test_settings();
        let prepared = prepared_model();
        let mut conn = open_test_database();
        let mut video = lightweight_video(r"\\EXAMPLE-NAS\Test\lightweight.mp4");
        let video_id = database::upsert_video(&conn, &video).unwrap();
        video.id = Some(video_id);
        insert_embeddings(&mut conn, video_id, 8);

        assert!(
            !try_reuse_existing_ai_index(&mut conn, &video, &settings, &prepared, false).unwrap()
        );
    }
}
