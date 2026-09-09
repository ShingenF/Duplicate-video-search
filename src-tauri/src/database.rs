use crate::models::{
    AiPairScoreRecord, DeleteIndexProgress, DeleteIndexSummary, FrameEmbeddingRecord,
    IndexRefreshSummary, ScanSession, StorageDatabaseStats, VideoRecord,
};
use crate::paths;
use anyhow::Context;
use rusqlite::{params, params_from_iter, types::Value, Connection, OptionalExtension, Row};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Duration;

pub fn open_database() -> anyhow::Result<Connection> {
    paths::ensure_data_dirs()?;
    let path = paths::database_path();
    let conn = Connection::open(&path).with_context(|| format!("open {}", path.display()))?;
    conn.busy_timeout(Duration::from_secs(30))?;
    migrate(&conn)?;
    Ok(conn)
}

pub(crate) fn migrate(conn: &Connection) -> anyhow::Result<()> {
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    conn.pragma_update(None, "temp_store", "MEMORY")?;
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE,
            file_name TEXT NOT NULL,
            parent_path TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            modified_unix_ms INTEGER NOT NULL,
            extension TEXT NOT NULL,
            container_format TEXT,
            duration_seconds REAL,
            width INTEGER,
            height INTEGER,
            bitrate INTEGER,
            codec TEXT,
            frame_rate REAL,
            audio_codec TEXT,
            partial_hash TEXT,
            sample_hashes_json TEXT NOT NULL,
            preview_images_json TEXT NOT NULL DEFAULT '[]',
            scan_status TEXT NOT NULL,
            error TEXT,
            quality_score REAL NOT NULL,
            scanned_at_unix_ms INTEGER NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_videos_duration ON videos(duration_seconds);
        CREATE INDEX IF NOT EXISTS idx_videos_partial_hash ON videos(partial_hash);
        CREATE INDEX IF NOT EXISTS idx_videos_quality ON videos(quality_score);

        CREATE TABLE IF NOT EXISTS scan_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            started_unix_ms INTEGER NOT NULL,
            completed_unix_ms INTEGER,
            total_files INTEGER NOT NULL DEFAULT 0,
            scanned INTEGER NOT NULL DEFAULT 0,
            failed INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS video_scan_sessions (
            session_id INTEGER NOT NULL,
            video_id INTEGER NOT NULL,
            PRIMARY KEY (session_id, video_id),
            FOREIGN KEY(session_id) REFERENCES scan_sessions(id) ON DELETE CASCADE,
            FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_video_scan_sessions_video ON video_scan_sessions(video_id);

        CREATE TABLE IF NOT EXISTS embedding_models (
            model_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            runtime TEXT NOT NULL,
            model_path TEXT NOT NULL,
            model_file_hash TEXT,
            created_unix_ms INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS frame_embeddings (
            video_id INTEGER NOT NULL,
            model_id TEXT NOT NULL,
            frame_index INTEGER NOT NULL,
            timestamp_seconds REAL NOT NULL,
            embedding BLOB NOT NULL,
            norm REAL NOT NULL,
            created_unix_ms INTEGER NOT NULL,
            PRIMARY KEY (video_id, model_id, frame_index),
            FOREIGN KEY(video_id) REFERENCES videos(id) ON DELETE CASCADE,
            FOREIGN KEY(model_id) REFERENCES embedding_models(model_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_frame_embeddings_model ON frame_embeddings(model_id);
        CREATE INDEX IF NOT EXISTS idx_frame_embeddings_video ON frame_embeddings(video_id);

        CREATE TABLE IF NOT EXISTS ai_match_edges (
            video_a_id INTEGER NOT NULL,
            video_b_id INTEGER NOT NULL,
            model_id TEXT NOT NULL,
            confidence REAL NOT NULL,
            matched_frame_count INTEGER NOT NULL,
            alignment_score REAL NOT NULL,
            clip_score REAL NOT NULL,
            verdict TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            created_unix_ms INTEGER NOT NULL,
            PRIMARY KEY (video_a_id, video_b_id, model_id),
            CHECK (video_a_id < video_b_id),
            FOREIGN KEY(video_a_id) REFERENCES videos(id) ON DELETE CASCADE,
            FOREIGN KEY(video_b_id) REFERENCES videos(id) ON DELETE CASCADE,
            FOREIGN KEY(model_id) REFERENCES embedding_models(model_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_ai_match_edges_confidence ON ai_match_edges(confidence);
        CREATE INDEX IF NOT EXISTS idx_ai_match_edges_verdict ON ai_match_edges(verdict);
        CREATE INDEX IF NOT EXISTS idx_ai_match_edges_video_b ON ai_match_edges(video_b_id);

        CREATE TABLE IF NOT EXISTS ai_pair_scores (
            video_a_id INTEGER NOT NULL,
            video_b_id INTEGER NOT NULL,
            model_id TEXT NOT NULL,
            cache_key TEXT NOT NULL,
            confidence REAL NOT NULL,
            matched_frame_count INTEGER NOT NULL,
            compared_frame_count INTEGER NOT NULL,
            average_similarity REAL NOT NULL,
            coverage REAL NOT NULL,
            created_unix_ms INTEGER NOT NULL,
            PRIMARY KEY (video_a_id, video_b_id, model_id, cache_key),
            CHECK (video_a_id < video_b_id),
            FOREIGN KEY(video_a_id) REFERENCES videos(id) ON DELETE CASCADE,
            FOREIGN KEY(video_b_id) REFERENCES videos(id) ON DELETE CASCADE,
            FOREIGN KEY(model_id) REFERENCES embedding_models(model_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_ai_pair_scores_model_key ON ai_pair_scores(model_id, cache_key);
        CREATE INDEX IF NOT EXISTS idx_ai_pair_scores_model_key_a ON ai_pair_scores(model_id, cache_key, video_a_id, video_b_id);
        CREATE INDEX IF NOT EXISTS idx_ai_pair_scores_confidence ON ai_pair_scores(model_id, cache_key, confidence);
        "#,
    )?;
    ensure_column(
        conn,
        "videos",
        "preview_images_json",
        "TEXT NOT NULL DEFAULT '[]'",
    )?;
    Ok(())
}

fn ensure_column(
    conn: &Connection,
    table: &str,
    column: &str,
    definition: &str,
) -> anyhow::Result<()> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table})"))?;
    let columns = stmt.query_map([], |row| row.get::<_, String>(1))?;
    for existing in columns {
        if existing? == column {
            return Ok(());
        }
    }
    conn.execute(
        &format!("ALTER TABLE {table} ADD COLUMN {column} {definition}"),
        [],
    )?;
    Ok(())
}

pub fn upsert_video(conn: &Connection, video: &VideoRecord) -> anyhow::Result<i64> {
    let existing = conn
        .query_row(
            "SELECT id, size_bytes, modified_unix_ms, partial_hash FROM videos WHERE path = ?1",
            params![video.path],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)? as u64,
                    row.get::<_, i64>(2)?,
                    row.get::<_, Option<String>>(3)?,
                ))
            },
        )
        .optional()?;
    let should_clear_ai =
        existing
            .as_ref()
            .is_some_and(|(_, size_bytes, modified_unix_ms, partial_hash)| {
                if *size_bytes != video.size_bytes {
                    return true;
                }
                match (
                    partial_hash.as_deref().filter(|value| !value.is_empty()),
                    video
                        .partial_hash
                        .as_deref()
                        .filter(|value| !value.is_empty()),
                ) {
                    (Some(existing_hash), Some(new_hash)) => existing_hash != new_hash,
                    _ => *modified_unix_ms != video.modified_unix_ms,
                }
            });

    let sample_hashes_json = serde_json::to_string(&video.sample_hashes)?;
    let preview_images_json = serde_json::to_string(&video.preview_images)?;
    conn.execute(
        r#"
        INSERT INTO videos (
            path, file_name, parent_path, size_bytes, modified_unix_ms, extension,
            container_format, duration_seconds, width, height, bitrate, codec,
            frame_rate, audio_codec, partial_hash, sample_hashes_json, preview_images_json,
            scan_status, error, quality_score, scanned_at_unix_ms
        )
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21)
        ON CONFLICT(path) DO UPDATE SET
            file_name = excluded.file_name,
            parent_path = excluded.parent_path,
            size_bytes = excluded.size_bytes,
            modified_unix_ms = excluded.modified_unix_ms,
            extension = excluded.extension,
            container_format = excluded.container_format,
            duration_seconds = excluded.duration_seconds,
            width = excluded.width,
            height = excluded.height,
            bitrate = excluded.bitrate,
            codec = excluded.codec,
            frame_rate = excluded.frame_rate,
            audio_codec = excluded.audio_codec,
            partial_hash = excluded.partial_hash,
            sample_hashes_json = excluded.sample_hashes_json,
            preview_images_json = excluded.preview_images_json,
            scan_status = excluded.scan_status,
            error = excluded.error,
            quality_score = excluded.quality_score,
            scanned_at_unix_ms = excluded.scanned_at_unix_ms
        "#,
        params![
            video.path,
            video.file_name,
            video.parent_path,
            video.size_bytes as i64,
            video.modified_unix_ms,
            video.extension,
            video.container_format,
            video.duration_seconds,
            video.width.map(|v| v as i64),
            video.height.map(|v| v as i64),
            video.bitrate.map(|v| v as i64),
            video.codec,
            video.frame_rate,
            video.audio_codec,
            video.partial_hash,
            sample_hashes_json,
            preview_images_json,
            video.scan_status,
            video.error,
            video.quality_score,
            video.scanned_at_unix_ms
        ],
    )?;

    let id = conn
        .query_row(
            "SELECT id FROM videos WHERE path = ?1",
            params![video.path],
            |row| row.get(0),
        )
        .optional()?
        .context("video was not found after upsert")?;
    if should_clear_ai {
        clear_video_ai_data(conn, id)?;
    }
    Ok(id)
}

pub fn upsert_video_preserving_content_identity(
    conn: &Connection,
    video: &VideoRecord,
) -> anyhow::Result<i64> {
    if let Some(existing) = find_video_by_path(conn, &video.path)? {
        if !same_content_identity(&existing, video, 1.0)
            && existing
                .partial_hash
                .as_deref()
                .filter(|value| !value.is_empty())
                .is_none()
        {
            if let Some(missing) = find_missing_video_by_content_identity(conn, video)? {
                let existing_id = existing
                    .id
                    .ok_or_else(|| anyhow::anyhow!("existing video is missing an id"))?;
                let missing_id = missing
                    .id
                    .ok_or_else(|| anyhow::anyhow!("matched video is missing an id"))?;
                if existing_id != missing_id {
                    conn.execute(
                        r#"
                        INSERT OR IGNORE INTO video_scan_sessions (session_id, video_id)
                        SELECT session_id, ?1
                        FROM video_scan_sessions
                        WHERE video_id = ?2
                        "#,
                        params![missing_id, existing_id],
                    )?;
                    delete_videos_and_move_paths(
                        conn,
                        &[existing_id],
                        &[(missing_id, PathBuf::from(&video.path))],
                    )?;
                }
            }
        }
    } else if let Some(missing) = find_missing_video_by_content_identity(conn, video)? {
        let video_id = missing
            .id
            .ok_or_else(|| anyhow::anyhow!("matched video is missing an id"))?;
        move_video_path(conn, video_id, Path::new(&video.path))?;
    }

    upsert_video(conn, video)
}

pub fn migrate_video_by_content_identity(
    conn: &Connection,
    candidate: &VideoRecord,
) -> anyhow::Result<Option<i64>> {
    if find_video_by_path(conn, &candidate.path)?.is_some() {
        return Ok(None);
    }
    let Some(missing) = find_missing_video_by_content_identity(conn, candidate)? else {
        return Ok(None);
    };
    let video_id = missing
        .id
        .ok_or_else(|| anyhow::anyhow!("matched video is missing an id"))?;
    move_video_path(conn, video_id, Path::new(&candidate.path))?;
    Ok(Some(video_id))
}

pub fn create_or_reset_scan_session(
    conn: &Connection,
    source: &str,
    started_unix_ms: i64,
) -> anyhow::Result<i64> {
    let mut stmt = conn.prepare(
        r#"
        SELECT id
        FROM scan_sessions
        WHERE lower(source) = lower(?1)
        ORDER BY id ASC
        "#,
    )?;
    let existing = stmt
        .query_map(params![source], |row| row.get::<_, i64>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    drop(stmt);

    if let Some(session_id) = existing.first().copied() {
        for duplicate_id in existing.iter().copied().skip(1) {
            conn.execute(
                r#"
                INSERT OR IGNORE INTO video_scan_sessions (session_id, video_id)
                SELECT ?1, video_id FROM video_scan_sessions WHERE session_id = ?2
                "#,
                params![session_id, duplicate_id],
            )?;
            conn.execute(
                "DELETE FROM video_scan_sessions WHERE session_id = ?1",
                params![duplicate_id],
            )?;
            conn.execute(
                "DELETE FROM scan_sessions WHERE id = ?1",
                params![duplicate_id],
            )?;
        }

        conn.execute(
            "DELETE FROM video_scan_sessions WHERE session_id = ?1",
            params![session_id],
        )?;
        conn.execute(
            r#"
            UPDATE scan_sessions
            SET source = ?2,
                started_unix_ms = ?3,
                completed_unix_ms = NULL,
                total_files = 0,
                scanned = 0,
                failed = 0
            WHERE id = ?1
            "#,
            params![session_id, source, started_unix_ms],
        )?;
        return Ok(session_id);
    }

    conn.execute(
        r#"
        INSERT INTO scan_sessions (source, started_unix_ms, total_files, scanned, failed)
        VALUES (?1, ?2, 0, 0, 0)
        "#,
        params![source, started_unix_ms],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn complete_scan_session(
    conn: &Connection,
    session_id: i64,
    completed_unix_ms: i64,
    total_files: usize,
    scanned: usize,
    failed: usize,
) -> anyhow::Result<()> {
    conn.execute(
        r#"
        UPDATE scan_sessions
        SET completed_unix_ms = ?2, total_files = ?3, scanned = ?4, failed = ?5
        WHERE id = ?1
        "#,
        params![
            session_id,
            completed_unix_ms,
            total_files as i64,
            scanned as i64,
            failed as i64
        ],
    )?;
    Ok(())
}

pub fn link_video_to_scan_session(
    conn: &Connection,
    session_id: i64,
    video_id: i64,
) -> anyhow::Result<()> {
    conn.execute(
        r#"
        INSERT OR IGNORE INTO video_scan_sessions (session_id, video_id)
        VALUES (?1, ?2)
        "#,
        params![session_id, video_id],
    )?;
    Ok(())
}

const VIDEO_SELECT_COLUMNS: &str = r#"
    id, path, file_name, parent_path, size_bytes, modified_unix_ms, extension,
    container_format, duration_seconds, width, height, bitrate, codec,
    frame_rate, audio_codec, partial_hash, sample_hashes_json, preview_images_json, scan_status,
    error, quality_score, scanned_at_unix_ms
"#;

fn row_to_video(row: &Row<'_>) -> rusqlite::Result<VideoRecord> {
    let sample_hashes_json: String = row.get(16)?;
    let sample_hashes =
        serde_json::from_str::<Vec<String>>(&sample_hashes_json).unwrap_or_else(|_| Vec::new());
    let preview_images_json: String = row.get(17)?;
    let preview_images =
        serde_json::from_str::<Vec<String>>(&preview_images_json).unwrap_or_else(|_| Vec::new());

    Ok(VideoRecord {
        id: row.get(0)?,
        path: row.get(1)?,
        file_name: row.get(2)?,
        parent_path: row.get(3)?,
        size_bytes: row.get::<_, i64>(4)? as u64,
        modified_unix_ms: row.get(5)?,
        extension: row.get(6)?,
        container_format: row.get(7)?,
        duration_seconds: row.get(8)?,
        width: row.get::<_, Option<i64>>(9)?.map(|v| v as u32),
        height: row.get::<_, Option<i64>>(10)?.map(|v| v as u32),
        bitrate: row.get::<_, Option<i64>>(11)?.map(|v| v as u64),
        codec: row.get(12)?,
        frame_rate: row.get(13)?,
        audio_codec: row.get(14)?,
        partial_hash: row.get(15)?,
        sample_hashes,
        preview_images,
        scan_status: row.get(18)?,
        error: row.get(19)?,
        quality_score: row.get(20)?,
        scanned_at_unix_ms: row.get(21)?,
    })
}

pub fn list_videos(conn: &Connection) -> anyhow::Result<Vec<VideoRecord>> {
    let mut stmt = conn.prepare(&format!(
        r#"
        SELECT {VIDEO_SELECT_COLUMNS}
        FROM videos
        ORDER BY quality_score DESC, path ASC
        "#
    ))?;

    let rows = stmt.query_map([], row_to_video)?;

    let mut videos = Vec::new();
    for row in rows {
        videos.push(row?);
    }
    Ok(videos)
}

pub fn find_video_by_path(conn: &Connection, path: &str) -> anyhow::Result<Option<VideoRecord>> {
    conn.query_row(
        &format!(
            r#"
            SELECT {VIDEO_SELECT_COLUMNS}
            FROM videos
            WHERE lower(path) = lower(?1)
            ORDER BY id DESC
            LIMIT 1
            "#
        ),
        params![path],
        row_to_video,
    )
    .optional()
    .map_err(Into::into)
}

pub fn find_missing_video_by_content_identity(
    conn: &Connection,
    candidate: &VideoRecord,
) -> anyhow::Result<Option<VideoRecord>> {
    let Some(partial_hash) = candidate
        .partial_hash
        .as_deref()
        .filter(|value| !value.is_empty())
    else {
        return Ok(None);
    };
    let Some(duration_seconds) = candidate.duration_seconds else {
        return Ok(None);
    };
    if candidate.size_bytes == 0 || candidate.extension.trim().is_empty() {
        return Ok(None);
    }

    let mut stmt = conn.prepare(&format!(
        r#"
        SELECT {VIDEO_SELECT_COLUMNS}
        FROM videos
        WHERE scan_status = 'ok'
          AND lower(path) <> lower(?1)
          AND size_bytes = ?2
          AND partial_hash = ?3
          AND lower(extension) = lower(?4)
          AND duration_seconds IS NOT NULL
          AND ABS(duration_seconds - ?5) <= 1.0
        ORDER BY scanned_at_unix_ms DESC, id DESC
        "#
    ))?;
    let rows = stmt.query_map(
        params![
            candidate.path,
            candidate.size_bytes as i64,
            partial_hash,
            candidate.extension,
            duration_seconds,
        ],
        row_to_video,
    )?;

    let mut missing = Vec::new();
    for row in rows {
        let video = row?;
        if same_storage_root(&video.path, &candidate.path)
            && !video_path_exists(&video.path)
            && same_content_identity(&video, candidate, 1.0)
        {
            missing.push(video);
            if missing.len() > 1 {
                return Ok(None);
            }
        }
    }

    Ok(missing.pop())
}

fn same_content_identity(left: &VideoRecord, right: &VideoRecord, duration_tolerance: f64) -> bool {
    if left.size_bytes != right.size_bytes
        || !left.extension.eq_ignore_ascii_case(&right.extension)
        || left
            .partial_hash
            .as_deref()
            .filter(|value| !value.is_empty())
            != right
                .partial_hash
                .as_deref()
                .filter(|value| !value.is_empty())
    {
        return false;
    }
    match (left.duration_seconds, right.duration_seconds) {
        (Some(left_duration), Some(right_duration)) => {
            (left_duration - right_duration).abs() <= duration_tolerance
        }
        _ => false,
    }
}

fn video_path_exists(path: &str) -> bool {
    let path = Path::new(path);
    crate::media::filesystem_input_path(path).exists()
}

fn same_storage_root(left: &str, right: &str) -> bool {
    storage_root_key(left).is_some_and(|left_root| {
        storage_root_key(right).is_some_and(|right_root| left_root == right_root)
    })
}

fn storage_root_key(path: &str) -> Option<String> {
    let text = path
        .replace('/', "\\")
        .trim_end_matches('\\')
        .to_lowercase();
    if let Some(rest) = text.strip_prefix(r"\\?\unc\") {
        let mut parts = rest.split('\\').filter(|part| !part.is_empty());
        let server = parts.next()?;
        let share = parts.next()?;
        return Some(format!(r"\\{server}\{share}"));
    }
    if let Some(rest) = text.strip_prefix(r"\\") {
        let mut parts = rest.split('\\').filter(|part| !part.is_empty());
        let server = parts.next()?;
        let share = parts.next()?;
        return Some(format!(r"\\{server}\{share}"));
    }
    if text.len() >= 3 && text.as_bytes()[1] == b':' && text.as_bytes()[2] == b'\\' {
        return Some(text[..3].to_string());
    }
    None
}

pub fn list_videos_for_sessions(
    conn: &Connection,
    session_ids: &[i64],
) -> anyhow::Result<Vec<VideoRecord>> {
    if session_ids.is_empty() {
        return Ok(Vec::new());
    }

    let expanded_session_ids = expand_session_ids_by_source(conn, session_ids)?;
    let mut ids = HashSet::new();
    let mut stmt =
        conn.prepare("SELECT DISTINCT video_id FROM video_scan_sessions WHERE session_id = ?1")?;
    for session_id in expanded_session_ids {
        let rows = stmt.query_map(params![session_id], |row| row.get::<_, i64>(0))?;
        for row in rows {
            ids.insert(row?);
        }
    }

    let videos = list_videos(conn)?;
    Ok(videos
        .into_iter()
        .filter(|video| video.id.is_some_and(|id| ids.contains(&id)))
        .collect())
}

pub fn list_scan_sessions(conn: &Connection) -> anyhow::Result<Vec<ScanSession>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT id, source, started_unix_ms, completed_unix_ms, total_files, scanned, failed
        FROM scan_sessions
        ORDER BY started_unix_ms DESC, id DESC
        "#,
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(ScanSession {
            id: row.get(0)?,
            source: row.get(1)?,
            started_unix_ms: row.get(2)?,
            completed_unix_ms: row.get(3)?,
            total_files: row.get::<_, i64>(4)? as usize,
            scanned: row.get::<_, i64>(5)? as usize,
            failed: row.get::<_, i64>(6)? as usize,
        })
    })?;

    let mut sessions = Vec::new();
    let mut seen_sources = HashSet::new();
    for row in rows {
        let session = row?;
        let key = session.source.replace('/', "\\").to_lowercase();
        if seen_sources.insert(key) {
            sessions.push(session);
        }
    }
    Ok(sessions)
}

pub fn delete_scan_session(conn: &Connection, session_id: i64) -> anyhow::Result<()> {
    let summary = delete_scan_sessions_with_progress(conn, &[session_id], |_| {})?;
    if summary.deleted_sessions == 0 {
        anyhow::bail!("scan session not found: {session_id}");
    }
    Ok(())
}

pub fn delete_scan_sessions(
    conn: &Connection,
    session_ids: &[i64],
) -> anyhow::Result<DeleteIndexSummary> {
    delete_scan_sessions_with_progress(conn, session_ids, |_| {})
}

pub fn delete_scan_sessions_with_progress<F>(
    conn: &Connection,
    session_ids: &[i64],
    mut on_progress: F,
) -> anyhow::Result<DeleteIndexSummary>
where
    F: FnMut(DeleteIndexProgress),
{
    let source_session_ids = expand_session_ids_by_source(conn, session_ids)?;
    if source_session_ids.is_empty() {
        return Ok(DeleteIndexSummary {
            requested_sessions: session_ids.len(),
            deleted_sessions: 0,
            affected_videos: 0,
            deleted_videos: 0,
            deleted_pair_scores: 0,
            deleted_match_edges: 0,
            deleted_frame_embeddings: 0,
        });
    }

    let phase_total = 8usize;
    let mut progress = |phase: &str,
                        phase_processed: usize,
                        deleted_sessions: usize,
                        affected_videos: usize,
                        deleted_videos: usize| {
        on_progress(DeleteIndexProgress {
            phase: phase.to_string(),
            phase_processed,
            phase_total,
            requested_sessions: source_session_ids.len(),
            deleted_sessions,
            affected_videos,
            deleted_videos,
        });
    };

    progress("preparing", 0, 0, 0, 0);
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ai_pair_scores_video_b ON ai_pair_scores(video_b_id)",
        [],
    )?;
    progress("collecting-sessions", 1, 0, 0, 0);

    let tx = conn.unchecked_transaction()?;
    tx.execute_batch(
        r#"
        DROP TABLE IF EXISTS temp.delete_session_ids;
        DROP TABLE IF EXISTS temp.delete_video_ids;
        DROP TABLE IF EXISTS temp.orphan_video_ids;
        CREATE TEMP TABLE delete_session_ids (
            session_id INTEGER PRIMARY KEY
        ) WITHOUT ROWID;
        CREATE TEMP TABLE delete_video_ids (
            video_id INTEGER PRIMARY KEY
        ) WITHOUT ROWID;
        CREATE TEMP TABLE orphan_video_ids (
            video_id INTEGER PRIMARY KEY
        ) WITHOUT ROWID;
        "#,
    )?;

    {
        let mut stmt =
            tx.prepare("INSERT OR IGNORE INTO temp.delete_session_ids (session_id) VALUES (?1)")?;
        for source_session_id in &source_session_ids {
            stmt.execute(params![source_session_id])?;
        }
    }

    tx.execute(
        r#"
        INSERT OR IGNORE INTO temp.delete_video_ids (video_id)
        SELECT video_id
        FROM video_scan_sessions
        WHERE session_id IN (SELECT session_id FROM temp.delete_session_ids)
        "#,
        [],
    )?;
    let affected_videos = count_table_rows(&tx, "temp.delete_video_ids")?;
    progress("unlinking-sessions", 2, 0, affected_videos, 0);

    tx.execute(
        "DELETE FROM video_scan_sessions WHERE session_id IN (SELECT session_id FROM temp.delete_session_ids)",
        [],
    )?;
    let deleted_sessions = tx.execute(
        "DELETE FROM scan_sessions WHERE id IN (SELECT session_id FROM temp.delete_session_ids)",
        [],
    )?;
    progress(
        "collecting-orphan-videos",
        3,
        deleted_sessions,
        affected_videos,
        0,
    );

    tx.execute(
        r#"
        INSERT OR IGNORE INTO temp.orphan_video_ids (video_id)
        SELECT deleted.video_id
        FROM temp.delete_video_ids AS deleted
        LEFT JOIN video_scan_sessions AS remaining ON remaining.video_id = deleted.video_id
        WHERE remaining.video_id IS NULL
        "#,
        [],
    )?;
    let deleted_videos = count_table_rows(&tx, "temp.orphan_video_ids")?;

    let mut deleted_pair_scores = 0usize;
    let mut deleted_match_edges = 0usize;
    let mut deleted_frame_embeddings = 0usize;
    if deleted_videos > 0 {
        progress(
            "deleting-pair-scores-left",
            4,
            deleted_sessions,
            affected_videos,
            deleted_videos,
        );
        deleted_pair_scores = deleted_pair_scores.saturating_add(tx.execute(
            "DELETE FROM ai_pair_scores WHERE video_a_id IN (SELECT video_id FROM temp.orphan_video_ids)",
            [],
        )?);

        progress(
            "deleting-pair-scores-right",
            5,
            deleted_sessions,
            affected_videos,
            deleted_videos,
        );
        deleted_pair_scores = deleted_pair_scores.saturating_add(tx.execute(
            "DELETE FROM ai_pair_scores WHERE video_b_id IN (SELECT video_id FROM temp.orphan_video_ids)",
            [],
        )?);

        progress(
            "deleting-ai-metadata",
            6,
            deleted_sessions,
            affected_videos,
            deleted_videos,
        );
        deleted_match_edges = deleted_match_edges.saturating_add(tx.execute(
            "DELETE FROM ai_match_edges WHERE video_a_id IN (SELECT video_id FROM temp.orphan_video_ids)",
            [],
        )?);
        deleted_match_edges = deleted_match_edges.saturating_add(tx.execute(
            "DELETE FROM ai_match_edges WHERE video_b_id IN (SELECT video_id FROM temp.orphan_video_ids)",
            [],
        )?);
        deleted_frame_embeddings = tx.execute(
            "DELETE FROM frame_embeddings WHERE video_id IN (SELECT video_id FROM temp.orphan_video_ids)",
            [],
        )?;

        progress(
            "deleting-videos",
            7,
            deleted_sessions,
            affected_videos,
            deleted_videos,
        );
        tx.execute(
            "DELETE FROM videos WHERE id IN (SELECT video_id FROM temp.orphan_video_ids)",
            [],
        )?;
    }

    tx.execute_batch(
        r#"
        DROP TABLE IF EXISTS temp.orphan_video_ids;
        DROP TABLE IF EXISTS temp.delete_video_ids;
        DROP TABLE IF EXISTS temp.delete_session_ids;
        "#,
    )?;
    tx.commit()?;

    progress(
        "completed",
        phase_total,
        deleted_sessions,
        affected_videos,
        deleted_videos,
    );

    Ok(DeleteIndexSummary {
        requested_sessions: session_ids.len(),
        deleted_sessions,
        affected_videos,
        deleted_videos,
        deleted_pair_scores,
        deleted_match_edges,
        deleted_frame_embeddings,
    })
}

fn count_table_rows(conn: &Connection, table: &str) -> anyhow::Result<usize> {
    let count = conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
        row.get::<_, i64>(0)
    })?;
    Ok(count.max(0) as usize)
}

pub fn refresh_index_sources(conn: &Connection) -> anyhow::Result<IndexRefreshSummary> {
    let sessions = list_scan_sessions(conn)?;
    let mut removed_missing_sources = 0usize;
    let mut skipped_unavailable_roots = 0usize;

    for session in sessions {
        let source_path = PathBuf::from(&session.source);
        if source_path.exists() {
            continue;
        }
        if !source_root_available(&source_path) {
            skipped_unavailable_roots = skipped_unavailable_roots.saturating_add(1);
            continue;
        }
        delete_scan_session(conn, session.id)?;
        removed_missing_sources = removed_missing_sources.saturating_add(1);
    }

    Ok(IndexRefreshSummary {
        removed_missing_sources,
        skipped_unavailable_roots,
    })
}

fn source_root_available(path: &Path) -> bool {
    let text = path.to_string_lossy().replace('/', "\\");
    if let Some(rest) = text.strip_prefix(r"\\") {
        let mut parts = rest.split('\\').filter(|part| !part.is_empty());
        let Some(server) = parts.next() else {
            return false;
        };
        let Some(share) = parts.next() else {
            return false;
        };
        return PathBuf::from(format!(r"\\{server}\{share}")).exists();
    }
    if text.len() >= 3 && text.as_bytes()[1] == b':' && text.as_bytes()[2] == b'\\' {
        return PathBuf::from(&text[..3]).exists();
    }
    path.parent().is_some_and(Path::exists)
}

pub fn clear_all_indexes(conn: &Connection) -> anyhow::Result<()> {
    let tx = conn.unchecked_transaction()?;
    tx.execute_batch(
        r#"
        DELETE FROM ai_pair_scores;
        DELETE FROM ai_match_edges;
        DELETE FROM frame_embeddings;
        DELETE FROM embedding_models;
        DELETE FROM video_scan_sessions;
        DELETE FROM scan_sessions;
        DELETE FROM videos;
        "#,
    )?;
    tx.commit()?;
    Ok(())
}

pub fn session_ids_for_video(conn: &Connection, video_id: i64) -> anyhow::Result<Vec<i64>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT session_id
        FROM video_scan_sessions
        WHERE video_id = ?1
        ORDER BY session_id ASC
        "#,
    )?;
    let rows = stmt.query_map(params![video_id], |row| row.get::<_, i64>(0))?;
    let mut session_ids = Vec::new();
    for row in rows {
        session_ids.push(row?);
    }
    Ok(session_ids)
}

pub fn session_exists(conn: &Connection, session_id: i64) -> anyhow::Result<bool> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM scan_sessions WHERE id = ?1",
        params![session_id],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

pub fn remove_video(conn: &Connection, video_id: i64) -> anyhow::Result<()> {
    conn.execute(
        "DELETE FROM video_scan_sessions WHERE video_id = ?1",
        params![video_id],
    )?;
    conn.execute("DELETE FROM videos WHERE id = ?1", params![video_id])?;
    Ok(())
}

pub fn move_video_path(conn: &Connection, video_id: i64, new_path: &Path) -> anyhow::Result<()> {
    let path_text = new_path.display().to_string();
    delete_orphan_path_conflicts(conn, video_id, &path_text)?;
    let conflicting_id = conn
        .query_row(
            "SELECT id FROM videos WHERE lower(path) = lower(?1) AND id <> ?2 LIMIT 1",
            params![path_text, video_id],
            |row| row.get::<_, i64>(0),
        )
        .optional()?;
    if let Some(conflicting_id) = conflicting_id {
        anyhow::bail!(
            "cannot move video index to {}; path is already owned by video id {}",
            new_path.display(),
            conflicting_id
        );
    }

    let updated = conn.execute(
        r#"
        UPDATE videos
        SET path = ?2,
            file_name = ?3,
            parent_path = ?4,
            scanned_at_unix_ms = ?5
        WHERE id = ?1
        "#,
        params![
            video_id,
            path_text,
            new_path
                .file_name()
                .map(|value| value.to_string_lossy().to_string())
                .unwrap_or_default(),
            new_path
                .parent()
                .map(|value| value.display().to_string())
                .unwrap_or_default(),
            crate::media::now_ms(),
        ],
    )?;
    if updated == 0 {
        anyhow::bail!("video id {video_id} was not found");
    }
    Ok(())
}

pub fn delete_videos_and_move_paths(
    conn: &Connection,
    video_ids: &[i64],
    migrations: &[(i64, PathBuf)],
) -> anyhow::Result<DeletedVideoRows> {
    if video_ids.is_empty() && migrations.is_empty() {
        return Ok(DeletedVideoRows::default());
    }

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ai_pair_scores_video_b ON ai_pair_scores(video_b_id)",
        [],
    )?;

    let mut ids = video_ids.to_vec();
    ids.sort_unstable();
    ids.dedup();
    let delete_set = ids.iter().copied().collect::<HashSet<_>>();
    let mut migration_paths = HashSet::new();
    let mut predeleted_orphans = DeletedVideoRows::default();
    for (video_id, path) in migrations {
        if delete_set.contains(video_id) {
            anyhow::bail!("cannot migrate removed video id {video_id}");
        }
        let path_text = path.display().to_string();
        add_deleted_rows(
            &mut predeleted_orphans,
            delete_orphan_path_conflicts_except(conn, *video_id, &path_text, &delete_set)?,
        );
        let key = path_text.to_lowercase();
        if !migration_paths.insert(key) {
            anyhow::bail!("multiple kept videos would migrate to {}", path.display());
        }
        let conflicting_id = conn
            .query_row(
                "SELECT id FROM videos WHERE lower(path) = lower(?1) AND id <> ?2 LIMIT 1",
                params![path_text, video_id],
                |row| row.get::<_, i64>(0),
            )
            .optional()?;
        if let Some(conflicting_id) = conflicting_id {
            if !delete_set.contains(&conflicting_id) {
                anyhow::bail!(
                    "cannot move video index to {}; path is already owned by video id {}",
                    path.display(),
                    conflicting_id
                );
            }
        }
    }

    let tx = conn.unchecked_transaction()?;
    tx.execute_batch(
        r#"
        DROP TABLE IF EXISTS temp.delete_video_ids;
        CREATE TEMP TABLE delete_video_ids (
            video_id INTEGER PRIMARY KEY
        ) WITHOUT ROWID;
        "#,
    )?;
    {
        let mut stmt =
            tx.prepare("INSERT OR IGNORE INTO temp.delete_video_ids (video_id) VALUES (?1)")?;
        for id in &ids {
            stmt.execute(params![id])?;
        }
    }

    let mut summary = predeleted_orphans;
    summary.pair_scores = summary.pair_scores.saturating_add(tx.execute(
        "DELETE FROM ai_pair_scores WHERE video_a_id IN (SELECT video_id FROM temp.delete_video_ids)",
        [],
    )?);
    summary.pair_scores = summary.pair_scores.saturating_add(tx.execute(
        "DELETE FROM ai_pair_scores WHERE video_b_id IN (SELECT video_id FROM temp.delete_video_ids)",
        [],
    )?);
    summary.match_edges = summary.match_edges.saturating_add(tx.execute(
        "DELETE FROM ai_match_edges WHERE video_a_id IN (SELECT video_id FROM temp.delete_video_ids)",
        [],
    )?);
    summary.match_edges = summary.match_edges.saturating_add(tx.execute(
        "DELETE FROM ai_match_edges WHERE video_b_id IN (SELECT video_id FROM temp.delete_video_ids)",
        [],
    )?);
    summary.frame_embeddings = summary.frame_embeddings.saturating_add(tx.execute(
        "DELETE FROM frame_embeddings WHERE video_id IN (SELECT video_id FROM temp.delete_video_ids)",
        [],
    )?);
    tx.execute(
        "DELETE FROM video_scan_sessions WHERE video_id IN (SELECT video_id FROM temp.delete_video_ids)",
        [],
    )?;
    summary.videos = summary.videos.saturating_add(tx.execute(
        "DELETE FROM videos WHERE id IN (SELECT video_id FROM temp.delete_video_ids)",
        [],
    )?);

    {
        let mut stmt = tx.prepare(
            r#"
            UPDATE videos
            SET path = ?2,
                file_name = ?3,
                parent_path = ?4,
                scanned_at_unix_ms = ?5
            WHERE id = ?1
            "#,
        )?;
        for (video_id, path) in migrations {
            let updated = stmt.execute(params![
                video_id,
                path.display().to_string(),
                path.file_name()
                    .map(|value| value.to_string_lossy().to_string())
                    .unwrap_or_default(),
                path.parent()
                    .map(|value| value.display().to_string())
                    .unwrap_or_default(),
                crate::media::now_ms(),
            ])?;
            if updated == 0 {
                anyhow::bail!("video id {video_id} was not found");
            }
        }
    }

    tx.execute_batch("DROP TABLE IF EXISTS temp.delete_video_ids;")?;
    tx.commit()?;
    Ok(summary)
}

pub fn upsert_embedding_model(
    conn: &Connection,
    model_id: &str,
    name: &str,
    dimension: usize,
    runtime: &str,
    model_path: &str,
    model_file_hash: &str,
    created_unix_ms: i64,
) -> anyhow::Result<()> {
    conn.execute(
        r#"
        INSERT INTO embedding_models (
            model_id, name, dimension, runtime, model_path, model_file_hash, created_unix_ms
        )
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
        ON CONFLICT(model_id) DO UPDATE SET
            name = excluded.name,
            dimension = excluded.dimension,
            runtime = excluded.runtime,
            model_path = excluded.model_path,
            model_file_hash = excluded.model_file_hash
        "#,
        params![
            model_id,
            name,
            dimension as i64,
            runtime,
            model_path,
            model_file_hash,
            created_unix_ms
        ],
    )?;
    Ok(())
}

pub fn count_frame_embeddings(
    conn: &Connection,
    video_id: i64,
    model_id: &str,
) -> anyhow::Result<usize> {
    let count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM frame_embeddings WHERE video_id = ?1 AND model_id = ?2",
        params![video_id, model_id],
        |row| row.get(0),
    )?;
    Ok(count as usize)
}

pub fn frame_embedding_counts_by_video(
    conn: &Connection,
    model_id: &str,
) -> anyhow::Result<HashMap<i64, usize>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT video_id, COUNT(*)
        FROM frame_embeddings
        WHERE model_id = ?1
        GROUP BY video_id
        "#,
    )?;
    let rows = stmt.query_map(params![model_id], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
    })?;
    let mut counts = HashMap::new();
    for row in rows {
        let (video_id, count) = row?;
        counts.insert(video_id, count.max(0) as usize);
    }
    Ok(counts)
}

pub fn find_embedding_source_by_video_identity(
    conn: &Connection,
    video: &VideoRecord,
    model_id: &str,
    expected_frames: usize,
) -> anyhow::Result<Option<i64>> {
    let Some(video_id) = video.id else {
        return Ok(None);
    };
    let Some(partial_hash) = video
        .partial_hash
        .as_deref()
        .filter(|value| !value.is_empty())
    else {
        return Ok(None);
    };
    let Some(duration_seconds) = video.duration_seconds else {
        return Ok(None);
    };
    let Some(width) = video.width else {
        return Ok(None);
    };
    let Some(height) = video.height else {
        return Ok(None);
    };

    conn.query_row(
        r#"
        SELECT v.id
        FROM videos v
        JOIN frame_embeddings fe
          ON fe.video_id = v.id
         AND fe.model_id = ?1
        WHERE v.id <> ?2
          AND v.scan_status = 'ok'
          AND v.partial_hash = ?3
          AND v.size_bytes = ?4
          AND ABS(COALESCE(v.duration_seconds, 0.0) - ?5) <= 0.001
          AND COALESCE(v.width, 0) = ?6
          AND COALESCE(v.height, 0) = ?7
          AND COALESCE(v.bitrate, 0) = ?8
          AND COALESCE(v.codec, '') = ?9
          AND ABS(COALESCE(v.frame_rate, 0.0) - ?10) <= 0.001
          AND COALESCE(v.audio_codec, '') = ?11
          AND COALESCE(v.container_format, '') = ?12
        GROUP BY v.id
        HAVING COUNT(fe.frame_index) >= ?13
        ORDER BY v.scanned_at_unix_ms DESC, v.id DESC
        LIMIT 1
        "#,
        params![
            model_id,
            video_id,
            partial_hash,
            video.size_bytes as i64,
            duration_seconds,
            width as i64,
            height as i64,
            video.bitrate.map(|value| value as i64).unwrap_or(0),
            video.codec.as_deref().unwrap_or_default(),
            video.frame_rate.unwrap_or(0.0),
            video.audio_codec.as_deref().unwrap_or_default(),
            video.container_format.as_deref().unwrap_or_default(),
            expected_frames as i64,
        ],
        |row| row.get(0),
    )
    .optional()
    .map_err(Into::into)
}

pub fn copy_frame_embeddings_from_video(
    conn: &mut Connection,
    source_video_id: i64,
    target_video_id: i64,
    model_id: &str,
) -> anyhow::Result<usize> {
    if source_video_id == target_video_id {
        return Ok(0);
    }
    let tx = conn.transaction()?;
    tx.execute(
        "DELETE FROM ai_pair_scores WHERE model_id = ?1 AND (video_a_id = ?2 OR video_b_id = ?2)",
        params![model_id, target_video_id],
    )?;
    tx.execute(
        "DELETE FROM ai_match_edges WHERE model_id = ?1 AND (video_a_id = ?2 OR video_b_id = ?2)",
        params![model_id, target_video_id],
    )?;
    tx.execute(
        "DELETE FROM frame_embeddings WHERE video_id = ?1 AND model_id = ?2",
        params![target_video_id, model_id],
    )?;
    let inserted = tx.execute(
        r#"
        INSERT INTO frame_embeddings (
            video_id, model_id, frame_index, timestamp_seconds, embedding, norm, created_unix_ms
        )
        SELECT ?1, model_id, frame_index, timestamp_seconds, embedding, norm, ?4
        FROM frame_embeddings
        WHERE video_id = ?2 AND model_id = ?3
        "#,
        params![
            target_video_id,
            source_video_id,
            model_id,
            crate::media::now_ms()
        ],
    )?;
    tx.commit()?;
    Ok(inserted)
}

fn clear_video_ai_data(conn: &Connection, video_id: i64) -> anyhow::Result<()> {
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ai_pair_scores_video_b ON ai_pair_scores(video_b_id)",
        [],
    )?;
    conn.execute(
        "DELETE FROM ai_pair_scores WHERE video_a_id = ?1 OR video_b_id = ?1",
        params![video_id],
    )?;
    conn.execute(
        "DELETE FROM ai_match_edges WHERE video_a_id = ?1 OR video_b_id = ?1",
        params![video_id],
    )?;
    conn.execute(
        "DELETE FROM frame_embeddings WHERE video_id = ?1",
        params![video_id],
    )?;
    Ok(())
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DeletedVideoRows {
    pub videos: usize,
    pub pair_scores: usize,
    pub match_edges: usize,
    pub frame_embeddings: usize,
}

fn add_deleted_rows(total: &mut DeletedVideoRows, rows: DeletedVideoRows) {
    total.videos = total.videos.saturating_add(rows.videos);
    total.pair_scores = total.pair_scores.saturating_add(rows.pair_scores);
    total.match_edges = total.match_edges.saturating_add(rows.match_edges);
    total.frame_embeddings = total.frame_embeddings.saturating_add(rows.frame_embeddings);
}

pub fn delete_videos_by_ids(
    conn: &Connection,
    video_ids: &[i64],
) -> anyhow::Result<DeletedVideoRows> {
    if video_ids.is_empty() {
        return Ok(DeletedVideoRows::default());
    }

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ai_pair_scores_video_b ON ai_pair_scores(video_b_id)",
        [],
    )?;

    let mut ids = video_ids.to_vec();
    ids.sort_unstable();
    ids.dedup();

    let tx = conn.unchecked_transaction()?;
    tx.execute_batch(
        r#"
        DROP TABLE IF EXISTS temp.delete_video_ids;
        CREATE TEMP TABLE delete_video_ids (
            video_id INTEGER PRIMARY KEY
        ) WITHOUT ROWID;
        "#,
    )?;
    {
        let mut stmt =
            tx.prepare("INSERT OR IGNORE INTO temp.delete_video_ids (video_id) VALUES (?1)")?;
        for id in &ids {
            stmt.execute(params![id])?;
        }
    }

    let mut summary = DeletedVideoRows::default();
    summary.pair_scores = summary.pair_scores.saturating_add(tx.execute(
        "DELETE FROM ai_pair_scores WHERE video_a_id IN (SELECT video_id FROM temp.delete_video_ids)",
        [],
    )?);
    summary.pair_scores = summary.pair_scores.saturating_add(tx.execute(
        "DELETE FROM ai_pair_scores WHERE video_b_id IN (SELECT video_id FROM temp.delete_video_ids)",
        [],
    )?);
    summary.match_edges = summary.match_edges.saturating_add(tx.execute(
        "DELETE FROM ai_match_edges WHERE video_a_id IN (SELECT video_id FROM temp.delete_video_ids)",
        [],
    )?);
    summary.match_edges = summary.match_edges.saturating_add(tx.execute(
        "DELETE FROM ai_match_edges WHERE video_b_id IN (SELECT video_id FROM temp.delete_video_ids)",
        [],
    )?);
    summary.frame_embeddings = tx.execute(
        "DELETE FROM frame_embeddings WHERE video_id IN (SELECT video_id FROM temp.delete_video_ids)",
        [],
    )?;
    tx.execute(
        "DELETE FROM video_scan_sessions WHERE video_id IN (SELECT video_id FROM temp.delete_video_ids)",
        [],
    )?;
    summary.videos = tx.execute(
        "DELETE FROM videos WHERE id IN (SELECT video_id FROM temp.delete_video_ids)",
        [],
    )?;
    tx.execute_batch("DROP TABLE IF EXISTS temp.delete_video_ids;")?;
    tx.commit()?;
    Ok(summary)
}

fn delete_orphan_path_conflicts(
    conn: &Connection,
    video_id: i64,
    path_text: &str,
) -> anyhow::Result<DeletedVideoRows> {
    delete_orphan_path_conflicts_except(conn, video_id, path_text, &HashSet::new())
}

fn delete_orphan_path_conflicts_except(
    conn: &Connection,
    video_id: i64,
    path_text: &str,
    ignored_ids: &HashSet<i64>,
) -> anyhow::Result<DeletedVideoRows> {
    let mut stmt = conn.prepare(
        r#"
        SELECT v.id
        FROM videos AS v
        WHERE lower(v.path) = lower(?1)
          AND v.id <> ?2
          AND NOT EXISTS (
              SELECT 1
              FROM video_scan_sessions AS linked
              WHERE linked.video_id = v.id
          )
        "#,
    )?;
    let ids = stmt
        .query_map(params![path_text, video_id], |row| row.get::<_, i64>(0))?
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|id| !ignored_ids.contains(id))
        .collect::<Vec<_>>();
    drop(stmt);
    delete_videos_by_ids(conn, &ids)
}

pub fn delete_orphan_videos(conn: &Connection) -> anyhow::Result<DeletedVideoRows> {
    let mut stmt = conn.prepare(
        r#"
        SELECT v.id
        FROM videos AS v
        LEFT JOIN video_scan_sessions AS linked ON linked.video_id = v.id
        WHERE linked.video_id IS NULL
        "#,
    )?;
    let ids = stmt
        .query_map([], |row| row.get::<_, i64>(0))?
        .collect::<Result<Vec<_>, _>>()?;
    drop(stmt);
    delete_videos_by_ids(conn, &ids)
}

pub fn storage_database_stats(conn: &Connection) -> anyhow::Result<StorageDatabaseStats> {
    Ok(StorageDatabaseStats {
        videos: table_row_count(conn, "videos")?,
        scan_sessions: table_row_count(conn, "scan_sessions")?,
        frame_embeddings: table_row_count(conn, "frame_embeddings")?,
        ai_pair_scores: table_row_count(conn, "ai_pair_scores")?,
        ai_match_edges: table_row_count(conn, "ai_match_edges")?,
        embedding_models: table_row_count(conn, "embedding_models")?,
    })
}

pub fn cleanup_stale_ai_rows(conn: &Connection) -> anyhow::Result<DeletedVideoRows> {
    cleanup_stale_ai_rows_with_active_cache(conn, None)
}

pub fn cleanup_stale_ai_rows_with_active_cache(
    conn: &Connection,
    active_cache_key: Option<&str>,
) -> anyhow::Result<DeletedVideoRows> {
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ai_pair_scores_video_b ON ai_pair_scores(video_b_id)",
        [],
    )?;
    let mut summary = DeletedVideoRows::default();
    let tx = conn.unchecked_transaction()?;
    summary.pair_scores = summary.pair_scores.saturating_add(tx.execute(
        r#"
        DELETE FROM ai_pair_scores
        WHERE NOT EXISTS (SELECT 1 FROM videos WHERE videos.id = ai_pair_scores.video_a_id)
           OR NOT EXISTS (SELECT 1 FROM videos WHERE videos.id = ai_pair_scores.video_b_id)
           OR NOT EXISTS (SELECT 1 FROM embedding_models WHERE embedding_models.model_id = ai_pair_scores.model_id)
        "#,
        [],
    )?);
    summary.match_edges = summary.match_edges.saturating_add(tx.execute(
        r#"
        DELETE FROM ai_match_edges
        WHERE NOT EXISTS (SELECT 1 FROM videos WHERE videos.id = ai_match_edges.video_a_id)
           OR NOT EXISTS (SELECT 1 FROM videos WHERE videos.id = ai_match_edges.video_b_id)
           OR NOT EXISTS (SELECT 1 FROM embedding_models WHERE embedding_models.model_id = ai_match_edges.model_id)
        "#,
        [],
    )?);
    summary.frame_embeddings = summary.frame_embeddings.saturating_add(tx.execute(
        r#"
        DELETE FROM frame_embeddings
        WHERE NOT EXISTS (SELECT 1 FROM videos WHERE videos.id = frame_embeddings.video_id)
           OR NOT EXISTS (SELECT 1 FROM embedding_models WHERE embedding_models.model_id = frame_embeddings.model_id)
        "#,
        [],
    )?);
    if let Some(active_cache_key) = active_cache_key {
        summary.pair_scores = summary.pair_scores.saturating_add(tx.execute(
            "DELETE FROM ai_pair_scores WHERE cache_key <> ?1",
            params![active_cache_key],
        )?);
    }
    tx.commit()?;
    Ok(summary)
}

fn table_row_count(conn: &Connection, table: &str) -> anyhow::Result<usize> {
    let count = conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
        row.get::<_, i64>(0)
    })?;
    Ok(count.max(0) as usize)
}

pub fn latest_embedding_model_id_for_path(
    conn: &Connection,
    model_path: &str,
) -> anyhow::Result<Option<String>> {
    conn.query_row(
        r#"
        SELECT model_id
        FROM embedding_models
        WHERE model_path = ?1
        ORDER BY created_unix_ms DESC
        LIMIT 1
        "#,
        params![model_path],
        |row| row.get(0),
    )
    .optional()
    .map_err(Into::into)
}

pub fn replace_frame_embeddings(
    conn: &mut Connection,
    video_id: i64,
    model_id: &str,
    embeddings: &[FrameEmbeddingRecord],
) -> anyhow::Result<()> {
    let tx = conn.transaction()?;
    tx.execute(
        "DELETE FROM ai_pair_scores WHERE model_id = ?1 AND (video_a_id = ?2 OR video_b_id = ?2)",
        params![model_id, video_id],
    )?;
    tx.execute(
        "DELETE FROM frame_embeddings WHERE video_id = ?1 AND model_id = ?2",
        params![video_id, model_id],
    )?;
    {
        let mut stmt = tx.prepare(
            r#"
            INSERT INTO frame_embeddings (
                video_id, model_id, frame_index, timestamp_seconds, embedding, norm, created_unix_ms
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            "#,
        )?;
        for embedding in embeddings {
            stmt.execute(params![
                video_id,
                model_id,
                embedding.frame_index as i64,
                embedding.timestamp_seconds,
                encode_f32_blob(&embedding.embedding),
                embedding.norm,
                crate::media::now_ms(),
            ])?;
        }
    }
    tx.commit()?;
    Ok(())
}

pub fn list_frame_embeddings(
    conn: &Connection,
    model_id: &str,
    video_ids: &[i64],
) -> anyhow::Result<Vec<FrameEmbeddingRecord>> {
    if video_ids.is_empty() {
        return Ok(Vec::new());
    }

    let mut embeddings = Vec::new();
    for chunk in video_ids.chunks(800) {
        let placeholders = sql_placeholders(chunk.len());
        let sql = format!(
            r#"
            SELECT video_id, model_id, frame_index, timestamp_seconds, embedding, norm
            FROM frame_embeddings
            WHERE model_id = ? AND video_id IN ({placeholders})
            ORDER BY video_id ASC, frame_index ASC
            "#
        );
        let mut values = Vec::with_capacity(chunk.len() + 1);
        values.push(Value::Text(model_id.to_string()));
        values.extend(chunk.iter().map(|id| Value::Integer(*id)));
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(values), |row| {
            let blob: Vec<u8> = row.get(4)?;
            Ok(FrameEmbeddingRecord {
                video_id: row.get(0)?,
                model_id: row.get(1)?,
                frame_index: row.get::<_, i64>(2)? as usize,
                timestamp_seconds: row.get(3)?,
                embedding: decode_f32_blob(&blob),
                norm: row.get(5)?,
            })
        })?;

        for row in rows {
            embeddings.push(row?);
        }
    }
    embeddings.sort_by_key(|item| (item.video_id, item.frame_index));
    Ok(embeddings)
}

pub fn list_ai_pair_scores_chunked<F>(
    conn: &mut Connection,
    model_id: &str,
    cache_key: &str,
    video_ids: &[i64],
    chunk_size: usize,
    mut on_chunk: F,
) -> anyhow::Result<Vec<AiPairScoreRecord>>
where
    F: FnMut(usize, usize),
{
    if video_ids.len() < 2 {
        return Ok(Vec::new());
    }

    let mut ids = video_ids.to_vec();
    ids.sort_unstable();
    ids.dedup();

    let mut scores = Vec::new();
    conn.execute_batch(
        r#"
        DROP TABLE IF EXISTS temp.current_match_video_ids;
        CREATE TEMP TABLE current_match_video_ids (
            video_id INTEGER PRIMARY KEY
        ) WITHOUT ROWID;
        "#,
    )?;

    {
        let tx = conn.transaction()?;
        {
            let mut stmt = tx.prepare(
                "INSERT OR IGNORE INTO temp.current_match_video_ids (video_id) VALUES (?1)",
            )?;
            for id in &ids {
                stmt.execute(params![id])?;
            }
        }
        tx.commit()?;
    }

    let chunk_size = chunk_size.clamp(1, 800);
    let mut processed_left = 0usize;
    for chunk in ids.chunks(chunk_size) {
        let placeholders = sql_placeholders(chunk.len());
        let sql = format!(
            r#"
            SELECT ps.video_a_id, ps.video_b_id, ps.model_id, ps.cache_key, ps.confidence,
                   ps.matched_frame_count, ps.compared_frame_count, ps.average_similarity, ps.coverage,
                   ps.created_unix_ms
            FROM ai_pair_scores AS ps INDEXED BY idx_ai_pair_scores_model_key_a
            JOIN temp.current_match_video_ids AS wanted_b ON wanted_b.video_id = ps.video_b_id
            WHERE ps.model_id = ? AND ps.cache_key = ?
              AND ps.video_a_id IN ({placeholders})
            ORDER BY ps.video_a_id ASC, ps.video_b_id ASC
            "#
        );
        let mut values = Vec::with_capacity(chunk.len() + 2);
        values.push(Value::Text(model_id.to_string()));
        values.push(Value::Text(cache_key.to_string()));
        values.extend(chunk.iter().map(|id| Value::Integer(*id)));

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map(params_from_iter(values), read_ai_pair_score_row)?;
        for row in rows {
            scores.push(row?);
        }
        processed_left = processed_left.saturating_add(chunk.len());
        on_chunk(processed_left.min(ids.len()), scores.len());
    }
    conn.execute_batch("DROP TABLE IF EXISTS temp.current_match_video_ids;")?;
    Ok(scores)
}

pub fn prune_ai_pair_score_cache_keys(
    conn: &Connection,
    model_id: &str,
    active_cache_key: &str,
) -> anyhow::Result<usize> {
    let mut stmt = conn.prepare(
        r#"
        SELECT cache_key
        FROM ai_pair_scores
        WHERE model_id = ?1
        GROUP BY cache_key
        "#,
    )?;
    let rows = stmt.query_map(params![model_id], |row| row.get::<_, String>(0))?;
    let mut stale_keys = Vec::new();
    for row in rows {
        let key = row?;
        if key != active_cache_key {
            stale_keys.push(key);
        }
    }
    drop(stmt);

    let mut deleted = 0usize;
    for key in stale_keys {
        deleted = deleted.saturating_add(conn.execute(
            "DELETE FROM ai_pair_scores WHERE model_id = ?1 AND cache_key = ?2",
            params![model_id, key],
        )?);
    }
    Ok(deleted)
}

fn read_ai_pair_score_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<AiPairScoreRecord> {
    Ok(AiPairScoreRecord {
        video_a_id: row.get(0)?,
        video_b_id: row.get(1)?,
        model_id: row.get(2)?,
        cache_key: row.get(3)?,
        confidence: row.get(4)?,
        matched: row.get::<_, i64>(5)? as usize,
        compared: row.get::<_, i64>(6)? as usize,
        average_similarity: row.get(7)?,
        coverage: row.get(8)?,
        created_unix_ms: row.get(9)?,
    })
}

pub fn upsert_ai_pair_scores(
    conn: &mut Connection,
    records: &[AiPairScoreRecord],
) -> anyhow::Result<()> {
    if records.is_empty() {
        return Ok(());
    }
    let tx = conn.transaction()?;
    {
        let mut stmt = tx.prepare(
            r#"
            INSERT INTO ai_pair_scores (
                video_a_id, video_b_id, model_id, cache_key, confidence,
                matched_frame_count, compared_frame_count, average_similarity, coverage,
                created_unix_ms
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            ON CONFLICT(video_a_id, video_b_id, model_id, cache_key) DO UPDATE SET
                confidence = excluded.confidence,
                matched_frame_count = excluded.matched_frame_count,
                compared_frame_count = excluded.compared_frame_count,
                average_similarity = excluded.average_similarity,
                coverage = excluded.coverage,
                created_unix_ms = excluded.created_unix_ms
            "#,
        )?;
        for record in records {
            stmt.execute(params![
                record.video_a_id,
                record.video_b_id,
                record.model_id,
                record.cache_key,
                record.confidence,
                record.matched as i64,
                record.compared as i64,
                record.average_similarity,
                record.coverage,
                record.created_unix_ms,
            ])?;
        }
    }
    tx.commit()?;
    Ok(())
}

fn sql_placeholders(count: usize) -> String {
    std::iter::repeat("?")
        .take(count)
        .collect::<Vec<_>>()
        .join(",")
}

fn encode_f32_blob(values: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(values.len() * 4);
    for value in values {
        blob.extend_from_slice(&value.to_le_bytes());
    }
    blob
}

fn decode_f32_blob(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn expand_session_ids_by_source(
    conn: &Connection,
    session_ids: &[i64],
) -> anyhow::Result<Vec<i64>> {
    let mut expanded = HashSet::new();
    for session_id in session_ids {
        let source: Option<String> = conn
            .query_row(
                "SELECT source FROM scan_sessions WHERE id = ?1",
                params![session_id],
                |row| row.get(0),
            )
            .optional()?;
        if let Some(source) = source {
            let mut stmt = conn.prepare(
                r#"
                SELECT id
                FROM scan_sessions
                WHERE lower(source) = lower(?1)
                "#,
            )?;
            let rows = stmt.query_map(params![source], |row| row.get::<_, i64>(0))?;
            for row in rows {
                expanded.insert(row?);
            }
        }
    }

    Ok(expanded.into_iter().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_video(path: &str) -> VideoRecord {
        VideoRecord {
            id: None,
            path: path.to_string(),
            file_name: path.rsplit('\\').next().unwrap_or(path).to_string(),
            parent_path: r"C:\Videos".to_string(),
            size_bytes: 1024,
            modified_unix_ms: 1,
            extension: "mp4".to_string(),
            container_format: Some("mp4".to_string()),
            duration_seconds: Some(10.0),
            width: Some(1920),
            height: Some(1080),
            bitrate: Some(1_000_000),
            codec: Some("h264".to_string()),
            frame_rate: Some(30.0),
            audio_codec: Some("aac".to_string()),
            partial_hash: None,
            sample_hashes: Vec::new(),
            preview_images: Vec::new(),
            scan_status: "ok".to_string(),
            error: None,
            quality_score: 1.0,
            scanned_at_unix_ms: 1,
        }
    }

    fn pair_score(a: i64, b: i64, cache_key: &str) -> AiPairScoreRecord {
        let (video_a_id, video_b_id) = if a < b { (a, b) } else { (b, a) };
        AiPairScoreRecord {
            video_a_id,
            video_b_id,
            model_id: "model".to_string(),
            cache_key: cache_key.to_string(),
            confidence: 0.9,
            matched: 8,
            compared: 16,
            average_similarity: 0.9,
            coverage: 0.5,
            created_unix_ms: 1,
        }
    }

    fn table_count(conn: &Connection, table: &str) -> i64 {
        conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
            row.get(0)
        })
        .unwrap()
    }

    #[test]
    fn refresh_index_sources_removes_deleted_local_source() {
        let conn = Connection::open_in_memory().unwrap();
        migrate(&conn).unwrap();
        let base = std::env::temp_dir().join(format!(
            "dvs-refresh-source-test-{}-{}",
            std::process::id(),
            crate::media::now_ms()
        ));
        let source = base.join("source");
        std::fs::create_dir_all(&source).unwrap();
        let session_id =
            create_or_reset_scan_session(&conn, &source.display().to_string(), 1).unwrap();
        complete_scan_session(&conn, session_id, 1, 0, 0, 0).unwrap();
        std::fs::remove_dir_all(&source).unwrap();

        let summary = refresh_index_sources(&conn).unwrap();
        assert_eq!(summary.removed_missing_sources, 1);
        assert_eq!(summary.skipped_unavailable_roots, 0);
        assert!(list_scan_sessions(&conn).unwrap().is_empty());

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn pair_score_cache_query_uses_participant_ids_and_prunes_stale_keys() {
        let mut conn = Connection::open_in_memory().unwrap();
        migrate(&conn).unwrap();
        upsert_embedding_model(
            &conn,
            "model",
            "test",
            4,
            "runtime",
            "model.onnx",
            "hash",
            1,
        )
        .unwrap();

        let ids = (0..5)
            .map(|index| {
                upsert_video(&conn, &test_video(&format!(r"C:\Videos\video-{index}.mp4"))).unwrap()
            })
            .collect::<Vec<_>>();

        upsert_ai_pair_scores(
            &mut conn,
            &[
                pair_score(ids[0], ids[1], "active"),
                pair_score(ids[0], ids[2], "active"),
                pair_score(ids[1], ids[3], "active"),
                pair_score(ids[2], ids[3], "active"),
                pair_score(ids[0], ids[1], "old"),
            ],
        )
        .unwrap();

        let participant_ids = [ids[0], ids[1], ids[3]];
        let mut chunks = Vec::new();
        let scores = list_ai_pair_scores_chunked(
            &mut conn,
            "model",
            "active",
            &participant_ids,
            1,
            |left_done, loaded| chunks.push((left_done, loaded)),
        )
        .unwrap();
        let pairs = scores
            .iter()
            .map(|score| (score.video_a_id, score.video_b_id))
            .collect::<HashSet<_>>();

        assert!(pairs.contains(&(ids[0], ids[1])));
        assert!(pairs.contains(&(ids[1], ids[3])));
        assert!(!pairs.contains(&(ids[0], ids[2])));
        assert!(!pairs.contains(&(ids[2], ids[3])));
        assert!(chunks.len() >= 2);

        let deleted = prune_ai_pair_score_cache_keys(&conn, "model", "active").unwrap();
        assert_eq!(deleted, 1);
        let old_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM ai_pair_scores WHERE model_id = 'model' AND cache_key = 'old'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(old_count, 0);
    }

    #[test]
    fn move_video_path_preserves_ai_rows() {
        let mut conn = Connection::open_in_memory().unwrap();
        migrate(&conn).unwrap();
        upsert_embedding_model(
            &conn,
            "model",
            "test",
            4,
            "runtime",
            "model.onnx",
            "hash",
            1,
        )
        .unwrap();

        let keep_id = upsert_video(&conn, &test_video(r"C:\Videos\keep.mp4")).unwrap();
        let other_id = upsert_video(&conn, &test_video(r"C:\Videos\other.mp4")).unwrap();
        replace_frame_embeddings(
            &mut conn,
            keep_id,
            "model",
            &[FrameEmbeddingRecord {
                video_id: keep_id,
                model_id: "model".to_string(),
                frame_index: 0,
                timestamp_seconds: 1.0,
                embedding: vec![1.0, 0.0, 0.0, 0.0],
                norm: 1.0,
            }],
        )
        .unwrap();
        upsert_ai_pair_scores(&mut conn, &[pair_score(keep_id, other_id, "active")]).unwrap();

        move_video_path(&conn, keep_id, Path::new(r"C:\Videos\renamed.mp4")).unwrap();

        let moved = list_videos(&conn)
            .unwrap()
            .into_iter()
            .find(|video| video.id == Some(keep_id))
            .unwrap();
        assert_eq!(moved.path, r"C:\Videos\renamed.mp4");
        assert_eq!(count_frame_embeddings(&conn, keep_id, "model").unwrap(), 1);
        assert_eq!(
            table_count(&conn, "ai_pair_scores"),
            1,
            "path migration must not drop pair-score rows for the kept id"
        );
    }

    #[test]
    fn move_video_path_removes_orphan_target_owner() {
        let mut conn = Connection::open_in_memory().unwrap();
        migrate(&conn).unwrap();
        upsert_embedding_model(
            &conn,
            "model",
            "test",
            4,
            "runtime",
            "model.onnx",
            "hash",
            1,
        )
        .unwrap();

        let keep_id = upsert_video(&conn, &test_video(r"C:\Videos\keep.mp4")).unwrap();
        let orphan_id = upsert_video(&conn, &test_video(r"C:\Videos\target.mp4")).unwrap();
        replace_frame_embeddings(
            &mut conn,
            keep_id,
            "model",
            &[FrameEmbeddingRecord {
                video_id: keep_id,
                model_id: "model".to_string(),
                frame_index: 0,
                timestamp_seconds: 1.0,
                embedding: vec![1.0, 0.0, 0.0, 0.0],
                norm: 1.0,
            }],
        )
        .unwrap();
        replace_frame_embeddings(
            &mut conn,
            orphan_id,
            "model",
            &[FrameEmbeddingRecord {
                video_id: orphan_id,
                model_id: "model".to_string(),
                frame_index: 0,
                timestamp_seconds: 1.0,
                embedding: vec![0.0, 1.0, 0.0, 0.0],
                norm: 1.0,
            }],
        )
        .unwrap();
        upsert_ai_pair_scores(&mut conn, &[pair_score(keep_id, orphan_id, "active")]).unwrap();

        move_video_path(&conn, keep_id, Path::new(r"C:\Videos\target.mp4")).unwrap();

        assert_eq!(table_count(&conn, "videos"), 1);
        let moved = find_video_by_path(&conn, r"C:\Videos\target.mp4")
            .unwrap()
            .unwrap();
        assert_eq!(moved.id, Some(keep_id));
        assert_eq!(count_frame_embeddings(&conn, keep_id, "model").unwrap(), 1);
        assert_eq!(
            count_frame_embeddings(&conn, orphan_id, "model").unwrap(),
            0
        );
        assert_eq!(table_count(&conn, "ai_pair_scores"), 0);
    }

    #[test]
    fn move_video_path_rejects_linked_target_owner() {
        let conn = Connection::open_in_memory().unwrap();
        migrate(&conn).unwrap();
        let session_id = create_or_reset_scan_session(&conn, r"C:\Videos", 2).unwrap();
        let keep_id = upsert_video(&conn, &test_video(r"C:\Videos\keep.mp4")).unwrap();
        let target_id = upsert_video(&conn, &test_video(r"C:\Videos\target.mp4")).unwrap();
        link_video_to_scan_session(&conn, session_id, target_id).unwrap();

        let error = move_video_path(&conn, keep_id, Path::new(r"C:\Videos\target.mp4"))
            .expect_err("linked target owner must not be auto-deleted");

        assert!(error
            .to_string()
            .contains("path is already owned by video id"));
        assert!(find_video_by_path(&conn, r"C:\Videos\keep.mp4")
            .unwrap()
            .is_some());
        assert!(find_video_by_path(&conn, r"C:\Videos\target.mp4")
            .unwrap()
            .is_some());
    }

    #[test]
    fn batch_delete_and_move_preserves_kept_ai_rows() {
        let mut conn = Connection::open_in_memory().unwrap();
        migrate(&conn).unwrap();
        upsert_embedding_model(
            &conn,
            "model",
            "test",
            4,
            "runtime",
            "model.onnx",
            "hash",
            1,
        )
        .unwrap();

        let keep_id = upsert_video(&conn, &test_video(r"C:\Videos\keep.mp4")).unwrap();
        let source_id = upsert_video(&conn, &test_video(r"C:\Videos\source.mp4")).unwrap();
        let extra_id = upsert_video(&conn, &test_video(r"C:\Videos\extra.mp4")).unwrap();
        replace_frame_embeddings(
            &mut conn,
            keep_id,
            "model",
            &[FrameEmbeddingRecord {
                video_id: keep_id,
                model_id: "model".to_string(),
                frame_index: 0,
                timestamp_seconds: 1.0,
                embedding: vec![1.0, 0.0, 0.0, 0.0],
                norm: 1.0,
            }],
        )
        .unwrap();
        replace_frame_embeddings(
            &mut conn,
            extra_id,
            "model",
            &[FrameEmbeddingRecord {
                video_id: extra_id,
                model_id: "model".to_string(),
                frame_index: 0,
                timestamp_seconds: 1.0,
                embedding: vec![0.0, 1.0, 0.0, 0.0],
                norm: 1.0,
            }],
        )
        .unwrap();
        upsert_ai_pair_scores(&mut conn, &[pair_score(keep_id, source_id, "active")]).unwrap();
        upsert_ai_pair_scores(&mut conn, &[pair_score(keep_id, extra_id, "active")]).unwrap();

        let summary = delete_videos_and_move_paths(
            &conn,
            &[source_id, extra_id],
            &[(keep_id, PathBuf::from(r"C:\Videos\source.mp4"))],
        )
        .unwrap();

        assert_eq!(summary.videos, 2);
        assert_eq!(summary.frame_embeddings, 1);
        assert_eq!(count_frame_embeddings(&conn, keep_id, "model").unwrap(), 1);
        let moved = list_videos(&conn)
            .unwrap()
            .into_iter()
            .find(|video| video.id == Some(keep_id))
            .unwrap();
        assert_eq!(moved.path, r"C:\Videos\source.mp4");
        assert_eq!(table_count(&conn, "ai_pair_scores"), 0);
    }

    #[test]
    fn batch_delete_and_move_counts_orphan_target_owner() {
        let conn = Connection::open_in_memory().unwrap();
        migrate(&conn).unwrap();

        let keep_id = upsert_video(&conn, &test_video(r"C:\Videos\keep.mp4")).unwrap();
        let orphan_id = upsert_video(&conn, &test_video(r"C:\Videos\target.mp4")).unwrap();

        let summary = delete_videos_and_move_paths(
            &conn,
            &[],
            &[(keep_id, PathBuf::from(r"C:\Videos\target.mp4"))],
        )
        .unwrap();

        assert_eq!(summary.videos, 1);
        assert_eq!(table_count(&conn, "videos"), 1);
        let moved = find_video_by_path(&conn, r"C:\Videos\target.mp4")
            .unwrap()
            .unwrap();
        assert_eq!(moved.id, Some(keep_id));
        assert!(list_videos(&conn)
            .unwrap()
            .into_iter()
            .all(|video| video.id != Some(orphan_id)));
    }

    #[test]
    fn external_move_reuses_missing_video_identity() {
        let mut conn = Connection::open_in_memory().unwrap();
        migrate(&conn).unwrap();
        upsert_embedding_model(
            &conn,
            "model",
            "test",
            4,
            "runtime",
            "model.onnx",
            "hash",
            1,
        )
        .unwrap();

        let mut old = test_video(r"C:\Videos\old-name.mp4");
        old.partial_hash = Some("same-content".to_string());
        old.modified_unix_ms = 10;
        let old_id = upsert_video(&conn, &old).unwrap();
        replace_frame_embeddings(
            &mut conn,
            old_id,
            "model",
            &[FrameEmbeddingRecord {
                video_id: old_id,
                model_id: "model".to_string(),
                frame_index: 0,
                timestamp_seconds: 1.0,
                embedding: vec![1.0, 0.0, 0.0, 0.0],
                norm: 1.0,
            }],
        )
        .unwrap();

        let mut moved = test_video(r"C:\Videos\new-name.mp4");
        moved.partial_hash = Some("same-content".to_string());
        moved.modified_unix_ms = 99;
        let moved_id = upsert_video_preserving_content_identity(&conn, &moved).unwrap();

        assert_eq!(moved_id, old_id);
        assert_eq!(table_count(&conn, "videos"), 1);
        assert_eq!(count_frame_embeddings(&conn, old_id, "model").unwrap(), 1);
        let video = find_video_by_path(&conn, r"C:\Videos\new-name.mp4")
            .unwrap()
            .unwrap();
        assert_eq!(video.id, Some(old_id));
        assert_eq!(video.modified_unix_ms, 99);
    }

    #[test]
    fn external_move_replaces_lightweight_placeholder() {
        let conn = Connection::open_in_memory().unwrap();
        migrate(&conn).unwrap();
        let session_id = create_or_reset_scan_session(&conn, r"C:\Videos", 1).unwrap();

        let mut old = test_video(r"C:\Videos\old-name.mp4");
        old.partial_hash = Some("same-content".to_string());
        let old_id = upsert_video(&conn, &old).unwrap();

        let mut placeholder = test_video(r"C:\Videos\new-name.mp4");
        placeholder.partial_hash = None;
        placeholder.duration_seconds = None;
        placeholder.sample_hashes.clear();
        placeholder.preview_images.clear();
        let placeholder_id = upsert_video(&conn, &placeholder).unwrap();
        link_video_to_scan_session(&conn, session_id, placeholder_id).unwrap();

        let mut full = test_video(r"C:\Videos\new-name.mp4");
        full.partial_hash = Some("same-content".to_string());
        let reused_id = upsert_video_preserving_content_identity(&conn, &full).unwrap();

        assert_eq!(reused_id, old_id);
        assert_eq!(table_count(&conn, "videos"), 1);
        assert!(find_video_by_path(&conn, r"C:\Videos\new-name.mp4")
            .unwrap()
            .is_some());
        let linked: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM video_scan_sessions WHERE session_id = ?1 AND video_id = ?2",
                params![session_id, old_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(linked, 1);
    }

    #[test]
    fn content_identity_migration_skips_occupied_target_path() {
        let conn = Connection::open_in_memory().unwrap();
        migrate(&conn).unwrap();

        let mut old = test_video(r"C:\Videos\old-name.mp4");
        old.partial_hash = Some("same-content".to_string());
        let old_id = upsert_video(&conn, &old).unwrap();

        let mut occupied = test_video(r"C:\Videos\new-name.mp4");
        occupied.partial_hash = Some("same-content".to_string());
        let occupied_id = upsert_video(&conn, &occupied).unwrap();

        let mut candidate = test_video(r"C:\Videos\new-name.mp4");
        candidate.partial_hash = Some("same-content".to_string());
        let migrated = migrate_video_by_content_identity(&conn, &candidate).unwrap();

        assert_eq!(migrated, None);
        assert_eq!(table_count(&conn, "videos"), 2);
        assert_eq!(
            find_video_by_path(&conn, r"C:\Videos\old-name.mp4")
                .unwrap()
                .unwrap()
                .id,
            Some(old_id)
        );
        assert_eq!(
            find_video_by_path(&conn, r"C:\Videos\new-name.mp4")
                .unwrap()
                .unwrap()
                .id,
            Some(occupied_id)
        );
    }

    #[test]
    fn can_copy_embeddings_from_matching_video_identity() {
        let mut conn = Connection::open_in_memory().unwrap();
        migrate(&conn).unwrap();
        upsert_embedding_model(
            &conn,
            "model",
            "test",
            4,
            "runtime",
            "model.onnx",
            "hash",
            1,
        )
        .unwrap();

        let mut source = test_video(r"C:\Videos\source.mp4");
        source.partial_hash = Some("same-content".to_string());
        let source_id = upsert_video(&conn, &source).unwrap();
        let mut target = test_video(r"C:\Videos\target.mp4");
        target.partial_hash = Some("same-content".to_string());
        target.modified_unix_ms = 99;
        let target_id = upsert_video(&conn, &target).unwrap();
        target.id = Some(target_id);

        replace_frame_embeddings(
            &mut conn,
            source_id,
            "model",
            &[
                FrameEmbeddingRecord {
                    video_id: source_id,
                    model_id: "model".to_string(),
                    frame_index: 0,
                    timestamp_seconds: 1.0,
                    embedding: vec![1.0, 0.0, 0.0, 0.0],
                    norm: 1.0,
                },
                FrameEmbeddingRecord {
                    video_id: source_id,
                    model_id: "model".to_string(),
                    frame_index: 1,
                    timestamp_seconds: 2.0,
                    embedding: vec![0.0, 1.0, 0.0, 0.0],
                    norm: 1.0,
                },
            ],
        )
        .unwrap();

        let source_match =
            find_embedding_source_by_video_identity(&conn, &target, "model", 2).unwrap();
        assert_eq!(source_match, Some(source_id));

        let copied =
            copy_frame_embeddings_from_video(&mut conn, source_id, target_id, "model").unwrap();
        assert_eq!(copied, 2);
        assert_eq!(
            count_frame_embeddings(&conn, target_id, "model").unwrap(),
            2
        );
    }

    #[test]
    fn delete_scan_session_bulk_removes_only_orphan_ai_rows() {
        let mut conn = Connection::open_in_memory().unwrap();
        migrate(&conn).unwrap();
        upsert_embedding_model(
            &conn,
            "model",
            "test",
            4,
            "runtime",
            "model.onnx",
            "hash",
            1,
        )
        .unwrap();

        let source_a = r"C:\Videos\A";
        let source_b = r"C:\Videos\B";
        let session_a = create_or_reset_scan_session(&conn, source_a, 1).unwrap();
        let session_b = create_or_reset_scan_session(&conn, source_b, 1).unwrap();
        let only_a = upsert_video(&conn, &test_video(r"C:\Videos\A\only-a.mp4")).unwrap();
        let shared = upsert_video(&conn, &test_video(r"C:\Videos\shared.mp4")).unwrap();
        let only_b = upsert_video(&conn, &test_video(r"C:\Videos\B\only-b.mp4")).unwrap();
        link_video_to_scan_session(&conn, session_a, only_a).unwrap();
        link_video_to_scan_session(&conn, session_a, shared).unwrap();
        link_video_to_scan_session(&conn, session_b, shared).unwrap();
        link_video_to_scan_session(&conn, session_b, only_b).unwrap();

        replace_frame_embeddings(
            &mut conn,
            only_a,
            "model",
            &[FrameEmbeddingRecord {
                video_id: only_a,
                model_id: "model".to_string(),
                frame_index: 0,
                timestamp_seconds: 0.0,
                embedding: vec![1.0, 0.0, 0.0, 0.0],
                norm: 1.0,
            }],
        )
        .unwrap();
        replace_frame_embeddings(
            &mut conn,
            shared,
            "model",
            &[FrameEmbeddingRecord {
                video_id: shared,
                model_id: "model".to_string(),
                frame_index: 0,
                timestamp_seconds: 0.0,
                embedding: vec![0.0, 1.0, 0.0, 0.0],
                norm: 1.0,
            }],
        )
        .unwrap();
        upsert_ai_pair_scores(
            &mut conn,
            &[
                pair_score(only_a, shared, "active"),
                pair_score(only_a, only_b, "active"),
                pair_score(shared, only_b, "active"),
            ],
        )
        .unwrap();

        delete_scan_session(&conn, session_a).unwrap();

        assert_eq!(table_count(&conn, "scan_sessions"), 1);
        assert_eq!(table_count(&conn, "videos"), 2);
        assert_eq!(table_count(&conn, "frame_embeddings"), 1);
        let videos = list_videos(&conn).unwrap();
        assert!(videos.iter().any(|video| video.id == Some(shared)));
        assert!(videos.iter().any(|video| video.id == Some(only_b)));
        assert!(!videos.iter().any(|video| video.id == Some(only_a)));

        let remaining_scores = list_ai_pair_scores_chunked(
            &mut conn,
            "model",
            "active",
            &[shared, only_b],
            10,
            |_, _| {},
        )
        .unwrap();
        assert_eq!(remaining_scores.len(), 1);
        assert_eq!(
            (
                remaining_scores[0].video_a_id,
                remaining_scores[0].video_b_id
            ),
            edge_key_for_test(shared, only_b)
        );
    }

    fn edge_key_for_test(left: i64, right: i64) -> (i64, i64) {
        if left < right {
            (left, right)
        } else {
            (right, left)
        }
    }
}
