use crate::database;
use crate::models::{
    AppSettings, BatchMergeTask, CompletedAiFrameCacheCleanupSummary, OperationHistoryEntry,
    OperationResult, ReplacementPlan, StaleVideoPruneSummary, StorageCleanupSummary,
    StorageUsageItem, StorageUsageSummary, VacuumSummary, VideoRecord,
};
use crate::paths;
use anyhow::{anyhow, Context};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
#[cfg(windows)]
use std::{ffi::OsStr, os::windows::ffi::OsStrExt};

const SETTINGS_FILE: &str = "settings.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct FileMoveRecord {
    video: VideoRecord,
    session_ids: Vec<i64>,
    from: String,
    to: String,
}

#[derive(Debug)]
struct BatchMergePlan {
    task_index: usize,
    keeper: VideoRecord,
    naming_source: VideoRecord,
    extras: Vec<VideoRecord>,
    target_path: PathBuf,
    final_target_path: PathBuf,
}

pub fn create_replacement_plan(
    high_quality_id: i64,
    naming_source_id: i64,
) -> anyhow::Result<ReplacementPlan> {
    if high_quality_id == naming_source_id {
        return Err(anyhow!(
            "high quality file and naming source must be different"
        ));
    }

    let conn = database::open_database()?;
    let videos = database::list_videos(&conn)?;
    let high_quality = find_video(&videos, high_quality_id)?;
    let naming_source = find_video(&videos, naming_source_id)?;
    let settings = get_app_settings()?;

    let high_quality_path = PathBuf::from(&high_quality.path);
    let target_path = PathBuf::from(&naming_source.path);
    if !is_path_allowed_by_settings(&high_quality_path, &settings)
        || !is_path_allowed_by_settings(&target_path, &settings)
    {
        return Err(anyhow!(
            "replacement plan contains a path outside {}",
            paths::ALLOWED_SOURCE
        ));
    }

    let backup_dir = PathBuf::from(&settings.backup_dir);
    let backup_path = backup_destination_path(&backup_dir, &target_path)?;
    let timestamp = now_ms();
    let plan_id = format!("replace-plan-{timestamp}");
    let plan_file_path = paths::data_dir()
        .join("operations")
        .join(format!("{plan_id}.json"));

    let steps = vec![
        format!("移动命名来源到备份文件夹: {}", backup_path.display()),
        format!(
            "移动高清文件到原位置并继承文件名: {}",
            target_path.display()
        ),
        "重新扫描来源目录并刷新索引数据库".to_string(),
    ];
    let warnings = vec![
        "备份文件会移动到 Settings 指定的备份文件夹，不在原位置追加后缀".to_string(),
        "跨目录移动依赖 NAS/SMB 对 rename 的支持，失败时不应继续删除任何文件".to_string(),
        "执行前应确认目标文件未被播放器、同步软件或其他进程占用".to_string(),
    ];

    let plan = ReplacementPlan {
        plan_id,
        high_quality_id,
        naming_source_id,
        high_quality_path: high_quality_path.display().to_string(),
        target_path: target_path.display().to_string(),
        backup_path: backup_path.display().to_string(),
        plan_file: plan_file_path.display().to_string(),
        steps,
        warnings,
    };

    write_plan(&plan_file_path, &plan)?;
    Ok(plan)
}

pub fn execute_replacement(
    high_quality_id: i64,
    naming_source_id: i64,
    confirmation: &str,
) -> anyhow::Result<OperationResult> {
    if confirmation != "REPLACE" {
        return Err(anyhow!("confirmation must be REPLACE"));
    }

    let plan = create_replacement_plan(high_quality_id, naming_source_id)?;
    let high_quality_path = PathBuf::from(&plan.high_quality_path);
    let target_path = PathBuf::from(&plan.target_path);
    let backup_path = PathBuf::from(&plan.backup_path);
    let conn = database::open_database()?;
    let videos = database::list_videos(&conn)?;
    let high_quality = find_video(&videos, high_quality_id)?.clone();
    let naming_source = find_video(&videos, naming_source_id)?.clone();
    let move_records = vec![
        build_move_record(&conn, &naming_source, &target_path, &backup_path)?,
        build_move_record(&conn, &high_quality, &high_quality_path, &target_path)?,
    ];

    ensure_existing_file(&high_quality_path, "high quality file")?;
    ensure_existing_file(&target_path, "target naming source")?;

    if let Some(parent) = backup_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create backup directory {}", parent.display()))?;
    }
    ensure_destination_available(&backup_path)?;
    move_file(&target_path, &backup_path)
        .with_context(|| format!("move target to backup {}", backup_path.display()))?;

    if let Err(error) = move_file(&high_quality_path, &target_path) {
        let _ = fs::rename(&backup_path, &target_path);
        return Err(anyhow!(
            "failed to move high quality file into target path; rollback attempted: {error}"
        ));
    }

    database::delete_videos_by_ids(&conn, &[naming_source_id])
        .context("remove replaced naming source from index")?;
    database::move_video_path(&conn, high_quality_id, &target_path)
        .context("migrate replacement video index path")?;

    let operation_id = format!("replace-operation-{}", now_ms());
    let log_file = paths::data_dir()
        .join("operations")
        .join("operation-log.jsonl");
    let messages = vec![
        format!("backup created at {}", backup_path.display()),
        format!("replacement moved into {}", target_path.display()),
        format!(
            "index migrated: kept video id {} at {} and removed video id {}",
            high_quality_id,
            target_path.display(),
            naming_source_id
        ),
    ];
    append_structured_operation_log(
        &log_file,
        json!({
            "operationId": operation_id,
            "action": "replace",
            "summary": format!("替换路径 {}", target_path.display()),
            "messages": messages,
            "moves": move_records,
            "reversible": true,
            "createdAtUnixMs": now_ms()
        }),
    )?;

    Ok(OperationResult {
        operation_id,
        status: "completed".to_string(),
        log_file: log_file.display().to_string(),
        messages,
    })
}

pub fn execute_merge_selection(
    high_quality_id: i64,
    naming_source_id: i64,
    extra_video_ids: Vec<i64>,
    disposal: &str,
    confirmation: &str,
    filename_source_id: Option<i64>,
) -> anyhow::Result<OperationResult> {
    if high_quality_id == naming_source_id
        && filename_source_id.unwrap_or(high_quality_id) == high_quality_id
    {
        return Err(anyhow!(
            "recommended keep file and path source must be different"
        ));
    }

    let settings = get_app_settings()?;
    let backup_mode = match disposal {
        "backup" => {
            if confirmation != "MERGE_BACKUP" {
                return Err(anyhow!("confirmation must be MERGE_BACKUP"));
            }
            true
        }
        "delete" => {
            if !settings.allow_direct_delete {
                return Err(anyhow!("direct delete is disabled in Settings"));
            }
            if confirmation != "MERGE_DELETE" {
                return Err(anyhow!("confirmation must be MERGE_DELETE"));
            }
            false
        }
        _ => return Err(anyhow!("unsupported merge disposal: {disposal}")),
    };

    let conn = database::open_database()?;
    let videos = database::list_videos(&conn)?;
    let high_quality = find_video(&videos, high_quality_id)?.clone();
    let naming_source = find_video(&videos, naming_source_id)?.clone();
    let path_source_moves = high_quality_id != naming_source_id;
    let filename_source = filename_source_id
        .map(|id| find_video(&videos, id).cloned())
        .transpose()?;
    let mut extra_ids = Vec::new();
    let mut seen = HashSet::new();
    for id in extra_video_ids {
        if id != high_quality_id && id != naming_source_id && seen.insert(id) {
            extra_ids.push(id);
        }
    }
    let extras = extra_ids
        .iter()
        .map(|id| find_video(&videos, *id).cloned())
        .collect::<anyhow::Result<Vec<_>>>()?;

    let high_quality_path = PathBuf::from(&high_quality.path);
    let target_path = PathBuf::from(&naming_source.path);
    let final_target_path = merge_target_path(&target_path, filename_source.as_ref())?;
    validate_operable_path(&high_quality_path, &settings)?;
    validate_operable_path(&target_path, &settings)?;
    validate_operable_path(&final_target_path, &settings)?;
    for video in &extras {
        validate_operable_path(&PathBuf::from(&video.path), &settings)?;
    }
    let mut stale_messages = Vec::new();
    if !existing_file_status(&high_quality_path)? {
        stale_messages.push(format!(
            "recommended keep file is missing; index will remain until manual refresh {}",
            high_quality.path
        ));
    }
    if !existing_file_status(&target_path)? {
        stale_messages.push(format!(
            "path source file is missing; index will remain until manual refresh {}",
            naming_source.path
        ));
    }
    if !stale_messages.is_empty() {
        return Ok(skipped_stale_result("merge-selection", stale_messages)?);
    }

    let mut existing_extras = Vec::new();
    let mut stale_extra_ids = Vec::new();
    let mut stale_extra_messages = Vec::new();
    for (id, video) in extra_ids.iter().zip(extras.iter()) {
        let path = PathBuf::from(&video.path);
        if existing_file_status(&path)? {
            existing_extras.push(video.clone());
        } else {
            stale_extra_ids.push(*id);
            stale_extra_messages.push(format!(
                "extra file is missing; index will be removed {}",
                video.path
            ));
        }
    }

    let mut merge_existing_paths = vec![high_quality_path.clone(), target_path.clone()];
    merge_existing_paths.extend(
        existing_extras
            .iter()
            .map(|video| PathBuf::from(&video.path)),
    );
    ensure_merge_destination_available(&final_target_path, &merge_existing_paths)?;
    let mut removed_id_set = existing_extras
        .iter()
        .filter_map(|video| video.id)
        .chain(stale_extra_ids.iter().copied())
        .collect::<HashSet<_>>();
    if path_source_moves {
        removed_id_set.insert(naming_source_id);
    }
    let final_target_key = path_key_for_path(&final_target_path);
    if let Some(conflict) = videos.iter().find(|video| {
        video.id != Some(high_quality_id)
            && path_key_for_path(Path::new(&video.path)) == final_target_key
            && !video.id.is_some_and(|id| removed_id_set.contains(&id))
    }) {
        let conflict_path = PathBuf::from(&conflict.path);
        if existing_file_status(&conflict_path)? {
            return Err(anyhow!(
                "merge target path is owned by unrelated video id {:?}: {}",
                conflict.id,
                conflict.path
            ));
        }
    }

    let backup_dir = PathBuf::from(&settings.backup_dir);
    let mut reserved_backup_paths = HashSet::new();
    let target_backup = if backup_mode && path_source_moves {
        Some(backup_destination_path_with_reserved(
            &backup_dir,
            &target_path,
            &mut reserved_backup_paths,
        )?)
    } else {
        None
    };
    let extra_backups = if backup_mode {
        existing_extras
            .iter()
            .map(|video| {
                let path = PathBuf::from(&video.path);
                backup_destination_path_with_reserved(
                    &backup_dir,
                    &path,
                    &mut reserved_backup_paths,
                )
                .map(|destination| (path, destination))
            })
            .collect::<anyhow::Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    let mut move_records = Vec::new();
    if backup_mode {
        if let Some(target_backup) = target_backup.as_ref() {
            move_records.push(build_move_record(
                &conn,
                &naming_source,
                &target_path,
                target_backup,
            )?);
        }
        move_records.push(build_move_record(
            &conn,
            &high_quality,
            &high_quality_path,
            &final_target_path,
        )?);
        for (video, (source, destination)) in existing_extras.iter().zip(extra_backups.iter()) {
            move_records.push(build_move_record(&conn, video, source, destination)?);
        }
    }

    if backup_mode {
        fs::create_dir_all(&backup_dir)
            .with_context(|| format!("create backup directory {}", backup_dir.display()))?;
        ensure_unique_destinations(
            target_backup
                .iter()
                .chain(extra_backups.iter().map(|(_, dest)| dest)),
        )?;
        if let Some(target_backup) = target_backup.as_ref() {
            ensure_destination_available(target_backup)?;
        }
        for (_, destination) in &extra_backups {
            ensure_destination_available(destination)?;
        }
        if path_source_moves {
            let target_backup = target_backup
                .as_ref()
                .ok_or_else(|| anyhow!("missing target backup path"))?;
            move_file(&target_path, target_backup)
                .with_context(|| format!("move target to backup {}", target_backup.display()))?;
        }
    } else if path_source_moves {
        fs::remove_file(&target_path)
            .with_context(|| format!("delete {}", target_path.display()))?;
    }

    if path_source_moves {
        if let Err(error) = move_file(&high_quality_path, &target_path) {
            if let Some(target_backup) = target_backup.as_ref() {
                let _ = fs::rename(target_backup, &target_path);
            }
            return Err(anyhow!(
                "failed to move recommended keep file into target path; rollback attempted when possible: {error}"
            ));
        }
    }

    let current_keeper_path = if path_source_moves {
        target_path.clone()
    } else {
        high_quality_path.clone()
    };

    let operation_id = format!("merge-selection-operation-{}", now_ms());
    let log_file = paths::data_dir()
        .join("operations")
        .join("operation-log.jsonl");
    let mut messages = Vec::new();
    if path_source_moves {
        if backup_mode {
            if let Some(target_backup) = target_backup.as_ref() {
                messages.push(format!(
                    "moved path source {} to backup {}",
                    naming_source.path,
                    target_backup.display()
                ));
            }
        } else {
            messages.push(format!("deleted path source {}", naming_source.path));
        }
    }
    messages.push(format!(
        "moved recommended keep {} into {}",
        high_quality.path,
        final_target_path.display()
    ));
    messages.extend(stale_extra_messages);

    if backup_mode {
        for (video, (source, destination)) in existing_extras.iter().zip(extra_backups.iter()) {
            if existing_file_status(source)? {
                move_file(source, destination)
                    .with_context(|| format!("move {} to backup", source.display()))?;
                messages.push(format!(
                    "moved {} to backup {}",
                    video.path,
                    destination.display()
                ));
            } else {
                messages.push(format!(
                    "extra file is missing; index will remain until manual refresh {}",
                    video.path
                ));
            }
        }
    } else {
        for video in &existing_extras {
            let source = PathBuf::from(&video.path);
            if existing_file_status(&source)? {
                fs::remove_file(&source).with_context(|| format!("delete {}", source.display()))?;
                messages.push(format!("deleted {}", video.path));
            } else {
                messages.push(format!(
                    "extra file is missing; index will remain until manual refresh {}",
                    video.path
                ));
            }
        }
    }

    if final_target_path != current_keeper_path {
        ensure_destination_available(&final_target_path)?;
        move_file(&current_keeper_path, &final_target_path).with_context(|| {
            format!(
                "rename kept video {} to {}",
                current_keeper_path.display(),
                final_target_path.display()
            )
        })?;
        messages.push(format!(
            "renamed kept video to filename source path {}",
            final_target_path.display()
        ));
    }

    let mut removed_ids = Vec::new();
    if path_source_moves {
        removed_ids.push(naming_source_id);
    }
    removed_ids.extend(existing_extras.iter().filter_map(|video| video.id));
    removed_ids.extend(stale_extra_ids);
    database::delete_videos_by_ids(&conn, &removed_ids)
        .context("remove merged-away videos from index")?;
    database::move_video_path(&conn, high_quality_id, &final_target_path)
        .context("migrate kept merge video index path")?;
    messages.push(format!(
        "index migrated: kept video id {} at {} and removed {} merged-away ids",
        high_quality_id,
        final_target_path.display(),
        removed_ids.len()
    ));

    append_structured_operation_log(
        &log_file,
        json!({
            "operationId": operation_id,
            "action": if backup_mode { "merge-backup" } else { "merge-delete" },
            "summary": if backup_mode {
                format!("并入并备份 {} 个文件", messages.len())
            } else {
                format!("并入并删除 {} 个文件", messages.len())
            },
            "messages": messages,
            "moves": move_records,
            "reversible": backup_mode,
            "createdAtUnixMs": now_ms()
        }),
    )?;
    Ok(OperationResult {
        operation_id,
        status: "completed".to_string(),
        log_file: log_file.display().to_string(),
        messages,
    })
}

pub fn execute_batch_merge_selection(
    tasks: Vec<BatchMergeTask>,
    disposal: &str,
    confirmation: &str,
) -> anyhow::Result<OperationResult> {
    if tasks.is_empty() {
        return Err(anyhow!("batch merge task list is empty"));
    }

    let settings = get_app_settings()?;
    let backup_mode = match disposal {
        "backup" => {
            if confirmation != "MERGE_BACKUP" {
                return Err(anyhow!("confirmation must be MERGE_BACKUP"));
            }
            true
        }
        "delete" => {
            if !settings.allow_direct_delete {
                return Err(anyhow!("direct delete is disabled in Settings"));
            }
            if confirmation != "MERGE_DELETE" {
                return Err(anyhow!("confirmation must be MERGE_DELETE"));
            }
            false
        }
        _ => return Err(anyhow!("unsupported merge disposal: {disposal}")),
    };

    let conn = database::open_database()?;
    let videos = database::list_videos(&conn)?;
    let videos_by_id = videos
        .iter()
        .filter_map(|video| video.id.map(|id| (id, video.clone())))
        .collect::<HashMap<_, _>>();
    let mut active_ids = videos_by_id.keys().copied().collect::<HashSet<_>>();
    let mut claimed_ids = HashSet::new();
    let mut plans = Vec::new();
    let mut messages = Vec::new();
    let mut planned_skipped = 0usize;

    for (task_index, task) in tasks.into_iter().enumerate() {
        let BatchMergeTask {
            keeper_id,
            naming_source_id,
            filename_source_id,
            extra_video_ids,
        } = task;
        let mut task_ids = vec![keeper_id, naming_source_id];
        if let Some(id) = filename_source_id {
            if id == keeper_id || id == naming_source_id || extra_video_ids.contains(&id) {
                task_ids.push(id);
            }
        }
        task_ids.extend(extra_video_ids.iter().copied());
        task_ids.sort_unstable();
        task_ids.dedup();
        if task_ids.iter().any(|id| claimed_ids.contains(id)) {
            planned_skipped = planned_skipped.saturating_add(1);
            messages.push(format!(
                "skipped task {} because one of its videos was already claimed by an earlier batch task",
                task_index + 1
            ));
            continue;
        }
        for id in &task_ids {
            claimed_ids.insert(*id);
        }

        let Some(keeper) = videos_by_id.get(&keeper_id).cloned() else {
            planned_skipped = planned_skipped.saturating_add(1);
            messages.push(format!(
                "skipped task {} because keeper id {} no longer exists",
                task_index + 1,
                keeper_id
            ));
            continue;
        };
        let Some(naming_source) = videos_by_id.get(&naming_source_id).cloned() else {
            planned_skipped = planned_skipped.saturating_add(1);
            messages.push(format!(
                "skipped task {} because naming source id {} no longer exists",
                task_index + 1,
                naming_source_id
            ));
            continue;
        };
        if !active_ids.contains(&keeper_id) || !active_ids.contains(&naming_source_id) {
            planned_skipped = planned_skipped.saturating_add(1);
            messages.push(format!(
                "skipped task {} because keeper or naming source was already removed",
                task_index + 1
            ));
            continue;
        }
        let filename_source = if let Some(id) = filename_source_id {
            let Some(video) = videos_by_id.get(&id).cloned() else {
                planned_skipped = planned_skipped.saturating_add(1);
                messages.push(format!(
                    "skipped task {} because filename source id {} no longer exists",
                    task_index + 1,
                    id
                ));
                continue;
            };
            Some(video)
        } else {
            None
        };

        let mut extra_ids = Vec::new();
        let mut seen = HashSet::new();
        for id in extra_video_ids {
            if id != keeper_id
                && id != naming_source_id
                && active_ids.contains(&id)
                && seen.insert(id)
            {
                extra_ids.push(id);
            }
        }
        if keeper_id == naming_source_id
            && extra_ids.is_empty()
            && filename_source_id.unwrap_or(keeper_id) == keeper_id
        {
            planned_skipped = planned_skipped.saturating_add(1);
            messages.push(format!(
                "skipped task {} because it has no removable extra videos",
                task_index + 1
            ));
            continue;
        }
        let extras = extra_ids
            .iter()
            .filter_map(|id| videos_by_id.get(id).cloned())
            .collect::<Vec<_>>();

        let keeper_path = PathBuf::from(&keeper.path);
        let target_path = PathBuf::from(&naming_source.path);
        let final_target_path = merge_target_path(&target_path, filename_source.as_ref())?;
        validate_operable_path(&keeper_path, &settings)?;
        validate_operable_path(&target_path, &settings)?;
        validate_operable_path(&final_target_path, &settings)?;
        for video in &extras {
            validate_operable_path(&PathBuf::from(&video.path), &settings)?;
        }

        let mut merge_existing_paths = vec![keeper_path.clone(), target_path.clone()];
        merge_existing_paths.extend(extras.iter().map(|video| PathBuf::from(&video.path)));
        ensure_merge_destination_available(&final_target_path, &merge_existing_paths)?;

        let target_key = path_key_for_path(&final_target_path);
        let mut task_removed = extras
            .iter()
            .filter_map(|video| video.id)
            .collect::<HashSet<_>>();
        if keeper_id != naming_source_id {
            task_removed.insert(naming_source_id);
        }
        if let Some(conflict) = videos.iter().find(|video| {
            video.id != Some(keeper_id)
                && path_key_for_path(Path::new(&video.path)) == target_key
                && !video.id.is_some_and(|id| task_removed.contains(&id))
        }) {
            let conflict_path = PathBuf::from(&conflict.path);
            if existing_file_status(&conflict_path)? {
                planned_skipped = planned_skipped.saturating_add(1);
                messages.push(format!(
                    "skipped task {} because target path is owned by unrelated video id {:?}: {}",
                    task_index + 1,
                    conflict.id,
                    conflict.path
                ));
                continue;
            }
        }

        plans.push(BatchMergePlan {
            task_index,
            keeper,
            naming_source,
            extras,
            target_path,
            final_target_path,
        });
    }

    if plans.is_empty() {
        return Ok(skipped_stale_result("batch-merge-selection", messages)?);
    }

    let backup_dir = PathBuf::from(&settings.backup_dir);
    if backup_mode {
        fs::create_dir_all(&backup_dir)
            .with_context(|| format!("create backup directory {}", backup_dir.display()))?;
    }
    let mut reserved_backup_paths = HashSet::new();
    let mut move_records = Vec::new();
    let mut removed_ids = Vec::new();
    let mut migrations = Vec::new();
    let mut completed = 0usize;
    let mut skipped = planned_skipped;
    let mut failed = 0usize;

    for plan in plans {
        let keeper_id = plan.keeper.id.expect("indexed video has id");
        let naming_source_id = plan.naming_source.id.expect("indexed video has id");
        let keeper_path = PathBuf::from(&plan.keeper.path);
        let target_path = plan.target_path.clone();
        let final_target_path = plan.final_target_path.clone();
        let merge_moves_path = keeper_id != naming_source_id;
        let keeper_changes_path = merge_moves_path || final_target_path != keeper_path;

        if !active_ids.contains(&keeper_id)
            || (merge_moves_path && !active_ids.contains(&naming_source_id))
        {
            skipped = skipped.saturating_add(1);
            messages.push(format!(
                "skipped task {} because keeper or naming source is no longer active",
                plan.task_index + 1
            ));
            continue;
        }
        if keeper_changes_path {
            if !existing_file_status(&keeper_path)? {
                skipped = skipped.saturating_add(1);
                messages.push(format!(
                    "skipped task {} because keeper file is missing: {}",
                    plan.task_index + 1,
                    plan.keeper.path
                ));
                continue;
            }
        }
        if merge_moves_path {
            if !existing_file_status(&target_path)? {
                skipped = skipped.saturating_add(1);
                messages.push(format!(
                    "skipped task {} because naming source file is missing: {}",
                    plan.task_index + 1,
                    plan.naming_source.path
                ));
                continue;
            }
        }
        let mut merge_existing_paths = vec![keeper_path.clone(), target_path.clone()];
        merge_existing_paths.extend(plan.extras.iter().map(|video| PathBuf::from(&video.path)));
        ensure_merge_destination_available(&final_target_path, &merge_existing_paths)?;

        let mut task_removed_ids = Vec::new();
        let mut task_inactive_ids = Vec::new();
        let mut task_migrations = Vec::new();
        let mut task_messages = Vec::new();
        let mut task_move_records = Vec::new();
        let target_backup = if backup_mode && merge_moves_path {
            let destination = backup_destination_path_with_reserved(
                &backup_dir,
                &target_path,
                &mut reserved_backup_paths,
            )?;
            ensure_destination_available(&destination)?;
            task_move_records.push(build_move_record(
                &conn,
                &plan.naming_source,
                &target_path,
                &destination,
            )?);
            Some(destination)
        } else {
            None
        };
        if backup_mode && keeper_changes_path {
            task_move_records.push(build_move_record(
                &conn,
                &plan.keeper,
                &keeper_path,
                &final_target_path,
            )?);
        }

        let mut extra_backup_paths = Vec::new();
        if backup_mode {
            for video in &plan.extras {
                let source = PathBuf::from(&video.path);
                if existing_file_status(&source)? {
                    let destination = backup_destination_path_with_reserved(
                        &backup_dir,
                        &source,
                        &mut reserved_backup_paths,
                    )?;
                    ensure_destination_available(&destination)?;
                    task_move_records.push(build_move_record(&conn, video, &source, &destination)?);
                    extra_backup_paths.push((video.clone(), source, destination));
                } else if let Some(id) = video.id {
                    task_removed_ids.push(id);
                    task_inactive_ids.push(id);
                    task_messages.push(format!(
                        "extra file is missing; index will be removed {}",
                        video.path
                    ));
                }
            }
        }

        let task_result = (|| -> anyhow::Result<usize> {
            let mut file_failures = 0usize;
            let mut current_keeper_path = keeper_path.clone();
            if merge_moves_path {
                if backup_mode {
                    let target_backup = target_backup
                        .as_ref()
                        .ok_or_else(|| anyhow!("missing target backup path"))?;
                    move_file(&target_path, target_backup).with_context(|| {
                        format!("move target to backup {}", target_backup.display())
                    })?;
                } else {
                    fs::remove_file(&target_path)
                        .with_context(|| format!("delete {}", target_path.display()))?;
                }

                if let Err(error) = move_file(&keeper_path, &target_path) {
                    if backup_mode {
                        if let Some(target_backup) = target_backup.as_ref() {
                            let _ = fs::rename(target_backup, &target_path);
                        }
                    }
                    return Err(anyhow!(
                        "failed to move keeper into target path; rollback attempted when possible: {error}"
                    ));
                }
                current_keeper_path = target_path.clone();
                task_removed_ids.push(naming_source_id);
                task_inactive_ids.push(naming_source_id);
                task_messages.push(format!(
                    "moved keeper {} into {}",
                    plan.keeper.path,
                    target_path.display()
                ));
            }

            if backup_mode {
                for (video, source, destination) in &extra_backup_paths {
                    if existing_file_status(source)? {
                        match move_file(source, destination) {
                            Ok(()) => {
                                if let Some(id) = video.id {
                                    task_removed_ids.push(id);
                                    task_inactive_ids.push(id);
                                }
                                task_messages.push(format!(
                                    "moved {} to backup {}",
                                    video.path,
                                    destination.display()
                                ));
                            }
                            Err(error) => {
                                file_failures = file_failures.saturating_add(1);
                                task_messages.push(format!(
                                    "failed to move {} to backup {}: {}",
                                    video.path,
                                    destination.display(),
                                    error
                                ));
                            }
                        }
                    }
                }
            } else {
                for video in &plan.extras {
                    let Some(id) = video.id else {
                        continue;
                    };
                    let source = PathBuf::from(&video.path);
                    if existing_file_status(&source)? {
                        match fs::remove_file(&source) {
                            Ok(()) => {
                                task_messages.push(format!("deleted {}", video.path));
                                task_removed_ids.push(id);
                                task_inactive_ids.push(id);
                            }
                            Err(error) => {
                                file_failures = file_failures.saturating_add(1);
                                task_messages
                                    .push(format!("failed to delete {}: {}", video.path, error));
                            }
                        }
                    } else {
                        task_messages.push(format!(
                            "extra file is missing; index will be removed {}",
                            video.path
                        ));
                        task_removed_ids.push(id);
                        task_inactive_ids.push(id);
                    }
                }
            }
            if final_target_path != current_keeper_path {
                ensure_destination_available(&final_target_path)?;
                move_file(&current_keeper_path, &final_target_path).with_context(|| {
                    format!(
                        "rename kept video {} to {}",
                        current_keeper_path.display(),
                        final_target_path.display()
                    )
                })?;
                task_messages.push(format!(
                    "renamed kept video into {}",
                    final_target_path.display()
                ));
            }
            if keeper_changes_path {
                task_migrations.push((keeper_id, final_target_path.clone()));
            }
            task_messages.push(format!(
                "kept video id {} at {}",
                keeper_id,
                final_target_path.display()
            ));
            Ok(file_failures)
        })();

        match task_result {
            Ok(file_failures) => {
                removed_ids.extend(task_removed_ids);
                for id in task_inactive_ids {
                    active_ids.remove(&id);
                }
                migrations.extend(task_migrations);
                move_records.extend(task_move_records);
                messages.extend(task_messages);
                if file_failures == 0 {
                    completed = completed.saturating_add(1);
                } else {
                    failed = failed.saturating_add(file_failures);
                    messages.push(format!(
                        "task {} completed with {} file-operation failure(s); completed moves were still indexed",
                        plan.task_index + 1,
                        file_failures
                    ));
                }
            }
            Err(error) => {
                failed = failed.saturating_add(1);
                messages.push(format!("failed task {}: {}", plan.task_index + 1, error));
            }
        }
    }

    let deleted = database::delete_videos_and_move_paths(&conn, &removed_ids, &migrations)
        .context("batch update video index")?;
    messages.push(format!(
        "batch index update: migrated {} kept ids, removed {} video ids, deleted {} pair-score rows and {} embeddings",
        migrations.len(),
        deleted.videos,
        deleted.pair_scores,
        deleted.frame_embeddings
    ));

    let operation_id = format!("batch-merge-selection-operation-{}", now_ms());
    let log_file = operation_log_path();
    append_structured_operation_log(
        &log_file,
        json!({
            "operationId": operation_id,
            "action": if backup_mode { "batch-merge-backup" } else { "batch-merge-delete" },
            "summary": if backup_mode {
                format!("批量合并并备份：完成 {}，跳过 {}，失败 {}", completed, skipped, failed)
            } else {
                format!("批量合并并删除：完成 {}，跳过 {}，失败 {}", completed, skipped, failed)
            },
            "messages": messages,
            "moves": move_records,
            "reversible": backup_mode,
            "createdAtUnixMs": now_ms()
        }),
    )?;

    Ok(OperationResult {
        operation_id,
        status: if failed == 0 {
            "completed".to_string()
        } else {
            "completed-with-failures".to_string()
        },
        log_file: log_file.display().to_string(),
        messages,
    })
}

pub fn open_video(path: &str) -> anyhow::Result<()> {
    let video_path = PathBuf::from(path);
    let settings = get_app_settings()?;
    if !is_path_allowed_by_settings(&video_path, &settings) {
        return Err(anyhow!("video path is outside {}", paths::ALLOWED_SOURCE));
    }
    ensure_existing_file(&video_path, "video file")?;

    let conn = database::open_database()?;
    let indexed = database::list_videos(&conn)?
        .into_iter()
        .any(|video| video.path.eq_ignore_ascii_case(path));
    if !indexed {
        return Err(anyhow!("video is not in the local index"));
    }

    open_path_with_default_app(&video_path)
        .with_context(|| format!("open video with default app {}", video_path.display()))
}

pub fn prune_missing_videos(
    conn: &rusqlite::Connection,
    videos: Vec<VideoRecord>,
) -> anyhow::Result<Vec<VideoRecord>> {
    let mut retained = Vec::new();
    let mut missing_ids = Vec::new();
    for video in videos {
        if video.scan_status != "ok" {
            retained.push(video);
            continue;
        }

        match existing_file_status(&PathBuf::from(&video.path)) {
            Ok(true) => retained.push(video),
            Ok(false) => {
                if let Some(id) = video.id {
                    missing_ids.push(id);
                }
            }
            Err(_) => retained.push(video),
        }
    }
    database::delete_videos_by_ids(conn, &missing_ids)?;
    Ok(retained)
}

pub fn prune_stale_videos(session_ids: Option<Vec<i64>>) -> anyhow::Result<StaleVideoPruneSummary> {
    let conn = database::open_database()?;
    let videos = match session_ids.as_deref() {
        Some(ids) => database::list_videos_for_sessions(&conn, ids)?,
        None => database::list_videos(&conn)?,
    };
    let checked_videos = videos.len();
    let mut missing_ids = Vec::new();

    for video in videos {
        if video.scan_status != "ok" {
            continue;
        }
        let Some(id) = video.id else {
            continue;
        };
        match existing_file_status(&PathBuf::from(&video.path)) {
            Ok(true) => {}
            Ok(false) => missing_ids.push(id),
            Err(_) => {}
        }
    }

    let missing = database::delete_videos_by_ids(&conn, &missing_ids)?;
    let orphan = database::delete_orphan_videos(&conn)?;

    Ok(StaleVideoPruneSummary {
        checked_videos,
        removed_missing_videos: missing.videos,
        removed_orphan_videos: orphan.videos,
        deleted_pair_scores: missing.pair_scores.saturating_add(orphan.pair_scores),
        deleted_match_edges: missing.match_edges.saturating_add(orphan.match_edges),
        deleted_frame_embeddings: missing
            .frame_embeddings
            .saturating_add(orphan.frame_embeddings),
    })
}

pub fn get_app_settings() -> anyhow::Result<AppSettings> {
    paths::ensure_data_dirs()?;
    let path = settings_path();
    if !path.exists() {
        return Ok(default_settings());
    }
    let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
    let mut settings = serde_json::from_reader::<_, AppSettings>(file)
        .with_context(|| format!("parse {}", path.display()))?;
    settings.ui_language = normalize_ui_language(&settings.ui_language);
    settings.ai_frame_cache_enabled = true;
    settings.local_preprocess_enabled = true;
    settings.local_preprocess_temp_dir =
        normalize_primary_cache_dir(&settings.local_preprocess_temp_dir);
    settings.local_preprocess_secondary_temp_dir =
        normalize_secondary_cache_dir(&settings.local_preprocess_secondary_temp_dir);
    settings.keeper_size_priority_duration_seconds =
        normalize_keeper_duration_seconds(settings.keeper_size_priority_duration_seconds);
    settings.ai_gpu_worker_count = settings.ai_gpu_worker_count.clamp(1, 64);
    settings.ai_match_worker_count = settings.ai_match_worker_count.clamp(1, 128);
    settings.ram_disk_size_mb = settings.ram_disk_size_mb.clamp(512, 1_048_576);
    settings.local_preprocess_video_workers = settings.local_preprocess_video_workers.clamp(1, 8);
    settings.local_preprocess_overlap_start_percent = settings
        .local_preprocess_overlap_start_percent
        .clamp(50, 99);
    settings.local_preprocess_process_workers =
        settings.local_preprocess_process_workers.clamp(1, 4);
    settings.local_preprocess_secondary_threshold_mb = settings
        .local_preprocess_secondary_threshold_mb
        .clamp(512, 1_048_576);
    settings.nas_ssh_preprocess_enabled = false;
    Ok(settings)
}

pub fn save_app_settings(settings: &AppSettings) -> anyhow::Result<AppSettings> {
    paths::ensure_data_dirs()?;
    let path = settings_path();
    let normalized = AppSettings {
        ui_language: normalize_ui_language(&settings.ui_language),
        backup_dir: if settings.backup_dir.trim().is_empty() {
            default_settings().backup_dir
        } else {
            settings.backup_dir.trim().to_string()
        },
        naming_source_dirs: normalize_path_list(&settings.naming_source_dirs),
        keeper_size_priority_duration_seconds: normalize_keeper_duration_seconds(
            settings.keeper_size_priority_duration_seconds,
        ),
        scan_worker_count: settings.scan_worker_count.clamp(1, 4),
        sample_hash_count: settings.sample_hash_count.clamp(3, 20),
        temporal_hash_threshold: settings.temporal_hash_threshold.clamp(6, 24),
        min_temporal_match_points: settings.min_temporal_match_points.clamp(1, 10),
        allowed_unmatched_sample_frames: settings.allowed_unmatched_sample_frames.clamp(0, 20),
        allow_direct_delete: settings.allow_direct_delete,
        restrict_scan_to_test_path: settings.restrict_scan_to_test_path,
        compare_within_same_folder: settings.compare_within_same_folder,
        ai_vision_enabled: settings.ai_vision_enabled,
        ai_model_path: if settings.ai_model_path.trim().is_empty() {
            default_settings().ai_model_path
        } else {
            settings.ai_model_path.trim().to_string()
        },
        ai_device: normalize_ai_device(&settings.ai_device),
        ai_frame_count: settings.ai_frame_count.clamp(8, 512),
        ai_batch_size: settings.ai_batch_size.clamp(1, 64),
        ai_similarity_threshold: settings.ai_similarity_threshold.clamp(0.0, 0.99),
        ai_min_matched_frames: settings.ai_min_matched_frames.clamp(1, 128),
        ai_clip_matching_enabled: settings.ai_clip_matching_enabled,
        ai_index_after_scan: settings.ai_index_after_scan,
        ai_extract_worker_count: settings.ai_extract_worker_count.clamp(1, 8),
        ai_gpu_worker_count: settings.ai_gpu_worker_count.clamp(1, 64),
        ai_match_worker_count: settings.ai_match_worker_count.clamp(1, 128),
        ai_frame_cache_enabled: true,
        delete_ai_frame_cache_after_index: settings.delete_ai_frame_cache_after_index,
        ram_disk_enabled: settings.ram_disk_enabled,
        ram_disk_size_mb: settings.ram_disk_size_mb.clamp(512, 1_048_576),
        ram_disk_setup_completed: settings.ram_disk_setup_completed,
        local_preprocess_enabled: true,
        local_preprocess_video_workers: settings.local_preprocess_video_workers.clamp(1, 8),
        local_preprocess_overlap_start_percent: settings
            .local_preprocess_overlap_start_percent
            .clamp(50, 99),
        local_preprocess_process_workers: settings.local_preprocess_process_workers.clamp(1, 4),
        local_preprocess_frame_workers: settings.local_preprocess_frame_workers.clamp(1, 64),
        local_preprocess_temp_dir: normalize_primary_cache_dir(&settings.local_preprocess_temp_dir),
        local_preprocess_secondary_temp_dir: normalize_secondary_cache_dir(
            &settings.local_preprocess_secondary_temp_dir,
        ),
        local_preprocess_secondary_threshold_mb: settings
            .local_preprocess_secondary_threshold_mb
            .clamp(512, 1_048_576),
        nas_ssh_preprocess_enabled: false,
        nas_ssh_host: String::new(),
        nas_ssh_user: String::new(),
        nas_ssh_password: String::new(),
        nas_ssh_host_key: String::new(),
        nas_ssh_port: 22,
        nas_remote_root: String::new(),
        nas_smb_root: crate::paths::ALLOWED_SOURCE.to_string(),
        nas_ssh_ffmpeg_workers: 1,
    };
    let file = File::create(&path).with_context(|| format!("create {}", path.display()))?;
    serde_json::to_writer_pretty(file, &normalized)?;
    Ok(normalized)
}

pub fn execute_file_action(
    video_ids: Vec<i64>,
    action: &str,
    confirmation: &str,
) -> anyhow::Result<OperationResult> {
    execute_file_action_with_options(video_ids, action, confirmation, false)
}

pub fn execute_file_action_with_options(
    video_ids: Vec<i64>,
    action: &str,
    confirmation: &str,
    defer_index_update: bool,
) -> anyhow::Result<OperationResult> {
    if video_ids.is_empty() {
        return Err(anyhow!("no videos selected"));
    }

    match action {
        "backup" => move_videos_to_backup(video_ids, confirmation, defer_index_update),
        "delete" => delete_videos(video_ids, confirmation, defer_index_update),
        _ => Err(anyhow!("unsupported file action: {action}")),
    }
}

pub fn pick_folder(initial_path: Option<String>) -> anyhow::Result<Option<String>> {
    Ok(pick_folders_native(initial_path, false)?.into_iter().next())
}

pub fn pick_folders(initial_path: Option<String>) -> anyhow::Result<Vec<String>> {
    pick_folders_native(initial_path, true)
}

fn pick_folders_native(
    initial_path: Option<String>,
    allow_multi: bool,
) -> anyhow::Result<Vec<String>> {
    #[cfg(windows)]
    {
        let script = r#"
$code = @'
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;

namespace DuplicateVideoSearchNativePicker
{
    [ComImport]
    [Guid("DC1C5A9C-E88A-4DDE-A5A1-60F82A20AEF7")]
    internal class FileOpenDialog { }

    [ComImport]
    [Guid("d57c7288-d4ad-4768-be02-9d969532d960")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    internal interface IFileOpenDialog
    {
        [PreserveSig]
        uint Show(IntPtr parent);
        void SetFileTypes(uint cFileTypes, IntPtr rgFilterSpec);
        void SetFileTypeIndex(uint iFileType);
        void GetFileTypeIndex(out uint piFileType);
        void Advise(IntPtr pfde, out uint pdwCookie);
        void Unadvise(uint dwCookie);
        void SetOptions(FOS fos);
        void GetOptions(out FOS pfos);
        void SetDefaultFolder(IShellItem psi);
        void SetFolder(IShellItem psi);
        void GetFolder(out IShellItem ppsi);
        void GetCurrentSelection(out IShellItem ppsi);
        void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string pszName);
        void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string pszName);
        void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string pszTitle);
        void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string pszText);
        void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string pszLabel);
        void GetResult(out IShellItem ppsi);
        void AddPlace(IShellItem psi, int fdap);
        void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string pszDefaultExtension);
        void Close(int hr);
        void SetClientGuid(ref Guid guid);
        void ClearClientData();
        void SetFilter(IntPtr pFilter);
        [PreserveSig]
        uint GetResults(out IShellItemArray ppenum);
        [PreserveSig]
        uint GetSelectedItems(out IShellItemArray ppsai);
    }

    [ComImport]
    [Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    internal interface IShellItem
    {
        void BindToHandler(IntPtr pbc, ref Guid bhid, ref Guid riid, out IntPtr ppv);
        void GetParent(out IShellItem ppsi);
        void GetDisplayName(SIGDN sigdnName, out IntPtr ppszName);
        void GetAttributes(uint sfgaoMask, out uint psfgaoAttribs);
        void Compare(IShellItem psi, uint hint, out int piOrder);
    }

    [ComImport]
    [Guid("b63ea76d-1f85-456f-a19c-48159efa858b")]
    [InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    internal interface IShellItemArray
    {
        void BindToHandler(IntPtr pbc, ref Guid bhid, ref Guid riid, out IntPtr ppvOut);
        void GetPropertyStore(int flags, ref Guid riid, out IntPtr ppv);
        void GetPropertyDescriptionList(IntPtr keyType, ref Guid riid, out IntPtr ppv);
        void GetAttributes(uint attribFlags, uint sfgaoMask, out uint psfgaoAttribs);
        void GetCount(out uint pdwNumItems);
        void GetItemAt(uint dwIndex, out IShellItem ppsi);
        void EnumItems(out IntPtr ppenumShellItems);
    }

    [Flags]
    internal enum FOS : uint
    {
        FOS_NOCHANGEDIR = 0x00000008,
        FOS_PICKFOLDERS = 0x00000020,
        FOS_FORCEFILESYSTEM = 0x00000040,
        FOS_ALLOWMULTISELECT = 0x00000200,
        FOS_PATHMUSTEXIST = 0x00000800,
        FOS_DONTADDTORECENT = 0x02000000
    }

    internal enum SIGDN : uint
    {
        DESKTOPABSOLUTEPARSING = 0x80028000,
        FILESYSPATH = 0x80058000
    }

    public static class FolderPicker
    {
        private static readonly Guid ShellItemGuid = typeof(IShellItem).GUID;
        private const uint ErrorCancelled = 0x800704C7;

        [DllImport("shell32.dll", CharSet = CharSet.Unicode, PreserveSig = false)]
        private static extern void SHCreateItemFromParsingName(
            [MarshalAs(UnmanagedType.LPWStr)] string pszPath,
            IntPtr pbc,
            [MarshalAs(UnmanagedType.LPStruct)] Guid riid,
            out IShellItem ppv);

        public static string Pick(string initialPath, bool allowMulti)
        {
            IFileOpenDialog dialog = (IFileOpenDialog)new FileOpenDialog();
            FOS options;
            dialog.GetOptions(out options);
            FOS pickerOptions = options | FOS.FOS_PICKFOLDERS | FOS.FOS_FORCEFILESYSTEM | FOS.FOS_PATHMUSTEXIST | FOS.FOS_NOCHANGEDIR | FOS.FOS_DONTADDTORECENT;
            if (allowMulti)
            {
                pickerOptions |= FOS.FOS_ALLOWMULTISELECT;
            }
            dialog.SetOptions(pickerOptions);
            dialog.SetTitle("Select folders");
            dialog.SetOkButtonLabel("Select");

            string initial = ResolveInitialDirectory(initialPath);
            if (!string.IsNullOrWhiteSpace(initial))
            {
                try
                {
                    IShellItem folder;
                    SHCreateItemFromParsingName(initial, IntPtr.Zero, ShellItemGuid, out folder);
                    dialog.SetFolder(folder);
                }
                catch
                {
                    // Invalid or unavailable initial locations should not block the picker.
                }
            }

            uint result = dialog.Show(IntPtr.Zero);
            if (result == ErrorCancelled)
            {
                return "";
            }
            if (result != 0)
            {
                Marshal.ThrowExceptionForHR(unchecked((int)result));
            }

            if (!allowMulti)
            {
                IShellItem item;
                dialog.GetResult(out item);
                return GetPath(item);
            }

            IShellItemArray items;
            uint resultsCode = dialog.GetResults(out items);
            if (resultsCode != 0)
            {
                resultsCode = dialog.GetSelectedItems(out items);
            }
            if (resultsCode != 0)
            {
                IShellItem item;
                dialog.GetResult(out item);
                return GetPath(item);
            }
            uint count;
            items.GetCount(out count);
            List<string> paths = new List<string>();
            for (uint index = 0; index < count; index++)
            {
                IShellItem item;
                items.GetItemAt(index, out item);
                string path = GetPath(item);
                if (!string.IsNullOrWhiteSpace(path))
                {
                    paths.Add(path);
                }
            }
            return string.Join(Environment.NewLine, paths.ToArray());
        }

        private static string ResolveInitialDirectory(string initialPath)
        {
            if (string.IsNullOrWhiteSpace(initialPath))
            {
                return "";
            }

            if (Directory.Exists(initialPath))
            {
                return initialPath;
            }

            if (File.Exists(initialPath))
            {
                string parent = Path.GetDirectoryName(initialPath);
                return parent ?? "";
            }

            return "";
        }

        private static string GetPath(IShellItem item)
        {
            try
            {
                return GetDisplayName(item, SIGDN.FILESYSPATH);
            }
            catch
            {
                return GetDisplayName(item, SIGDN.DESKTOPABSOLUTEPARSING);
            }
        }

        private static string GetDisplayName(IShellItem item, SIGDN sigdn)
        {
            IntPtr pointer;
            item.GetDisplayName(sigdn, out pointer);
            try
            {
                return Marshal.PtrToStringUni(pointer) ?? "";
            }
            finally
            {
                Marshal.FreeCoTaskMem(pointer);
            }
        }
    }
}
'@
Add-Type -TypeDefinition $code
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::Write([DuplicateVideoSearchNativePicker.FolderPicker]::Pick($env:DVS_PICKER_INITIAL, $env:DVS_PICKER_MULTI -eq "1"))
"#;
        let mut command = Command::new("powershell.exe");
        apply_no_window(&mut command);
        if let Some(initial_path) = initial_path {
            command.env("DVS_PICKER_INITIAL", initial_path);
        }
        if allow_multi {
            command.env("DVS_PICKER_MULTI", "1");
        }
        let output = command
            .args(["-NoProfile", "-Sta", "-Command", script])
            .output()
            .context("open Windows folder picker")?;
        if !output.status.success() {
            return Err(anyhow!(
                "folder picker failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        Ok(String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(str::trim)
            .filter(|path| !path.is_empty())
            .map(ToOwned::to_owned)
            .collect())
    }

    #[cfg(not(windows))]
    {
        let _ = initial_path;
        let _ = allow_multi;
        Ok(Vec::new())
    }
}

pub fn list_operation_history() -> anyhow::Result<Vec<OperationHistoryEntry>> {
    let values = read_operation_log_values()?;
    let rolled_back: HashSet<String> = values
        .iter()
        .filter_map(|value| value.get("rollbackOf")?.as_str().map(ToString::to_string))
        .collect();
    let mut entries = Vec::new();

    for value in values {
        if value.get("rollbackOf").is_some() {
            continue;
        }
        let Some(operation_id) = value.get("operationId").and_then(|item| item.as_str()) else {
            continue;
        };
        let action = value
            .get("action")
            .and_then(|item| item.as_str())
            .unwrap_or_else(|| {
                if value.get("planId").is_some() {
                    "replace"
                } else {
                    "operation"
                }
            })
            .to_string();
        let messages = json_string_array(value.get("messages"));
        let has_moves = value
            .get("moves")
            .and_then(|item| item.as_array())
            .is_some_and(|items| !items.is_empty());
        let reversible = value
            .get("reversible")
            .and_then(|item| item.as_bool())
            .unwrap_or(false)
            && has_moves
            && !rolled_back.contains(operation_id);
        let summary = value
            .get("summary")
            .and_then(|item| item.as_str())
            .map(ToString::to_string)
            .unwrap_or_else(|| summarize_history_entry(&action, messages.len()));
        entries.push(OperationHistoryEntry {
            operation_id: operation_id.to_string(),
            action,
            summary,
            created_at_unix_ms: value
                .get("createdAtUnixMs")
                .and_then(|item| item.as_i64())
                .unwrap_or(0),
            reversible,
            rolled_back: rolled_back.contains(operation_id),
            messages,
        });
    }

    entries.sort_by(|left, right| right.created_at_unix_ms.cmp(&left.created_at_unix_ms));
    Ok(entries)
}

pub fn rollback_operation(operation_id: &str) -> anyhow::Result<OperationResult> {
    let values = read_operation_log_values()?;
    if values
        .iter()
        .any(|value| value.get("rollbackOf").and_then(|item| item.as_str()) == Some(operation_id))
    {
        return Err(anyhow!("operation has already been rolled back"));
    }
    let target = values
        .iter()
        .find(|value| value.get("operationId").and_then(|item| item.as_str()) == Some(operation_id))
        .ok_or_else(|| anyhow!("operation was not found: {operation_id}"))?;
    if !target
        .get("reversible")
        .and_then(|item| item.as_bool())
        .unwrap_or(false)
    {
        return Err(anyhow!("operation is not reversible"));
    }
    let moves: Vec<FileMoveRecord> = serde_json::from_value(
        target
            .get("moves")
            .cloned()
            .ok_or_else(|| anyhow!("operation does not include rollback records"))?,
    )?;
    if moves.is_empty() {
        return Err(anyhow!("operation does not include rollback records"));
    }

    let settings = get_app_settings()?;
    for record in moves.iter().rev() {
        let current_path = PathBuf::from(&record.to);
        let original_path = PathBuf::from(&record.from);
        validate_operable_path(&original_path, &settings)?;
        if !current_path.exists() {
            return Err(anyhow!(
                "rollback source file does not exist: {}",
                current_path.display()
            ));
        }
        if original_path.exists() {
            return Err(anyhow!(
                "rollback destination already exists: {}",
                original_path.display()
            ));
        }
    }

    let mut messages = Vec::new();
    for record in moves.iter().rev() {
        let current_path = PathBuf::from(&record.to);
        let original_path = PathBuf::from(&record.from);
        if let Some(parent) = original_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create rollback directory {}", parent.display()))?;
        }
        move_file(&current_path, &original_path).with_context(|| {
            format!(
                "rollback move {} to {}",
                current_path.display(),
                original_path.display()
            )
        })?;
        messages.push(format!(
            "restored {} to {}",
            current_path.display(),
            original_path.display()
        ));
    }

    let conn = database::open_database()?;
    for record in &moves {
        let video_id = database::upsert_video(&conn, &record.video)?;
        for session_id in &record.session_ids {
            if database::session_exists(&conn, *session_id)? {
                database::link_video_to_scan_session(&conn, *session_id, video_id)?;
            }
        }
    }

    let rollback_id = format!("rollback-operation-{}", now_ms());
    let log_file = operation_log_path();
    append_structured_operation_log(
        &log_file,
        json!({
            "operationId": rollback_id,
            "action": "rollback",
            "rollbackOf": operation_id,
            "summary": format!("回滚 {}", operation_id),
            "messages": messages,
            "reversible": false,
            "createdAtUnixMs": now_ms()
        }),
    )?;

    Ok(OperationResult {
        operation_id: rollback_id,
        status: "completed".to_string(),
        log_file: log_file.display().to_string(),
        messages,
    })
}

fn find_video(videos: &[VideoRecord], id: i64) -> anyhow::Result<&VideoRecord> {
    videos
        .iter()
        .find(|video| video.id == Some(id))
        .ok_or_else(|| anyhow!("video id {id} was not found in the local index"))
}

fn existing_file_status(path: &Path) -> anyhow::Result<bool> {
    match fs::metadata(path) {
        Ok(metadata) if metadata.is_file() => Ok(true),
        Ok(_) => Err(anyhow!("path is not a video file: {}", path.display())),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
        Err(error) => Err(anyhow!(
            "cannot access video file {}: {}",
            path.display(),
            error
        )),
    }
}

fn ensure_existing_file(path: &Path, label: &str) -> anyhow::Result<()> {
    if existing_file_status(path)? {
        Ok(())
    } else {
        Err(anyhow!("{label} does not exist: {}", path.display()))
    }
}

#[cfg(windows)]
fn open_path_with_default_app(path: &Path) -> anyhow::Result<()> {
    use windows_sys::Win32::UI::Shell::ShellExecuteW;
    use windows_sys::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;

    let operation = wide_null("open");
    let file = path
        .as_os_str()
        .encode_wide()
        .chain([0])
        .collect::<Vec<_>>();
    let result = unsafe {
        ShellExecuteW(
            std::ptr::null_mut(),
            operation.as_ptr(),
            file.as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            SW_SHOWNORMAL,
        )
    };
    if result as isize <= 32 {
        return Err(anyhow!(
            "ShellExecuteW failed with code {}",
            result as isize
        ));
    }
    Ok(())
}

#[cfg(windows)]
fn wide_null(value: &str) -> Vec<u16> {
    OsStr::new(value).encode_wide().chain([0]).collect()
}

#[cfg(not(windows))]
fn open_path_with_default_app(path: &Path) -> anyhow::Result<()> {
    let mut command = Command::new("xdg-open");
    command.arg(path).spawn().context("open with xdg-open")?;
    Ok(())
}

fn skipped_stale_result(action: &str, messages: Vec<String>) -> anyhow::Result<OperationResult> {
    let operation_id = format!("skip-stale-operation-{}", now_ms());
    let log_file = operation_log_path();
    append_structured_operation_log(
        &log_file,
        json!({
            "operationId": operation_id,
            "action": "skip-stale",
            "summary": format!("跳过失效任务 {}", action),
            "messages": messages,
            "reversible": false,
            "createdAtUnixMs": now_ms()
        }),
    )?;

    Ok(OperationResult {
        operation_id,
        status: "skipped-stale".to_string(),
        log_file: log_file.display().to_string(),
        messages,
    })
}

pub fn is_path_allowed_by_settings(path: &Path, settings: &AppSettings) -> bool {
    !settings.restrict_scan_to_test_path || paths::is_allowed_scan_path(path)
}

fn move_videos_to_backup(
    video_ids: Vec<i64>,
    confirmation: &str,
    defer_index_update: bool,
) -> anyhow::Result<OperationResult> {
    if confirmation != "MOVE" {
        return Err(anyhow!("confirmation must be MOVE"));
    }

    let settings = get_app_settings()?;
    let backup_dir = PathBuf::from(&settings.backup_dir);
    fs::create_dir_all(&backup_dir)
        .with_context(|| format!("create backup directory {}", backup_dir.display()))?;

    let conn = database::open_database()?;
    let videos = database::list_videos(&conn)?;
    let operation_id = format!("backup-operation-{}", now_ms());
    let log_file = paths::data_dir()
        .join("operations")
        .join("operation-log.jsonl");
    let mut messages = Vec::new();
    let mut moves = Vec::new();
    let mut move_records = Vec::new();
    let mut stale_ids = Vec::new();

    let mut reserved_backup_paths = HashSet::new();
    for video_id in video_ids {
        let video = find_video(&videos, video_id)?;
        let source = PathBuf::from(&video.path);
        if !is_path_allowed_by_settings(&source, &settings) {
            return Err(anyhow!("video path is outside {}", paths::ALLOWED_SOURCE));
        }
        if !existing_file_status(&source)? {
            stale_ids.push(video_id);
            messages.push(format!("removed stale missing index {}", source.display()));
            continue;
        }
        let destination = backup_destination_path_with_reserved(
            &backup_dir,
            &source,
            &mut reserved_backup_paths,
        )?;
        move_records.push(build_move_record(&conn, video, &source, &destination)?);
        moves.push((video_id, source, destination));
    }

    ensure_unique_destinations(moves.iter().map(|(_, _, destination)| destination))?;
    for (_, _, destination) in &moves {
        ensure_destination_available(destination)?;
    }

    for (video_id, source, destination) in moves {
        move_file(&source, &destination)?;
        messages.push(format!(
            "moved {} to {}",
            source.display(),
            destination.display()
        ));
        stale_ids.push(video_id);
    }
    if defer_index_update {
        messages.push(
            "index updated; current similarity list will update after manual refresh".to_string(),
        );
    }
    database::delete_videos_by_ids(&conn, &stale_ids)?;

    append_structured_operation_log(
        &log_file,
        json!({
            "operationId": operation_id,
            "action": "backup",
            "summary": format!("移入备份 {} 个文件", messages.len()),
            "messages": messages,
            "moves": move_records,
            "reversible": true,
            "createdAtUnixMs": now_ms()
        }),
    )?;
    Ok(OperationResult {
        operation_id,
        status: "completed".to_string(),
        log_file: log_file.display().to_string(),
        messages,
    })
}

fn delete_videos(
    video_ids: Vec<i64>,
    confirmation: &str,
    defer_index_update: bool,
) -> anyhow::Result<OperationResult> {
    let settings = get_app_settings()?;
    if !settings.allow_direct_delete {
        return Err(anyhow!("direct delete is disabled in Settings"));
    }
    if confirmation != "DELETE" {
        return Err(anyhow!("confirmation must be DELETE"));
    }

    let conn = database::open_database()?;
    let videos = database::list_videos(&conn)?;
    let operation_id = format!("delete-operation-{}", now_ms());
    let log_file = paths::data_dir()
        .join("operations")
        .join("operation-log.jsonl");
    let mut messages = Vec::new();

    for video_id in video_ids {
        let video = find_video(&videos, video_id)?;
        let source = PathBuf::from(&video.path);
        if !is_path_allowed_by_settings(&source, &settings) {
            return Err(anyhow!("video path is outside {}", paths::ALLOWED_SOURCE));
        }
        if existing_file_status(&source)? {
            fs::remove_file(&source).with_context(|| format!("delete {}", source.display()))?;
            messages.push(format!("deleted {}", source.display()));
        } else {
            messages.push(format!("removed stale missing index {}", source.display()));
        }
        database::delete_videos_by_ids(&conn, &[video_id])?;
    }
    if defer_index_update {
        messages.push(
            "index updated; current similarity list will update after manual refresh".to_string(),
        );
    }

    append_structured_operation_log(
        &log_file,
        json!({
            "operationId": operation_id,
            "action": "delete",
            "summary": format!("直接删除 {} 个文件", messages.len()),
            "messages": messages,
            "reversible": false,
            "createdAtUnixMs": now_ms()
        }),
    )?;
    Ok(OperationResult {
        operation_id,
        status: "completed".to_string(),
        log_file: log_file.display().to_string(),
        messages,
    })
}

fn default_settings() -> AppSettings {
    AppSettings {
        ui_language: "zh".to_string(),
        backup_dir: paths::data_dir().join("backups").display().to_string(),
        naming_source_dirs: Vec::new(),
        keeper_size_priority_duration_seconds: 300.0,
        scan_worker_count: 1,
        sample_hash_count: 11,
        temporal_hash_threshold: 8,
        min_temporal_match_points: 3,
        allowed_unmatched_sample_frames: 14,
        allow_direct_delete: false,
        restrict_scan_to_test_path: true,
        compare_within_same_folder: false,
        ai_vision_enabled: true,
        ai_model_path: "models\\dinov2-small-dynamic\\model.onnx".to_string(),
        ai_device: "auto".to_string(),
        ai_frame_count: 128,
        ai_batch_size: 32,
        ai_similarity_threshold: 0.86,
        ai_min_matched_frames: 8,
        ai_clip_matching_enabled: true,
        ai_index_after_scan: true,
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
        local_preprocess_temp_dir: default_primary_cache_dir(),
        local_preprocess_secondary_temp_dir: default_secondary_cache_dir(),
        local_preprocess_secondary_threshold_mb: 16 * 1024,
        nas_ssh_preprocess_enabled: false,
        nas_ssh_host: String::new(),
        nas_ssh_user: String::new(),
        nas_ssh_password: String::new(),
        nas_ssh_host_key: String::new(),
        nas_ssh_port: 22,
        nas_remote_root: String::new(),
        nas_smb_root: crate::paths::ALLOWED_SOURCE.to_string(),
        nas_ssh_ffmpeg_workers: 1,
    }
}

pub fn storage_usage() -> anyhow::Result<StorageUsageSummary> {
    paths::ensure_data_dirs()?;
    let data_dir = paths::data_dir();
    let database_path = paths::database_path();
    let sqlite_wal = database_path.with_extension("sqlite-wal");
    let sqlite_shm = database_path.with_extension("sqlite-shm");
    let known_dirs = [
        ("aiFrameCache", data_dir.join("ai-frame-cache")),
        ("thumbnails", data_dir.join("thumbnails")),
        ("backups", data_dir.join("backups")),
        ("operations", data_dir.join("operations")),
        ("reports", data_dir.join("reports")),
        ("tools", data_dir.join("tools")),
    ];

    let mut items = Vec::new();
    for (key, path) in known_dirs {
        items.push(storage_usage_item(key, &path)?);
    }
    items.push(storage_usage_item("database", &database_path)?);
    items.push(storage_usage_item("databaseWal", &sqlite_wal)?);
    items.push(storage_usage_item("databaseShm", &sqlite_shm)?);

    let total_bytes = items
        .iter()
        .fold(0u64, |sum, item| sum.saturating_add(item.bytes));
    let conn = database::open_database()?;
    Ok(StorageUsageSummary {
        data_dir: data_dir.display().to_string(),
        total_bytes,
        items,
        database_stats: database::storage_database_stats(&conn)?,
    })
}

pub fn cleanup_storage() -> anyhow::Result<StorageCleanupSummary> {
    paths::ensure_data_dirs()?;
    let conn = database::open_database()?;
    let videos = database::list_videos(&conn)?;
    let settings = get_app_settings()?;
    let ai_cleanup = crate::media::prune_stale_ai_frame_cache(&videos, settings.ai_frame_count)?;
    let thumbnail_cleanup = crate::media::prune_unreferenced_thumbnails(&videos)?;
    let orphan = database::delete_orphan_videos(&conn)?;
    let active_cache_key = crate::ai::current_ai_match_cache_key(&settings);
    let stale = database::cleanup_stale_ai_rows_with_active_cache(&conn, Some(&active_cache_key))?;

    Ok(StorageCleanupSummary {
        deleted_ai_frame_cache_files: ai_cleanup.files,
        deleted_ai_frame_cache_bytes: ai_cleanup.bytes,
        deleted_thumbnail_files: thumbnail_cleanup.files,
        deleted_thumbnail_bytes: thumbnail_cleanup.bytes,
        removed_orphan_videos: orphan.videos,
        deleted_pair_scores: orphan.pair_scores.saturating_add(stale.pair_scores),
        deleted_match_edges: orphan.match_edges.saturating_add(stale.match_edges),
        deleted_frame_embeddings: orphan
            .frame_embeddings
            .saturating_add(stale.frame_embeddings),
    })
}

pub fn cleanup_completed_ai_frame_cache() -> anyhow::Result<CompletedAiFrameCacheCleanupSummary> {
    paths::ensure_data_dirs()?;
    let conn = database::open_database()?;
    let videos = database::list_videos(&conn)?;
    let settings = get_app_settings()?;
    let model_path = crate::ai::resolve_model_path(&settings.ai_model_path);
    let model_path_text = model_path.display().to_string();
    let Some(model_id) = database::latest_embedding_model_id_for_path(&conn, &model_path_text)?
    else {
        return Ok(CompletedAiFrameCacheCleanupSummary {
            checked_videos: videos.len(),
            eligible_videos: 0,
            deleted_files: 0,
            deleted_bytes: 0,
        });
    };
    let embedding_counts = database::frame_embedding_counts_by_video(&conn, &model_id)?;
    let mut summary = CompletedAiFrameCacheCleanupSummary {
        checked_videos: videos.len(),
        eligible_videos: 0,
        deleted_files: 0,
        deleted_bytes: 0,
    };

    for video in videos.iter().filter(|video| video.scan_status == "ok") {
        let Some(video_id) = video.id else {
            continue;
        };
        let expected = crate::ai::expected_ai_frame_count(video, settings.ai_frame_count);
        if embedding_counts.get(&video_id).copied().unwrap_or(0) < expected {
            continue;
        }
        summary.eligible_videos = summary.eligible_videos.saturating_add(1);
        let cleanup = crate::media::delete_ai_frame_cache_for_video(video, expected)?;
        summary.deleted_files = summary.deleted_files.saturating_add(cleanup.files);
        summary.deleted_bytes = summary.deleted_bytes.saturating_add(cleanup.bytes);
    }

    Ok(summary)
}

pub fn vacuum_database() -> anyhow::Result<VacuumSummary> {
    paths::ensure_data_dirs()?;
    let path = paths::database_path();
    let before_bytes = file_len(&path);
    let conn = database::open_database()?;
    conn.execute_batch(
        "PRAGMA wal_checkpoint(TRUNCATE); VACUUM; PRAGMA wal_checkpoint(TRUNCATE);",
    )?;
    drop(conn);
    let after_bytes = file_len(&path);
    Ok(VacuumSummary {
        before_bytes,
        after_bytes,
        reclaimed_bytes: before_bytes.saturating_sub(after_bytes),
    })
}

fn storage_usage_item(key: &str, path: &Path) -> anyhow::Result<StorageUsageItem> {
    let (bytes, file_count) = path_usage(path)?;
    Ok(StorageUsageItem {
        key: key.to_string(),
        path: path.display().to_string(),
        bytes,
        file_count,
    })
}

fn path_usage(path: &Path) -> anyhow::Result<(u64, usize)> {
    let metadata = match fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok((0, 0)),
        Err(error) => return Err(error).with_context(|| format!("read {}", path.display())),
    };
    if metadata.is_file() {
        return Ok((metadata.len(), 1));
    }
    if !metadata.is_dir() {
        return Ok((0, 0));
    }

    let mut bytes = 0u64;
    let mut file_count = 0usize;
    for entry in walkdir::WalkDir::new(path)
        .into_iter()
        .filter_map(Result::ok)
    {
        if entry.file_type().is_file() {
            if let Ok(metadata) = entry.metadata() {
                bytes = bytes.saturating_add(metadata.len());
                file_count = file_count.saturating_add(1);
            }
        }
    }
    Ok((bytes, file_count))
}

fn file_len(path: &Path) -> u64 {
    fs::metadata(path)
        .map(|metadata| metadata.len())
        .unwrap_or(0)
}

fn normalize_ui_language(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "en" | "english" => "en".to_string(),
        _ => "zh".to_string(),
    }
}

fn normalize_path_list(paths: &[String]) -> Vec<String> {
    let mut normalized = Vec::new();
    let mut seen = HashSet::new();
    for path in paths {
        let trimmed = path.trim();
        if trimmed.is_empty() {
            continue;
        }
        let key = trimmed
            .replace('/', "\\")
            .trim_end_matches('\\')
            .to_lowercase();
        if seen.insert(key) {
            normalized.push(trimmed.to_string());
        }
    }
    normalized
}

fn normalize_keeper_duration_seconds(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 86_400.0)
    } else {
        300.0
    }
}

fn normalize_ai_device(device: &str) -> String {
    match device.trim().to_ascii_lowercase().as_str() {
        "gpu" => "gpu".to_string(),
        "cpu" => "cpu".to_string(),
        _ => "auto".to_string(),
    }
}

fn settings_path() -> PathBuf {
    paths::data_dir().join(SETTINGS_FILE)
}

fn default_primary_cache_dir() -> String {
    r"Z:\TEMP".to_string()
}

fn default_secondary_cache_dir() -> String {
    r"D:\TEMP".to_string()
}

fn normalize_primary_cache_dir(value: &str) -> String {
    if value.trim().is_empty() {
        default_primary_cache_dir()
    } else {
        value.trim().to_string()
    }
}

fn normalize_secondary_cache_dir(value: &str) -> String {
    if value.trim().is_empty() {
        default_secondary_cache_dir()
    } else {
        value.trim().to_string()
    }
}

fn merge_target_path(
    path_source_path: &Path,
    filename_source: Option<&VideoRecord>,
) -> anyhow::Result<PathBuf> {
    let Some(filename_source) = filename_source else {
        return Ok(path_source_path.to_path_buf());
    };
    let parent = path_source_path.parent().ok_or_else(|| {
        anyhow!(
            "path source has no parent directory: {}",
            path_source_path.display()
        )
    })?;
    let file_name = Path::new(&filename_source.path)
        .file_name()
        .map(|value| value.to_os_string())
        .or_else(|| {
            let trimmed = filename_source.file_name.trim();
            (!trimmed.is_empty()).then(|| std::ffi::OsString::from(trimmed))
        })
        .ok_or_else(|| anyhow!("filename source is missing a file name"))?;
    Ok(parent.join(file_name))
}

fn ensure_merge_destination_available(
    destination: &Path,
    allowed_existing_paths: &[PathBuf],
) -> anyhow::Result<()> {
    let destination_key = path_key_for_path(destination);
    if allowed_existing_paths
        .iter()
        .any(|path| path_key_for_path(path) == destination_key)
    {
        return Ok(());
    }
    ensure_destination_available(destination)
}

fn path_key_for_path(path: &Path) -> String {
    path.to_string_lossy()
        .replace('/', "\\")
        .trim_end_matches('\\')
        .to_ascii_lowercase()
}

fn backup_destination_path(backup_dir: &Path, source: &Path) -> anyhow::Result<PathBuf> {
    let mut reserved = HashSet::new();
    backup_destination_path_with_reserved(backup_dir, source, &mut reserved)
}

fn backup_destination_path_with_reserved(
    backup_dir: &Path,
    source: &Path,
    reserved: &mut HashSet<String>,
) -> anyhow::Result<PathBuf> {
    let file_name = source
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| anyhow!("source path has no file name: {}", source.display()))?;
    let stem = source
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .unwrap_or(file_name);
    let extension = source.extension().and_then(|value| value.to_str());

    for attempt in 0..10_000 {
        let candidate_name = if attempt == 0 {
            file_name.to_string()
        } else if let Some(extension) = extension.filter(|value| !value.is_empty()) {
            format!("{stem} ({attempt}).{extension}")
        } else {
            format!("{file_name} ({attempt})")
        };
        let candidate = backup_dir.join(candidate_name);
        let key = candidate.to_string_lossy().to_lowercase();
        if reserved.contains(&key) {
            continue;
        }
        match fs::metadata(&candidate) {
            Ok(_) => continue,
            Err(error) if error.kind() == ErrorKind::NotFound => {
                reserved.insert(key);
                return Ok(candidate);
            }
            Err(error) => {
                return Err(anyhow!(
                    "cannot inspect backup destination {}: {}",
                    candidate.display(),
                    error
                ));
            }
        }
    }

    Err(anyhow!(
        "cannot find an available backup filename for {} in {}",
        source.display(),
        backup_dir.display()
    ))
}

fn ensure_destination_available(destination: &Path) -> anyhow::Result<()> {
    match fs::metadata(destination) {
        Ok(_) => {
            return Err(anyhow!(
                "backup destination already exists; move or rename it first: {}",
                destination.display()
            ));
        }
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => {
            return Err(anyhow!(
                "cannot inspect backup destination {}: {}",
                destination.display(),
                error
            ));
        }
    }
    Ok(())
}

fn ensure_unique_destinations<'a>(
    destinations: impl Iterator<Item = &'a PathBuf>,
) -> anyhow::Result<()> {
    let mut seen = HashSet::new();
    for destination in destinations {
        let key = destination.to_string_lossy().to_lowercase();
        if !seen.insert(key) {
            return Err(anyhow!(
                "multiple selected videos would use the same backup filename: {}",
                destination.display()
            ));
        }
    }
    Ok(())
}

fn validate_operable_path(path: &Path, settings: &AppSettings) -> anyhow::Result<()> {
    if !is_path_allowed_by_settings(path, settings) {
        return Err(anyhow!("video path is outside {}", paths::ALLOWED_SOURCE));
    }
    Ok(())
}

fn move_file(source: &Path, destination: &Path) -> anyhow::Result<()> {
    match fs::rename(source, destination) {
        Ok(()) => Ok(()),
        Err(rename_error) => {
            fs::copy(source, destination).with_context(|| {
                format!(
                    "copy {} to {} after rename failed: {rename_error}",
                    source.display(),
                    destination.display()
                )
            })?;
            fs::remove_file(source)
                .with_context(|| format!("remove source after backup {}", source.display()))?;
            Ok(())
        }
    }
}

fn write_plan(path: &Path, plan: &ReplacementPlan) -> anyhow::Result<()> {
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    serde_json::to_writer_pretty(file, plan)?;
    Ok(())
}

fn build_move_record(
    conn: &rusqlite::Connection,
    video: &VideoRecord,
    from: &Path,
    to: &Path,
) -> anyhow::Result<FileMoveRecord> {
    let video_id = video
        .id
        .ok_or_else(|| anyhow!("video record is missing an id: {}", video.path))?;
    Ok(FileMoveRecord {
        video: video.clone(),
        session_ids: database::session_ids_for_video(conn, video_id)?,
        from: from.display().to_string(),
        to: to.display().to_string(),
    })
}

fn operation_log_path() -> PathBuf {
    paths::data_dir()
        .join("operations")
        .join("operation-log.jsonl")
}

fn read_operation_log_values() -> anyhow::Result<Vec<serde_json::Value>> {
    paths::ensure_data_dirs()?;
    let path = operation_log_path();
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = File::open(&path).with_context(|| format!("open {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut values = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(&line) {
            values.push(value);
        }
    }
    Ok(values)
}

fn json_string_array(value: Option<&serde_json::Value>) -> Vec<String> {
    value
        .and_then(|item| item.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(ToString::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn summarize_history_entry(action: &str, message_count: usize) -> String {
    match action {
        "backup" => format!("移入备份 {message_count} 个文件"),
        "delete" => format!("直接删除 {message_count} 个文件"),
        "merge-backup" => format!("并入并备份 {message_count} 个步骤"),
        "merge-delete" => format!("并入并删除 {message_count} 个步骤"),
        "replace" => "替换路径".to_string(),
        _ => format!("操作 {message_count} 个步骤"),
    }
}

fn append_structured_operation_log(path: &Path, entry: serde_json::Value) -> anyhow::Result<()> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open {}", path.display()))?;
    writeln!(file, "{}", serde_json::to_string(&entry)?)?;
    Ok(())
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(windows)]
fn apply_no_window(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x08000000;
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
fn apply_no_window(_command: &mut Command) {}
