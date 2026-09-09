use anyhow::Context;
use duplicate_video_search_lib::{
    ai, database, local_preprocess, matching, media,
    models::{MatchGroup, MatchItem, VideoRecord},
    nas_preprocess, operations, paths, ram_disk,
};
use serde_json::json;
use std::collections::HashSet;
use std::env;
use std::fs;

fn main() -> anyhow::Result<()> {
    let mut args = env::args().skip(1);
    let command = args.next().unwrap_or_else(|| "status".to_string());

    // Keep CLI lifecycle identical to the GUI's AI-index worker: ensure the
    // configured ImDisk cache is mounted before any command starts, and let
    // TaskRamDisk's Drop implementation release it on every exit path.
    let settings = operations::get_app_settings()?;
    let _ram_disk = ram_disk::TaskRamDisk::activate(&settings)?;

    match command.as_str() {
        "status" => {
            let status = media::tool_status()?;
            println!("{}", serde_json::to_string_pretty(&status)?);
        }
        "scan" => {
            let source = args
                .next()
                .unwrap_or_else(|| paths::ALLOWED_SOURCE.to_string());
            let summary = media::scan_source(&source)?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        "groups" => {
            let conn = database::open_database()?;
            let videos = database::list_videos(&conn)?;
            let min_confidence = args
                .next()
                .and_then(|value| value.parse::<f64>().ok())
                .unwrap_or(0.90);
            let temporal_hash_threshold = args
                .next()
                .and_then(|value| value.parse::<u32>().ok())
                .unwrap_or(8);
            let min_temporal_match_points = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(3);
            let allowed_unmatched_sample_frames = args
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(14);
            let groups = matching::build_match_groups_with_options(
                &videos,
                matching::MatchOptions {
                    compare_within_same_folder: settings.compare_within_same_folder,
                    min_confidence,
                    temporal_hash_threshold,
                    min_temporal_match_points,
                    allowed_unmatched_sample_frames,
                    keeper_size_priority_duration_seconds: 300.0,
                },
            );
            println!("{}", serde_json::to_string_pretty(&groups)?);
        }
        "ai-groups" => {
            let mut conn = database::open_database()?;
            let videos = database::list_videos(&conn)?;
            let settings = operations::get_app_settings()?;
            let min_confidence = args
                .next()
                .and_then(|value| value.parse::<f64>().ok())
                .unwrap_or(0.90);
            let groups = ai::build_ai_match_groups(&mut conn, &videos, &settings, min_confidence)?;
            println!("{}", serde_json::to_string_pretty(&groups)?);
        }
        "ai-pair-debug" => {
            let left_path = args.next().context("missing left video path")?;
            let right_path = args.next().context("missing right video path")?;
            let mut conn = database::open_database()?;
            let settings = operations::get_app_settings()?;
            let report = ai::debug_ai_pair(&mut conn, &left_path, &right_path, &settings)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        "videos" => {
            let conn = database::open_database()?;
            let videos = database::list_videos(&conn)?;
            println!("{}", serde_json::to_string_pretty(&videos)?);
        }
        "clear-index" => {
            let values = args.collect::<Vec<_>>();
            let keep_cache = values.iter().any(|value| value == "--keep-cache");
            let conn = database::open_database()?;
            database::clear_all_indexes(&conn)?;
            let removed_cache_dirs = if keep_cache {
                Vec::new()
            } else {
                clear_generated_caches()?
            };
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "status": "completed",
                    "keepCache": keep_cache,
                    "removedCacheDirs": removed_cache_dirs
                }))?
            );
        }
        "ai-index" => {
            let values = args.collect::<Vec<_>>();
            let force_rebuild = values.iter().any(|value| value == "--force");
            let session_ids = values
                .iter()
                .filter(|value| value.as_str() != "--force")
                .filter_map(|value| value.parse::<i64>().ok())
                .collect::<Vec<_>>();
            let summary = ai::build_ai_index_with_progress(
                if session_ids.is_empty() {
                    None
                } else {
                    Some(session_ids)
                },
                force_rebuild,
                |progress| {
                    let done = progress
                        .processed
                        .saturating_add(progress.skipped)
                        .saturating_add(progress.failed)
                        .min(progress.total_videos);
                    eprintln!(
                        "{} {}/{} total {} failed {}",
                        progress.phase,
                        done,
                        progress.total_videos,
                        progress.total_videos,
                        progress.failed
                    );
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        "merge-all-ai-groups" => {
            let values = args.collect::<Vec<_>>();
            let mut min_confidence = 0.90;
            let mut disposal = "backup".to_string();
            let mut limit: Option<usize> = None;
            let mut index = 0usize;
            while index < values.len() {
                match values[index].as_str() {
                    "--confidence" => {
                        if let Some(value) = values.get(index + 1) {
                            if let Ok(parsed) = value.parse::<f64>() {
                                min_confidence = parsed;
                            }
                        }
                        index += 2;
                    }
                    "--disposal" => {
                        if let Some(value) = values.get(index + 1) {
                            disposal = value.to_string();
                        }
                        index += 2;
                    }
                    "--limit" => {
                        if let Some(value) = values.get(index + 1) {
                            limit = value.parse::<usize>().ok();
                        }
                        index += 2;
                    }
                    value => {
                        if let Ok(parsed) = value.parse::<f64>() {
                            min_confidence = parsed;
                        }
                        index += 1;
                    }
                }
            }
            let summary = merge_all_ai_groups(min_confidence, &disposal, limit)?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        "nas-preprocess" => {
            let values = args.collect::<Vec<_>>();
            let mut limit: Option<usize> = None;
            let mut session_ids = Vec::new();
            let mut video_ids = Vec::new();
            let mut index = 0usize;
            while index < values.len() {
                if values[index] == "--limit" {
                    if let Some(value) = values.get(index + 1) {
                        limit = value.parse::<usize>().ok();
                    }
                    index += 2;
                    continue;
                }
                if values[index] == "--id" {
                    if let Some(value) = values.get(index + 1) {
                        if let Ok(video_id) = value.parse::<i64>() {
                            video_ids.push(video_id);
                        }
                    }
                    index += 2;
                    continue;
                }
                if let Ok(session_id) = values[index].parse::<i64>() {
                    session_ids.push(session_id);
                }
                index += 1;
            }
            let session_ids = session_ids.into_iter().collect::<Vec<_>>();
            let conn = database::open_database()?;
            let mut videos = if session_ids.is_empty() {
                database::list_videos(&conn)?
            } else {
                database::list_videos_for_sessions(&conn, &session_ids)?
            };
            if !video_ids.is_empty() {
                videos.retain(|video| video.id.is_some_and(|id| video_ids.contains(&id)));
            }
            if let Some(limit) = limit {
                videos.truncate(limit);
            }
            let settings = operations::get_app_settings()?;
            let summary =
                nas_preprocess::preprocess_ai_frame_cache(&videos, &settings, |progress| {
                    eprintln!(
                        "nas-preprocess {}/{} skipped {} failed {}",
                        progress.imported, progress.total_videos, progress.skipped, progress.failed
                    );
                })?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        "local-preprocess" => {
            let values = args.collect::<Vec<_>>();
            let mut limit: Option<usize> = None;
            let mut session_ids = Vec::new();
            let mut video_ids = Vec::new();
            let mut options = local_preprocess::LocalPreprocessOptions::default();
            let mut index = 0usize;
            while index < values.len() {
                match values[index].as_str() {
                    "--limit" => {
                        if let Some(value) = values.get(index + 1) {
                            limit = value.parse::<usize>().ok();
                        }
                        index += 2;
                    }
                    "--id" => {
                        if let Some(value) = values.get(index + 1) {
                            if let Ok(video_id) = value.parse::<i64>() {
                                video_ids.push(video_id);
                            }
                        }
                        index += 2;
                    }
                    "--workers" => {
                        if let Some(value) = values.get(index + 1) {
                            if let Ok(workers) = value.parse::<usize>() {
                                options.video_workers = workers;
                            }
                        }
                        index += 2;
                    }
                    "--overlap-start-percent" => {
                        if let Some(value) = values.get(index + 1) {
                            if let Ok(percent) = value.parse::<usize>() {
                                options.overlap_start_percent = percent;
                            }
                        }
                        index += 2;
                    }
                    "--process-workers" => {
                        if let Some(value) = values.get(index + 1) {
                            if let Ok(workers) = value.parse::<usize>() {
                                options.process_workers = workers;
                            }
                        }
                        index += 2;
                    }
                    "--frame-workers" => {
                        if let Some(value) = values.get(index + 1) {
                            if let Ok(workers) = value.parse::<usize>() {
                                options.frame_workers = workers;
                            }
                        }
                        index += 2;
                    }
                    "--temp" => {
                        if let Some(value) = values.get(index + 1) {
                            options.temp_dir = Some(value.into());
                        }
                        index += 2;
                    }
                    "--secondary-temp" => {
                        if let Some(value) = values.get(index + 1) {
                            options.secondary_temp_dir = Some(value.into());
                        }
                        index += 2;
                    }
                    "--secondary-threshold-mb" => {
                        if let Some(value) = values.get(index + 1) {
                            if let Ok(threshold_mb) = value.parse::<usize>() {
                                options.secondary_threshold_mb = threshold_mb;
                            }
                        }
                        index += 2;
                    }
                    value => {
                        if let Ok(session_id) = value.parse::<i64>() {
                            session_ids.push(session_id);
                        }
                        index += 1;
                    }
                }
            }

            let conn = database::open_database()?;
            let mut videos = if session_ids.is_empty() {
                database::list_videos(&conn)?
            } else {
                database::list_videos_for_sessions(&conn, &session_ids)?
            };
            if !video_ids.is_empty() {
                videos.retain(|video| video.id.is_some_and(|id| video_ids.contains(&id)));
            }
            if let Some(limit) = limit {
                videos.truncate(limit);
            }
            let settings = operations::get_app_settings()?;
            let summary = local_preprocess::preprocess_ai_frame_cache(
                &videos,
                &settings,
                options,
                |progress| {
                    let done = progress
                        .imported
                        .saturating_add(progress.skipped)
                        .saturating_add(progress.failed)
                        .min(progress.total_videos);
                    eprintln!(
                        "local-preprocess {} {}/{} total {} skipped {} failed {}",
                        progress.phase,
                        done,
                        progress.total_videos,
                        progress.total_videos,
                        progress.skipped,
                        progress.failed
                    );
                    if let Some(path) = progress.current_path {
                        eprintln!("  {path}");
                    }
                },
            )?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
        }
        "plan" => {
            let high_quality_id = args
                .next()
                .ok_or_else(|| anyhow::anyhow!("missing high quality video id"))?
                .parse::<i64>()?;
            let naming_source_id = args
                .next()
                .ok_or_else(|| anyhow::anyhow!("missing naming source video id"))?
                .parse::<i64>()?;
            let plan = operations::create_replacement_plan(high_quality_id, naming_source_id)?;
            println!("{}", serde_json::to_string_pretty(&plan)?);
        }
        _ => {
            eprintln!(
                "usage: dvs-cli [status|scan|videos|groups|ai-groups|ai-index|clear-index|merge-all-ai-groups|nas-preprocess|local-preprocess|plan] [source|ids|confidence threshold] [--workers 1|2] [--overlap-start-percent n] [--process-workers n]"
            );
            std::process::exit(2);
        }
    }

    Ok(())
}

#[derive(Debug)]
enum MergeTask {
    Merge {
        title: String,
        keeper_id: i64,
        source_id: i64,
        extra_ids: Vec<i64>,
    },
    File {
        title: String,
        ids: Vec<i64>,
    },
}

fn clear_generated_caches() -> anyhow::Result<Vec<String>> {
    let mut removed = Vec::new();
    for dir_name in [
        "ai-frame-cache",
        "thumbnails",
        "local-video-staging",
        "nas-frame-import",
    ] {
        let target = paths::data_dir().join(dir_name);
        if target.exists() {
            fs::remove_dir_all(&target)
                .with_context(|| format!("remove generated cache {}", target.display()))?;
            removed.push(target.display().to_string());
        }
    }
    paths::ensure_data_dirs()?;
    Ok(removed)
}

fn merge_all_ai_groups(
    min_confidence: f64,
    disposal: &str,
    limit: Option<usize>,
) -> anyhow::Result<serde_json::Value> {
    let disposal = match disposal {
        "backup" | "delete" => disposal,
        _ => "backup",
    };
    let mut conn = database::open_database()?;
    let videos = database::list_videos(&conn)?;
    let settings = operations::get_app_settings()?;
    let scan_source_dirs = database::list_scan_sessions(&conn)?
        .into_iter()
        .map(|session| session.source)
        .collect::<Vec<_>>();
    let mut groups = ai::build_ai_match_groups(
        &mut conn,
        &videos,
        &settings,
        min_confidence.clamp(0.0, 1.0),
    )?;
    groups.sort_by(|left, right| {
        right
            .confidence
            .total_cmp(&left.confidence)
            .then_with(|| right.reclaimable_bytes.cmp(&left.reclaimable_bytes))
    });
    if let Some(limit) = limit {
        groups.truncate(limit);
    }

    let mut tasks = Vec::new();
    let mut planned_skipped = 0usize;
    let mut claimed_video_ids = HashSet::new();
    for group in &groups {
        let Some(task) = plan_group_task(group, &settings.naming_source_dirs, &scan_source_dirs)
        else {
            continue;
        };
        let group_video_ids = group
            .items
            .iter()
            .filter_map(|item| item.video.id)
            .collect::<Vec<_>>();
        if group_video_ids
            .iter()
            .any(|id| claimed_video_ids.contains(id))
        {
            planned_skipped += 1;
            continue;
        }
        for id in group_video_ids {
            claimed_video_ids.insert(id);
        }
        tasks.push(task);
    }

    let mut completed = 0usize;
    let mut skipped = planned_skipped;
    let mut failed = 0usize;
    let mut failures = Vec::new();
    for task in &tasks {
        let indexed_ids = current_video_ids()?;
        let result = match task {
            MergeTask::Merge {
                keeper_id,
                source_id,
                extra_ids,
                ..
            } => {
                let active_extra_ids = extra_ids
                    .iter()
                    .copied()
                    .filter(|id| indexed_ids.contains(id))
                    .collect::<Vec<_>>();
                if !indexed_ids.contains(keeper_id) || !indexed_ids.contains(source_id) {
                    skipped += 1;
                    continue;
                }
                operations::execute_merge_selection(
                    *keeper_id,
                    *source_id,
                    active_extra_ids,
                    disposal,
                    if disposal == "backup" {
                        "MERGE_BACKUP"
                    } else {
                        "MERGE_DELETE"
                    },
                    None,
                )
            }
            MergeTask::File { ids, .. } => {
                let active_ids = ids
                    .iter()
                    .copied()
                    .filter(|id| indexed_ids.contains(id))
                    .collect::<Vec<_>>();
                if active_ids.is_empty() {
                    skipped += 1;
                    continue;
                }
                operations::execute_file_action(
                    active_ids,
                    disposal,
                    if disposal == "backup" {
                        "MOVE"
                    } else {
                        "DELETE"
                    },
                )
            }
        };

        match result {
            Ok(outcome) if outcome.status == "skipped-stale" => skipped += 1,
            Ok(_) => completed += 1,
            Err(error) => {
                failed += 1;
                if failures.len() < 8 {
                    failures.push(format!("{}: {error}", task_title(task)));
                }
            }
        }
    }

    Ok(json!({
        "status": if failed == 0 { "completed" } else { "completed-with-failures" },
        "confidence": min_confidence.clamp(0.0, 1.0),
        "disposal": disposal,
        "totalGroups": groups.len(),
        "plannedTasks": tasks.len(),
        "completed": completed,
        "skipped": skipped,
        "failed": failed,
        "failures": failures
    }))
}

fn current_video_ids() -> anyhow::Result<HashSet<i64>> {
    let conn = database::open_database()?;
    Ok(database::list_videos(&conn)?
        .into_iter()
        .filter_map(|video| video.id)
        .collect())
}

fn plan_group_task(
    group: &MatchGroup,
    naming_source_dirs: &[String],
    scan_source_dirs: &[String],
) -> Option<MergeTask> {
    let keeper_id = group
        .recommended_video_id
        .or_else(|| group.items.iter().find_map(|item| item.video.id))?;
    let source_id = naming_source_for_group(group, naming_source_dirs, scan_source_dirs)?
        .video
        .id?;
    let non_keeper_ids = group
        .items
        .iter()
        .filter_map(|item| item.video.id)
        .filter(|id| *id != keeper_id)
        .collect::<Vec<_>>();
    if non_keeper_ids.is_empty() {
        return None;
    }
    if source_id != keeper_id {
        Some(MergeTask::Merge {
            title: group.title.clone(),
            keeper_id,
            source_id,
            extra_ids: non_keeper_ids
                .into_iter()
                .filter(|id| *id != source_id)
                .collect(),
        })
    } else {
        Some(MergeTask::File {
            title: group.title.clone(),
            ids: non_keeper_ids,
        })
    }
}

fn naming_source_for_group<'a>(
    group: &'a MatchGroup,
    naming_source_dirs: &[String],
    scan_source_dirs: &[String],
) -> Option<&'a MatchItem> {
    group.items.iter().min_by(|left, right| {
        let left_priority = naming_folder_priority(left, naming_source_dirs);
        let right_priority = naming_folder_priority(right, naming_source_dirs);
        file_name_chinese_prefix_rank(&left.video.file_name)
            .cmp(&file_name_chinese_prefix_rank(&right.video.file_name))
            .then_with(|| {
                path_source_rename_suffix_priority(left, &group.items, scan_source_dirs).cmp(
                    &path_source_rename_suffix_priority(right, &group.items, scan_source_dirs),
                )
            })
            .then_with(|| left_priority.cmp(&right_priority))
            .then_with(|| {
                score_readable_file_name(&right.video.file_name)
                    .cmp(&score_readable_file_name(&left.video.file_name))
            })
            .then_with(|| {
                right
                    .video
                    .quality_score
                    .total_cmp(&left.video.quality_score)
            })
    })
}

fn file_name_chinese_prefix_rank(name: &str) -> usize {
    let (base, _) = split_file_name(name);
    let base = base.trim_start();
    let digit_end = base
        .char_indices()
        .take_while(|(_, ch)| ch.is_ascii_digit())
        .map(|(index, ch)| index + ch.len_utf8())
        .last()
        .unwrap_or(0);
    if digit_end == 0 {
        return 2;
    }
    let rest = base[digit_end..].trim_start();
    if let Some(inside) = rest
        .strip_prefix('\u{3010}')
        .and_then(|value| value.split('\u{3011}').next())
    {
        if inside.chars().any(is_cjk) {
            return 0;
        }
    }
    if rest.chars().next().is_some_and(is_cjk) {
        return 1;
    }
    2
}

fn is_cjk(ch: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&ch)
}

fn path_source_rename_suffix_priority(
    item: &MatchItem,
    items: &[MatchItem],
    scan_source_dirs: &[String],
) -> usize {
    let (base, extension) = split_file_name(&item.video.file_name);
    let base_key = base.to_lowercase();
    if let Some(suffixed_base) = rename_suffix_base(&item.video.file_name) {
        let has_unsuffixed_sibling = items.iter().any(|candidate| {
            if std::ptr::eq(candidate, item)
                || !videos_share_scan_source(&item.video, &candidate.video, scan_source_dirs)
            {
                return false;
            }
            let (candidate_base, candidate_extension) = split_file_name(&candidate.video.file_name);
            candidate_extension.eq_ignore_ascii_case(extension)
                && candidate_base.to_lowercase() == suffixed_base
        });
        if has_unsuffixed_sibling {
            return 2;
        }
    }

    let has_suffixed_sibling = items.iter().any(|candidate| {
        if std::ptr::eq(candidate, item)
            || !videos_share_scan_source(&item.video, &candidate.video, scan_source_dirs)
        {
            return false;
        }
        let (_, candidate_extension) = split_file_name(&candidate.video.file_name);
        candidate_extension.eq_ignore_ascii_case(extension)
            && rename_suffix_base(&candidate.video.file_name).as_deref() == Some(base_key.as_str())
    });

    if has_suffixed_sibling {
        0
    } else {
        1
    }
}

fn videos_share_scan_source(
    left: &VideoRecord,
    right: &VideoRecord,
    scan_source_dirs: &[String],
) -> bool {
    if scan_source_dirs.is_empty() {
        return path_key(&left.parent_path) == path_key(&right.parent_path);
    }
    scan_source_dirs
        .iter()
        .any(|source| path_inside_dir(&left.path, source) && path_inside_dir(&right.path, source))
}

fn split_file_name(name: &str) -> (&str, &str) {
    name.rsplit_once('.').unwrap_or((name, ""))
}

fn rename_suffix_base(name: &str) -> Option<String> {
    let (base, _) = split_file_name(name);
    let trimmed = base.trim_end();
    let stripped = trimmed
        .strip_suffix(')')
        .or_else(|| trimmed.strip_suffix('\u{ff09}'))?;
    let (open_index, open_char) = stripped
        .char_indices()
        .rev()
        .find(|(_, ch)| *ch == '(' || *ch == '\u{ff08}')?;
    let suffix = &stripped[open_index + open_char.len_utf8()..];
    if suffix.is_empty() || !suffix.chars().all(|ch| ch.is_ascii_digit()) || suffix == "0" {
        return None;
    }
    let original = stripped[..open_index].trim_end();
    if original.is_empty() {
        None
    } else {
        Some(original.to_lowercase())
    }
}

fn naming_folder_priority(item: &MatchItem, dirs: &[String]) -> usize {
    dirs.iter()
        .position(|dir| {
            path_inside_dir(&item.video.path, dir) || path_inside_dir(&item.video.parent_path, dir)
        })
        .unwrap_or(usize::MAX)
}

fn path_inside_dir(path: &str, dir: &str) -> bool {
    let file_key = path_key(path);
    let dir_key = path_key(dir);
    !dir_key.is_empty() && (file_key == dir_key || file_key.starts_with(&format!("{dir_key}\\")))
}

fn path_key(path: &str) -> String {
    path.trim()
        .replace('/', "\\")
        .trim_end_matches('\\')
        .to_ascii_lowercase()
}

fn score_readable_file_name(name: &str) -> i64 {
    let base = name.rsplit_once('.').map(|(base, _)| base).unwrap_or(name);
    let char_count = base.chars().count() as i64;
    let mut score = char_count.min(80);
    if base
        .chars()
        .any(|ch| ('\u{4e00}'..='\u{9fff}').contains(&ch))
    {
        score += 80;
    }
    if base.chars().any(|ch| ch.is_ascii_digit()) {
        score += 18;
    }
    if !base.is_empty() && base.chars().all(|ch| ch.is_ascii_digit()) {
        score -= 70;
    }
    if char_count < 6 {
        score -= 20;
    }
    score
}

fn task_title(task: &MergeTask) -> &str {
    match task {
        MergeTask::Merge { title, .. } | MergeTask::File { title, .. } => title,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_video(id: i64, path: &str) -> VideoRecord {
        let file_name = path.rsplit('\\').next().unwrap_or(path).to_string();
        let parent_path = path
            .rsplit_once('\\')
            .map(|(parent, _)| parent.to_string())
            .unwrap_or_default();
        let extension = file_name
            .rsplit_once('.')
            .map(|(_, extension)| extension.to_string())
            .unwrap_or_default();
        VideoRecord {
            id: Some(id),
            path: path.to_string(),
            file_name,
            parent_path,
            size_bytes: 1024,
            modified_unix_ms: 1,
            extension,
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

    fn test_item(id: i64, path: &str, quality_rank: usize) -> MatchItem {
        MatchItem {
            video: test_video(id, path),
            role: String::new(),
            quality_rank,
            match_ranges: Vec::new(),
            match_detail: None,
        }
    }

    fn test_group(items: Vec<MatchItem>) -> MatchGroup {
        MatchGroup {
            id: "group-test".to_string(),
            title: "test".to_string(),
            kind: "test".to_string(),
            confidence: 1.0,
            recommended_video_id: None,
            reclaimable_bytes: 0,
            item_count: items.len(),
            evidence: Vec::new(),
            report: String::new(),
            items,
        }
    }

    #[test]
    fn naming_source_prefers_unsuffixed_duplicate_suffix_original() {
        let group = test_group(vec![
            test_item(1, r"C:\Videos\Movie (1).mp4", 1),
            test_item(2, r"C:\Videos\Movie.mp4", 2),
            test_item(3, r"C:\Videos\Readable Long Title.mp4", 3),
        ]);

        let source = naming_source_for_group(&group, &[], &[]).unwrap();

        assert_eq!(source.video.id, Some(2));
    }

    #[test]
    fn naming_source_prefers_unsuffixed_full_width_duplicate_suffix_original() {
        let group = test_group(vec![
            test_item(1, r"C:\Videos\Clip（1）.mp4", 1),
            test_item(2, r"C:\Videos\Clip.mp4", 2),
        ]);

        let source = naming_source_for_group(&group, &[], &[]).unwrap();

        assert_eq!(source.video.id, Some(2));
    }

    #[test]
    fn naming_source_prefers_unsuffixed_duplicate_under_same_scan_source() {
        let group = test_group(vec![
            test_item(1, r"C:\Videos\A\Clip (1).mp4", 1),
            test_item(2, r"C:\Videos\B\Clip.mp4", 2),
            test_item(3, r"C:\Other\Clip.mp4", 3),
        ]);
        let scan_sources = vec![r"C:\Videos".to_string()];

        let source = naming_source_for_group(&group, &[], &scan_sources).unwrap();

        assert_eq!(source.video.id, Some(2));
    }

    #[test]
    fn naming_source_prefers_numeric_bracket_chinese_before_suffix_rule() {
        let group = test_group(vec![
            test_item(1, r"C:\Videos\channel@example (10).mp4", 1),
            test_item(2, r"C:\Videos\channel@example.mp4", 2),
            test_item(3, r"C:\Videos\91【示例】旅行的回忆.mp4", 3),
        ]);

        let source = naming_source_for_group(&group, &[], &[]).unwrap();

        assert_eq!(source.video.id, Some(3));
    }

    #[test]
    fn naming_source_prefers_numeric_chinese_before_suffix_rule_without_bracket_match() {
        let group = test_group(vec![
            test_item(1, r"C:\Videos\channel@example (10).mp4", 1),
            test_item(2, r"C:\Videos\channel@example.mp4", 2),
            test_item(3, r"C:\Videos\91旅行的回忆.mp4", 3),
        ]);

        let source = naming_source_for_group(&group, &[], &[]).unwrap();

        assert_eq!(source.video.id, Some(3));
    }

    #[test]
    fn naming_source_suffix_rule_precedes_configured_source_dirs() {
        let group = test_group(vec![
            test_item(1, r"C:\Videos\Preferred\Clip (1).mp4", 1),
            test_item(2, r"C:\Videos\Other\Clip.mp4", 2),
        ]);
        let naming_dirs = vec![r"C:\Videos\Preferred".to_string()];
        let scan_sources = vec![r"C:\Videos".to_string()];

        let source = naming_source_for_group(&group, &naming_dirs, &scan_sources).unwrap();

        assert_eq!(source.video.id, Some(2));
    }
}
