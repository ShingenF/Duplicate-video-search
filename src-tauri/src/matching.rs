use crate::models::{MatchGroup, MatchItem, VideoRecord};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
struct PairMatch {
    left: usize,
    right: usize,
    kind: String,
    confidence: f64,
    evidence: Vec<String>,
}

#[derive(Debug, Clone, Copy, Default)]
struct TemporalMatch {
    ordered_points: usize,
    informative_left: usize,
    informative_right: usize,
    average_distance: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct MatchScore {
    count: usize,
    total_distance: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct MatchOptions {
    pub compare_within_same_folder: bool,
    pub min_confidence: f64,
    pub temporal_hash_threshold: u32,
    pub min_temporal_match_points: usize,
    pub allowed_unmatched_sample_frames: usize,
    pub keeper_size_priority_duration_seconds: f64,
}

impl MatchOptions {
    fn normalized(self) -> Self {
        Self {
            compare_within_same_folder: self.compare_within_same_folder,
            min_confidence: self.min_confidence.clamp(0.0, 1.0),
            temporal_hash_threshold: self.temporal_hash_threshold.clamp(6, 24),
            min_temporal_match_points: self.min_temporal_match_points.clamp(1, 10),
            allowed_unmatched_sample_frames: self.allowed_unmatched_sample_frames.clamp(0, 20),
            keeper_size_priority_duration_seconds: if self
                .keeper_size_priority_duration_seconds
                .is_finite()
            {
                self.keeper_size_priority_duration_seconds
                    .clamp(0.0, 86_400.0)
            } else {
                300.0
            },
        }
    }
}

impl Default for MatchOptions {
    fn default() -> Self {
        Self {
            compare_within_same_folder: false,
            min_confidence: 0.90,
            temporal_hash_threshold: 8,
            min_temporal_match_points: 3,
            allowed_unmatched_sample_frames: 14,
            keeper_size_priority_duration_seconds: 300.0,
        }
    }
}

#[derive(Debug)]
struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
        }
    }

    fn find(&mut self, value: usize) -> usize {
        if self.parent[value] != value {
            self.parent[value] = self.find(self.parent[value]);
        }
        self.parent[value]
    }

    fn union(&mut self, left: usize, right: usize) {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root != right_root {
            self.parent[right_root] = left_root;
        }
    }
}

pub fn build_match_groups(videos: &[VideoRecord]) -> Vec<MatchGroup> {
    build_match_groups_with_min_confidence(videos, 0.90)
}

pub fn build_match_groups_with_min_confidence(
    videos: &[VideoRecord],
    min_confidence: f64,
) -> Vec<MatchGroup> {
    build_match_groups_with_options(
        videos,
        MatchOptions {
            min_confidence,
            ..MatchOptions::default()
        },
    )
}

pub fn build_match_groups_with_options(
    videos: &[VideoRecord],
    options: MatchOptions,
) -> Vec<MatchGroup> {
    let options = options.normalized();
    let min_confidence = options.min_confidence;
    let grouping_confidence = min_confidence.min(0.50);
    let mut pairs = Vec::new();
    for left in 0..videos.len() {
        for right in (left + 1)..videos.len() {
            if options.compare_within_same_folder
                && !crate::paths::same_parent_folder(&videos[left].path, &videos[right].path)
            {
                continue;
            }
            if let Some(pair) = compare_pair(videos, left, right, options)
                .filter(|pair| pair.confidence >= grouping_confidence)
            {
                pairs.push(pair);
            }
        }
    }

    let mut union = UnionFind::new(videos.len());
    for pair in &pairs {
        union.union(pair.left, pair.right);
    }

    let mut grouped: HashMap<usize, Vec<usize>> = HashMap::new();
    for pair in &pairs {
        let root = union.find(pair.left);
        grouped
            .entry(root)
            .or_default()
            .extend([pair.left, pair.right]);
    }

    let mut groups = Vec::new();
    for (index, (_, mut members)) in grouped.into_iter().enumerate() {
        members.sort_unstable();
        members.dedup();
        if members.len() < 2 {
            continue;
        }

        let member_set = members.iter().copied().collect::<HashSet<_>>();
        let group_pairs = pairs
            .iter()
            .filter(|pair| member_set.contains(&pair.left) && member_set.contains(&pair.right))
            .collect::<Vec<_>>();
        let visible_pairs = group_pairs
            .iter()
            .copied()
            .filter(|pair| pair.confidence >= min_confidence)
            .collect::<Vec<_>>();
        if visible_pairs.is_empty() {
            continue;
        }

        let mut sorted_members = members.clone();
        sorted_members.sort_by(|a, b| {
            compare_quality(
                &videos[*b],
                &videos[*a],
                options.keeper_size_priority_duration_seconds,
            )
        });

        let recommended = sorted_members.first().copied();
        let recommended_video_id = recommended.and_then(|idx| videos[idx].id);
        let reclaimable_bytes = sorted_members
            .iter()
            .skip(1)
            .map(|idx| videos[*idx].size_bytes)
            .sum::<u64>();
        let confidence = max_confidence(&visible_pairs);
        let kind = dominant_kind(&group_pairs);
        let evidence = collect_evidence(&group_pairs);
        let title = group_title(&sorted_members, videos);

        let items = sorted_members
            .iter()
            .enumerate()
            .map(|(rank, idx)| MatchItem {
                video: videos[*idx].clone(),
                role: if Some(*idx) == recommended {
                    "推荐保留".to_string()
                } else if kind.contains("片段") {
                    "疑似片段".to_string()
                } else {
                    "可替换命名来源".to_string()
                },
                quality_rank: rank + 1,
                match_ranges: Vec::new(),
                match_detail: None,
            })
            .collect::<Vec<_>>();

        let report = build_report(
            index + 1,
            &kind,
            confidence,
            &evidence,
            &items,
            reclaimable_bytes,
        );

        groups.push(MatchGroup {
            id: format!("group-{index:03}", index = index + 1),
            title,
            kind,
            confidence,
            recommended_video_id,
            reclaimable_bytes,
            item_count: items.len(),
            evidence,
            report,
            items,
        });
    }

    groups.sort_by(|a, b| {
        b.reclaimable_bytes.cmp(&a.reclaimable_bytes).then_with(|| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    });
    groups
}

fn compare_quality(
    left: &VideoRecord,
    right: &VideoRecord,
    size_priority_seconds: f64,
) -> Ordering {
    if durations_within(left, right, size_priority_seconds) {
        return left
            .size_bytes
            .cmp(&right.size_bytes)
            .then_with(|| pixel_count(left).cmp(&pixel_count(right)))
            .then_with(|| bitrate_bucket(left.bitrate).cmp(&bitrate_bucket(right.bitrate)))
            .then_with(|| {
                frame_rate_bucket(left.frame_rate).cmp(&frame_rate_bucket(right.frame_rate))
            })
            .then_with(|| right.path.cmp(&left.path));
    }
    quality_sort_key(left)
        .cmp(&quality_sort_key(right))
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
    (left_duration - right_duration).abs() <= seconds
}

fn quality_sort_key(video: &VideoRecord) -> (i64, u64, u64, u64, i64) {
    (
        duration_bucket(video.duration_seconds),
        pixel_count(video),
        bitrate_bucket(video.bitrate),
        size_bucket(video.size_bytes),
        frame_rate_bucket(video.frame_rate),
    )
}

fn duration_bucket(duration: Option<f64>) -> i64 {
    duration
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(|value| (value / 10.0).round() as i64)
        .unwrap_or(0)
}

fn bitrate_bucket(bitrate: Option<u64>) -> u64 {
    bitrate.unwrap_or(0) / 100_000
}

fn size_bucket(size_bytes: u64) -> u64 {
    size_bytes / 1_048_576
}

fn frame_rate_bucket(frame_rate: Option<f64>) -> i64 {
    frame_rate
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(|value| (value * 10.0).round() as i64)
        .unwrap_or(0)
}

fn pixel_count(video: &VideoRecord) -> u64 {
    video.width.unwrap_or(0) as u64 * video.height.unwrap_or(0) as u64
}

fn group_title(members: &[usize], videos: &[VideoRecord]) -> String {
    members
        .iter()
        .map(|idx| {
            let stem = file_stem(&videos[*idx].file_name);
            (title_score(&stem), stem)
        })
        .max_by(|a, b| a.0.cmp(&b.0))
        .map(|(_, title)| title)
        .unwrap_or_else(|| "Untitled group".to_string())
}

fn file_stem(file_name: &str) -> String {
    file_name
        .rsplit_once('.')
        .map(|(stem, _)| stem)
        .unwrap_or(file_name)
        .trim()
        .to_string()
}

fn title_score(title: &str) -> i32 {
    let has_cjk = title
        .chars()
        .any(|ch| ('\u{4e00}'..='\u{9fff}').contains(&ch));
    let has_digit = title.chars().any(|ch| ch.is_ascii_digit());
    let numeric_only = title.chars().all(|ch| ch.is_ascii_digit());
    let readable_len = (4..=90).contains(&title.chars().count());

    let mut score = 0;
    if has_cjk {
        score += 100;
    }
    if has_digit {
        score += 15;
    }
    if readable_len {
        score += 10;
    }
    if !numeric_only {
        score += 30;
    }
    score
}

fn compare_pair(
    videos: &[VideoRecord],
    left: usize,
    right: usize,
    options: MatchOptions,
) -> Option<PairMatch> {
    let a = &videos[left];
    let b = &videos[right];

    if a.scan_status != "ok" || b.scan_status != "ok" {
        return None;
    }

    let duration_delta = duration_delta(a, b);
    if a.partial_hash.is_some()
        && a.partial_hash == b.partial_hash
        && a.size_bytes == b.size_bytes
        && duration_delta.unwrap_or(0.0) < 0.25
    {
        return Some(PairMatch {
            left,
            right,
            kind: "完全重复".to_string(),
            confidence: 1.0,
            evidence: vec!["分段文件哈希一致".to_string(), "时长和大小一致".to_string()],
        });
    }

    let aligned = aligned_hash_distance(&a.sample_hashes, &b.sample_hashes);
    let temporal_hash_threshold = options.temporal_hash_threshold.clamp(6, 24);
    let temporal_match =
        temporal_hash_match(&a.sample_hashes, &b.sample_hashes, temporal_hash_threshold);
    let close_points = temporal_match.ordered_points;
    let informative_min = temporal_match
        .informative_left
        .min(temporal_match.informative_right);
    let temporal_coverage = if informative_min == 0 {
        0.0
    } else {
        close_points as f64 / informative_min as f64
    };
    let required_temporal_points = required_temporal_points(
        informative_min,
        options.min_temporal_match_points,
        options.allowed_unmatched_sample_frames,
    );
    let aligned_close_points =
        aligned_close_points(&a.sample_hashes, &b.sample_hashes, temporal_hash_threshold);
    let title_overlap = title_overlap_score(a, b);
    if let Some((average, compared)) = aligned {
        let duration_ratio = duration_ratio(a, b).unwrap_or(0.0);
        if compared >= 2 && average <= 10.0 && duration_ratio >= 0.94 {
            let confidence =
                (0.98 - average / 90.0 - (1.0 - duration_ratio) / 2.0).clamp(0.72, 0.98);
            return Some(PairMatch {
                left,
                right,
                kind: "疑似同源转码".to_string(),
                confidence,
                evidence: vec![
                    format!("抽样帧平均汉明距离 {:.1}", average),
                    format!("时长重合比例 {:.0}%", duration_ratio * 100.0),
                ],
            });
        }
        if compared >= 4 && aligned_close_points >= 2 && average <= 14.0 && duration_ratio >= 0.985
        {
            let confidence =
                (0.58 + aligned_close_points as f64 * 0.05 - average / 120.0).clamp(0.50, 0.72);
            return Some(PairMatch {
                left,
                right,
                kind: "疑似同源/错位抽样".to_string(),
                confidence,
                evidence: vec![
                    format!("对齐抽样近似点 {} 个", aligned_close_points),
                    format!("抽样帧平均汉明距离 {:.1}", average),
                    format!("时长重合比例 {:.0}%", duration_ratio * 100.0),
                ],
            });
        }
    }

    let ratio = duration_ratio(a, b).unwrap_or(1.0);
    if close_points >= required_temporal_points && (ratio < 0.94 || temporal_coverage >= 0.35) {
        let confidence = if ratio >= 0.94 {
            (0.58 + close_points as f64 * 0.045 - temporal_match.average_distance / 120.0)
                .clamp(0.50, 0.86)
        } else {
            (0.62 + close_points as f64 * 0.045 - temporal_match.average_distance / 120.0)
                .clamp(0.50, 0.84)
        };
        return Some(PairMatch {
            left,
            right,
            kind: "疑似片段/剪辑".to_string(),
            confidence,
            evidence: vec![
                format!("有序相似帧 {} 个", close_points),
                format!(
                    "允许未匹配帧 {} 个",
                    options.allowed_unmatched_sample_frames.clamp(0, 20)
                ),
                format!("时长比例 {:.0}%", ratio * 100.0),
            ],
        });
    }

    if close_points >= 2 && title_overlap >= 0.22 && ratio >= 0.60 {
        let confidence = (0.48 + close_points as f64 * 0.035 + title_overlap * 0.24
            - temporal_match.average_distance / 160.0)
            .clamp(0.50, 0.68);
        return Some(PairMatch {
            left,
            right,
            kind: "疑似片段/标题强相关".to_string(),
            confidence,
            evidence: vec![
                format!("有序相似帧 {} 个", close_points),
                format!("标题关键词重合 {:.0}%", title_overlap * 100.0),
                format!("时长比例 {:.0}%", ratio * 100.0),
            ],
        });
    }

    if close_points >= 1 && title_overlap >= 0.65 && ratio >= 0.60 {
        let confidence = (0.46 + title_overlap * 0.20 + close_points as f64 * 0.03
            - temporal_match.average_distance / 180.0)
            .clamp(0.50, 0.60);
        return Some(PairMatch {
            left,
            right,
            kind: "疑似片段/标题强相关".to_string(),
            confidence,
            evidence: vec![
                format!("有序相似帧 {} 个", close_points),
                format!("标题关键词重合 {:.0}%", title_overlap * 100.0),
                format!("时长比例 {:.0}%", ratio * 100.0),
            ],
        });
    }

    None
}

fn required_temporal_points(
    informative_min: usize,
    min_temporal_match_points: usize,
    allowed_unmatched_sample_frames: usize,
) -> usize {
    if informative_min == 0 {
        return min_temporal_match_points.clamp(1, 10);
    }
    informative_min
        .saturating_sub(allowed_unmatched_sample_frames.clamp(0, 20))
        .max(min_temporal_match_points.clamp(1, 10))
}

fn aligned_hash_distance(left: &[String], right: &[String]) -> Option<(f64, usize)> {
    let count = left.len().min(right.len());
    if count == 0 {
        return None;
    }

    let mut total = 0u32;
    let mut compared = 0usize;
    for index in 0..count {
        if !is_informative_hash(&left[index]) || !is_informative_hash(&right[index]) {
            continue;
        }
        if let Some(distance) = hamming_hex(&left[index], &right[index]) {
            total += distance;
            compared += 1;
        }
    }

    if compared == 0 {
        None
    } else {
        Some((total as f64 / compared as f64, compared))
    }
}

fn temporal_hash_match(left: &[String], right: &[String], threshold: u32) -> TemporalMatch {
    let left_points = informative_hash_points(left);
    let right_points = informative_hash_points(right);
    let left_len = left_points.len();
    let right_len = right_points.len();
    if left_len == 0 || right_len == 0 {
        return TemporalMatch {
            informative_left: left_len,
            informative_right: right_len,
            ..TemporalMatch::default()
        };
    }

    let mut dp = vec![vec![MatchScore::default(); right_len + 1]; left_len + 1];
    for left_index in 0..left_len {
        for right_index in 0..right_len {
            let mut best = better_match_score(
                dp[left_index][right_index + 1],
                dp[left_index + 1][right_index],
            );
            if let Some(distance) =
                hamming_hex(&left_points[left_index].1, &right_points[right_index].1)
            {
                if distance <= threshold {
                    best = better_match_score(
                        best,
                        MatchScore {
                            count: dp[left_index][right_index].count + 1,
                            total_distance: dp[left_index][right_index].total_distance + distance,
                        },
                    );
                }
            }
            dp[left_index + 1][right_index + 1] = best;
        }
    }

    let score = dp[left_len][right_len];
    TemporalMatch {
        ordered_points: score.count,
        informative_left: left_len,
        informative_right: right_len,
        average_distance: if score.count == 0 {
            64.0
        } else {
            score.total_distance as f64 / score.count as f64
        },
    }
}

fn better_match_score(left: MatchScore, right: MatchScore) -> MatchScore {
    if left.count > right.count
        || (left.count == right.count && left.total_distance <= right.total_distance)
    {
        left
    } else {
        right
    }
}

fn informative_hash_points(hashes: &[String]) -> Vec<(usize, String)> {
    hashes
        .iter()
        .enumerate()
        .filter(|(_, hash)| is_informative_hash(hash))
        .map(|(index, hash)| (index, hash.clone()))
        .collect()
}

fn is_informative_hash(hash: &str) -> bool {
    if hash.len() != 16 {
        return false;
    }
    let Some(value) = u64::from_str_radix(hash, 16).ok() else {
        return false;
    };
    let ones = value.count_ones();
    if !(12..=52).contains(&ones) {
        return false;
    }
    let mut byte_values = HashSet::new();
    for chunk in hash.as_bytes().chunks(2) {
        if let Ok(text) = std::str::from_utf8(chunk) {
            byte_values.insert(text.to_string());
        }
    }
    byte_values.len() > 3
}

fn aligned_close_points(left: &[String], right: &[String], threshold: u32) -> usize {
    left.iter()
        .zip(right.iter())
        .filter(|(left, right)| {
            if !is_informative_hash(left) || !is_informative_hash(right) {
                return false;
            }
            hamming_hex(left, right)
                .map(|distance| distance <= threshold)
                .unwrap_or(false)
        })
        .count()
}

fn title_overlap_score(left: &VideoRecord, right: &VideoRecord) -> f64 {
    let left_tokens = title_bigrams(&file_stem(&left.file_name));
    let right_tokens = title_bigrams(&file_stem(&right.file_name));
    if left_tokens.is_empty() || right_tokens.is_empty() {
        return 0.0;
    }
    let shared = left_tokens.intersection(&right_tokens).count();
    shared as f64 / left_tokens.len().min(right_tokens.len()) as f64
}

fn title_bigrams(title: &str) -> HashSet<String> {
    let chars = title
        .chars()
        .filter(|ch| ('\u{4e00}'..='\u{9fff}').contains(ch))
        .collect::<Vec<_>>();
    chars
        .windows(2)
        .map(|window| window.iter().collect::<String>())
        .filter(|token| !is_weak_title_bigram(token))
        .collect()
}

fn is_weak_title_bigram(token: &str) -> bool {
    matches!(token, "视频" | "合集" | "示例")
}

fn hamming_hex(left: &str, right: &str) -> Option<u32> {
    let left = u64::from_str_radix(left, 16).ok()?;
    let right = u64::from_str_radix(right, 16).ok()?;
    Some((left ^ right).count_ones())
}

fn duration_delta(left: &VideoRecord, right: &VideoRecord) -> Option<f64> {
    Some((left.duration_seconds? - right.duration_seconds?).abs())
}

fn duration_ratio(left: &VideoRecord, right: &VideoRecord) -> Option<f64> {
    let a = left.duration_seconds?;
    let b = right.duration_seconds?;
    let longer = a.max(b);
    let shorter = a.min(b);
    if longer <= 0.0 {
        None
    } else {
        Some(shorter / longer)
    }
}

fn max_confidence(pairs: &[&PairMatch]) -> f64 {
    if pairs.is_empty() {
        return 0.0;
    }
    pairs.iter().map(|pair| pair.confidence).fold(0.0, f64::max)
}

fn dominant_kind(pairs: &[&PairMatch]) -> String {
    if pairs.iter().any(|pair| pair.kind == "疑似片段/剪辑") {
        "疑似片段/剪辑".to_string()
    } else if pairs.iter().any(|pair| pair.kind == "疑似同源转码") {
        "疑似同源转码".to_string()
    } else {
        "完全重复".to_string()
    }
}

fn collect_evidence(pairs: &[&PairMatch]) -> Vec<String> {
    let mut evidence = Vec::new();
    for pair in pairs {
        for item in &pair.evidence {
            if !evidence.contains(item) {
                evidence.push(item.clone());
            }
            if evidence.len() >= 5 {
                return evidence;
            }
        }
    }
    evidence
}

fn build_report(
    index: usize,
    kind: &str,
    confidence: f64,
    evidence: &[String],
    items: &[MatchItem],
    reclaimable_bytes: u64,
) -> String {
    let mut lines = Vec::new();
    lines.push("Duplicate Video Search 报告".to_string());
    lines.push(format!("组别: group-{index:03}"));
    lines.push(format!("类型: {kind}"));
    lines.push(format!("置信度: {:.0}%", confidence * 100.0));
    lines.push(format!("估算可释放: {}", human_bytes(reclaimable_bytes)));
    lines.push(String::new());
    lines.push("文件:".to_string());
    for item in items {
        let video = &item.video;
        lines.push(format!("- {}: {}", item.role, video.path));
        lines.push(format!(
            "  规格: {}x{}, {:?}, {}, {}, {}",
            video.width.unwrap_or(0),
            video.height.unwrap_or(0),
            video.codec.as_deref().unwrap_or("-"),
            format_duration(video.duration_seconds),
            human_bytes(video.size_bytes),
            video
                .bitrate
                .map(|value| format!("{:.1} Mbps", value as f64 / 1_000_000.0))
                .unwrap_or_else(|| "-".to_string())
        ));
    }
    lines.push(String::new());
    lines.push("证据:".to_string());
    for item in evidence {
        lines.push(format!("- {item}"));
    }
    lines.join("\n")
}

fn format_duration(value: Option<f64>) -> String {
    value
        .map(|seconds| format!("{seconds:.1}s"))
        .unwrap_or_else(|| "-".to_string())
}

fn human_bytes(value: u64) -> String {
    let units = ["B", "KB", "MB", "GB", "TB"];
    let mut size = value as f64;
    let mut index = 0usize;
    while size >= 1024.0 && index < units.len() - 1 {
        size /= 1024.0;
        index += 1;
    }
    if index == 0 {
        format!("{} {}", value, units[index])
    } else {
        format!("{size:.1} {}", units[index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn video(id: i64, hashes: Vec<&str>, duration: f64, width: u32, size: u64) -> VideoRecord {
        VideoRecord {
            id: Some(id),
            path: format!("\\\\EXAMPLE-NAS\\Test\\video-{id}.mp4"),
            file_name: format!("video-{id}.mp4"),
            parent_path: "\\\\EXAMPLE-NAS\\Test".to_string(),
            size_bytes: size,
            modified_unix_ms: 0,
            extension: "mp4".to_string(),
            container_format: Some("mp4".to_string()),
            duration_seconds: Some(duration),
            width: Some(width),
            height: Some(1080),
            bitrate: Some(8_000_000),
            codec: Some("h264".to_string()),
            frame_rate: Some(30.0),
            audio_codec: Some("aac".to_string()),
            partial_hash: None,
            sample_hashes: hashes.into_iter().map(str::to_string).collect(),
            preview_images: Vec::new(),
            scan_status: "ok".to_string(),
            error: None,
            quality_score: width as f64 * 1000.0 + duration,
            scanned_at_unix_ms: 0,
        }
    }

    fn video_with_bitrate(
        id: i64,
        hashes: Vec<&str>,
        duration: f64,
        width: u32,
        size: u64,
        bitrate: u64,
    ) -> VideoRecord {
        let mut record = video(id, hashes, duration, width, size);
        record.bitrate = Some(bitrate);
        record.quality_score = 0.0;
        record
    }

    #[test]
    fn same_folder_filter_excludes_cross_folder_and_nested_duplicates() {
        let mut videos = (1..=4).map(|id| {
            let mut item = video(id, vec![], 100.0, 1920, 1000);
            item.partial_hash = Some("identical".into());
            item
        }).collect::<Vec<_>>();
        videos[0].path = r"C:\Videos\a.mp4".into();
        videos[1].path = "c:/videos/b.mp4".into();
        videos[2].path = r"C:\Videos\Child\c.mp4".into();
        videos[3].path = r"D:\Videos\d.mp4".into();
        assert_eq!(build_match_groups(&videos)[0].item_count, 4);
        let groups = build_match_groups_with_options(&videos, MatchOptions {
            compare_within_same_folder: true,
            ..MatchOptions::default()
        });
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].item_count, 2);
        assert!(groups[0].items.iter().all(|item| item.video.id.unwrap() <= 2));
        assert!(crate::paths::same_parent_folder(r"\\NAS\Share\a.mp4", "//nas/share/b.mp4"));
        assert!(!crate::paths::same_parent_folder(r"\\NAS\Share\a.mp4", r"\\NAS\Share2\b.mp4"));
    }

    #[test]
    fn groups_similar_transcodes() {
        let videos = vec![
            video(
                1,
                vec!["b50807e078ba934f", "59e0eb0094527faf"],
                100.0,
                3840,
                1000,
            ),
            video(
                2,
                vec!["b50807e078ba934e", "59e0eb0094527fae"],
                99.0,
                1920,
                900,
            ),
        ];
        let groups = build_match_groups(&videos);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].recommended_video_id, Some(1));
    }

    #[test]
    fn ignores_unrelated_hashes() {
        let videos = vec![
            video(
                1,
                vec!["b50807e078ba934f", "a8d19a3165630e9e"],
                100.0,
                1920,
                1000,
            ),
            video(
                2,
                vec!["2e979c77e32508d8", "d16863881cdaf727"],
                100.0,
                1920,
                900,
            ),
        ];
        assert!(build_match_groups(&videos).is_empty());
    }

    #[test]
    fn recommends_higher_bitrate_when_duration_and_resolution_are_equivalent() {
        let videos = vec![
            video_with_bitrate(
                1,
                vec!["b50807e078ba934f", "59e0eb0094527faf"],
                100.0,
                1920,
                400,
                4_000_000,
            ),
            video_with_bitrate(
                2,
                vec!["b50807e078ba934e", "59e0eb0094527fae"],
                100.4,
                1920,
                600,
                8_000_000,
            ),
        ];
        let groups = build_match_groups(&videos);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].recommended_video_id, Some(2));
    }

    #[test]
    fn recommends_larger_file_when_duration_is_close() {
        let videos = vec![
            video_with_bitrate(
                1,
                vec!["b50807e078ba934f", "59e0eb0094527faf"],
                5_000.0,
                3840,
                400,
                10_000_000,
            ),
            video_with_bitrate(
                2,
                vec!["b50807e078ba934e", "59e0eb0094527fae"],
                5_260.0,
                1920,
                900,
                4_000_000,
            ),
        ];
        let groups = build_match_groups(&videos);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].recommended_video_id, Some(2));
    }

    #[test]
    fn confidence_threshold_splits_weak_bridge_pairs() {
        let videos = vec![
            video(
                1,
                vec!["b50807e078ba934f", "59e0eb0094527faf"],
                100.0,
                1920,
                1000,
            ),
            video(
                2,
                vec!["b50807e078ba934e", "59e0eb0094527fae"],
                100.0,
                1920,
                900,
            ),
            video(
                3,
                vec!["b50807e078ba934f", "2e979c77e32508d8", "d16863881cdaf727"],
                90.0,
                1920,
                800,
            ),
        ];
        let groups = build_match_groups_with_min_confidence(&videos, 0.90);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].item_count, 2);
    }

    #[test]
    fn detects_offset_same_duration_samples_at_low_threshold() {
        let videos = vec![
            video(
                1,
                vec![
                    "73d49d546e8af1a6",
                    "55cfa06812feb547",
                    "75df28e101692280",
                    "7ec5f63fbcdc1369",
                    "cefda23f45b15983",
                ],
                1837.16,
                1920,
                1_453_384_252,
            ),
            video(
                2,
                vec![
                    "33f499146e9ae113",
                    "3daac9403ae6ad4f",
                    "c15bb27b05616a90",
                    "7ec5f63fbcdc1369",
                    "ceffa23f75b07987",
                ],
                1830.2,
                1920,
                802_675_344,
            ),
        ];

        let groups = build_match_groups_with_min_confidence(&videos, 0.50);
        assert_eq!(groups.len(), 1);
        assert!(groups[0].confidence >= 0.50);
    }

    #[test]
    fn detects_title_supported_partial_clip_at_low_threshold() {
        let mut organized = video(
            1,
            vec![
                "2db6cb02780cd321",
                "e9721fd62cc88371",
                "5ec5a8fb51b0cbf9",
                "7e4f772c2eca8191",
                "37a4c500719a7cd9",
            ],
            2082.986009,
            1920,
            325_496_004,
        );
        organized.file_name = "42【示例】旅行记录，山间步道，湖边日落，城市夜景，海边散步，森林露营.mp4".to_string();
        let mut incoming = video(
            2,
            vec![
                "fd274b82b81c57e5",
                "6cb3df94ba7e75d7",
                "7e47630e2eca8193",
                "3f80c504711a74dd",
                "a73c5198e2064dbf",
            ],
            2709.64,
            1920,
            621_869_135,
        );
        incoming.file_name = "A001 示例：山间步道+湖边日落+城市夜景+海边散步+森林露营.mp4".to_string();

        let groups = build_match_groups_with_min_confidence(&[organized, incoming], 0.50);
        assert_eq!(groups.len(), 1);
        assert!(groups[0].confidence >= 0.50);
    }

    #[test]
    fn allowed_unmatched_frames_controls_temporal_clip_matching() {
        let left_hashes = vec![
            "b50807e078ba934f",
            "59e0eb0094527faf",
            "2e979c77e32508d8",
            "d16863881cdaf727",
            "a8227ca4a95c24c7",
            "b46bca47c66ee15f",
            "068e924c07f04aa7",
            "bd176dd5bc693d3e",
            "b71569dfb4ad8050",
            "0cf1be88cb3e0ae9",
            "f296625495db5c26",
            "3eb426feb7427e9d",
            "6dc2eb263cabf350",
            "adf242cf79d969d7",
            "8354fd78f151d660",
            "967d08beb15eda2c",
            "80247980ca43ddfa",
            "b61f098bc3321836",
            "fd572d157ca95d9e",
            "46cef22c27d04aa7",
        ];
        let right_hashes = vec![
            "b50807e078ba934f",
            "59e0eb0094527faf",
            "2e979c77e32508d8",
            "d16863881cdaf727",
            "a8227ca4a95c24c7",
            "b46bca47c66ee15f",
            "57dd835b56a3db38",
            "38b22cf4b94c7497",
            "8f7afd79d77ff04f",
            "47f6d7da1fbf388e",
            "c0b5fa550df89105",
            "673d8195120bf701",
            "2c91a195f0328741",
            "b5089b66b47ef701",
            "199143ac18c81bc7",
            "5df2e194f15169d7",
            "a61f14ff6bf89105",
            "2e971455c1e92abe",
            "ac919146b7ac7e73",
            "6b0ffbcd958ed4bf",
        ];
        let videos = vec![
            video(1, left_hashes.clone(), 100.0, 1920, 1_000),
            video(2, right_hashes.clone(), 50.0, 1920, 900),
        ];

        let relaxed = build_match_groups_with_options(
            &videos,
            MatchOptions {
                min_confidence: 0.50,
                temporal_hash_threshold: 8,
                min_temporal_match_points: 3,
                allowed_unmatched_sample_frames: 14,
                keeper_size_priority_duration_seconds: 300.0,
                compare_within_same_folder: false,
            },
        );
        assert_eq!(relaxed.len(), 1);

        let strict = build_match_groups_with_options(
            &videos,
            MatchOptions {
                min_confidence: 0.50,
                temporal_hash_threshold: 8,
                min_temporal_match_points: 3,
                allowed_unmatched_sample_frames: 13,
                keeper_size_priority_duration_seconds: 300.0,
                compare_within_same_folder: false,
            },
        );
        assert!(strict.is_empty());
    }

    #[test]
    fn does_not_promote_low_information_style_matches_to_high_confidence() {
        let mut first = video(
            1,
            vec![
                "a13ad5006a024bbf",
                "975c39f0d22e758f",
                "970c61a8c2367d97",
                "65bc5184b2416b9f",
                "648d7180a6416b99",
                "42c9a46f10e0bf59",
                "66bd5988a6c36892",
                "64bc5da0d363e89e",
                "66bdd96473c3e81e",
                "43d09d542eef2416",
                "970c61e8d22e2597",
            ],
            1690.0,
            1280,
            1_514_829_974,
        );
        first.file_name = "series-3.mp4".to_string();
        let mut second = video(
            2,
            vec![
                "970c71a0c2267d87",
                "970c21e0da2e658f",
                "971469a8da366d8f",
                "971439a0da367d8f",
                "971479a0da367dc7",
                "970c71b8d23665d7",
                "970c69a0da3e7d8f",
                "970c61a0da3e7587",
                "970c69a0da367587",
                "970c69a0d23e7d8f",
                "970479a0da3e7d8f",
            ],
            1975.8,
            1920,
            1_269_859_894,
        );
        second.file_name = "series-6.mp4".to_string();

        let groups = build_match_groups_with_min_confidence(&[first, second], 0.89);
        assert!(groups.is_empty());
    }
}
