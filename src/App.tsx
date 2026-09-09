import {
  AlertTriangle,
  Archive,
  ArrowRightLeft,
  ChevronDown,
  ChevronRight,
  CheckCircle2,
  CheckSquare,
  Clipboard,
  Database,
  FileVideo,
  Folder,
  FolderOpen,
  FolderPlus,
  GripVertical,
  History,
  ImageIcon,
  MemoryStick,
  Play,
  RefreshCw,
  RotateCcw,
  Save,
  Search,
  Settings,
  ShieldCheck,
  Square,
  Trash2,
  X,
} from "lucide-react";
import { convertFileSrc } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { type CSSProperties, type KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  buildAiIndex,
  cancelCurrentWork,
  cleanupCompletedAiFrameCache,
  cleanupStorage,
  configureRamDisk,
  executeMergeSelection,
  executeBatchMergeSelection,
  deleteScanSessions,
  executeFileAction,
  getAiModelStatus,
  getAppSettings,
  getRamDiskStatus,
  getStorageUsage,
  getToolStatus,
  listMatchGroups,
  listOperationHistory,
  listScanSessions,
  listVideos,
  openRamDiskDriverDownload,
  openVideo,
  pickFolder,
  pickFolders,
  pruneStaleVideos,
  refreshIndexSources,
  releaseRamDisk,
  rollbackOperation,
  saveAppSettings,
  scanSources,
  vacuumDatabase,
} from "./api";
import type {
  AppSettings,
  AiIndexProgress,
  AiIndexSummary,
  AiModelStatus,
  BatchMergeTask,
  CompletedAiFrameCacheCleanupSummary,
  DeleteIndexProgress,
  MatchGroup,
  MatchItem,
  MatchRefreshProgress,
  OperationHistoryEntry,
  RamDiskStatus,
  ScanProgress,
  ScanSession,
  ScanSummary,
  StorageCleanupSummary,
  StorageUsageSummary,
  ToolStatus,
  VideoRecord,
} from "./types";

type View = "matches" | "library" | "ai" | "history" | "settings";
type GroupSort = "reclaimable" | "confidence" | "files";
type MatchFocusPane = "groups" | "videos";
type BatchDisposal = "backup" | "delete";

type ConfirmDialog = {
  title: string;
  body: string;
  confirmLabel: string;
  danger?: boolean;
};

type BatchProgress = {
  phase: string;
  total: number;
  processed: number;
  completed: number;
  skipped: number;
  failed: number;
  currentTitle?: string;
};

type RuntimeLogLevel = "active" | "success" | "warning" | "error" | "info";

type RuntimeLogEntry = {
  id: number;
  key: string;
  time: string;
  level: RuntimeLogLevel;
  title: string;
  detail?: string;
};

type BatchTaskLogStatus = "pending" | "running" | "completed" | "skipped" | "failed";

type BatchTaskLogEntry = {
  id: number;
  backendIndex?: number;
  title: string;
  status: BatchTaskLogStatus;
  reason?: string;
  keeperId?: number;
  namingSourceId?: number;
  filenameSourceId?: number | null;
  extraCount?: number;
};

type ScopeTreeNode = {
  key: string;
  label: string;
  path: string;
  children: ScopeTreeNode[];
  session?: ScanSession;
  sessionIds: number[];
};

type ScopeTreeRow = {
  key: string;
  node: ScopeTreeNode;
  depth: number;
};

type SelectionModifiers = {
  shiftKey: boolean;
  ctrlKey: boolean;
  metaKey: boolean;
};

const isTauri = () => "__TAURI_INTERNALS__" in window;

const SETTINGS_TEXT = {
  zh: {
    settingsTitle: "操作设置",
    aiTitle: "AI 索引与视觉匹配",
    matchesNav: "相似结果",
    libraryNav: "索引库",
    aiNav: "AI",
    historyNav: "日志",
    settingsNav: "设置",
    scanPlaceholder: "输入视频目录或测试集路径",
    chooseScanFolder: "选择扫描文件夹",
    addPath: "添加路径",
    stopCurrentWork: "终止当前扫描和 AI 索引任务",
    stopping: "正在终止",
    stopScan: "终止扫描",
    scanning: "扫描中",
    startScan: "开始扫描",
    scanQueue: "待扫描路径",
    scanQueueCount: "个路径将依次扫描",
    removePath: "移除路径",
    allIndexes: "全部索引",
    noPathSelected: "未选择路径",
    pathUnit: "个路径",
    indexed: "已索引",
    similarGroups: "相似组",
    reclaimable: "可释放",
    compareScope: "比对范围",
    saveSettings: "保存设置",
    saveAiSettings: "保存 AI 设置",
    checkModel: "检查模型",
    rebuildAiIndex: "重建 AI 索引",
    buildAiIndex: "构建 AI 索引",
    select: "选择",
    add: "添加",
    delete: "删除",
    language: "界面语言",
    languageHelp: "切换后设置页和 AI 设置页会使用同一种语言显示。",
    chinese: "中文",
    english: "English",
    backupDir: "备份文件夹",
    backupDirHelp: "“移入备份”会把文件移动到这里。",
    keeperWindow: "推荐保留体积优先窗口（分钟）",
    keeperWindowHelp: "两个候选视频时长差不超过这个值时，优先推荐保留体积最大的文件。默认 5 分钟。",
    primaryCache: "一级缓存路径",
    primaryCacheHelp: "默认 Z:\\TEMP。能放进一级缓存盘的视频会优先使用这里；空间不足时会等待释放。",
    ramDisk: "自动内存缓存盘",
    ramDiskHelp: "首次配置会申请管理员权限、检查 ImDisk 驱动，并注册按需挂载/卸载任务。",
    ramDiskSize: "内存缓存容量 MB",
    ramDiskSizeHelp: "建议给 Windows 和 AI 推理保留足够内存；超大视频仍可转入二级缓存。",
    configureRamDisk: "应用内存盘设置",
    secondaryCache: "二级缓存路径",
    secondaryCacheHelp: "默认 D:\\TEMP。单个视频超过二级缓存阈值时，会自动改用这里。",
    namingDirs: "命名规范文件夹",
    namingDirsPlaceholder: "已整理、希望继承路径和文件名的文件夹",
    namingDirsHelp: "相似组里只要有候选位于这些文件夹内，默认就继承这里的路径和文件名；列表顺序越靠前优先级越高。",
    allowDirectDelete: "允许直接删除",
    allowDirectDeleteHelp: "关闭时，所有“仅删除”按钮都会保持不可用。",
    restrictTestPath: "锁定在测试路径",
    restrictTestPathHelp: "关闭后可扫描和操作测试路径之外的路径，用于真实环境 Beta 测试。",
    storageTitle: "存储占用",
    storageHelp: "这里只读取本地 data 目录和 SQLite 行数，不会扫描 NAS 视频源。",
    refreshStorage: "刷新容量",
    cleanupStorage: "清理旧缓存",
    cleanupCompletedFrameCache: "删除已完成 AI 索引的帧缓存",
    vacuumDatabase: "压缩数据库",
    storageTotal: "总计",
    storageRows: "数据库行数",
    storageNotLoaded: "点击刷新容量查看当前占用。",
    cleaningSummary: "最近一次清理",
    cleanupConfirmTitle: "清理旧缓存",
    cleanupConfirmBody: "将删除旧 AI 帧缓存、未引用缩略图、孤儿索引和非当前匹配参数的 pair-score。不会删除任何视频源文件。",
    cleanupConfirmLabel: "清理",
    cleanupCompletedFrameCacheConfirmTitle: "删除已完成 AI 索引的帧缓存",
    cleanupCompletedFrameCacheConfirmBody: "只会删除已经拥有当前模型 frame_embeddings 的视频帧缓存，不会删除源视频、缩略图或 embeddings。刷新相似结果仍然可用；以后强制重建 AI 索引会重新抽帧。",
    cleanupCompletedFrameCacheConfirmLabel: "删除帧缓存",
    completedFrameCacheCleanupSummary: "已完成 AI 帧缓存清理",
    vacuumConfirmTitle: "压缩数据库",
    vacuumConfirmBody: "VACUUM 会重写 SQLite 文件，期间数据库会被锁定。建议在没有扫描或刷新匹配时执行。",
    vacuumConfirmLabel: "压缩",
    aiVision: "启用 AI 视觉匹配",
    aiVisionHelp: "使用本地 ONNX 模型生成帧向量，只在本机处理视频帧。",
    aiAfterScan: "扫描后同步构建 AI 索引",
    aiAfterScanHelp: "扫描完成后自动为本次扫描路径建立 AI 索引，减少重复等待。",
    deleteFrameCacheAfterIndex: "AI 索引完成后删除帧缓存",
    deleteFrameCacheAfterIndexHelp: "成功写入 frame_embeddings 后删除该视频的 RGB 帧缓存，可显著降低 data\\ai-frame-cache 占用；强制重建 AI 索引时会重新抽帧。",
    localPipeline: "本地暂存流水线",
    localPipelineHelp: "扫描阶段只记录路径；AI 索引会把视频复制到本机缓存盘，在本机补齐普通索引元数据并抽帧，处理完成后自动删除暂存文件。",
    localVideoWorkers: "本地下载 workers",
    localVideoWorkersHelp: "1 为逐个下载。2 会在前一个接近完成时启动下一个。3-8 适合许多小视频。",
    overlapStart: "交叉下载启动进度 %",
    overlapStartHelp: "下载 workers 为 2-8 时生效。默认 95。",
    localProcessWorkers: "本地处理 workers",
    localProcessWorkersHelp: "已下载视频的元数据和普通索引处理并行数。默认 2。",
    gpuAiWorkers: "GPU AI workers",
    gpuAiWorkersHelp: "GPU/Auto 模式下同时跑 AI 的视频数量。默认 4，最大 64。",
    localFrameWorkers: "FFmpeg 帧处理线程",
    localFrameWorkersHelp: "每个暂存视频内部的 FFmpeg 并行度。默认 16，最大 64。",
    aiMatchWorkers: "AI 匹配 workers",
    aiMatchWorkersHelp: "刷新相似结果时使用的 CPU 线程数。默认 8。",
    primaryCacheFolder: "一级缓存文件夹",
    primaryCacheFolderHelp: "默认 Z:\\TEMP。能放入一级缓存的视频会继续使用这里，并发下载会等待空间释放。",
    secondaryCacheFolder: "二级缓存文件夹",
    secondaryCacheFolderHelp: "默认 D:\\TEMP。超过下面阈值的单个视频使用这里。",
    secondaryThreshold: "二级缓存阈值 MB",
    secondaryThresholdHelp: "单个视频大于这个值时使用二级缓存。一级缓存卡住时可调低。",
    aiModelPath: "AI 模型路径",
    aiModelPathReady: "可用",
    aiModelPathNotReady: "不可用",
    aiModelPathHelp: "模型文件应保存在当前项目的 models 文件夹下。",
    aiDevice: "AI 运行设备",
    auto: "自动",
    aiDeviceHelp: "GPU 会优先使用 DirectML；不可用时建议切回 CPU 验证。",
    aiFrameCount: "AI 抽帧数量",
    aiFrameCountHelp: "默认 128。增加后更容易识别剪辑片段，但索引更慢、缓存更大。",
    aiBatchSize: "AI 批大小",
    aiBatchSizeHelp: "默认 32。动态 batch 模型可提高到 64；显存不足时调低。",
    aiSimilarityThreshold: "AI 帧相似阈值",
    aiSimilarityThresholdHelp: "默认 0.86。越低越容易召回，0 会进入极宽松测试模式。",
    aiMinMatchedFrames: "AI 最少匹配帧",
    aiMinMatchedFramesHelp: "默认 8。提高会减少风格相近但内容不同的视频误配。",
    compareWithinSameFolder: "只比对同文件夹内文件",
    compareWithinSameFolderHelp: "仅比对直接位于同一文件夹的文件，子文件夹分别比对。保存后刷新匹配结果生效。",
    aiClipMatching: "启用 AI 片段匹配",
    aiClipMatchingHelp: "允许不同时间点的相似帧按顺序匹配。",
    lastAiIndex: "最近一次 AI 索引",
    noAiIndex: "还没有运行 AI 索引。",
  },
  en: {
    settingsTitle: "Operation Settings",
    aiTitle: "AI Index and Visual Matching",
    matchesNav: "Matches",
    libraryNav: "Library",
    aiNav: "AI",
    historyNav: "Logs",
    settingsNav: "Settings",
    scanPlaceholder: "Enter a video folder or test set path",
    chooseScanFolder: "Choose scan folder",
    addPath: "Add Path",
    stopCurrentWork: "Stop current scan and AI index task",
    stopping: "Stopping",
    stopScan: "Stop Scan",
    scanning: "Scanning",
    startScan: "Start Scan",
    scanQueue: "Scan Queue",
    scanQueueCount: "paths will scan in order",
    removePath: "Remove path",
    allIndexes: "All indexes",
    noPathSelected: "No path selected",
    pathUnit: "paths",
    indexed: "Indexed",
    similarGroups: "Similar Groups",
    reclaimable: "Reclaimable",
    compareScope: "Compare Scope",
    saveSettings: "Save Settings",
    saveAiSettings: "Save AI Settings",
    checkModel: "Check Model",
    rebuildAiIndex: "Rebuild AI Index",
    buildAiIndex: "Build AI Index",
    select: "Select",
    add: "Add",
    delete: "Delete",
    language: "Interface Language",
    languageHelp: "After switching, Settings and AI Settings use one consistent language.",
    chinese: "中文",
    english: "English",
    backupDir: "Backup Folder",
    backupDirHelp: "Move to Backup sends files here.",
    keeperWindow: "Prefer Larger File Window (minutes)",
    keeperWindowHelp: "When durations differ by no more than this value, prefer keeping the larger file. Default is 5 minutes.",
    primaryCache: "Primary Cache Path",
    primaryCacheHelp: "Default is Z:\\TEMP. Videos that fit the primary cache use it first; downloads wait when space is insufficient.",
    ramDisk: "Automatic RAM Cache Disk",
    ramDiskHelp: "First-time setup requests administrator permission, checks the ImDisk driver, and registers on-demand mount/release tasks.",
    ramDiskSize: "RAM Cache Size (MB)",
    ramDiskSizeHelp: "Leave enough memory for Windows and AI inference. Oversized videos can still use secondary cache.",
    configureRamDisk: "Apply RAM Disk Settings",
    secondaryCache: "Secondary Cache Path",
    secondaryCacheHelp: "Default is D:\\TEMP. A single video above the secondary threshold uses this path.",
    namingDirs: "Naming Source Folders",
    namingDirsPlaceholder: "Folders whose path and file names should be inherited",
    namingDirsHelp: "If a candidate is inside one of these folders, its path and file name are preferred. Earlier folders have higher priority.",
    allowDirectDelete: "Allow Direct Delete",
    allowDirectDeleteHelp: "When disabled, all delete-only buttons stay unavailable.",
    restrictTestPath: "Lock to Test Path",
    restrictTestPathHelp: "Disable this to scan and operate outside the test path during real beta testing.",
    storageTitle: "Storage Usage",
    storageHelp: "This reads the local data directory and SQLite row counts only. It does not scan NAS video sources.",
    refreshStorage: "Refresh Usage",
    cleanupStorage: "Clean Old Cache",
    cleanupCompletedFrameCache: "Delete Completed AI Frame Cache",
    vacuumDatabase: "Vacuum Database",
    storageTotal: "Total",
    storageRows: "Database Rows",
    storageNotLoaded: "Click Refresh Usage to inspect current storage.",
    cleaningSummary: "Last Cleanup",
    cleanupConfirmTitle: "Clean Old Cache",
    cleanupConfirmBody: "This deletes old AI frame cache, unreferenced thumbnails, orphan indexes, and pair-score rows from non-current match settings. It does not delete source videos.",
    cleanupConfirmLabel: "Clean",
    cleanupCompletedFrameCacheConfirmTitle: "Delete Completed AI Frame Cache",
    cleanupCompletedFrameCacheConfirmBody: "This only deletes frame cache for videos that already have current-model frame_embeddings. It does not delete source videos, thumbnails, or embeddings. Match refresh still works; forced AI rebuilds will extract frames again.",
    cleanupCompletedFrameCacheConfirmLabel: "Delete Frame Cache",
    completedFrameCacheCleanupSummary: "Completed AI Frame Cache Cleanup",
    vacuumConfirmTitle: "Vacuum Database",
    vacuumConfirmBody: "VACUUM rewrites SQLite and locks the database during the operation. Run it when scans and match refreshes are idle.",
    vacuumConfirmLabel: "Vacuum",
    aiVision: "Enable AI Visual Matching",
    aiVisionHelp: "Generate frame embeddings with a local ONNX model. Video frames stay on this machine.",
    aiAfterScan: "Build AI Index After Scan",
    aiAfterScanHelp: "After a scan, automatically build the AI index for this scan scope.",
    deleteFrameCacheAfterIndex: "Delete Frame Cache After AI Index",
    deleteFrameCacheAfterIndexHelp: "After frame_embeddings are written successfully, delete that video's RGB frame cache. This greatly reduces data\\ai-frame-cache usage; forced AI rebuilds will extract frames again.",
    localPipeline: "Local Staging Pipeline",
    localPipelineHelp: "Scanning records paths first. AI indexing copies videos to the local cache disk, fills normal metadata, extracts frames, then removes staged files.",
    localVideoWorkers: "Local Download Workers",
    localVideoWorkersHelp: "1 downloads serially. 2 starts the next near the end. 3-8 helps when there are many small videos.",
    overlapStart: "Download Overlap Start %",
    overlapStartHelp: "Used when download workers is 2-8. Default is 95.",
    localProcessWorkers: "Local Process Workers",
    localProcessWorkersHelp: "Parallel metadata and normal index processing for downloaded videos. Default is 2.",
    gpuAiWorkers: "GPU AI Workers",
    gpuAiWorkersHelp: "Videos processed by AI at the same time in GPU/Auto mode. Default is 4, max is 64.",
    localFrameWorkers: "FFmpeg Frame Threads",
    localFrameWorkersHelp: "FFmpeg parallelism inside each staged video. Default is 16, max is 64.",
    aiMatchWorkers: "AI Match Workers",
    aiMatchWorkersHelp: "CPU threads used when refreshing similarity results. Default is 8.",
    primaryCacheFolder: "Primary Cache Folder",
    primaryCacheFolderHelp: "Default is Z:\\TEMP. Videos that fit this disk keep using it, and downloads wait for free space.",
    secondaryCacheFolder: "Secondary Cache Folder",
    secondaryCacheFolderHelp: "Default is D:\\TEMP. Single videos larger than the threshold below use this disk.",
    secondaryThreshold: "Secondary Cache Threshold MB",
    secondaryThresholdHelp: "Single videos larger than this use the secondary cache. Lower it if primary cache pauses.",
    aiModelPath: "AI Model Path",
    aiModelPathReady: "Ready",
    aiModelPathNotReady: "Not Ready",
    aiModelPathHelp: "The model file should live under the current project's models folder.",
    aiDevice: "AI Device",
    auto: "Auto",
    aiDeviceHelp: "GPU prioritizes DirectML. Switch to CPU when GPU is unavailable or for verification.",
    aiFrameCount: "AI Frame Count",
    aiFrameCountHelp: "Default is 128. More frames improve clip recall but slow indexing and increase cache size.",
    aiBatchSize: "AI Batch Size",
    aiBatchSizeHelp: "Default is 32. Dynamic batch models can use 64; lower it if VRAM is insufficient.",
    aiSimilarityThreshold: "AI Frame Similarity Threshold",
    aiSimilarityThresholdHelp: "Default is 0.86. Lower values increase recall. 0 enters a very loose test mode.",
    aiMinMatchedFrames: "AI Minimum Matched Frames",
    aiMinMatchedFramesHelp: "Default is 8. Higher values reduce false positives from visually similar but different videos.",
    compareWithinSameFolder: "Compare only files in the same folder",
    compareWithinSameFolderHelp: "Compare files with the same direct parent folder; subfolders are separate. Save and refresh matches to apply.",
    aiClipMatching: "Enable AI Clip Matching",
    aiClipMatchingHelp: "Allows similar frames at different timestamps to match in order.",
    lastAiIndex: "Last AI Index",
    noAiIndex: "No AI index has been run yet.",
  },
} as const;

type SettingsLanguage = keyof typeof SETTINGS_TEXT;

function normalizeSettingsLanguage(value: string | undefined | null): SettingsLanguage {
  return value === "en" ? "en" : "zh";
}

function storageItemLabel(key: string, language: SettingsLanguage): string {
  const zh: Record<string, string> = {
    aiFrameCache: "AI 帧缓存",
    thumbnails: "缩略图",
    backups: "备份文件",
    operations: "操作记录",
    reports: "报告",
    tools: "工具文件",
    database: "SQLite 数据库",
    databaseWal: "SQLite WAL",
    databaseShm: "SQLite SHM",
  };
  const en: Record<string, string> = {
    aiFrameCache: "AI Frame Cache",
    thumbnails: "Thumbnails",
    backups: "Backups",
    operations: "Operation Logs",
    reports: "Reports",
    tools: "Tool Files",
    database: "SQLite Database",
    databaseWal: "SQLite WAL",
    databaseShm: "SQLite SHM",
  };
  return (language === "en" ? en : zh)[key] ?? key;
}

function fileCountText(count: number, language: SettingsLanguage): string {
  return language === "en"
    ? `${count.toLocaleString()} files`
    : `${count.toLocaleString()} 个文件`;
}

function storageDbLabel(key: string, language: SettingsLanguage): string {
  const zh: Record<string, string> = {
    videos: "视频",
    sessions: "扫描源",
    embeddings: "帧向量",
    pairScores: "相似缓存",
    edges: "匹配边",
    models: "模型",
  };
  const en: Record<string, string> = {
    videos: "videos",
    sessions: "sessions",
    embeddings: "embeddings",
    pairScores: "pair-score",
    edges: "edges",
    models: "models",
  };
  return (language === "en" ? en : zh)[key] ?? key;
}

function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let size = value;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size.toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function formatDuration(seconds: number | null): string {
  if (!seconds || seconds <= 0) return "-";
  const total = Math.round(seconds);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h > 0
    ? `${h}:${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`
    : `${m}:${s.toString().padStart(2, "0")}`;
}

function formatDate(value: number | null): string {
  if (!value) return "未完成";
  return new Date(value).toLocaleString();
}

function formatResolution(video: VideoRecord): string {
  return video.width && video.height ? `${video.width}x${video.height}` : "-";
}

function confidence(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function clampConfidencePercent(value: number): number {
  if (!Number.isFinite(value)) return 90;
  return Math.min(100, Math.max(0, value));
}

function previewSrc(path: string): string {
  return isTauri() ? convertFileSrc(path) : path;
}

function shortPath(path: string): string {
  if (path.length <= 82) return path;
  return `...${path.slice(-79)}`;
}

function compactProgressFileName(path: string, maxLength = 72): string {
  const normalized = path.replace(/\\/g, "/");
  const fileName = normalized.split("/").filter(Boolean).pop() ?? path;
  if (fileName.length <= maxLength) return fileName;

  const dotIndex = fileName.lastIndexOf(".");
  const extension = dotIndex > 0 ? fileName.slice(dotIndex) : "";
  const tailLength = Math.min(
    Math.max(extension.length + 8, 14),
    Math.floor(maxLength * 0.4),
  );
  const headLength = Math.max(12, maxLength - tailLength - 3);
  return `${fileName.slice(0, headLength)}...${fileName.slice(-tailLength)}`;
}

function formatMatchRefreshProgress(progress: MatchRefreshProgress): string {
  const processed = Math.min(progress.processedPairs, progress.totalPairs);
  const percent = matchRefreshPercent(progress);
  const phaseSpeed = Number.isFinite(progress.phasePairsPerSecond)
    ? Math.round(progress.phasePairsPerSecond).toLocaleString()
    : "0";
  const phaseTotal = progress.phaseTotal > 0 ? progress.phaseTotal : progress.totalPairs;
  const phaseProcessed = Math.min(progress.phaseProcessed, phaseTotal);
  const phaseText =
    phaseTotal > 0
      ? `${phaseProcessed.toLocaleString()}/${phaseTotal.toLocaleString()} (${percent}%)`
      : `${processed.toLocaleString()}/${progress.totalPairs.toLocaleString()} (${percent}%)`;
  const groupText =
    progress.groups > 0 ? ` · groups ${progress.groups.toLocaleString()}` : "";
  return `${progress.phase} ${phaseText} · ${phaseSpeed} pair/s · overall ${processed.toLocaleString()}/${progress.totalPairs.toLocaleString()} · cache ${progress.cachedPairs.toLocaleString()} · computed ${progress.computedPairs.toLocaleString()}${groupText}`;
}

function matchRefreshPercent(progress: MatchRefreshProgress): number {
  if (progress.phaseTotal > 0) {
    return Math.min(
      100,
      Math.max(0, Math.round((progress.phaseProcessed / progress.phaseTotal) * 100)),
    );
  }
  if (progress.totalPairs <= 0) return 100;
  return Math.min(100, Math.max(0, Math.round((progress.processedPairs / progress.totalPairs) * 100)));
}

function formatDeleteIndexProgress(progress: DeleteIndexProgress): string {
  const phaseLabels: Record<string, string> = {
    preparing: "准备删除索引",
    "collecting-sessions": "收集路径范围",
    "unlinking-sessions": "解除路径关联",
    "collecting-orphan-videos": "查找仅属于此范围的视频",
    "deleting-pair-scores-left": "删除相似缓存 A 侧引用",
    "deleting-pair-scores-right": "删除相似缓存 B 侧引用",
    "deleting-ai-metadata": "删除 AI 向量与匹配元数据",
    "deleting-videos": "删除视频索引记录",
    completed: "索引删除完成",
  };
  const label = phaseLabels[progress.phase] ?? progress.phase;
  const total = Math.max(progress.phaseTotal, 1);
  const processed = Math.min(progress.phaseProcessed, total);
  return `${label} ${processed}/${total} · 路径 ${progress.deletedSessions}/${progress.requestedSessions} · 相关视频 ${progress.affectedVideos} · 待清理 ${progress.deletedVideos}`;
}

function pathKey(path: string): string {
  return path.trim().replace(/\//g, "\\").replace(/\\+$/, "").toLowerCase();
}

function uniquePaths(paths: string[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const path of paths) {
    const trimmed = path.trim();
    if (!trimmed) continue;
    const key = pathKey(trimmed);
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(trimmed);
  }
  return result;
}

function uniqueNumbers(values: number[]): number[] {
  return Array.from(new Set(values));
}

function sameNumberList(left: number[] | null, right: number[] | null): boolean {
  if (left === null || right === null) return left === right;
  if (left.length !== right.length) return false;
  return left.every((value, index) => value === right[index]);
}

function pathSegments(path: string): string[] {
  const normalized = path.trim().replace(/\//g, "\\").replace(/\\+$/, "");
  if (!normalized) return [];
  if (normalized.startsWith("\\\\")) {
    const parts = normalized.replace(/^\\+/, "").split("\\").filter(Boolean);
    if (parts.length >= 2) return [`\\\\${parts[0]}\\${parts[1]}`, ...parts.slice(2)];
    return parts;
  }
  return normalized.split("\\").filter(Boolean);
}

function joinPathSegments(segments: string[]): string {
  if (segments.length === 0) return "";
  return segments.reduce((current, segment, index) => {
    if (index === 0) return segment;
    return current.endsWith("\\") ? `${current}${segment}` : `${current}\\${segment}`;
  }, "");
}

function buildScopeTree(sessions: ScanSession[]): ScopeTreeNode[] {
  const roots = new Map<string, ScopeTreeNode>();

  function getNode(map: Map<string, ScopeTreeNode>, label: string, path: string): ScopeTreeNode {
    const key = pathKey(path);
    let node = map.get(key);
    if (!node) {
      node = { key, label, path, children: [], sessionIds: [] };
      map.set(key, node);
    }
    return node;
  }

  function getChild(parent: ScopeTreeNode, label: string, path: string): ScopeTreeNode {
    const key = pathKey(path);
    let child = parent.children.find((item) => item.key === key);
    if (!child) {
      child = { key, label, path, children: [], sessionIds: [] };
      parent.children.push(child);
    }
    return child;
  }

  for (const session of sessions) {
    const segments = pathSegments(session.source);
    if (segments.length === 0) continue;
    let current: ScopeTreeNode | null = null;
    for (let index = 0; index < segments.length; index += 1) {
      const partial = joinPathSegments(segments.slice(0, index + 1));
      current =
        index === 0
          ? getNode(roots, segments[index], partial)
          : getChild(current as ScopeTreeNode, segments[index], partial);
    }
    if (current) {
      current.session = session;
      current.path = session.source;
    }
  }

  function finalize(nodes: ScopeTreeNode[]): ScopeTreeNode[] {
    return nodes
      .map((node) => {
        const children = finalize(node.children);
        const sessionIds = uniqueNumbers([
          ...(node.session ? [node.session.id] : []),
          ...children.flatMap((child) => child.sessionIds),
        ]);
        return { ...node, children, sessionIds };
      })
      .sort((left, right) => left.label.localeCompare(right.label, undefined, { numeric: true }));
  }

  return finalize(Array.from(roots.values()));
}

function flattenScopeTree(
  nodes: ScopeTreeNode[],
  collapsedKeys: Set<string>,
  depth = 0,
): ScopeTreeRow[] {
  const rows: ScopeTreeRow[] = [];
  for (const node of nodes) {
    rows.push({ key: node.key, node, depth });
    if (node.children.length > 0 && !collapsedKeys.has(node.key)) {
      rows.push(...flattenScopeTree(node.children, collapsedKeys, depth + 1));
    }
  }
  return rows;
}

function collectCollapsibleScopeKeys(nodes: ScopeTreeNode[]): string[] {
  return nodes.flatMap((node) => [
    ...(node.children.length > 0 ? [node.key] : []),
    ...collectCollapsibleScopeKeys(node.children),
  ]);
}

function clampNumberInput(raw: string, min: number, max: number, fallback: number): number {
  const parsed = Number(raw);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(max, Math.max(min, parsed));
}

function numberInputText(value: number): string {
  return Number.isFinite(value) ? String(value) : "";
}

function pathInsideDir(path: string, dir: string): boolean {
  const fileKey = pathKey(path);
  const dirKey = pathKey(dir);
  return Boolean(dirKey) && (fileKey === dirKey || fileKey.startsWith(`${dirKey}\\`));
}

function namingFolderPriority(item: MatchItem, dirs: string[]): number {
  const index = dirs.findIndex(
    (dir) => pathInsideDir(item.video.path, dir) || pathInsideDir(item.video.parentPath, dir),
  );
  return index === -1 ? Number.MAX_SAFE_INTEGER : index;
}

function scoreReadableFileName(name: string): number {
  const base = name.replace(/\.[^.]+$/, "");
  let score = Math.min(base.length, 80);
  if (/^\d{1,3}【[^】]+】[\u4e00-\u9fff]/.test(base)) score += 260;
  if (/[\u4e00-\u9fff]/.test(base)) score += 80;
  if (/\d/.test(base)) score += 18;
  if (/^\d+$/.test(base)) score -= 70;
  if (base.length < 6) score -= 20;
  return score;
}

function fileNameChinesePrefixRank(name: string): number {
  const base = name.replace(/\.[^.]+$/, "").trimStart();
  if (/^\d+\s*【[^】]*[\u4e00-\u9fff][^】]*】/.test(base)) return 0;
  if (/^\d+\s*[\u4e00-\u9fff]/.test(base)) return 1;
  return 2;
}

function splitFileName(name: string): { base: string; extension: string } {
  const match = /^(.*?)(\.[^.]*)?$/.exec(name);
  return {
    base: match?.[1] ?? name,
    extension: (match?.[2] ?? "").toLowerCase(),
  };
}

function renameSuffixBase(name: string): string | null {
  const { base } = splitFileName(name);
  const trimmed = base.trimEnd();
  const closeLength = trimmed.endsWith(")") || trimmed.endsWith("\uFF09") ? 1 : 0;
  if (closeLength === 0) return null;
  const withoutClose = trimmed.slice(0, -closeLength);
  const openIndex = Math.max(withoutClose.lastIndexOf("("), withoutClose.lastIndexOf("\uFF08"));
  if (openIndex === -1) return null;
  const suffix = withoutClose.slice(openIndex + 1);
  if (!/^[1-9]\d*$/.test(suffix)) return null;
  const originalBase = withoutClose.slice(0, openIndex).trimEnd();
  return originalBase ? originalBase.toLowerCase() : null;
}

function videosShareScanSource(
  left: VideoRecord,
  right: VideoRecord,
  scanSourcePaths: string[],
): boolean {
  if (scanSourcePaths.length === 0) {
    return pathKey(left.parentPath) === pathKey(right.parentPath);
  }
  return scanSourcePaths.some(
    (source) => pathInsideDir(left.path, source) && pathInsideDir(right.path, source),
  );
}

function pathSourceRenameSuffixPriority(
  item: MatchItem,
  items: MatchItem[],
  scanSourcePaths: string[],
): number {
  const itemName = splitFileName(item.video.fileName);
  const baseKey = itemName.base.toLowerCase();
  const suffixedBase = renameSuffixBase(item.video.fileName);
  const hasUnsuffixedSibling = suffixedBase
    ? items.some((candidate) => {
        const candidateName = splitFileName(candidate.video.fileName);
        return (
          candidate !== item &&
          videosShareScanSource(item.video, candidate.video, scanSourcePaths) &&
          candidateName.extension === itemName.extension &&
          candidateName.base.toLowerCase() === suffixedBase
        );
      })
    : false;
  if (hasUnsuffixedSibling) return 2;

  const hasSuffixedSibling = items.some((candidate) => {
    if (candidate === item || !videosShareScanSource(item.video, candidate.video, scanSourcePaths)) {
      return false;
    }
    const candidateName = splitFileName(candidate.video.fileName);
    return (
      candidateName.extension === itemName.extension &&
      renameSuffixBase(candidate.video.fileName) === baseKey
    );
  });
  return hasSuffixedSibling ? 0 : 1;
}

function compareNamingSourceItems(
  left: MatchItem,
  right: MatchItem,
  items: MatchItem[],
  namingSourceDirs: string[],
  scanSourcePaths: string[],
): number {
  const prefixDelta =
    fileNameChinesePrefixRank(left.video.fileName) - fileNameChinesePrefixRank(right.video.fileName);
  if (prefixDelta !== 0) return prefixDelta;

  const renameSuffixDelta =
    pathSourceRenameSuffixPriority(left, items, scanSourcePaths) -
    pathSourceRenameSuffixPriority(right, items, scanSourcePaths);
  if (renameSuffixDelta !== 0) return renameSuffixDelta;

  const priorityDelta =
    namingFolderPriority(left, namingSourceDirs) - namingFolderPriority(right, namingSourceDirs);
  if (priorityDelta !== 0) return priorityDelta;

  const readableDelta =
    scoreReadableFileName(right.video.fileName) - scoreReadableFileName(left.video.fileName);
  if (readableDelta !== 0) return readableDelta;

  return left.qualityRank - right.qualityRank;
}

function videoId(video: VideoRecord): number | null {
  return typeof video.id === "number" ? video.id : null;
}

function toggleNumber(list: number[], value: number): number[] {
  return list.includes(value) ? list.filter((item) => item !== value) : [...list, value];
}

async function writeClipboardText(text: string) {
  try {
    await navigator.clipboard.writeText(text);
    return;
  } catch {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "true");
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    textarea.style.top = "0";
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    const copied = document.execCommand("copy");
    document.body.removeChild(textarea);
    if (!copied) throw new Error("clipboard write failed");
  }
}

function NumberSettingInput({
  value,
  min,
  max,
  step,
  fallback,
  onCommit,
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  fallback: number;
  onCommit: (value: number) => void;
}) {
  const [text, setText] = useState(numberInputText(value));
  const focused = useRef(false);

  useEffect(() => {
    if (!focused.current) setText(numberInputText(value));
  }, [value]);

  function commit() {
    const next = clampNumberInput(text, min, max, fallback);
    setText(numberInputText(next));
    onCommit(next);
  }

  return (
    <input
      type="number"
      min={min}
      max={max}
      step={step}
      value={text}
      onFocus={() => {
        focused.current = true;
      }}
      onChange={(event) => setText(event.target.value)}
      onBlur={() => {
        focused.current = false;
        commit();
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.currentTarget.blur();
        }
      }}
    />
  );
}

function ScopeSelectionBox({ checked, partial }: { checked: boolean; partial: boolean }) {
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (ref.current) ref.current.indeterminate = partial;
  }, [partial]);

  return (
    <input
      ref={ref}
      className="scope-tree-check"
      type="checkbox"
      checked={checked}
      readOnly
      tabIndex={-1}
    />
  );
}

function PreviewStrip({ video }: { video: VideoRecord }) {
  if (video.previewImages.length === 0) {
    return (
      <div className="preview-empty">
        <ImageIcon size={18} />
      </div>
    );
  }

  return (
    <div className="preview-strip">
      {video.previewImages.slice(0, 3).map((path) => (
        <img key={path} src={previewSrc(path)} alt="" />
      ))}
    </div>
  );
}

function VideoSpec({ video }: { video: VideoRecord }) {
  return (
    <div className="spec-line">
      <span>{formatResolution(video)}</span>
      <span>{video.codec ?? "-"}</span>
      <span>{formatDuration(video.durationSeconds)}</span>
      <span>{formatBytes(video.sizeBytes)}</span>
    </div>
  );
}

function SimilarityHitMap({ item }: { item: MatchItem }) {
  const detail = item.matchDetail;
  const points = (detail?.points ?? [])
    .map((point) => ({
      ...point,
      fraction: Math.min(1, Math.max(0, point.fraction)),
      similarity: Math.min(1, Math.max(0, point.similarity)),
    }))
    .filter((point) => Number.isFinite(point.fraction));

  if (!detail || points.length === 0) return null;

  const formatRangeTime = (seconds: number) => (seconds > 0 ? formatDuration(seconds) : "0:00");
  const title = `${detail.relationExplanation}；命中 ${detail.hitCount} 点，显示 ${detail.displayedHitCount} 点，平均相似度 ${confidence(
    detail.averageSimilarity,
  )}，区间 ${formatRangeTime(detail.startSeconds)}-${formatRangeTime(detail.endSeconds)}`;

  return (
    <div className="similarity-hit-map" title={title}>
      <div className="similarity-hit-head">
        <span className={`relation-chip relation-${detail.relationType}`}>{detail.relationLabel}</span>
        <span>{detail.hitCount} 点 · {confidence(detail.averageSimilarity)}</span>
      </div>
      <div className="similarity-hit-track" aria-label="有序相似帧命中点">
        {points.map((point, index) => {
          const style: CSSProperties = {
            left: `${point.fraction * 100}%`,
            opacity: 0.38 + point.similarity * 0.62,
          };
          return <span className="similarity-hit-point" style={style} key={`${index}-${point.fraction}`} />;
        })}
      </div>
      <div className="similarity-hit-meta">
        <span>{formatRangeTime(detail.startSeconds)}-{formatRangeTime(detail.endSeconds)}</span>
        <span>对比 {detail.peerFileName}</span>
      </div>
      <small className="similarity-hit-explain">{detail.relationExplanation}</small>
    </div>
  );
}

function CandidateToggle({
  checked,
  label,
  compact = false,
  disabled = false,
  onClick,
}: {
  checked: boolean;
  label: string;
  compact?: boolean;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className={`candidate-toggle ${checked ? "checked" : ""} ${compact ? "compact" : ""}`}
      aria-pressed={checked}
      title={label}
      disabled={disabled}
      onClick={(event) => {
        event.stopPropagation();
        if (disabled) return;
        onClick();
      }}
    >
      {checked ? <CheckSquare size={16} /> : <Square size={16} />}
      {!compact && <span>{label}</span>}
    </button>
  );
}

function namingSourceForGroup(
  group: MatchGroup | null,
  overrides: Record<string, number>,
  namingSourceDirs: string[] = [],
  recommendedVideoId?: number | null,
  scanSourcePaths: string[] = [],
): MatchItem | null {
  if (!group) return null;
  const override = overrides[group.id];
  const overrideItem = group.items.find((item) => item.video.id === override);
  if (overrideItem) return overrideItem;

  return [...group.items].sort((left, right) =>
    compareNamingSourceItems(left, right, group.items, namingSourceDirs, scanSourcePaths),
  )[0] ?? null;
}

function recommendedIdForGroup(
  group: MatchGroup | null,
  overrides: Record<string, number>,
): number | null {
  if (!group) return null;
  const override = overrides[group.id];
  return group.items.some((item) => item.video.id === override)
    ? override
    : group.recommendedVideoId;
}

function isRemoteNavBlockedTarget(target: EventTarget | null): boolean {
  const element = target instanceof HTMLElement ? target : null;
  if (!element) return false;
  if (element.isContentEditable) return true;
  return Boolean(element.closest("input, textarea, select, button, a, [contenteditable='true']"));
}

export default function App() {
  const [view, setView] = useState<View>("matches");
  const [status, setStatus] = useState<ToolStatus | null>(null);
  const [settings, setSettings] = useState<AppSettings | null>(null);
  const [source, setSource] = useState("");
  const [scanPaths, setScanPaths] = useState<string[]>([]);
  const [videos, setVideos] = useState<VideoRecord[]>([]);
  const [scanSessions, setScanSessions] = useState<ScanSession[]>([]);
  const [operationHistory, setOperationHistory] = useState<OperationHistoryEntry[]>([]);
  const [scopeSessionIds, setScopeSessionIds] = useState<number[] | null>([]);
  const [groups, setGroups] = useState<MatchGroup[]>([]);
  const [selectedGroupId, setSelectedGroupId] = useState<string | null>(null);
  const [batchGroupIds, setBatchGroupIds] = useState<number[]>([]);
  const [selectedVideoIds, setSelectedVideoIds] = useState<number[]>([]);
  const [matchFocusPane, setMatchFocusPane] = useState<MatchFocusPane>("groups");
  const [focusedVideoId, setFocusedVideoId] = useState<number | null>(null);
  const [namingOverrides, setNamingOverrides] = useState<Record<string, number>>({});
  const [keeperOverrides, setKeeperOverrides] = useState<Record<string, number>>({});
  const [filenameOverrides, setFilenameOverrides] = useState<Record<string, number>>({});
  const [temporarilyIgnoredVideoIds, setTemporarilyIgnoredVideoIds] = useState<number[]>([]);
  const [minConfidence, setMinConfidence] = useState(0.9);
  const [minConfidenceText, setMinConfidenceText] = useState("90");
  const [maxConfidence, setMaxConfidence] = useState(1);
  const [maxConfidenceText, setMaxConfidenceText] = useState("100");
  const [aiMatchThresholdText, setAiMatchThresholdText] = useState("86");
  const [aiMinMatchedFramesText, setAiMinMatchedFramesText] = useState("8");
  const [matchFiltersExpanded, setMatchFiltersExpanded] = useState(false);
  const [batchDisposal, setBatchDisposal] = useState<BatchDisposal>("backup");
  const [groupSort, setGroupSort] = useState<GroupSort>("reclaimable");
  const [scanSummary, setScanSummary] = useState<ScanSummary | null>(null);
  const [scanProgress, setScanProgress] = useState<ScanProgress | null>(null);
  const [batchProgress, setBatchProgress] = useState<BatchProgress | null>(null);
  const [refreshProgress, setRefreshProgress] = useState<string | null>(null);
  const [refreshProgressPercent, setRefreshProgressPercent] = useState<number | null>(null);
  const [operationProgress, setOperationProgress] = useState<string | null>(null);
  const [aiModelStatus, setAiModelStatus] = useState<AiModelStatus | null>(null);
  const [aiIndexProgress, setAiIndexProgress] = useState<AiIndexProgress | null>(null);
  const [aiIndexSummary, setAiIndexSummary] = useState<AiIndexSummary | null>(null);
  const [batchTaskLogs, setBatchTaskLogs] = useState<BatchTaskLogEntry[]>([]);
  const [manualCopyText, setManualCopyText] = useState<string | null>(null);
  const [isScanning, setIsScanning] = useState(false);
  const [isIndexingAi, setIsIndexingAi] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [isExecuting, setIsExecuting] = useState(false);
  const [isMeasuringStorage, setIsMeasuringStorage] = useState(false);
  const [isCleaningStorage, setIsCleaningStorage] = useState(false);
  const [isVacuumingDatabase, setIsVacuumingDatabase] = useState(false);
  const [isPickingFolder, setIsPickingFolder] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [runtimeLogs, setRuntimeLogs] = useState<RuntimeLogEntry[]>([]);
  const [groupListWidth, setGroupListWidth] = useState(430);
  const [settingsDraft, setSettingsDraft] = useState<AppSettings | null>(null);
  const [ramDiskStatus, setRamDiskStatus] = useState<RamDiskStatus | null>(null);
  const [ramDiskSizeMb, setRamDiskSizeMb] = useState(16 * 1024);
  const [isConfiguringRamDisk, setIsConfiguringRamDisk] = useState(false);
  const [ramDiskSetupDismissed, setRamDiskSetupDismissed] = useState(false);
  const [storageUsage, setStorageUsage] = useState<StorageUsageSummary | null>(null);
  const [storageCleanupSummary, setStorageCleanupSummary] = useState<StorageCleanupSummary | null>(null);
  const [completedFrameCacheCleanupSummary, setCompletedFrameCacheCleanupSummary] =
    useState<CompletedAiFrameCacheCleanupSummary | null>(null);
  const [vacuumSummary, setVacuumSummary] = useState<string | null>(null);
  const [namingFolderInput, setNamingFolderInput] = useState("");
  const [collapsedScopeKeys, setCollapsedScopeKeys] = useState<Set<string>>(() => new Set());
  const [scopeAnchorKey, setScopeAnchorKey] = useState<string | null>(null);
  const [confirmDialog, setConfirmDialog] = useState<ConfirmDialog | null>(null);
  const [isSettingsSaveBarVisible, setIsSettingsSaveBarVisible] = useState(true);
  const confirmResolver = useRef<((value: boolean) => void) | null>(null);
  const lastSettingsScrollTop = useRef(0);
  const groupRowRefs = useRef<Record<string, HTMLElement | null>>({});
  const videoCardRefs = useRef<Record<number, HTMLElement | null>>({});
  const runtimeLogSeq = useRef(0);
  const knownCollapsibleScopeKeys = useRef<Set<string>>(new Set());

  const activeGroups = useMemo(() => {
    if (temporarilyIgnoredVideoIds.length === 0) return groups;
    const ignored = new Set(temporarilyIgnoredVideoIds);
    return groups.flatMap((group) => {
      const items = group.items.filter(
        (item) => item.video.id === null || !ignored.has(item.video.id),
      );
      if (items.length < 2) return [];
      const recommendedStillVisible = items.some((item) => item.video.id === group.recommendedVideoId);
      const fallbackRecommended = [...items].sort((left, right) => left.qualityRank - right.qualityRank)[0]
        ?.video.id ?? null;
      return [{
        ...group,
        items,
        itemCount: items.length,
        recommendedVideoId: recommendedStillVisible ? group.recommendedVideoId : fallbackRecommended,
      }];
    });
  }, [groups, temporarilyIgnoredVideoIds]);

  const selectedGroup = useMemo(
    () => activeGroups.find((group) => group.id === selectedGroupId) ?? activeGroups[0] ?? null,
    [activeGroups, selectedGroupId],
  );

  const settingsLanguage = normalizeSettingsLanguage(settingsDraft?.uiLanguage ?? settings?.uiLanguage);
  const settingText = SETTINGS_TEXT[settingsLanguage];
  const settingsDirty = useMemo(
    () => Boolean(settings && settingsDraft && JSON.stringify(settings) !== JSON.stringify(settingsDraft)),
    [settings, settingsDraft],
  );

  const selectedRecommendedId = useMemo(
    () => recommendedIdForGroup(selectedGroup, keeperOverrides),
    [selectedGroup, keeperOverrides],
  );

  const allScopeSessionIds = useMemo(() => scanSessions.map((session) => session.id), [scanSessions]);
  const scanSourcePaths = useMemo(() => scanSessions.map((session) => session.source), [scanSessions]);

  const namingSource = useMemo(
    () =>
      namingSourceForGroup(
        selectedGroup,
        namingOverrides,
        settings?.namingSourceDirs ?? [],
        selectedRecommendedId,
        scanSourcePaths,
      ),
    [selectedGroup, namingOverrides, settings, selectedRecommendedId, scanSourcePaths],
  );

  const selectedScopeIds = useMemo(
    () => (scopeSessionIds === null ? allScopeSessionIds : scopeSessionIds),
    [allScopeSessionIds, scopeSessionIds],
  );

  const selectedScopeSet = useMemo(() => new Set(selectedScopeIds), [selectedScopeIds]);

  const scopeTree = useMemo(() => buildScopeTree(scanSessions), [scanSessions]);

  const scopeRows = useMemo(
    () => flattenScopeTree(scopeTree, collapsedScopeKeys),
    [collapsedScopeKeys, scopeTree],
  );

  const collapsibleScopeKeys = useMemo(() => collectCollapsibleScopeKeys(scopeTree), [scopeTree]);

  const totalReclaimable = useMemo(
    () => activeGroups.reduce((sum, group) => sum + group.reclaimableBytes, 0),
    [activeGroups],
  );

  const selectedVideos = useMemo(
    () => videos.filter((video) => video.id !== null && selectedVideoIds.includes(video.id)),
    [videos, selectedVideoIds],
  );

  const displayedGroups = useMemo(() => {
    return [...activeGroups].sort((left, right) => {
      if (groupSort === "confidence") {
        return right.confidence - left.confidence || right.reclaimableBytes - left.reclaimableBytes;
      }
      if (groupSort === "files") {
        return right.itemCount - left.itemCount || right.confidence - left.confidence;
      }
      return right.reclaimableBytes - left.reclaimableBytes || right.confidence - left.confidence;
    });
  }, [activeGroups, groupSort]);

  const visibleGroupIds = useMemo(
    () =>
      displayedGroups
        .map((group) => Number(group.id.replace("group-", "")))
        .filter((id) => Number.isFinite(id)),
    [displayedGroups],
  );

  const currentGroupVideoIds = useMemo(
    () =>
      selectedGroup?.items
        .map((item) => videoId(item.video))
        .filter((id): id is number => id !== null) ?? [],
    [selectedGroup],
  );

  const focusedCurrentVideoId =
    focusedVideoId !== null && currentGroupVideoIds.includes(focusedVideoId)
      ? focusedVideoId
      : currentGroupVideoIds[0] ?? null;

  const allCurrentGroupVideosSelected =
    currentGroupVideoIds.length > 0 &&
    currentGroupVideoIds.every((id) => selectedVideoIds.includes(id));

  const allVisibleGroupsQueued =
    visibleGroupIds.length > 0 && visibleGroupIds.every((id) => batchGroupIds.includes(id));

  useEffect(() => {
    setBatchGroupIds((current) => {
      const next = current.filter((id) => visibleGroupIds.includes(id));
      return sameNumberList(current, next) ? current : next;
    });
  }, [visibleGroupIds]);

  const scanPercent = useMemo(() => {
    const scanned = scanProgress?.scanned ?? scanSummary?.scanned ?? 0;
    const total = scanProgress?.totalFiles ?? scanSummary?.totalFiles ?? 0;
    if (total === 0) return 0;
    return Math.min(100, Math.round((scanned / total) * 100));
  }, [scanProgress, scanSummary]);

  const batchPercent = useMemo(() => {
    if (!batchProgress || batchProgress.total === 0) return 0;
    return Math.min(100, Math.round((batchProgress.processed / batchProgress.total) * 100));
  }, [batchProgress]);

  const batchProgressIndeterminate =
    Boolean(batchProgress) && isExecuting && batchProgress?.phase.includes("后端");

  const canDismissScanProgress = !isScanning && (Boolean(scanProgress) || Boolean(scanSummary));
  const canDismissBatchProgress = Boolean(batchProgress) && !isExecuting;
  const canDismissOperationProgress = Boolean(operationProgress) && !isExecuting;
  const canDismissRefreshProgress =
    Boolean(refreshProgress) && refreshProgressPercent !== null && refreshProgressPercent >= 100;
  const canDismissAiIndexProgress =
    !isIndexingAi && (Boolean(aiIndexProgress) || Boolean(aiIndexSummary));

  const aiIndexCounts = useMemo(() => {
    const total = aiIndexProgress?.totalVideos ?? aiIndexSummary?.totalVideos ?? 0;
    const done = aiIndexProgress
      ? aiIndexProgress.processed +
        aiIndexProgress.skipped +
        (aiIndexProgress.insufficientFrames ?? 0) +
        aiIndexProgress.failed
      : (aiIndexSummary?.processed ?? 0) +
        (aiIndexSummary?.skipped ?? 0) +
        (aiIndexSummary?.insufficientFrames ?? 0) +
        (aiIndexSummary?.failed ?? 0);
    const shownDone = total > 0 ? Math.min(done, total) : done;
    const started = aiIndexProgress
      ? Math.min(total, Math.max(shownDone, aiIndexProgress.started ?? shownDone))
      : shownDone;
    const prepared = aiIndexProgress
      ? Math.min(total, Math.max(shownDone, aiIndexProgress.prepared ?? shownDone))
      : shownDone;
    const weightedDone = Math.min(
      total,
      shownDone + Math.max(0, prepared - shownDone) * 0.7 + Math.max(0, started - prepared) * 0.15,
    );
    return {
      total,
      done,
      shownDone,
      started,
      prepared,
      weightedDone,
    };
  }, [aiIndexProgress, aiIndexSummary]);

  const aiIndexPercent = useMemo(() => {
    if (aiIndexCounts.total === 0) return 0;
    return Math.min(100, (aiIndexCounts.weightedDone / aiIndexCounts.total) * 100);
  }, [aiIndexCounts]);

  const queuedScanTargets = useMemo(
    () => uniquePaths([...scanPaths, source.trim()].filter(Boolean)),
    [scanPaths, source],
  );

  function normalizedMinConfidence(): number {
    return clampConfidencePercent(Number(minConfidenceText)) / 100;
  }

  function normalizedConfidenceRange(): [number, number] {
    const nextMin = clampConfidencePercent(Number(minConfidenceText)) / 100;
    const nextMax = clampConfidencePercent(Number(maxConfidenceText)) / 100;
    return nextMin <= nextMax ? [nextMin, nextMax] : [nextMax, nextMin];
  }

  function normalizedAiMatchThreshold(): number {
    return Math.min(0.99, Math.max(0, Number(aiMatchThresholdText) / 100 || 0));
  }

  function normalizedAiMinMatchedFrames(): number {
    const value = Math.round(Number(aiMinMatchedFramesText));
    return Number.isFinite(value) ? Math.min(128, Math.max(1, value)) : 8;
  }

  function runtimeLogTime(): string {
    return new Date().toLocaleTimeString("zh-CN", {
      hour12: false,
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  }

  function upsertRuntimeLog(
    key: string,
    title: string,
    detail: string | undefined,
    level: RuntimeLogLevel = "info",
  ) {
    setRuntimeLogs((current) => {
      const existing = current.find((entry) => entry.key === key);
      const nextEntry: RuntimeLogEntry = {
        id: existing?.id ?? ++runtimeLogSeq.current,
        key,
        time: runtimeLogTime(),
        level,
        title,
        detail,
      };
      return [nextEntry, ...current.filter((entry) => entry.key !== key)].slice(0, 80);
    });
  }

  function pushRuntimeLog(
    title: string,
    detail: string | undefined,
    level: RuntimeLogLevel = "info",
  ) {
    const key = `event-${++runtimeLogSeq.current}`;
    setRuntimeLogs((current) =>
      [
        {
          id: runtimeLogSeq.current,
          key,
          time: runtimeLogTime(),
          level,
          title,
          detail,
        },
        ...current,
      ].slice(0, 80),
    );
  }

  function clearRuntimeLogs() {
    setRuntimeLogs([]);
  }

  function waitForPaint(): Promise<void> {
    return new Promise((resolve) => {
      window.requestAnimationFrame(() => window.requestAnimationFrame(() => resolve()));
    });
  }

  function backendTaskMessages(messages: string[]) {
    const byIndex = new Map<number, { status: BatchTaskLogStatus; reason: string }>();
    for (const message of messages) {
      const match = /\b(?:skipped|failed) task (\d+)\b/i.exec(message);
      if (!match) continue;
      const index = Number(match[1]);
      if (!Number.isFinite(index)) continue;
      byIndex.set(index, {
        status: /^failed\b/i.test(message) ? "failed" : "skipped",
        reason: message,
      });
    }
    for (const message of messages) {
      const match = /\btask (\d+) completed with .*failure/i.exec(message);
      if (!match) continue;
      const index = Number(match[1]);
      if (!Number.isFinite(index)) continue;
      byIndex.set(index, { status: "failed", reason: message });
    }
    return byIndex;
  }

  function failureLikeMessages(messages: string[]): string[] {
    return messages.filter((message) => {
      const lower = message.toLowerCase();
      return (
        lower.includes("failed") ||
        lower.includes("error") ||
        lower.includes("skipped") ||
        lower.includes("missing") ||
        lower.includes("cannot") ||
        lower.includes("locked") ||
        message.includes("失败") ||
        message.includes("跳过") ||
        message.includes("缺失") ||
        message.includes("无法")
      );
    });
  }

  function batchTaskStatusLabel(status: BatchTaskLogStatus): string {
    switch (status) {
      case "pending":
        return "待处理";
      case "running":
        return "后端处理中";
      case "completed":
        return "已完成";
      case "skipped":
        return "已跳过";
      case "failed":
        return "失败";
    }
  }

  function logLevelFromMessage(message: string): RuntimeLogLevel {
    const lower = message.toLowerCase();
    if (
      message.includes("失败") ||
      message.includes("错误") ||
      message.includes("无法") ||
      lower.includes("error") ||
      lower.includes("failed") ||
      lower.includes("locked")
    ) {
      return "error";
    }
    if (message.includes("跳过") || lower.includes("skipped")) return "warning";
    if (
      message.includes("完成") ||
      message.includes("已") ||
      lower.includes("completed") ||
      lower.includes("success")
    ) {
      return "success";
    }
    return "info";
  }

  function logScanProgress(progress: ScanProgress) {
    const done = progress.totalFiles > 0 ? `${progress.scanned}/${progress.totalFiles}` : "0/0";
    const detail = `${progress.phase} · ${done} · 失败 ${progress.failed}${
      progress.currentPath ? ` · ${shortPath(progress.currentPath)}` : ""
    }`;
    upsertRuntimeLog(
      "scan-progress",
      progress.phase === "completed" ? "扫描完成" : "扫描",
      detail,
      progress.phase === "completed" ? "success" : progress.phase === "cancelled" ? "warning" : "active",
    );
  }

  function logAiIndexProgress(progress: AiIndexProgress) {
    const insufficientFrames = progress.insufficientFrames ?? 0;
    const done = progress.processed + progress.skipped + insufficientFrames + progress.failed;
    const detail = `${progress.phase} · ${Math.min(done, progress.totalVideos)}/${progress.totalVideos} · 处理 ${progress.processed} · 跳过 ${progress.skipped} · 抽帧不足 ${insufficientFrames} · 失败 ${progress.failed}${
      progress.currentPath ? ` · ${shortPath(progress.currentPath)}` : ""
    }`;
    upsertRuntimeLog(
      "ai-index-progress",
      progress.phase === "completed" ? "AI 索引完成" : "AI 索引",
      detail,
      progress.phase === "completed" ? "success" : "active",
    );
  }

  function logBatchProgress(progress: BatchProgress) {
    const detail = `${progress.phase} · ${progress.processed}/${progress.total} · 完成 ${progress.completed} · 跳过 ${progress.skipped} · 失败 ${progress.failed}${
      progress.currentTitle ? ` · ${shortPath(progress.currentTitle)}` : ""
    }`;
    upsertRuntimeLog(
      "batch-progress",
      progress.processed >= progress.total ? "批量处理完成" : "批量处理",
      detail,
      progress.failed > 0 ? "warning" : progress.processed >= progress.total ? "success" : "active",
    );
  }

  async function refreshIndexData(
    nextScope = scopeSessionIds,
    options: { pruneMissing?: boolean; clearMatches?: boolean } = {},
  ) {
    setRefreshProgress(options.pruneMissing ? "正在刷新索引路径" : "正在读取索引");
    setRefreshProgressPercent(null);
    try {
      const pruneSummary = options.pruneMissing ? await refreshIndexSources() : null;
      const [toolStatus, appSettings, sessions, history] = await Promise.all([
        getToolStatus(),
        getAppSettings(),
        listScanSessions(),
        listOperationHistory(),
      ]);

      const validSessionIds = new Set(sessions.map((session) => session.id));
      const nextValidScope =
        nextScope === null ? null : nextScope.filter((id) => validSessionIds.has(id));
      const nextVideos = await listVideos(nextValidScope ?? undefined);
      const nextVideoIds = new Set(nextVideos.map((video) => video.id).filter((id) => id !== null));

      setStatus(toolStatus);
      setSettings(appSettings);
      setSettingsDraft(appSettings);
      setScanSessions(sessions);
      setOperationHistory(history);
      setVideos(nextVideos);
      if (!sameNumberList(nextScope, nextValidScope)) {
        setScopeSessionIds(nextValidScope);
      }
      setSelectedVideoIds((ids) => ids.filter((id) => nextVideoIds.has(id)));
      if (options.clearMatches) {
        setGroups([]);
        setSelectedGroupId(null);
        setBatchGroupIds([]);
        setFilenameOverrides({});
        setTemporarilyIgnoredVideoIds([]);
      } else {
        setGroups((current) => {
          const filtered = current.filter((group) =>
            group.items.every((item) => item.video.id !== null && nextVideoIds.has(item.video.id)),
          );
          if (
            selectedGroupId &&
            !filtered.some((group) => group.id === selectedGroupId)
          ) {
            setSelectedGroupId(filtered[0]?.id ?? null);
          }
          const nextGroupIds = filtered
            .map((group) => Number(group.id.replace("group-", "")))
            .filter((id) => Number.isFinite(id));
          setBatchGroupIds((ids) => ids.filter((id) => nextGroupIds.includes(id)));
          return filtered;
        });
      }

      if (pruneSummary) {
        const parts = [];
        if (pruneSummary.removedMissingSources > 0) {
          parts.push(`已移除 ${pruneSummary.removedMissingSources} 个失效路径`);
        }
        if (pruneSummary.skippedUnavailableRoots > 0) {
          parts.push(`跳过 ${pruneSummary.skippedUnavailableRoots} 个根路径不可访问的索引源`);
        }
        setNotice(parts.length > 0 ? parts.join("，") : "索引路径已刷新，没有发现失效路径");
      }
    } finally {
      setRefreshProgress(null);
      setRefreshProgressPercent(null);
    }
  }

  async function refreshAll(
    nextScope = scopeSessionIds,
    confidenceOverride = minConfidence,
    maxConfidenceOverride = maxConfidence,
  ) {
    const apiScope = nextScope ?? undefined;
    setRefreshProgress("正在清理已移动或孤立的视频索引");
    setRefreshProgressPercent(null);
    try {
      const pruneSummary = await pruneStaleVideos(apiScope);
      if (
        pruneSummary.removedMissingVideos > 0 ||
        pruneSummary.removedOrphanVideos > 0
      ) {
        setRefreshProgress(
          `已清理 ${pruneSummary.removedMissingVideos} 个缺失视频、${pruneSummary.removedOrphanVideos} 个孤立索引，正在刷新相似结果`,
        );
      } else {
        setRefreshProgress("正在读取索引、AI 相似分组和工具状态");
      }
      const [
        toolStatus,
        appSettings,
        sessions,
        nextVideos,
        nextGroups,
        history,
      ] = await Promise.all([
        getToolStatus(),
        getAppSettings(),
        listScanSessions(),
        listVideos(apiScope),
        listMatchGroups(apiScope, confidenceOverride, maxConfidenceOverride),
        listOperationHistory(),
      ]);

      setStatus(toolStatus);
      setSettings(appSettings);
      setSettingsDraft(appSettings);
      setScanSessions(sessions);
      setOperationHistory(history);
      setVideos(nextVideos);
      setGroups(nextGroups);
      setFilenameOverrides({});
      setTemporarilyIgnoredVideoIds([]);
      setSelectedVideoIds((ids) =>
        ids.filter((id) => nextVideos.some((video) => video.id === id)),
      );
      const nextGroupIds = nextGroups
        .map((group) => Number(group.id.replace("group-", "")))
        .filter((id) => Number.isFinite(id));
      setBatchGroupIds((ids) => ids.filter((id) => nextGroupIds.includes(id)));

      if (nextGroups.length === 0) {
        setSelectedGroupId(null);
      } else if (!selectedGroupId || !nextGroups.some((group) => group.id === selectedGroupId)) {
        setSelectedGroupId(nextGroups[0].id);
      }
    } finally {
      setRefreshProgress(null);
      setRefreshProgressPercent(null);
    }
  }

  async function refreshStartup() {
    const [appSettings, sessions, history, nextVideos] = await Promise.all([
      getAppSettings(),
      listScanSessions(),
      listOperationHistory(),
      listVideos([]),
    ]);
    let nextRamDiskStatus = await getRamDiskStatus();
    if (
      appSettings.ramDiskEnabled &&
      appSettings.ramDiskSetupCompleted &&
      nextRamDiskStatus.ready
    ) {
      try {
        nextRamDiskStatus = await releaseRamDisk();
      } catch {
        nextRamDiskStatus = await getRamDiskStatus();
      }
    }
    setSettings(appSettings);
    setSettingsDraft(appSettings);
    setRamDiskStatus(nextRamDiskStatus);
    setRamDiskSizeMb(appSettings.ramDiskSizeMb);
    setScanSessions(sessions);
    setOperationHistory(history);
    setVideos(nextVideos);
  }

  useEffect(() => {
    refreshStartup().catch((error) => setNotice(String(error)));
  }, []);

  useEffect(() => {
    if (!settings) return;
    setAiMatchThresholdText(String(Math.round(settings.aiSimilarityThreshold * 100)));
    setAiMinMatchedFramesText(String(settings.aiMinMatchedFrames));
    setRamDiskSizeMb(settings.ramDiskSizeMb);
  }, [settings]);

  useEffect(() => {
    if (view !== "ai" && view !== "settings") return;
    lastSettingsScrollTop.current = 0;
    setIsSettingsSaveBarVisible(true);
  }, [view]);

  useEffect(() => {
    if (settingsDirty && (view === "ai" || view === "settings")) {
      setIsSettingsSaveBarVisible(true);
    }
  }, [settingsDirty, view]);

  useEffect(() => {
    setCollapsedScopeKeys((current) => {
      const next = new Set(current);
      let changed = false;
      for (const key of collapsibleScopeKeys) {
        if (!knownCollapsibleScopeKeys.current.has(key)) {
          next.add(key);
          changed = true;
        }
      }
      knownCollapsibleScopeKeys.current = new Set(collapsibleScopeKeys);
      return changed ? next : current;
    });
  }, [collapsibleScopeKeys]);

  useEffect(() => {
    if (!notice) return;
    pushRuntimeLog("消息", notice, logLevelFromMessage(notice));
  }, [notice]);

  useEffect(() => {
    const element = selectedGroupId ? groupRowRefs.current[selectedGroupId] : null;
    element?.scrollIntoView({ block: "nearest" });
  }, [selectedGroupId, displayedGroups]);

  useEffect(() => {
    if (!isTauri()) return;
    let dispose: (() => void) | undefined;
    listen<ScanProgress>("scan-progress", (event) => {
      setScanProgress(event.payload);
      logScanProgress(event.payload);
    })
      .then((unlisten) => {
        dispose = unlisten;
      })
      .catch((error) => setNotice(String(error)));

    return () => {
      dispose?.();
    };
  }, []);

  useEffect(() => {
    if (!isTauri()) return;
    let dispose: (() => void) | undefined;
    listen<AiIndexProgress>("ai-index-progress", (event) => {
      setAiIndexProgress(event.payload);
      logAiIndexProgress(event.payload);
    })
      .then((unlisten) => {
        dispose = unlisten;
      })
      .catch((error) => setNotice(String(error)));

    return () => {
      dispose?.();
    };
  }, []);

  useEffect(() => {
    if (!isTauri()) return;
    let dispose: (() => void) | undefined;
    listen<MatchRefreshProgress>("match-refresh-progress", (event) => {
      const text = formatMatchRefreshProgress(event.payload);
      const percent = matchRefreshPercent(event.payload);
      setRefreshProgress(text);
      setRefreshProgressPercent(percent);
      upsertRuntimeLog(
        "match-refresh-progress",
        percent >= 100 ? "相似结果刷新完成" : "相似结果刷新",
        text,
        percent >= 100 ? "success" : "active",
      );
    })
      .then((unlisten) => {
        dispose = unlisten;
      })
      .catch((error) => setNotice(String(error)));

    return () => {
      dispose?.();
    };
  }, []);

  useEffect(() => {
    if (!isTauri()) return;
    let dispose: (() => void) | undefined;
    listen<DeleteIndexProgress>("delete-index-progress", (event) => {
      const text = formatDeleteIndexProgress(event.payload);
      setOperationProgress(text);
      upsertRuntimeLog(
        "delete-index-progress",
        event.payload.phase === "completed" ? "索引删除完成" : "索引删除",
        text,
        event.payload.phase === "completed" ? "success" : "active",
      );
    })
      .then((unlisten) => {
        dispose = unlisten;
      })
      .catch((error) => setNotice(String(error)));

    return () => {
      dispose?.();
    };
  }, []);

  function beginResize(event: React.PointerEvent<HTMLDivElement>) {
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = groupListWidth;
    const move = (moveEvent: PointerEvent) => {
      const nextWidth = Math.min(640, Math.max(340, startWidth + moveEvent.clientX - startX));
      setGroupListWidth(nextWidth);
    };
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  }

  function selectGroup(groupId: string) {
    setSelectedGroupId(groupId);
    setSelectedVideoIds([]);
    setFocusedVideoId(null);
  }

  function focusGroupElement(groupId: string | null | undefined) {
    if (!groupId) return;
    window.requestAnimationFrame(() => {
      const element = groupRowRefs.current[groupId];
      element?.focus({ preventScroll: true });
      element?.scrollIntoView({ block: "nearest" });
    });
  }

  function focusVideoElement(id: number | null | undefined) {
    if (id === null || id === undefined) return;
    window.requestAnimationFrame(() => {
      const element = videoCardRefs.current[id];
      element?.focus({ preventScroll: true });
      element?.scrollIntoView({ block: "nearest" });
    });
  }

  function switchToGroupPane() {
    setMatchFocusPane("groups");
    focusGroupElement(selectedGroup?.id ?? displayedGroups[0]?.id);
  }

  function switchToVideoPane(preferredId: number | null = focusedCurrentVideoId) {
    const nextId =
      preferredId !== null && currentGroupVideoIds.includes(preferredId)
        ? preferredId
        : currentGroupVideoIds[0] ?? null;
    if (nextId === null) return;
    setFocusedVideoId(nextId);
    setMatchFocusPane("videos");
    focusVideoElement(nextId);
  }

  function selectAdjacentGroup(delta: number) {
    if (displayedGroups.length === 0) return;
    const currentIndex = displayedGroups.findIndex((group) => group.id === selectedGroup?.id);
    const fallbackIndex = delta > 0 ? 0 : displayedGroups.length - 1;
    const nextIndex =
      currentIndex === -1
        ? fallbackIndex
        : Math.min(displayedGroups.length - 1, Math.max(0, currentIndex + delta));
    const nextGroupId = displayedGroups[nextIndex].id;
    setMatchFocusPane("groups");
    selectGroup(nextGroupId);
    focusGroupElement(nextGroupId);
  }

  function selectAdjacentVideo(delta: number) {
    if (currentGroupVideoIds.length === 0) return;
    const currentIndex = focusedCurrentVideoId === null
      ? -1
      : currentGroupVideoIds.indexOf(focusedCurrentVideoId);
    const fallbackIndex = delta > 0 ? 0 : currentGroupVideoIds.length - 1;
    const nextIndex =
      currentIndex === -1
        ? fallbackIndex
        : Math.min(currentGroupVideoIds.length - 1, Math.max(0, currentIndex + delta));
    const nextId = currentGroupVideoIds[nextIndex];
    setFocusedVideoId(nextId);
    setMatchFocusPane("videos");
    focusVideoElement(nextId);
  }

  function toggleFocusedMatchSelection() {
    if (matchFocusPane === "videos") {
      const id = focusedCurrentVideoId;
      const item = selectedGroup?.items.find((candidate) => candidate.video.id === id);
      if (item) toggleVideo(item.video);
      return;
    }
    if (selectedGroup) toggleBatchGroup(selectedGroup);
  }

  function handleRemoteMatchKey(event: globalThis.KeyboardEvent | KeyboardEvent<HTMLElement>) {
    if (view !== "matches" || confirmDialog || manualCopyText) return;
    if (isRemoteNavBlockedTarget(event.target)) return;
    const isSpace = event.key === " " || event.key === "Spacebar";
    if (
      event.key !== "ArrowLeft" &&
      event.key !== "ArrowRight" &&
      event.key !== "ArrowUp" &&
      event.key !== "ArrowDown" &&
      !isSpace
    ) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();

    if (event.key === "ArrowLeft") {
      switchToGroupPane();
      return;
    }
    if (event.key === "ArrowRight") {
      switchToVideoPane();
      return;
    }
    if (event.key === "ArrowUp" || event.key === "ArrowDown") {
      const delta = event.key === "ArrowDown" ? 1 : -1;
      if (matchFocusPane === "videos") {
        selectAdjacentVideo(delta);
      } else {
        selectAdjacentGroup(delta);
      }
      return;
    }
    if (isSpace) {
      toggleFocusedMatchSelection();
    }
  }

  function handleGroupListKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    handleRemoteMatchKey(event);
  }

  useEffect(() => {
    const handler = (event: globalThis.KeyboardEvent) => handleRemoteMatchKey(event);
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });

  useEffect(() => {
    if (currentGroupVideoIds.length === 0) {
      setFocusedVideoId(null);
      return;
    }
    setFocusedVideoId((current) =>
      current !== null && currentGroupVideoIds.includes(current) ? current : currentGroupVideoIds[0],
    );
  }, [currentGroupVideoIds]);

  useEffect(() => {
    if (matchFocusPane === "videos") {
      focusVideoElement(focusedCurrentVideoId);
    }
  }, [focusedCurrentVideoId, matchFocusPane, selectedGroup?.id]);

  async function refreshMatches() {
    const [nextMinConfidence, nextMaxConfidence] = normalizedConfidenceRange();
    const nextAiThreshold = normalizedAiMatchThreshold();
    const nextMinMatchedFrames = normalizedAiMinMatchedFrames();
    setMinConfidence(nextMinConfidence);
    setMaxConfidence(nextMaxConfidence);
    setMinConfidenceText(String(Math.round(nextMinConfidence * 100)));
    setMaxConfidenceText(String(Math.round(nextMaxConfidence * 100)));
    setAiMatchThresholdText(String(Math.round(nextAiThreshold * 100)));
    setAiMinMatchedFramesText(String(nextMinMatchedFrames));
    setNotice(null);
    try {
      const current = await ensureSettingsSavedBeforeTask(
        settingsLanguage === "en" ? "refreshing matches" : "刷新相似结果",
      );
      if (!current) return;
      if (
        current &&
        (Math.abs(current.aiSimilarityThreshold - nextAiThreshold) > 0.0001 ||
          current.aiMinMatchedFrames !== nextMinMatchedFrames)
      ) {
        const saved = await saveAppSettings({
          ...current,
          aiSimilarityThreshold: nextAiThreshold,
          aiMinMatchedFrames: nextMinMatchedFrames,
        });
        setSettings(saved);
        setSettingsDraft(saved);
      }
      await refreshAll(scopeSessionIds, nextMinConfidence, nextMaxConfidence);
      setNotice("相似结果已刷新");
    } catch (error) {
      setNotice(String(error));
    }
  }

  async function checkAiModel() {
    setNotice(null);
    try {
      const nextStatus = await getAiModelStatus();
      setAiModelStatus(nextStatus);
      setNotice(nextStatus.message);
    } catch (error) {
      setNotice(String(error));
    }
  }

  async function handleBuildAiIndex(forceRebuild = false) {
    setNotice(null);
    setAiIndexSummary(null);
    const originalScopeSessionIds = scopeSessionIds;

    const activeSettings = await ensureSettingsSavedBeforeTask(
      settingsLanguage === "en" ? "building the AI index" : "构建 AI 索引",
    );
    if (!activeSettings) return;

    if (!activeSettings?.aiVisionEnabled) {
      setNotice("请先在 AI 页面启用 AI 视觉匹配");
      return;
    }

    if (forceRebuild) {
      const confirmed = await requestConfirm({
        title: "重建 AI 索引",
        body: "这会重新计算当前比对范围内已有视频的 AI 向量，不会修改视频文件。",
        confirmLabel: "重建",
      });
      if (!confirmed) return;
    }

    setIsIndexingAi(true);
    setAiIndexProgress({
      phase: "starting",
      totalVideos: 0,
      processed: 0,
      skipped: 0,
      insufficientFrames: 0,
      failed: 0,
      started: 0,
      prepared: 0,
      currentPath: null,
    });

    try {
      const nextStatus = await getAiModelStatus();
      setAiModelStatus(nextStatus);
      if (!nextStatus.ready) {
        setNotice(nextStatus.message);
        return;
      }

      let aiBuildSessionIds = originalScopeSessionIds ?? undefined;
      const explicitScanTargets = queuedScanTargets;
      const scopeScanTargets =
        explicitScanTargets.length > 0
          ? explicitScanTargets
          : uniquePaths(
              scanSessions
                .filter((session) =>
                  originalScopeSessionIds === null
                    ? true
                    : originalScopeSessionIds.includes(session.id),
                )
                .map((session) => session.source),
            );
      if (scopeScanTargets.length > 0) {
        setIsScanning(true);
        setScanSummary(null);
        setScanProgress({
          source: scopeScanTargets[0],
          phase: "scan-before-ai-index",
          totalFiles: 0,
          scanned: 0,
          failed: 0,
          currentPath: null,
        });
        const summaries = await scanSources(scopeScanTargets);
        const newSessionIds = summaries
          .map((item) => item.sessionId)
          .filter((id): id is number => id !== null);
        const scanSummary: ScanSummary = {
          sessionId: summaries.length === 1 ? summaries[0].sessionId : null,
          source: summaries.length === 1 ? summaries[0].source : `${summaries.length} 个路径`,
          databasePath: summaries[0]?.databasePath ?? status?.databasePath ?? "data\\index.sqlite",
          totalFiles: summaries.reduce((sum, item) => sum + item.totalFiles, 0),
          scanned: summaries.reduce((sum, item) => sum + item.scanned, 0),
          failed: summaries.reduce((sum, item) => sum + item.failed, 0),
          elapsedMs: summaries.reduce((sum, item) => sum + item.elapsedMs, 0),
        };
        setScanSummary(scanSummary);
        setScanProgress({
          source: scanSummary.source,
          phase: "completed",
          totalFiles: scanSummary.totalFiles,
          scanned: scanSummary.scanned,
          failed: scanSummary.failed,
          currentPath: null,
        });
        setIsScanning(false);
        const shouldRestrictToNewSessions =
          originalScopeSessionIds !== null || explicitScanTargets.length > 0;
        if (shouldRestrictToNewSessions && newSessionIds.length > 0) {
          aiBuildSessionIds = newSessionIds;
          setScopeSessionIds(newSessionIds);
          await refreshIndexData(newSessionIds, { clearMatches: true });
        } else {
          aiBuildSessionIds = undefined;
          await refreshIndexData(null, { clearMatches: true });
        }
      }

      const summary = await buildAiIndex(aiBuildSessionIds, forceRebuild);
      setAiIndexSummary(summary);
      setAiIndexProgress({
        phase: "completed",
        totalVideos: summary.totalVideos,
        processed: summary.processed,
        skipped: summary.skipped,
        insufficientFrames: summary.insufficientFrames ?? 0,
        failed: summary.failed,
        started: summary.totalVideos,
        prepared: summary.totalVideos,
        currentPath: null,
      });
      const insufficientFrames = summary.insufficientFrames ?? 0;
      const allSkipped =
        summary.totalVideos > 0 &&
        summary.processed === 0 &&
        summary.failed === 0 &&
        summary.skipped === summary.totalVideos;
      setNotice(
        allSkipped
          ? `AI 索引已是最新：${summary.skipped} 个视频已有向量，未重复处理`
          : `${forceRebuild ? "AI 索引重建完成" : "AI 索引完成"}：处理 ${summary.processed}，跳过 ${summary.skipped}，抽帧不足 ${insufficientFrames}，失败 ${summary.failed}`,
      );
      await refreshIndexData(aiBuildSessionIds ?? null);
      setNotice((current) => `${current ?? "AI 索引完成"}，相似结果请手动刷新`);
      setAiIndexProgress({
        phase: "completed",
        totalVideos: summary.totalVideos,
        processed: summary.processed,
        skipped: summary.skipped,
        insufficientFrames: summary.insufficientFrames ?? 0,
        failed: summary.failed,
        started: summary.totalVideos,
        prepared: summary.totalVideos,
        currentPath: null,
      });
    } catch (error) {
      setNotice(String(error));
    } finally {
      await releaseRamDiskAfterTask();
      setIsScanning(false);
      setIsIndexingAi(false);
      setIsCancelling(false);
    }
  }

  function requestConfirm(dialog: ConfirmDialog): Promise<boolean> {
    confirmResolver.current?.(false);
    setConfirmDialog(dialog);
    return new Promise((resolve) => {
      confirmResolver.current = resolve;
    });
  }

  function settleConfirm(value: boolean) {
    confirmResolver.current?.(value);
    confirmResolver.current = null;
    setConfirmDialog(null);
  }

  function addScanPath(path = source) {
    const trimmed = path.trim();
    if (!trimmed) {
      setNotice("请输入或选择要扫描的路径");
      return;
    }
    setScanPaths((current) => uniquePaths([...current, trimmed]));
    if (path === source) setSource("");
  }

  function removeScanPath(path: string) {
    setScanPaths((current) => current.filter((item) => pathKey(item) !== pathKey(path)));
    if (pathKey(source) === pathKey(path)) setSource("");
  }

  async function chooseScanFolder() {
    setIsPickingFolder(true);
    try {
      const selected = await pickFolders(source.trim() || undefined);
      if (selected.length > 0) {
        setSource(selected[0]);
        setScanPaths((current) => uniquePaths([...current, ...selected]));
      }
    } catch (error) {
      setNotice(String(error));
    } finally {
      setIsPickingFolder(false);
    }
  }

  async function chooseBackupFolder() {
    setIsPickingFolder(true);
    try {
      const selected = await pickFolder(settingsDraft?.backupDir || undefined);
      if (selected && settingsDraft) {
        setSettingsDraft({ ...settingsDraft, backupDir: selected });
      }
    } catch (error) {
      setNotice(String(error));
    } finally {
      setIsPickingFolder(false);
    }
  }

  async function chooseLocalPreprocessTempFolder() {
    setIsPickingFolder(true);
    try {
      const selected = await pickFolder(settingsDraft?.localPreprocessTempDir || undefined);
      if (selected && settingsDraft) {
        setSettingsDraft({ ...settingsDraft, localPreprocessTempDir: selected });
      }
    } catch (error) {
      setNotice(String(error));
    } finally {
      setIsPickingFolder(false);
    }
  }

  async function chooseLocalPreprocessSecondaryTempFolder() {
    setIsPickingFolder(true);
    try {
      const selected = await pickFolder(settingsDraft?.localPreprocessSecondaryTempDir || undefined);
      if (selected && settingsDraft) {
        setSettingsDraft({ ...settingsDraft, localPreprocessSecondaryTempDir: selected });
      }
    } catch (error) {
      setNotice(String(error));
    } finally {
      setIsPickingFolder(false);
    }
  }

  function addNamingFolder(path = namingFolderInput) {
    const trimmed = path.trim();
    if (!trimmed || !settingsDraft) return;
    setSettingsDraft({
      ...settingsDraft,
      namingSourceDirs: uniquePaths([...(settingsDraft.namingSourceDirs ?? []), trimmed]),
    });
    setNamingFolderInput("");
  }

  function removeNamingFolder(path: string) {
    if (!settingsDraft) return;
    setSettingsDraft({
      ...settingsDraft,
      namingSourceDirs: (settingsDraft.namingSourceDirs ?? []).filter(
        (item) => pathKey(item) !== pathKey(path),
      ),
    });
  }

  async function chooseNamingFolder() {
    setIsPickingFolder(true);
    try {
      const selected = await pickFolder(namingFolderInput || undefined);
      if (selected && settingsDraft) {
        setSettingsDraft({
          ...settingsDraft,
          namingSourceDirs: uniquePaths([...(settingsDraft.namingSourceDirs ?? []), selected]),
        });
        setNamingFolderInput("");
      }
    } catch (error) {
      setNotice(String(error));
    } finally {
      setIsPickingFolder(false);
    }
  }

  async function handleScan() {
    const targets = queuedScanTargets;
    if (targets.length === 0) {
      setNotice("请输入或选择要扫描的路径");
      return;
    }

    const activeSettings = await ensureSettingsSavedBeforeTask(
      settingsLanguage === "en" ? "starting the scan" : "开始扫描",
    );
    if (!activeSettings) return;

    setIsScanning(true);
    setNotice(null);
    setScanSummary(null);
    setScanProgress({
      source: targets[0],
      phase: "starting",
      totalFiles: 0,
      scanned: 0,
      failed: 0,
      currentPath: null,
    });

    try {
      const summaries = await scanSources(targets);
      const summary: ScanSummary = {
        sessionId: summaries.length === 1 ? summaries[0].sessionId : null,
        source: summaries.length === 1 ? summaries[0].source : `${summaries.length} 个路径`,
        databasePath: summaries[0]?.databasePath ?? status?.databasePath ?? "data\\index.sqlite",
        totalFiles: summaries.reduce((sum, item) => sum + item.totalFiles, 0),
        scanned: summaries.reduce((sum, item) => sum + item.scanned, 0),
        failed: summaries.reduce((sum, item) => sum + item.failed, 0),
        elapsedMs: summaries.reduce((sum, item) => sum + item.elapsedMs, 0),
      };
      setScanSummary(summary);
      setScanProgress({
        source: summary.source,
        phase: "completed",
        totalFiles: summary.totalFiles,
        scanned: summary.scanned,
        failed: summary.failed,
        currentPath: null,
      });
      const scannedSessionIds = summaries
        .map((item) => item.sessionId)
        .filter((id): id is number => id !== null);
      setScopeSessionIds(scannedSessionIds);
      setScanPaths([]);
      setSource("");
      if (false && settings?.aiVisionEnabled && settings?.aiIndexAfterScan) {
        const sessionIds = summaries
          .map((item) => item.sessionId)
          .filter((id): id is number => id !== null);
        if (sessionIds.length > 0) {
          setIsIndexingAi(true);
          setAiIndexSummary(null);
          try {
            const aiSummary = await buildAiIndex(sessionIds);
            setAiIndexSummary(aiSummary);
            setAiIndexProgress({
              phase: "completed",
              totalVideos: aiSummary.totalVideos,
              processed: aiSummary.processed,
              skipped: aiSummary.skipped,
              insufficientFrames: aiSummary.insufficientFrames ?? 0,
              failed: aiSummary.failed,
              started: aiSummary.totalVideos,
              prepared: aiSummary.totalVideos,
              currentPath: null,
            });
          } catch (error) {
            setNotice(`扫描已完成，但 AI 索引失败：${String(error)}`);
          } finally {
            setIsIndexingAi(false);
          }
        }
      }
      await refreshIndexData(scannedSessionIds, { clearMatches: true });
      setNotice("扫描完成，索引已更新；AI 索引和相似结果请手动触发");
    } catch (error) {
      setNotice(String(error));
    } finally {
      await releaseRamDiskAfterTask();
      setIsScanning(false);
      setIsCancelling(false);
    }
  }

  async function handleCancelCurrentWork() {
    if (!isScanning && !isIndexingAi) return;
    setIsCancelling(true);
    setNotice("正在终止扫描任务，当前正在执行的文件步骤会尽快结束");
    try {
      await cancelCurrentWork();
    } catch (error) {
      setNotice(String(error));
      setIsCancelling(false);
    }
  }

  async function releaseRamDiskAfterTask() {
    try {
      const nextStatus = await releaseRamDisk();
      setRamDiskStatus(nextStatus);
    } catch (error) {
      const message = `内存缓存盘卸载失败：${String(error)}`;
      setNotice((currentNotice) => (currentNotice ? `${currentNotice}；${message}` : message));
      try {
        setRamDiskStatus(await getRamDiskStatus());
      } catch {
        // Keep the release failure visible.
      }
    }
  }

  async function handleRescanSession(session: ScanSession) {
    const activeSettings = await ensureSettingsSavedBeforeTask(
      settingsLanguage === "en" ? "rescanning" : "重新扫描",
    );
    if (!activeSettings) return;

    setIsScanning(true);
    setNotice(null);
    setScanSummary(null);
    setScanProgress({
      source: session.source,
      phase: "starting",
      totalFiles: 0,
      scanned: 0,
      failed: 0,
      currentPath: null,
    });

    try {
      const [summary] = await scanSources([session.source]);
      setScanSummary(summary);
      setScanProgress({
        source: summary.source,
        phase: "completed",
        totalFiles: summary.totalFiles,
        scanned: summary.scanned,
        failed: summary.failed,
        currentPath: null,
      });
      await refreshIndexData(scopeSessionIds, { clearMatches: true });
      setNotice(`已重新扫描：${shortPath(session.source)}`);
    } catch (error) {
      setNotice(String(error));
    } finally {
      await releaseRamDiskAfterTask();
      setIsScanning(false);
      setIsCancelling(false);
    }
  }

  async function copyReport() {
    if (!selectedGroup) return;
    try {
      await writeClipboardText(selectedGroup.report);
      setNotice("报告已复制");
    } catch {
      setManualCopyText(selectedGroup.report);
      setNotice("报告已生成，请手动复制");
    }
    window.setTimeout(() => setNotice(null), 1800);
  }

  async function handleOpenVideo(video: VideoRecord) {
    try {
      await openVideo(video.path);
    } catch (error) {
      setNotice(String(error));
    }
  }

  async function handleSelectedResolution(disposal: "backup" | "delete") {
    if (!selectedGroup) return;
    const selectedItems = selectedGroup.items.filter((item) => {
      const id = videoId(item.video);
      return id !== null && selectedVideoIds.includes(id);
    });
    if (selectedItems.length < 2) {
      setNotice("请至少选择两个视频；底部按钮只会在已选视频范围内执行并入。");
      return;
    }

    if (disposal === "delete" && !settings?.allowDirectDelete) {
      setNotice("仅删除已关闭，请先在 Settings 中启用");
      return;
    }

    const selectedIdSet = new Set(
      selectedItems
        .map((item) => videoId(item.video))
        .filter((id): id is number => id !== null),
    );
    const overriddenKeeperId = keeperOverrides[selectedGroup.id];
    const selectedKeeper =
      selectedItems.find((item) => item.video.id === overriddenKeeperId) ??
      selectedItems.find((item) => item.video.id === selectedGroup.recommendedVideoId) ??
      [...selectedItems].sort((left, right) => left.qualityRank - right.qualityRank)[0];
    const keeperId = selectedKeeper ? videoId(selectedKeeper.video) : null;

    if (!selectedKeeper || keeperId === null) {
      setNotice("已选范围内缺少可保留的视频 ID");
      return;
    }
    const selectedNonKeepers = selectedItems.filter((item) => videoId(item.video) !== keeperId);
    if (selectedNonKeepers.length === 0) {
      setNotice("请同时选择要合并处理的候选视频");
      return;
    }

    const namingSourceDirs = settings?.namingSourceDirs ?? [];
    const overriddenSourceId = namingOverrides[selectedGroup.id];
    const suggestedSource = namingSourceForGroup(
      selectedGroup,
      namingOverrides,
      namingSourceDirs,
      keeperId,
      scanSourcePaths,
    );
    const targetItem =
      selectedItems.find((item) => item.video.id === overriddenSourceId) ??
      selectedItems.find((item) => item.video.id === suggestedSource?.video.id) ??
      [...selectedItems].sort((left, right) =>
        compareNamingSourceItems(left, right, selectedItems, namingSourceDirs, scanSourcePaths),
      )[0];
    const targetId = videoId(targetItem.video);
    const filenameSourceId = filenameOverrides[selectedGroup.id] ?? null;

    if (targetId === null) {
      setNotice("已选范围内的路径来源视频缺少可用 ID");
      return;
    }

    if (!selectedIdSet.has(targetId)) {
      setNotice("路径来源必须在本次已选视频范围内");
      return;
    }

    if (targetId === keeperId && (filenameSourceId === null || filenameSourceId === keeperId)) {
      await runFileAction(
        disposal,
        selectedNonKeepers
          .map((item) => videoId(item.video))
          .filter((id): id is number => id !== null),
        true,
      );
      return;
    }

    const extraIds = selectedNonKeepers
      .map((item) => videoId(item.video))
      .filter((id): id is number => id !== null && id !== targetId);

    const ok = await requestConfirm({
      title: disposal === "backup" ? "并入并备份已选视频" : "并入并删除已选视频",
      body:
        disposal === "backup"
          ? `只在 ${selectedItems.length} 个已选视频内处理：保留 ${selectedKeeper.video.fileName}，并入 ${targetItem.video.fileName} 的路径；未选中的候选视频保持不变。被替换的视频会移入备份文件夹。`
          : `只在 ${selectedItems.length} 个已选视频内处理：保留 ${selectedKeeper.video.fileName}，并入 ${targetItem.video.fileName} 的路径；未选中的候选视频保持不变。被替换的视频会直接删除。`,
      confirmLabel: disposal === "backup" ? "并入并备份" : "并入并删除",
      danger: disposal === "delete",
    });
    if (!ok) return;

    setOperationProgress(
      disposal === "backup" ? "正在并入已选视频并移动备份" : "正在并入已选视频并删除候选",
    );
    setIsExecuting(true);
    try {
      const result = await executeMergeSelection(
        keeperId,
        targetId,
        extraIds,
        disposal,
        disposal === "backup" ? "MERGE_BACKUP" : "MERGE_DELETE",
        filenameSourceId,
      );
      setNotice(`${result.status}: ${result.messages.length} 个文件已处理；当前相似列表将在手动刷新后更新`);
      setSelectedVideoIds([]);
    } catch (error) {
      setNotice(String(error));
    } finally {
      setOperationProgress(null);
      setIsExecuting(false);
    }
  }

  async function handleBatchGroupResolution(disposal: "backup" | "delete") {
    const selectedGroups = activeGroups.filter((group) => {
      const numeric = Number(group.id.replace("group-", ""));
      return batchGroupIds.includes(numeric);
    });
    if (selectedGroups.length === 0) {
      setNotice("请先在左侧相似结果里勾选要批量处理的组");
      return;
    }
    if (disposal === "delete" && !settings?.allowDirectDelete) {
      setNotice("仅删除已关闭，请先在 Settings 中启用");
      return;
    }

    const tasks: Array<BatchMergeTask & { title: string }> = [];
    const plannedTaskLogs: BatchTaskLogEntry[] = [];
    let batchLogId = 0;
    let plannedSkipped = 0;
    const claimedVideoIds = new Set<number>();

    for (const group of selectedGroups) {
      const recommendedId = recommendedIdForGroup(group, keeperOverrides);
      const keeper = group.items.find((item) => item.video.id === recommendedId);
      const keeperId = keeper ? videoId(keeper.video) : null;
      if (keeperId === null) {
        plannedTaskLogs.push({
          id: ++batchLogId,
          title: group.title,
          status: "skipped",
          reason: "缺少可保留的视频 ID",
        });
        plannedSkipped += 1;
        continue;
      }
      const groupVideoIds = group.items
        .map((item) => videoId(item.video))
        .filter((id): id is number => id !== null);
      if (groupVideoIds.some((id) => claimedVideoIds.has(id))) {
        plannedSkipped += 1;
        plannedTaskLogs.push({
          id: ++batchLogId,
          title: group.title,
          status: "skipped",
          reason: "与前序批量任务引用了相同视频",
          keeperId,
        });
        continue;
      }

      const sourceItem = namingSourceForGroup(
        group,
        namingOverrides,
        settings?.namingSourceDirs ?? [],
        recommendedId,
        scanSourcePaths,
      );
      const sourceId = sourceItem ? videoId(sourceItem.video) : null;
      const nonKeeperIds = group.items
        .map((item) => videoId(item.video))
        .filter((id): id is number => id !== null && id !== keeperId);
      if (nonKeeperIds.length === 0) {
        plannedSkipped += 1;
        plannedTaskLogs.push({
          id: ++batchLogId,
          title: group.title,
          status: "skipped",
          reason: "没有可合并处理的非保留候选",
          keeperId,
        });
        continue;
      }
      groupVideoIds.forEach((id) => claimedVideoIds.add(id));

      const backendIndex = tasks.length + 1;
      tasks.push({
        title: group.title,
        keeperId,
        namingSourceId: sourceId ?? keeperId,
        filenameSourceId: filenameOverrides[group.id] ?? null,
        extraVideoIds:
          sourceId !== null && sourceId !== keeperId
            ? nonKeeperIds.filter((id) => id !== sourceId)
            : nonKeeperIds,
      });
      plannedTaskLogs.push({
        id: ++batchLogId,
        backendIndex,
        title: group.title,
        status: "pending",
        keeperId,
        namingSourceId: sourceId ?? keeperId,
        filenameSourceId: filenameOverrides[group.id] ?? null,
        extraCount:
          sourceId !== null && sourceId !== keeperId
            ? nonKeeperIds.filter((id) => id !== sourceId).length
            : nonKeeperIds.length,
      });
    }

    if (tasks.length === 0) {
      setBatchTaskLogs(plannedTaskLogs);
      setNotice(
        plannedSkipped > 0
          ? `所选相似组都与前序任务引用了相同视频，已跳过 ${plannedSkipped} 组`
          : "所选相似组没有可处理的候选视频",
      );
      return;
    }

    const ok = await requestConfirm({
      title: disposal === "backup" ? "批量备份替换" : "批量删除替换",
      body:
        disposal === "backup"
          ? `将按当前保留项和路径源处理 ${tasks.length} 个相似组；被替换或移除的视频会进入备份文件夹。`
          : `将按当前保留项和路径源处理 ${tasks.length} 个相似组；被替换或移除的视频会直接删除。默认取消。`,
      confirmLabel: disposal === "backup" ? "批量备份替换" : "批量删除替换",
      danger: disposal === "delete",
    });
    if (!ok) return;

    setIsExecuting(true);
    setNotice(null);
    let completed = 0;
    let skipped = 0;
    let failed = 0;
    const publishBatchProgress = (phase: string, currentTitle?: string) => {
      const nextProgress = {
        phase,
        total: selectedGroups.length,
        processed: Math.min(
          selectedGroups.length,
          plannedSkipped + completed + skipped + failed,
        ),
        completed,
        skipped: plannedSkipped + skipped,
        failed,
        currentTitle,
      };
      setBatchProgress(nextProgress);
      logBatchProgress(nextProgress);
    };
    publishBatchProgress("准备批量处理");
    try {
      publishBatchProgress("后端批量处理中");
      setBatchTaskLogs((current) =>
        current.map((entry) =>
          entry.status === "pending" ? { ...entry, status: "running" } : entry,
        ),
      );
      await waitForPaint();
      const result = await executeBatchMergeSelection(
        tasks.map(({ title: _title, ...task }) => task),
        disposal,
        disposal === "backup" ? "MERGE_BACKUP" : "MERGE_DELETE",
      );
      const taskMessages = backendTaskMessages(result.messages);
      let backendCompleted = 0;
      let backendSkipped = 0;
      let backendFailed = 0;
      setBatchTaskLogs((current) =>
        current.map((entry) => {
          if (entry.status === "skipped" && !entry.backendIndex) return entry;
          const backendMessage = entry.backendIndex ? taskMessages.get(entry.backendIndex) : undefined;
          if (backendMessage) {
            if (backendMessage.status === "failed") backendFailed += 1;
            if (backendMessage.status === "skipped") backendSkipped += 1;
            return {
              ...entry,
              status: backendMessage.status,
              reason: backendMessage.reason,
            };
          }
          if (entry.status === "running" || entry.status === "pending") {
            backendCompleted += 1;
            return { ...entry, status: "completed" };
          }
          return entry;
        }),
      );
      completed = backendCompleted;
      skipped = backendSkipped;
      failed = backendFailed;
      setBatchGroupIds([]);
      setSelectedVideoIds([]);
      publishBatchProgress("后端批量处理完成");
      pushRuntimeLog(
        "后端批量处理完成",
        result.messages.join("\n"),
        result.status === "completed-with-failures" ? "warning" : "success",
      );
      setNotice(
        `${result.status}: ${result.messages.length} 条记录；批量任务已由后端一次性处理，当前相似列表将在手动刷新后更新`,
      );
      return;
      /*
      for (const task of tasks) {
        publishBatchProgress("正在处理", task.title);
        const indexedIds = new Set(
          (await listVideos(scopeSessionIds ?? undefined))
            .map((video) => videoId(video))
            .filter((id): id is number => id !== null),
        );
        const activeTask =
          task.type === "merge"
            ? {
                ...task,
                extraIds: task.extraIds.filter((id) => indexedIds.has(id)),
              }
            : {
                ...task,
                ids: task.ids.filter((id) => indexedIds.has(id)),
              };
        if (
          (activeTask.type === "merge" &&
            (!indexedIds.has(activeTask.keeperId) || !indexedIds.has(activeTask.sourceId))) ||
          (activeTask.type === "file" && activeTask.ids.length === 0)
        ) {
          skipped += 1;
          publishBatchProgress("已跳过失效任务", task.title);
          continue;
        }

        try {
          if (activeTask.type === "merge") {
            const result = await executeMergeSelection(
              activeTask.keeperId,
              activeTask.sourceId,
              activeTask.extraIds,
              disposal,
              disposal === "backup" ? "MERGE_BACKUP" : "MERGE_DELETE",
            );
            if (result.status === "skipped-stale") {
              skipped += 1;
              publishBatchProgress("已跳过失效任务", task.title);
              continue;
            }
          } else {
            const result = await executeFileAction(
              activeTask.ids,
              disposal,
              disposal === "backup" ? "MOVE" : "DELETE",
              true,
            );
            if (result.status === "skipped-stale") {
              skipped += 1;
              publishBatchProgress("已跳过失效任务", task.title);
              continue;
            }
          }
          completed += 1;
          publishBatchProgress("已完成当前任务", task.title);
        } catch (error) {
          failures.push(`${task.title}: ${String(error)}`);
          publishBatchProgress("当前任务失败", task.title);
        }
      }
      setBatchGroupIds([]);
      setSelectedVideoIds([]);
      publishBatchProgress("批量处理完成");
      setNotice(
        failures.length === 0
          ? `批量处理完成：${completed}/${selectedGroups.length} 组${plannedSkipped + skipped > 0 ? `，跳过 ${plannedSkipped + skipped} 个冲突或失效任务` : ""}；当前相似列表将在手动刷新后更新`
          : `批量处理完成 ${completed}/${selectedGroups.length} 组，跳过 ${plannedSkipped + skipped} 组，失败 ${failures.length} 组：${failures[0]}`,
      );
      */
    } finally {
      setIsExecuting(false);
    }
  }

  async function runFileAction(
    action: "backup" | "delete",
    ids = selectedVideoIds,
    deferIndexUpdate = false,
  ) {
    const validIds = ids.filter((id) => Number.isFinite(id));
    if (validIds.length === 0) {
      setNotice("请先选择候选视频");
      return;
    }

    let confirmation = "MOVE";
    if (action === "backup") {
      const ok = await requestConfirm({
        title: "移入备份",
        body: `将 ${validIds.length} 个候选视频移入备份文件夹：${settings?.backupDir ?? "data\\backups"}`,
        confirmLabel: "移入备份",
      });
      if (!ok) return;
    } else {
      if (!settings?.allowDirectDelete) {
        setNotice("直接删除已关闭，请先在 Settings 中启用");
        return;
      }
      const ok = await requestConfirm({
        title: "仅删除",
        body: `将仅删除 ${validIds.length} 个候选视频，不执行路径并入，也不会进入备份文件夹。默认取消。`,
        confirmLabel: "仅删除",
        danger: true,
      });
      if (!ok) return;
      confirmation = "DELETE";
    }

    setOperationProgress(action === "backup" ? "正在移入备份文件夹" : "正在删除候选视频");
    setIsExecuting(true);
    try {
      const result = await executeFileAction(validIds, action, confirmation, deferIndexUpdate);
      setNotice(
        deferIndexUpdate
          ? `${result.status}: ${result.messages.length} 个文件已处理；当前相似列表将在手动刷新后更新`
          : `${result.status}: ${result.messages.length} 个文件已处理`,
      );
      setSelectedVideoIds((current) => current.filter((id) => !validIds.includes(id)));
      if (!deferIndexUpdate) {
        await refreshIndexData(scopeSessionIds, { clearMatches: true });
      }
    } catch (error) {
      setNotice(String(error));
    } finally {
      setOperationProgress(null);
      setIsExecuting(false);
    }
  }

  async function saveSettingsDraft(showNotice = true): Promise<AppSettings | null> {
    if (!settingsDraft) return settings;
    try {
      const saved = await saveAppSettings(settingsDraft);
      setSettings(saved);
      setSettingsDraft(saved);
      if (showNotice) setNotice(settingsLanguage === "en" ? "Settings saved" : "设置已保存");
      return saved;
    } catch (error) {
      setNotice(String(error));
      return null;
    }
  }

  async function ensureSettingsSavedBeforeTask(taskLabel: string): Promise<AppSettings | null> {
    if (!settingsDirty) return settings ?? settingsDraft;
    const confirmed = await requestConfirm({
      title: settingsLanguage === "en" ? "Unsaved settings" : "设置尚未保存",
      body:
        settingsLanguage === "en"
          ? `There are unsaved setting changes. Save them before ${taskLabel} so this task uses the latest settings.`
          : `检测到尚未保存的设置更改。${taskLabel}前需要先保存，确保本次任务使用最新设置。`,
      confirmLabel: settingsLanguage === "en" ? "Save and continue" : "保存并继续",
    });
    if (!confirmed) return null;
    return saveSettingsDraft(false);
  }

  function handleSettingsScroll(scrollTop: number) {
    const delta = scrollTop - lastSettingsScrollTop.current;
    if (Math.abs(delta) >= 6) {
      setIsSettingsSaveBarVisible(delta > 0);
      lastSettingsScrollTop.current = scrollTop;
    }
  }

  async function handleConfigureRamDisk() {
    const sizeMb = Math.min(1_048_576, Math.max(512, Math.round(ramDiskSizeMb || 16 * 1024)));
    setRamDiskSizeMb(sizeMb);
    setIsConfiguringRamDisk(true);
    setNotice(null);
    try {
      const nextStatus = await configureRamDisk(sizeMb);
      const saved = await getAppSettings();
      setRamDiskStatus(nextStatus);
      setSettings(saved);
      setSettingsDraft(saved);
      setRamDiskSetupDismissed(false);
      setNotice(nextStatus.message);
    } catch (error) {
      setNotice(String(error));
      try {
        setRamDiskStatus(await getRamDiskStatus());
      } catch {
        // Keep the original setup error visible.
      }
    } finally {
      setIsConfiguringRamDisk(false);
    }
  }

  async function recheckRamDisk() {
    setIsConfiguringRamDisk(true);
    try {
      const next = await getRamDiskStatus();
      setRamDiskStatus(next);
      setNotice(next.message);
    } catch (error) {
      setNotice(String(error));
    } finally {
      setIsConfiguringRamDisk(false);
    }
  }

  async function skipRamDiskSetup() {
    const current = settings ?? settingsDraft;
    if (!current) {
      setRamDiskSetupDismissed(true);
      return;
    }
    try {
      const saved = await saveAppSettings({
        ...current,
        ramDiskEnabled: false,
        ramDiskSetupCompleted: true,
        localPreprocessTempDir: current.localPreprocessSecondaryTempDir,
      });
      setSettings(saved);
      setSettingsDraft(saved);
      setRamDiskSetupDismissed(true);
      setNotice("已暂时停用自动内存盘；可以稍后在设置中重新启用");
    } catch (error) {
      setNotice(String(error));
    }
  }

  async function refreshStoragePanel() {
    setIsMeasuringStorage(true);
    setNotice(null);
    try {
      const usage = await getStorageUsage();
      setStorageUsage(usage);
      setNotice(`${settingText.storageTitle}: ${formatBytes(usage.totalBytes)}`);
    } catch (error) {
      setNotice(String(error));
    } finally {
      setIsMeasuringStorage(false);
    }
  }

  async function handleCleanupStorage() {
    const ok = await requestConfirm({
      title: settingText.cleanupConfirmTitle,
      body: settingText.cleanupConfirmBody,
      confirmLabel: settingText.cleanupConfirmLabel,
      danger: true,
    });
    if (!ok) return;

    setIsCleaningStorage(true);
    setNotice(null);
    try {
      const summary = await cleanupStorage();
      setStorageCleanupSummary(summary);
      setVacuumSummary(null);
      const usage = await getStorageUsage();
      setStorageUsage(usage);
      setNotice(
        `${settingText.cleanupStorage}: ${formatBytes(
          summary.deletedAiFrameCacheBytes + summary.deletedThumbnailBytes,
        )}, ${storageDbLabel("pairScores", settingsLanguage)} ${summary.deletedPairScores.toLocaleString()}`,
      );
    } catch (error) {
      setNotice(String(error));
    } finally {
      setIsCleaningStorage(false);
    }
  }

  async function handleCleanupCompletedFrameCache() {
    const ok = await requestConfirm({
      title: settingText.cleanupCompletedFrameCacheConfirmTitle,
      body: settingText.cleanupCompletedFrameCacheConfirmBody,
      confirmLabel: settingText.cleanupCompletedFrameCacheConfirmLabel,
      danger: true,
    });
    if (!ok) return;

    setIsCleaningStorage(true);
    setNotice(null);
    try {
      const summary = await cleanupCompletedAiFrameCache();
      setCompletedFrameCacheCleanupSummary(summary);
      setVacuumSummary(null);
      const usage = await getStorageUsage();
      setStorageUsage(usage);
      setNotice(
        `${settingText.cleanupCompletedFrameCache}: ${formatBytes(
          summary.deletedBytes,
        )}, ${fileCountText(summary.deletedFiles, settingsLanguage)}`,
      );
    } catch (error) {
      setNotice(String(error));
    } finally {
      setIsCleaningStorage(false);
    }
  }

  async function handleVacuumDatabase() {
    const ok = await requestConfirm({
      title: settingText.vacuumConfirmTitle,
      body: settingText.vacuumConfirmBody,
      confirmLabel: settingText.vacuumConfirmLabel,
      danger: true,
    });
    if (!ok) return;

    setIsVacuumingDatabase(true);
    setNotice(null);
    try {
      const summary = await vacuumDatabase();
      setVacuumSummary(formatBytes(summary.reclaimedBytes));
      const usage = await getStorageUsage();
      setStorageUsage(usage);
      setNotice(`${settingText.vacuumDatabase}: ${formatBytes(summary.reclaimedBytes)}`);
    } catch (error) {
      setNotice(String(error));
    } finally {
      setIsVacuumingDatabase(false);
    }
  }

  function commitScopeSelection(ids: number[]) {
    const valid = uniqueNumbers(ids).filter((id) => allScopeSessionIds.includes(id));
    setScopeSessionIds(valid);
  }

  function toggleScopeCollapse(key: string) {
    setCollapsedScopeKeys((current) => {
      const next = new Set(current);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  }

  function collapseAllScopes() {
    setCollapsedScopeKeys(new Set(collapsibleScopeKeys));
  }

  function expandAllScopes() {
    setCollapsedScopeKeys(new Set());
  }

  function selectScopeRow(row: ScopeTreeRow, modifiers: SelectionModifiers) {
    if (row.node.sessionIds.length === 0) return;
    let next = new Set(selectedScopeSet);
    const rowIds = row.node.sessionIds;

    if (modifiers.shiftKey && scopeAnchorKey) {
      const anchorIndex = scopeRows.findIndex((item) => item.key === scopeAnchorKey);
      const rowIndex = scopeRows.findIndex((item) => item.key === row.key);
      if (anchorIndex !== -1 && rowIndex !== -1) {
        const [start, end] =
          anchorIndex <= rowIndex ? [anchorIndex, rowIndex] : [rowIndex, anchorIndex];
        const rangeIds = uniqueNumbers(
          scopeRows.slice(start, end + 1).flatMap((item) => item.node.sessionIds),
        );
        const fullySelected = rangeIds.length > 0 && rangeIds.every((id) => next.has(id));
        for (const id of rangeIds) {
          if (fullySelected) {
            next.delete(id);
          } else {
            next.add(id);
          }
        }
      }
    } else {
      const fullySelected = rowIds.every((id) => next.has(id));
      for (const id of rowIds) {
        if (fullySelected) {
          next.delete(id);
        } else {
          next.add(id);
        }
      }
    }

    setScopeAnchorKey(row.key);
    commitScopeSelection(Array.from(next));
  }

  function setAllVisibleGroups(selected: boolean) {
    setBatchGroupIds((current) =>
      selected
        ? Array.from(new Set([...current, ...visibleGroupIds]))
        : current.filter((id) => !visibleGroupIds.includes(id)),
    );
  }

  function toggleBatchGroup(group: MatchGroup) {
    const numeric = Number(group.id.replace("group-", ""));
    if (!Number.isFinite(numeric)) return;
    setBatchGroupIds((current) => toggleNumber(current, numeric));
  }

  function setCurrentGroupVideos(selected: boolean) {
    setSelectedVideoIds((current) =>
      selected
        ? Array.from(new Set([...current, ...currentGroupVideoIds]))
        : current.filter((id) => !currentGroupVideoIds.includes(id)),
    );
  }

  function toggleVideo(video: VideoRecord) {
    const id = videoId(video);
    if (id === null) return;
    setSelectedVideoIds((current) => toggleNumber(current, id));
  }

  function setNamingSource(group: MatchGroup, item: MatchItem) {
    const id = videoId(item.video);
    if (id === null) {
      setNotice("当前文件缺少可用 ID");
      return;
    }
    setNamingOverrides((current) => ({ ...current, [group.id]: id }));
    setNotice(`已将 ${item.video.fileName} 设为路径源`);
    window.setTimeout(() => setNotice(null), 1800);
  }

  function setFilenameSource(group: MatchGroup, item: MatchItem) {
    const id = videoId(item.video);
    if (id === null) {
      setNotice("当前文件缺少可用 ID");
      return;
    }
    setFilenameOverrides((current) => ({ ...current, [group.id]: id }));
    setNotice(`已将 ${item.video.fileName} 设为文件名来源`);
    window.setTimeout(() => setNotice(null), 1800);
  }

  function temporarilyIgnoreVideo(item: MatchItem) {
    const id = videoId(item.video);
    if (id === null) {
      setNotice("当前文件缺少可用 ID");
      return;
    }
    setTemporarilyIgnoredVideoIds((current) => (current.includes(id) ? current : [...current, id]));
    setSelectedVideoIds((current) => current.filter((value) => value !== id));
    setFilenameOverrides((current) =>
      Object.fromEntries(Object.entries(current).filter(([, value]) => value !== id)),
    );
    setNotice(`本轮暂不修改 ${item.video.fileName}；刷新比对后会重新参与`);
    window.setTimeout(() => setNotice(null), 2200);
  }

  function setKeeper(group: MatchGroup, item: MatchItem) {
    const id = videoId(item.video);
    if (id === null) {
      setNotice("当前文件缺少可用 ID");
      return;
    }
    setKeeperOverrides((current) => ({ ...current, [group.id]: id }));
    setNotice(`已将 ${item.video.fileName} 设为保留`);
    window.setTimeout(() => setNotice(null), 1800);
  }

  async function handleDeleteScopeNode(node: ScopeTreeNode) {
    const ids = uniqueNumbers(node.sessionIds);
    if (ids.length === 0) return;
    const ok = await requestConfirm({
      title: "删除索引源",
      body:
        ids.length === 1
          ? `只删除本地索引记录，不删除 NAS 上的视频文件：${node.path}`
          : `将删除这个范围下的 ${ids.length} 个本地索引源，不删除 NAS 上的视频文件：${node.path}`,
      confirmLabel: ids.length === 1 ? "删除索引" : "删除此范围",
      danger: true,
    });
    if (!ok) return;

    const nextScope = scopeSessionIds === null
      ? null
      : scopeSessionIds.filter((id) => !ids.includes(id));
    setIsExecuting(true);
    setNotice(null);
    setOperationProgress(`准备删除 ${ids.length} 个索引源`);
    try {
      const summary = await deleteScanSessions(ids);
      setScopeSessionIds(nextScope);
      setOperationProgress("正在读取删除后的索引状态");
      await refreshIndexData(nextScope, { clearMatches: true });
      setNotice(
        `索引源已删除：路径 ${summary.deletedSessions}/${summary.requestedSessions}，视频 ${summary.deletedVideos}/${summary.affectedVideos}`,
      );
    } catch (error) {
      setNotice(String(error));
    } finally {
      setOperationProgress(null);
      setIsExecuting(false);
    }
  }

  async function handleRefreshIndexSources() {
    try {
      await refreshIndexData(scopeSessionIds, { pruneMissing: true, clearMatches: true });
    } catch (error) {
      setNotice(String(error));
    }
  }

  async function handleLightRefresh() {
    try {
      await refreshIndexData(scopeSessionIds);
    } catch (error) {
      setNotice(String(error));
    }
  }

  async function handleRollback(entry: OperationHistoryEntry) {
    const ok = await requestConfirm({
      title: "回滚操作",
      body: `将按历史记录反向移动文件并恢复索引：${entry.summary}`,
      confirmLabel: "回滚",
      danger: true,
    });
    if (!ok) return;

    setIsExecuting(true);
    try {
      const result = await rollbackOperation(entry.operationId);
      setNotice(`${result.status}: ${result.messages.length} 个步骤已回滚`);
      await refreshIndexData(scopeSessionIds, { clearMatches: true });
    } catch (error) {
      setNotice(String(error));
    } finally {
      setIsExecuting(false);
    }
  }

  const activeScopeLabel =
    scopeSessionIds === null
      ? settingText.allIndexes
      : scopeSessionIds.length === 0
        ? settingText.noPathSelected
        : `${scopeSessionIds.length} ${settingText.pathUnit}`;

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <FileVideo size={20} strokeWidth={2.2} />
          </div>
          <div>
            <strong>Duplicate Video Search</strong>
            <span>v2.5.5 · settings save guard</span>
          </div>
        </div>

        <nav className="nav">
          <button className={`nav-item ${view === "matches" ? "active" : ""}`} onClick={() => setView("matches")}>
            <ShieldCheck size={18} />
            {settingText.matchesNav}
          </button>
          <button className={`nav-item ${view === "library" ? "active" : ""}`} onClick={() => setView("library")}>
            <Database size={18} />
            {settingText.libraryNav}
          </button>
          <button className={`nav-item ${view === "ai" ? "active" : ""}`} onClick={() => setView("ai")}>
            <ImageIcon size={18} />
            {settingText.aiNav}
          </button>
          <button className={`nav-item ${view === "history" ? "active" : ""}`} onClick={() => setView("history")}>
            <History size={18} />
            {settingText.historyNav}
          </button>
          <button className={`nav-item ${view === "settings" ? "active" : ""}`} onClick={() => setView("settings")}>
            <Settings size={18} />
            {settingText.settingsNav}
          </button>
        </nav>

        <div className="status-box">
          <div className="status-row">
            <span>FFmpeg</span>
            {status?.ffmpegPath ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}
          </div>
          <div className="status-row">
            <span>FFprobe</span>
            {status?.ffprobePath ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}
          </div>
          <div className="path-line">{status?.databasePath ?? "data\\index.sqlite"}</div>
        </div>
      </aside>

      <main className="workspace">
        <header className="toolbar">
          <div className="source-control">
            <FolderOpen size={18} />
            <input
              value={source}
              onChange={(event) => setSource(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") addScanPath();
              }}
              placeholder={settingText.scanPlaceholder}
            />
            <button
              className="inline-icon-button"
              onClick={chooseScanFolder}
              disabled={isPickingFolder || isScanning}
              title={settingText.chooseScanFolder}
            >
              <FolderPlus size={17} />
            </button>
          </div>
          <button className="secondary-button" onClick={() => addScanPath()} disabled={isScanning || !source.trim()}>
            <FolderPlus size={18} />
            {settingText.addPath}
          </button>
          {(isScanning || isIndexingAi) && (
            <button
              className="secondary-button"
              onClick={handleCancelCurrentWork}
              disabled={isCancelling}
              title={settingText.stopCurrentWork}
            >
              <Square size={18} />
              {isCancelling ? settingText.stopping : settingText.stopScan}
            </button>
          )}
          <button className="primary-button" onClick={handleScan} disabled={isScanning || queuedScanTargets.length === 0}>
            <Search size={18} />
            {isScanning ? settingText.scanning : settingText.startScan}
          </button>
        </header>

        {scanPaths.length > 0 && (
          <section className="scan-queue" aria-label="scan paths">
            <div className="scan-queue-head">
              <strong>{settingText.scanQueue}</strong>
              <span>{queuedScanTargets.length} {settingText.scanQueueCount}</span>
            </div>
            <div className="path-chip-list">
              {queuedScanTargets.map((path) => (
                <span className="path-chip" key={path}>
                  {shortPath(path)}
                  {scanPaths.some((item) => pathKey(item) === pathKey(path)) && (
                    <button onClick={() => removeScanPath(path)} disabled={isScanning} title={settingText.removePath}>
                      <X size={14} />
                    </button>
                  )}
                </span>
              ))}
            </div>
          </section>
        )}

        <section className="metrics" aria-label="metrics">
          <div className="metric">
            <span>{settingText.indexed}</span>
            <strong>{videos.length}</strong>
          </div>
          <div className="metric">
            <span>{settingText.similarGroups}</span>
            <strong>{groups.length}</strong>
          </div>
          <div className="metric">
            <span>{settingText.reclaimable}</span>
            <strong>{formatBytes(totalReclaimable)}</strong>
          </div>
          <div className="metric wide">
            <span>{settingText.compareScope}</span>
            <strong>{activeScopeLabel}</strong>
          </div>
        </section>

        {(isScanning || scanProgress || scanSummary) && (
          <section className="scan-panel">
            <div className="scan-panel-head">
              <span>{isScanning ? "正在扫描" : "最近一次扫描"}</span>
              <strong>
                {scanProgress?.scanned ?? scanSummary?.scanned ?? 0}/
                {scanProgress?.totalFiles ?? scanSummary?.totalFiles ?? 0}
              </strong>
              {canDismissScanProgress && (
                <button
                  className="progress-dismiss"
                  onClick={() => {
                    setScanProgress(null);
                    setScanSummary(null);
                  }}
                  title="关闭进度"
                >
                  <X size={14} />
                </button>
              )}
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${scanPercent}%` }} />
            </div>
            <div className="scan-panel-foot">
              <span>{scanProgress?.currentPath ? shortPath(scanProgress.currentPath) : scanProgress?.phase ?? "idle"}</span>
              <span>
                总共 {scanProgress?.totalFiles ?? scanSummary?.totalFiles ?? 0} · 失败{" "}
                {scanProgress?.failed ?? scanSummary?.failed ?? 0}
              </span>
            </div>
          </section>
        )}

        {batchProgress && (
          <section className="scan-panel batch-progress-panel">
            <div className="scan-panel-head">
              <span>{isExecuting ? "正在批量处理" : "最近一次批量处理"}</span>
              <strong>
                {batchProgress.processed}/{batchProgress.total}
              </strong>
              {canDismissBatchProgress && (
                <button className="progress-dismiss" onClick={() => setBatchProgress(null)} title="关闭进度">
                  <X size={14} />
                </button>
              )}
            </div>
            <div className="progress-track">
              <div
                className={`progress-fill ${batchProgressIndeterminate ? "indeterminate" : ""}`}
                style={batchProgressIndeterminate ? undefined : { width: `${batchPercent}%` }}
              />
            </div>
            <div className="scan-panel-foot">
              <span>
                {batchProgress.currentTitle
                  ? shortPath(batchProgress.currentTitle)
                  : batchProgress.phase}
              </span>
              <span>
                完成 {batchProgress.completed} · 跳过 {batchProgress.skipped} · 失败{" "}
                {batchProgress.failed}
              </span>
            </div>
          </section>
        )}

        {operationProgress && !batchProgress && (
          <section className="scan-panel operation-progress-panel">
            <div className="scan-panel-head">
              <span>正在处理文件操作</span>
              <strong>进行中</strong>
              {canDismissOperationProgress && (
                <button className="progress-dismiss" onClick={() => setOperationProgress(null)} title="关闭进度">
                  <X size={14} />
                </button>
              )}
            </div>
            <div className="progress-track">
              <div className="progress-fill indeterminate" />
            </div>
            <div className="scan-panel-foot">
              <span>{operationProgress}</span>
              <span>当前操作完成后界面会更新状态</span>
            </div>
          </section>
        )}

        {refreshProgress && !isExecuting && (
          <section className="scan-panel refresh-progress-panel">
            <div className="scan-panel-head">
              <span>正在刷新</span>
              <strong>进行中</strong>
              {canDismissRefreshProgress && (
                <button
                  className="progress-dismiss"
                  onClick={() => {
                    setRefreshProgress(null);
                    setRefreshProgressPercent(null);
                  }}
                  title="关闭进度"
                >
                  <X size={14} />
                </button>
              )}
            </div>
            <div className="progress-track">
              <div
                className={`progress-fill ${refreshProgressPercent === null ? "indeterminate" : ""}`}
                style={
                  refreshProgressPercent === null
                    ? undefined
                    : { width: `${refreshProgressPercent}%` }
                }
              />
            </div>
            <div className="scan-panel-foot">
              <span>{refreshProgress}</span>
              <span>刷新期间不会执行文件写入</span>
            </div>
          </section>
        )}

        {(isIndexingAi || aiIndexProgress || aiIndexSummary) && (
          <section className="scan-panel ai-progress-panel">
            <div className="scan-panel-head">
              <span>{isIndexingAi ? "正在构建 AI 索引" : "最近一次 AI 索引"}</span>
              <strong>
                {aiIndexCounts.prepared}/{aiIndexCounts.total}
              </strong>
              {canDismissAiIndexProgress && (
                <button
                  className="progress-dismiss"
                  onClick={() => {
                    setAiIndexProgress(null);
                    setAiIndexSummary(null);
                  }}
                  title="关闭进度"
                >
                  <X size={14} />
                </button>
              )}
            </div>
            <div className="progress-track">
              <div
                className={`progress-fill ${isIndexingAi ? "active" : ""}`}
                style={{ width: `${aiIndexPercent}%` }}
              />
            </div>
            <div className="scan-panel-foot">
              <span
                className="progress-current-file"
                title={aiIndexProgress?.currentPath ?? aiIndexProgress?.phase ?? "idle"}
              >
                {aiIndexProgress?.currentPath
                  ? compactProgressFileName(aiIndexProgress.currentPath)
                  : aiIndexProgress?.phase ?? "idle"}
              </span>
              <span>
                总共 {aiIndexCounts.total} · 完成 {aiIndexCounts.prepared} · 处理{" "}
                {aiIndexProgress?.processed ?? aiIndexSummary?.processed ?? 0} · 跳过{" "}
                {aiIndexProgress?.skipped ?? aiIndexSummary?.skipped ?? 0} · 抽帧不足{" "}
                {aiIndexProgress?.insufficientFrames ?? aiIndexSummary?.insufficientFrames ?? 0} · 失败{" "}
                {aiIndexProgress?.failed ?? aiIndexSummary?.failed ?? 0}
              </span>
            </div>
          </section>
        )}

        {notice && <div className="notice">{notice}</div>}

        {view === "matches" && (
          <section
            className="content-grid"
            style={
              {
                gridTemplateColumns: `${groupListWidth}px 10px minmax(0, 1fr)`,
                "--group-list-width": `${groupListWidth}px`,
              } as CSSProperties
            }
          >
            <div className="group-list">
              <div className="section-title">
                <span>相似结果</span>
                <div className="section-actions">
                  <small>{groups.length} groups</small>
                  <button
                    className="tiny-button"
                    onClick={refreshMatches}
                    disabled={isScanning || isExecuting || Boolean(refreshProgress)}
                    title="重新读取索引并刷新相似结果"
                  >
                    <RefreshCw size={13} />
                    刷新
                  </button>
                </div>
              </div>
              <div className="match-filter-shell">
                <button
                  className="match-filter-toggle"
                  type="button"
                  aria-expanded={matchFiltersExpanded}
                  aria-controls="match-filter-fields"
                  onClick={() => setMatchFiltersExpanded((value) => !value)}
                >
                  <span>筛选与匹配</span>
                  <small>
                    {Math.round(Number(minConfidenceText) || 0)}–{Math.round(Number(maxConfidenceText) || 0)}%
                    · AI {Math.round(Number(aiMatchThresholdText) || 0)}% · {normalizedAiMinMatchedFrames()} 帧
                  </small>
                  {matchFiltersExpanded ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
                </button>
                {matchFiltersExpanded && (
                  <div className="match-controls" id="match-filter-fields">
                    <label>
                      <span>最低相似度</span>
                      <div className="percent-input">
                        <input
                          type="number"
                          min={0}
                          max={100}
                          step={1}
                          value={minConfidenceText}
                          onChange={(event) => setMinConfidenceText(event.target.value)}
                          onBlur={() => {
                            const next = normalizedMinConfidence();
                            setMinConfidenceText(String(Math.round(next * 100)));
                          }}
                          onKeyDown={(event) => {
                            if (event.key === "Enter") refreshMatches();
                          }}
                        />
                        <span>%</span>
                      </div>
                    </label>
                    <label>
                      <span>最高相似度</span>
                      <div className="percent-input">
                        <input
                          type="number"
                          min={0}
                          max={100}
                          step={1}
                          value={maxConfidenceText}
                          onChange={(event) => setMaxConfidenceText(event.target.value)}
                          onBlur={() => {
                            const next = clampConfidencePercent(Number(maxConfidenceText)) / 100;
                            setMaxConfidenceText(String(Math.round(next * 100)));
                          }}
                          onKeyDown={(event) => {
                            if (event.key === "Enter") refreshMatches();
                          }}
                        />
                        <span>%</span>
                      </div>
                    </label>
                    <label>
                      <span>AI 识别阈值</span>
                      <div className="percent-input">
                        <input
                          type="number"
                          min={0}
                          max={99}
                          step={1}
                          value={aiMatchThresholdText}
                          onChange={(event) => setAiMatchThresholdText(event.target.value)}
                          onBlur={() =>
                            setAiMatchThresholdText(String(Math.round(normalizedAiMatchThreshold() * 100)))
                          }
                          onKeyDown={(event) => {
                            if (event.key === "Enter") refreshMatches();
                          }}
                        />
                        <span>%</span>
                      </div>
                    </label>
                    <label>
                      <span>最低匹配帧数</span>
                      <input
                        type="number"
                        min={1}
                        max={128}
                        step={1}
                        value={aiMinMatchedFramesText}
                        onChange={(event) => setAiMinMatchedFramesText(event.target.value)}
                        onBlur={() => setAiMinMatchedFramesText(String(normalizedAiMinMatchedFrames()))}
                        onKeyDown={(event) => {
                          if (event.key === "Enter") refreshMatches();
                        }}
                      />
                    </label>
                    <label>
                      <span>排序</span>
                      <select
                        value={groupSort}
                        onChange={(event) => setGroupSort(event.target.value as GroupSort)}
                      >
                        <option value="reclaimable">可释放</option>
                        <option value="confidence">相似度</option>
                        <option value="files">文件数</option>
                      </select>
                    </label>
                  </div>
                )}
              </div>
              <div className="batch-actions">
                <span>{batchGroupIds.length} 组已勾选</span>
                <div className="batch-replace-control">
                  <button
                    className={batchDisposal === "delete" ? "mini-danger" : "mini-button"}
                    onClick={() => handleBatchGroupResolution(batchDisposal)}
                    disabled={
                      batchGroupIds.length === 0 ||
                      isExecuting ||
                      (batchDisposal === "delete" && !settings?.allowDirectDelete)
                    }
                    title={
                      batchDisposal === "delete" && !settings?.allowDirectDelete
                        ? "在 Settings 中启用仅删除"
                        : undefined
                    }
                  >
                    <ArrowRightLeft size={15} />
                    批量替换
                  </button>
                  <button
                    className={`batch-mode-toggle ${batchDisposal === "delete" ? "danger" : ""}`}
                    type="button"
                    onClick={() =>
                      setBatchDisposal((value) => (value === "backup" ? "delete" : "backup"))
                    }
                    disabled={isExecuting}
                    title="切换替换后原文件的处理方式"
                  >
                    {batchDisposal === "backup" ? <Archive size={14} /> : <Trash2 size={14} />}
                    {batchDisposal === "backup" ? "备份" : "删除"}
                  </button>
                </div>
              </div>
              <div className="group-list-header-row">
                <CandidateToggle
                  checked={allVisibleGroupsQueued}
                  label={allVisibleGroupsQueued ? "取消全选" : "全选"}
                  compact
                  disabled={visibleGroupIds.length === 0}
                  onClick={() => setAllVisibleGroups(!allVisibleGroupsQueued)}
                />
                <span>相似组</span>
                <span>相似度</span>
              </div>
              {groups.length === 0 ? (
                <div className="empty-state">暂无相似组</div>
              ) : (
                <div className="group-scroll" onKeyDown={handleGroupListKeyDown}>
                  {displayedGroups.map((group) => {
                    const numeric = Number(group.id.replace("group-", ""));
                    const isBatchSelected = batchGroupIds.includes(numeric);
                    const isSelected = group.id === selectedGroup?.id;
                    return (
                      <article
                        key={group.id}
                        ref={(element) => {
                          groupRowRefs.current[group.id] = element;
                        }}
                        className={`group-row ${isSelected ? "selected" : ""}`}
                        onClick={() => {
                          setMatchFocusPane("groups");
                          selectGroup(group.id);
                        }}
                        onFocus={() => setMatchFocusPane("groups")}
                        onKeyDown={(event) => {
                          if (event.key === "Enter") {
                            event.preventDefault();
                            event.stopPropagation();
                            selectGroup(group.id);
                          } else if (event.key === " " || event.key === "Spacebar") {
                            event.preventDefault();
                            event.stopPropagation();
                            setMatchFocusPane("groups");
                            if (isSelected) {
                              toggleBatchGroup(group);
                            } else {
                              selectGroup(group.id);
                            }
                          }
                        }}
                        role="button"
                        tabIndex={0}
                      >
                        <CandidateToggle
                          checked={isBatchSelected}
                          label="加入批量替换队列"
                          compact
                          onClick={() => toggleBatchGroup(group)}
                        />
                        <div className="group-row-main">
                          <span className="kind-pill">{group.kind}</span>
                          <strong>{group.title}</strong>
                          <span>{group.itemCount} files</span>
                        </div>
                        <div className="row-meta">
                          <b>{confidence(group.confidence)}</b>
                          <span>{formatBytes(group.reclaimableBytes)}</span>
                        </div>
                      </article>
                    );
                  })}
                </div>
              )}
            </div>

            <div className="splitter" onPointerDown={beginResize} title="拖动调整相似结果宽度">
              <GripVertical size={16} />
            </div>

            <div className="detail-pane">
              {selectedGroup ? (
                <>
                  <div className="detail-head">
                    <div>
                      <span className="eyebrow">{selectedGroup.kind} · {confidence(selectedGroup.confidence)}</span>
                      <h1>{selectedGroup.title}</h1>
                    </div>
                    <button className="secondary-button" onClick={copyReport}>
                      <Clipboard size={18} />
                      复制报告
                    </button>
                  </div>

                  <div className="evidence-line">
                    {selectedGroup.evidence.map((item) => (
                      <span key={item}>{item}</span>
                    ))}
                  </div>

                  <div className="candidate-toolbar">
                    <button
                      className="mini-button"
                      onClick={() => setCurrentGroupVideos(!allCurrentGroupVideosSelected)}
                      disabled={currentGroupVideoIds.length === 0}
                    >
                      {allCurrentGroupVideosSelected ? <CheckSquare size={15} /> : <Square size={15} />}
                      {allCurrentGroupVideosSelected ? "取消全选" : "全选候选"}
                    </button>
                    <button
                      className="mini-button"
                      onClick={() => setSelectedVideoIds([])}
                      disabled={selectedVideoIds.length === 0}
                    >
                      清空选择
                    </button>
                  </div>

                  <div className="video-cards">
                    {selectedGroup.items.map((item) => {
                      const id = videoId(item.video);
                      const isChecked = id !== null && selectedVideoIds.includes(id);
                      const isKeeper = item.video.id === selectedRecommendedId;
                      const isNamingSource = item.video.id === namingSource?.video.id;
                      const isFilenameSource = item.video.id === filenameOverrides[selectedGroup.id];
                      const isFocusedVideo =
                        id !== null && matchFocusPane === "videos" && focusedCurrentVideoId === id;
                      return (
                        <article
                          className={`video-card ${isChecked ? "selected" : ""} ${isFocusedVideo ? "focused" : ""}`}
                          key={item.video.path}
                          ref={(element) => {
                            if (id !== null) videoCardRefs.current[id] = element;
                          }}
                          role="button"
                          tabIndex={0}
                          onClick={(event) => {
                            if (id === null || isRemoteNavBlockedTarget(event.target)) return;
                            setFocusedVideoId(id);
                            setMatchFocusPane("videos");
                          }}
                          onFocus={() => {
                            if (id === null) return;
                            setFocusedVideoId(id);
                            setMatchFocusPane("videos");
                          }}
                          onKeyDown={(event) => handleRemoteMatchKey(event)}
                        >
                          <CandidateToggle
                            checked={isChecked}
                            label="选择"
                            onClick={() => toggleVideo(item.video)}
                          />
                          <PreviewStrip video={item.video} />
                          <div className="video-card-body">
                            <div className="video-card-head">
                              <div className="role-stack">
                                {isKeeper && <span className="role keep">推荐保留</span>}
                                {isNamingSource && <span className="role inherit">推荐继承路径自</span>}
                                {isFilenameSource && <span className="role inherit">当前文件名源</span>}
                                {!isKeeper && !isNamingSource && !isFilenameSource && (
                                  <span className="role">
                                    {item.video.id === selectedGroup.recommendedVideoId
                                      ? "原推荐保留"
                                      : item.role}
                                  </span>
                                )}
                              </div>
                              <div className="card-actions">
                                {!isKeeper && (
                                  <button
                                    className="mini-button"
                                    onClick={(event) => {
                                      event.stopPropagation();
                                      setKeeper(selectedGroup, item);
                                    }}
                                  >
                                    设为保留
                                  </button>
                                )}
                                <button
                                  className={`mini-button ${isNamingSource ? "active" : ""}`}
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    setNamingSource(selectedGroup, item);
                                  }}
                                  disabled={isNamingSource}
                                >
                                  {isNamingSource ? "当前路径源" : "设为路径源"}
                                </button>
                                <button
                                  className={`mini-button ${isFilenameSource ? "active" : ""}`}
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    setFilenameSource(selectedGroup, item);
                                  }}
                                  disabled={isFilenameSource}
                                >
                                  {isFilenameSource ? "当前文件名" : "只保留文件名"}
                                </button>
                                <button
                                  className="mini-button"
                                  onClick={(event) => {
                                    event.stopPropagation();
                                    temporarilyIgnoreVideo(item);
                                  }}
                                >
                                  暂时不修改
                                </button>
                                <button className="icon-text-button" onClick={() => handleOpenVideo(item.video)}>
                                  <Play size={16} />
                                  播放
                                </button>
                              </div>
                            </div>
                            <strong>{item.video.fileName}</strong>
                            <small>{item.video.parentPath}</small>
                            <VideoSpec video={item.video} />
                            <SimilarityHitMap item={item} />
                          </div>
                        </article>
                      );
                    })}
                  </div>

                  <div className="action-bar">
                    <button
                      className="secondary-button"
                      onClick={() => handleSelectedResolution("backup")}
                      disabled={selectedVideoIds.length < 2 || isExecuting}
                    >
                      <Archive size={18} />
                      并入并备份
                    </button>
                    <button
                      className="danger-button"
                      onClick={() => handleSelectedResolution("delete")}
                      disabled={selectedVideoIds.length < 2 || !settings?.allowDirectDelete || isExecuting}
                      title={!settings?.allowDirectDelete ? "在 Settings 中启用仅删除" : undefined}
                    >
                      <ArrowRightLeft size={18} />
                      并入并删除
                    </button>
                  </div>
                </>
              ) : (
                <div className="empty-state">等待扫描结果</div>
              )}
            </div>
          </section>
        )}

        {view === "library" && (
          <section className="page-panel library-page">
            <div className="detail-head">
              <div>
                <span className="eyebrow">library</span>
                <h1>索引库与比对范围</h1>
              </div>
              <button
                className="secondary-button"
                onClick={handleRefreshIndexSources}
                disabled={Boolean(refreshProgress)}
              >
                <RefreshCw size={18} />
                {refreshProgress ? "刷新中" : "刷新路径"}
              </button>
            </div>

            <div className="scope-panel">
              <div className="section-title">
                <span>索引源路径</span>
                <small>{activeScopeLabel}</small>
              </div>
              <div className="scope-actions">
                <button className="mini-button" onClick={() => setScopeSessionIds(null)}>
                  全部参与比对
                </button>
                <button className="mini-button" onClick={() => setScopeSessionIds([])}>
                  清空范围
                </button>
                <button
                  className="mini-button"
                  onClick={collapseAllScopes}
                  disabled={collapsibleScopeKeys.length === 0}
                >
                  全部折叠
                </button>
                <button
                  className="mini-button"
                  onClick={expandAllScopes}
                  disabled={collapsedScopeKeys.size === 0}
                >
                  全部展开
                </button>
              </div>
              {scanSessions.length === 0 ? (
                <div className="empty-inline">当前数据库没有路径索引。添加一个或多个扫描路径后会显示在这里。</div>
              ) : (
                <div className="scope-tree" role="tree" aria-label="Index sources">
                  {scopeRows.map((row) => {
                    const selectedCount = row.node.sessionIds.filter((id) => selectedScopeSet.has(id)).length;
                    const checked =
                      row.node.sessionIds.length > 0 && selectedCount === row.node.sessionIds.length;
                    const partial = selectedCount > 0 && !checked;
                    const session = row.node.session;
                    const expanded = row.node.children.length > 0 && !collapsedScopeKeys.has(row.key);
                    return (
                      <div
                        key={row.key}
                        className={`scope-tree-row ${checked ? "selected" : ""} ${partial ? "partial" : ""}`}
                        role="treeitem"
                        aria-selected={checked}
                        aria-expanded={row.node.children.length > 0 ? expanded : undefined}
                        tabIndex={0}
                        style={{ "--tree-depth": row.depth } as CSSProperties}
                        onClick={(event) => selectScopeRow(row, event)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter" || event.key === " ") {
                            event.preventDefault();
                            selectScopeRow(row, event);
                          }
                        }}
                      >
                        <div className="scope-tree-label">
                          <div className="scope-tree-leading">
                            {row.node.children.length > 0 ? (
                              <button
                                className="tree-expander"
                                onClick={(event) => {
                                  event.stopPropagation();
                                  toggleScopeCollapse(row.key);
                                }}
                                title={expanded ? "折叠" : "展开"}
                              >
                                {expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
                              </button>
                            ) : (
                              <span className="tree-expander-spacer" />
                            )}
                            <ScopeSelectionBox checked={checked} partial={partial} />
                            <Folder size={16} />
                          </div>
                          <div className="scope-tree-main">
                            <strong>{row.node.label}</strong>
                            <small>{row.node.path}</small>
                          </div>
                        </div>
                        <div className="scope-tree-aside">
                          <div className="scope-tree-meta">
                          <span>{row.node.sessionIds.length} 个路径</span>
                          {session ? (
                            <>
                              <span>{formatDate(session.completedUnixMs ?? session.startedUnixMs)}</span>
                              <span>
                                {session.scanned}/{session.totalFiles} · 失败 {session.failed}
                              </span>
                            </>
                          ) : null}
                        </div>
                        {row.node.sessionIds.length > 0 ? (
                          <div className="session-actions">
                            {session ? (
                              <button
                                className="mini-button"
                                onClick={(event) => {
                                  event.stopPropagation();
                                  handleRescanSession(session);
                                }}
                                disabled={isScanning || isExecuting}
                              >
                                <RefreshCw size={14} />
                                重新扫描
                              </button>
                            ) : null}
                            <button
                              className="mini-danger"
                              onClick={(event) => {
                                event.stopPropagation();
                                handleDeleteScopeNode(row.node);
                              }}
                              disabled={isExecuting}
                            >
                              {row.node.sessionIds.length === 1 ? "删除索引源" : "删除此范围"}
                            </button>
                          </div>
                        ) : null}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

          </section>
        )}

        {view === "history" && (
          <section className="page-panel log-page">
            <div className="detail-head">
              <div>
                <span className="eyebrow">logs</span>
                <h1>Logs</h1>
              </div>
              <div className="detail-actions">
                <button
                  className="secondary-button"
                  onClick={clearRuntimeLogs}
                  disabled={runtimeLogs.length === 0}
                >
                  Clear runtime logs
                </button>
                <button
                  className="secondary-button"
                  onClick={handleLightRefresh}
                  disabled={Boolean(refreshProgress)}
                >
                  <RefreshCw size={18} />
                  {refreshProgress ? "Refreshing" : "Refresh history"}
                </button>
              </div>
            </div>

            <div className="log-page-grid">
              <section className="log-panel">
                <div className="log-panel-head">
                  <div>
                    <strong>Runtime logs</strong>
                    <small>{runtimeLogs.length} entries</small>
                  </div>
                </div>
                {runtimeLogs.length === 0 ? (
                  <div className="empty-inline">Waiting for scan, refresh, AI index, or file operation status.</div>
                ) : (
                  <div className="log-entry-list">
                    {runtimeLogs.map((entry) => (
                      <article className={`log-entry ${entry.level}`} key={entry.id}>
                        <div className="log-entry-head">
                          <span className="live-log-dot" />
                          <strong>{entry.title}</strong>
                          <time>{entry.time}</time>
                        </div>
                        {entry.detail && <pre className="log-detail">{entry.detail}</pre>}
                      </article>
                    ))}
                  </div>
                )}
              </section>

              <section className="log-panel">
                <div className="log-panel-head">
                  <div>
                    <strong>Batch task details</strong>
                    <small>{batchTaskLogs.length} items</small>
                  </div>
                  {batchProgress && (
                    <span className="log-summary-pill">
                      {batchProgress.processed}/{batchProgress.total} · completed {batchProgress.completed} · skipped {batchProgress.skipped} · failed {batchProgress.failed}
                    </span>
                  )}
                </div>
                {batchTaskLogs.length === 0 ? (
                  <div className="empty-inline">No batch replacement task has run yet.</div>
                ) : (
                  <div className="batch-log-list">
                    {batchTaskLogs.map((entry) => (
                      <article className={`batch-log-row ${entry.status}`} key={entry.id}>
                        <div className="batch-log-main">
                          <div className="batch-log-title">
                            <span>{batchTaskStatusLabel(entry.status)}</span>
                            <strong>{entry.title}</strong>
                          </div>
                          <small>
                            keep {entry.keeperId ?? "-"} · path source {entry.namingSourceId ?? "-"} · filename source {entry.filenameSourceId ?? "-"} · extras {entry.extraCount ?? 0}
                          </small>
                          {entry.reason && <pre className="log-detail">{entry.reason}</pre>}
                        </div>
                      </article>
                    ))}
                  </div>
                )}
              </section>

              <section className="log-panel log-panel-wide">
                <div className="log-panel-head">
                  <div>
                    <strong>Operation history</strong>
                    <small>{operationHistory.length} entries</small>
                  </div>
                </div>
                {operationHistory.length === 0 ? (
                  <div className="empty-state">No delete, merge, or backup operation has run yet.</div>
                ) : (
                  <div className="history-list">
                    {operationHistory.map((entry) => {
                      const importantMessages = failureLikeMessages(entry.messages);
                      return (
                        <article className="history-row" key={entry.operationId}>
                          <div className="history-main">
                            <div className="history-title">
                              <span>{entry.action}</span>
                              <strong>{entry.summary}</strong>
                              {importantMessages.length > 0 && (
                                <b className="history-alert">{importantMessages.length} issues</b>
                              )}
                            </div>
                            <small>{formatDate(entry.createdAtUnixMs)}</small>
                            <small>{entry.operationId}</small>
                            {importantMessages.length > 0 && (
                              <details open>
                                <summary>Failed, skipped, missing, or blocked items</summary>
                                <ul>
                                  {importantMessages.map((message, index) => (
                                    <li key={`important-${index}`}>{message}</li>
                                  ))}
                                </ul>
                              </details>
                            )}
                            {entry.messages.length > 0 && (
                              <details open={importantMessages.length > 0}>
                                <summary>{entry.messages.length} full records</summary>
                                <ul>
                                  {entry.messages.map((message, index) => (
                                    <li key={`message-${index}`}>{message}</li>
                                  ))}
                                </ul>
                              </details>
                            )}
                          </div>
                          {entry.rolledBack ? (
                            <span className="history-status">Rolled back</span>
                          ) : entry.reversible ? (
                            <button
                              className="secondary-button"
                              onClick={() => handleRollback(entry)}
                              disabled={isExecuting}
                            >
                              <RotateCcw size={17} />
                              Roll back
                            </button>
                          ) : (
                            <span className="history-status muted">Not reversible</span>
                          )}
                        </article>
                      );
                    })}
                  </div>
                )}
              </section>
            </div>
          </section>
        )}

        {view === "ai" && (
          <section className="page-panel settings-page">
            <div
              className="settings-page-scroll"
              onScroll={(event) => handleSettingsScroll(event.currentTarget.scrollTop)}
            >
            <div className="detail-head">
              <div>
                <span className="eyebrow">ai</span>
                <h1>{settingText.aiTitle}</h1>
              </div>
              <div className="detail-actions">
                <button className="secondary-button" onClick={checkAiModel} disabled={isIndexingAi}>
                  <RefreshCw size={18} />
                  {settingText.checkModel}
                </button>
                <button
                  className="secondary-button"
                  onClick={() => handleBuildAiIndex(true)}
                  disabled={!settingsDraft?.aiVisionEnabled || isIndexingAi}
                >
                  <RefreshCw size={18} />
                  {settingText.rebuildAiIndex}
                </button>
                <button
                  className="primary-button"
                  onClick={() => handleBuildAiIndex(false)}
                  disabled={!settingsDraft?.aiVisionEnabled || isIndexingAi}
                >
                  <ImageIcon size={18} />
                  {settingText.buildAiIndex}
                </button>
              </div>
            </div>

            {settingsDraft && (
              <div className="settings-grid">
                <label className="toggle-row">
                  <input
                    type="checkbox"
                    checked={settingsDraft.aiVisionEnabled}
                    onChange={(event) =>
                      setSettingsDraft({
                        ...settingsDraft,
                        aiVisionEnabled: event.target.checked,
                      })
                    }
                  />
                  <span>
                    {settingText.aiVision}
                    <small>{settingText.aiVisionHelp}</small>
                  </span>
                </label>

                <label className="toggle-row">
                  <input
                    type="checkbox"
                    checked={settingsDraft.aiIndexAfterScan}
                    onChange={(event) =>
                      setSettingsDraft({
                        ...settingsDraft,
                        aiIndexAfterScan: event.target.checked,
                      })
                    }
                  />
                  <span>
                    {settingText.aiAfterScan}
                    <small>{settingText.aiAfterScanHelp}</small>
                  </span>
                </label>

                <label className="toggle-row">
                  <input
                    type="checkbox"
                    checked={settingsDraft.deleteAiFrameCacheAfterIndex}
                    onChange={(event) =>
                      setSettingsDraft({
                        ...settingsDraft,
                        deleteAiFrameCacheAfterIndex: event.target.checked,
                      })
                    }
                  />
                  <span>
                    {settingText.deleteFrameCacheAfterIndex}
                    <small>{settingText.deleteFrameCacheAfterIndexHelp}</small>
                  </span>
                </label>

                <div className="setting-field">
                  <span>{settingText.localPipeline}</span>
                  <small>{settingText.localPipelineHelp}</small>
                </div>

                <label className="setting-field compact-setting">
                  <span>{settingText.localVideoWorkers}</span>
                  <NumberSettingInput
                    value={settingsDraft.localPreprocessVideoWorkers}
                    min={1}
                    max={8}
                    step={1}
                    fallback={1}
                    onCommit={(value) =>
                      setSettingsDraft({ ...settingsDraft, localPreprocessVideoWorkers: value })
                    }
                  />
                  <small>{settingText.localVideoWorkersHelp}</small>
                </label>

                <label className="setting-field compact-setting">
                  <span>{settingText.overlapStart}</span>
                  <NumberSettingInput
                    value={settingsDraft.localPreprocessOverlapStartPercent}
                    min={50}
                    max={99}
                    step={1}
                    fallback={95}
                    onCommit={(value) =>
                      setSettingsDraft({ ...settingsDraft, localPreprocessOverlapStartPercent: value })
                    }
                  />
                  <small>{settingText.overlapStartHelp}</small>
                </label>

                <label className="setting-field compact-setting">
                  <span>{settingText.localProcessWorkers}</span>
                  <NumberSettingInput
                    value={settingsDraft.localPreprocessProcessWorkers}
                    min={1}
                    max={4}
                    step={1}
                    fallback={2}
                    onCommit={(value) =>
                      setSettingsDraft({ ...settingsDraft, localPreprocessProcessWorkers: value })
                    }
                  />
                  <small>{settingText.localProcessWorkersHelp}</small>
                </label>

                <label className="setting-field compact-setting">
                  <span>{settingText.gpuAiWorkers}</span>
                  <NumberSettingInput
                    value={settingsDraft.aiGpuWorkerCount}
                    min={1}
                    max={64}
                    step={1}
                    fallback={4}
                    onCommit={(value) => setSettingsDraft({ ...settingsDraft, aiGpuWorkerCount: value })}
                  />
                  <small>{settingText.gpuAiWorkersHelp}</small>
                </label>

                <label className="setting-field compact-setting">
                  <span>{settingText.localFrameWorkers}</span>
                  <NumberSettingInput
                    value={settingsDraft.localPreprocessFrameWorkers}
                    min={1}
                    max={64}
                    step={1}
                    fallback={16}
                    onCommit={(value) =>
                      setSettingsDraft({ ...settingsDraft, localPreprocessFrameWorkers: value })
                    }
                  />
                  <small>{settingText.localFrameWorkersHelp}</small>
                </label>

                <label className="setting-field compact-setting">
                  <span>{settingText.aiMatchWorkers}</span>
                  <NumberSettingInput
                    value={settingsDraft.aiMatchWorkerCount}
                    min={1}
                    max={128}
                    step={1}
                    fallback={8}
                    onCommit={(value) => setSettingsDraft({ ...settingsDraft, aiMatchWorkerCount: value })}
                  />
                  <small>{settingText.aiMatchWorkersHelp}</small>
                </label>

                <label className="toggle-row">
                  <input
                    type="checkbox"
                    checked={settingsDraft.ramDiskEnabled}
                    onChange={(event) =>
                      setSettingsDraft({ ...settingsDraft, ramDiskEnabled: event.target.checked })
                    }
                  />
                  <span>
                    {settingText.ramDisk}
                    <small>{ramDiskStatus?.message ?? settingText.ramDiskHelp}</small>
                  </span>
                </label>

                {settingsDraft.ramDiskEnabled && (
                  <div className="setting-field ram-disk-setting">
                    <span>{settingText.ramDiskSize}</span>
                    <NumberSettingInput
                      value={settingsDraft.ramDiskSizeMb}
                      min={512}
                      max={1048576}
                      step={512}
                      fallback={16384}
                      onCommit={(value) => {
                        setSettingsDraft({ ...settingsDraft, ramDiskSizeMb: value });
                        setRamDiskSizeMb(value);
                      }}
                    />
                    <button
                      className="secondary-button"
                      onClick={handleConfigureRamDisk}
                      disabled={isConfiguringRamDisk}
                    >
                      <MemoryStick size={17} />
                      {isConfiguringRamDisk ? "正在配置" : settingText.configureRamDisk}
                    </button>
                    <small>{settingText.ramDiskSizeHelp}</small>
                  </div>
                )}

                <label className="setting-field">
                  <span>{settingText.primaryCacheFolder}</span>
                  <div className="path-input-row">
                    <input
                      value={settingsDraft.localPreprocessTempDir}
                      readOnly={settingsDraft.ramDiskEnabled}
                      onChange={(event) =>
                        setSettingsDraft({ ...settingsDraft, localPreprocessTempDir: event.target.value })
                      }
                      placeholder="Z:\\TEMP"
                    />
                    {!settingsDraft.ramDiskEnabled && (
                      <button
                        className="secondary-button"
                        onClick={chooseLocalPreprocessTempFolder}
                        disabled={isPickingFolder}
                      >
                        <FolderOpen size={17} />
                        {settingText.select}
                      </button>
                    )}
                  </div>
                  <small>{settingText.primaryCacheFolderHelp}</small>
                </label>

                <label className="setting-field">
                  <span>{settingText.secondaryCacheFolder}</span>
                  <div className="path-input-row">
                    <input
                      value={settingsDraft.localPreprocessSecondaryTempDir}
                      onChange={(event) =>
                        setSettingsDraft({
                          ...settingsDraft,
                          localPreprocessSecondaryTempDir: event.target.value,
                        })
                      }
                      placeholder="D:\\TEMP"
                    />
                    <button
                      className="secondary-button"
                      onClick={chooseLocalPreprocessSecondaryTempFolder}
                      disabled={isPickingFolder}
                    >
                      <FolderOpen size={17} />
                      {settingText.select}
                    </button>
                  </div>
                  <small>{settingText.secondaryCacheFolderHelp}</small>
                </label>

                <label className="setting-field compact-setting">
                  <span>{settingText.secondaryThreshold}</span>
                  <NumberSettingInput
                    value={settingsDraft.localPreprocessSecondaryThresholdMb}
                    min={512}
                    max={1048576}
                    step={512}
                    fallback={16384}
                    onCommit={(value) =>
                      setSettingsDraft({ ...settingsDraft, localPreprocessSecondaryThresholdMb: value })
                    }
                  />
                  <small>{settingText.secondaryThresholdHelp}</small>
                </label>

                <label className="setting-field">
                  <span>{settingText.aiModelPath}</span>
                  <input
                    value={settingsDraft.aiModelPath}
                    onChange={(event) =>
                      setSettingsDraft({ ...settingsDraft, aiModelPath: event.target.value })
                    }
                    placeholder="models\\dinov2-small-dynamic\\model.onnx"
                  />
                  <small>
                    {aiModelStatus
                      ? `${aiModelStatus.ready ? settingText.aiModelPathReady : settingText.aiModelPathNotReady}: ${aiModelStatus.message}`
                      : settingText.aiModelPathHelp}
                  </small>
                </label>

                <label className="setting-field compact-setting">
                  <span>{settingText.aiDevice}</span>
                  <select
                    value={settingsDraft.aiDevice}
                    onChange={(event) =>
                      setSettingsDraft({ ...settingsDraft, aiDevice: event.target.value })
                    }
                  >
                    <option value="auto">{settingText.auto}</option>
                    <option value="gpu">GPU</option>
                    <option value="cpu">CPU</option>
                  </select>
                  <small>{settingText.aiDeviceHelp}</small>
                </label>

                <label className="setting-field compact-setting">
                  <span>{settingText.aiFrameCount}</span>
                  <NumberSettingInput
                    value={settingsDraft.aiFrameCount}
                    min={8}
                    max={512}
                    step={1}
                    fallback={128}
                    onCommit={(value) => setSettingsDraft({ ...settingsDraft, aiFrameCount: value })}
                  />
                  <small>{settingText.aiFrameCountHelp}</small>
                </label>

                <label className="setting-field compact-setting">
                  <span>{settingText.aiBatchSize}</span>
                  <NumberSettingInput
                    value={settingsDraft.aiBatchSize}
                    min={1}
                    max={64}
                    step={1}
                    fallback={32}
                    onCommit={(value) => setSettingsDraft({ ...settingsDraft, aiBatchSize: value })}
                  />
                  <small>{settingText.aiBatchSizeHelp}</small>
                </label>

                <label className="setting-field compact-setting">
                  <span>{settingText.aiSimilarityThreshold}</span>
                  <NumberSettingInput
                    value={settingsDraft.aiSimilarityThreshold}
                    min={0}
                    max={0.99}
                    step={0.01}
                    fallback={0.86}
                    onCommit={(value) => setSettingsDraft({ ...settingsDraft, aiSimilarityThreshold: value })}
                  />
                  <small>{settingText.aiSimilarityThresholdHelp}</small>
                </label>

                <label className="setting-field compact-setting">
                  <span>{settingText.aiMinMatchedFrames}</span>
                  <NumberSettingInput
                    value={settingsDraft.aiMinMatchedFrames}
                    min={1}
                    max={128}
                    step={1}
                    fallback={8}
                    onCommit={(value) => setSettingsDraft({ ...settingsDraft, aiMinMatchedFrames: value })}
                  />
                  <small>{settingText.aiMinMatchedFramesHelp}</small>
                </label>

                <label className="toggle-row">
                  <input
                    type="checkbox"
                    checked={settingsDraft.compareWithinSameFolder}
                    onChange={(event) =>
                      setSettingsDraft({ ...settingsDraft, compareWithinSameFolder: event.target.checked })
                    }
                  />
                  <span>
                    {settingText.compareWithinSameFolder}
                    <small>{settingText.compareWithinSameFolderHelp}</small>
                  </span>
                </label>

                <label className="toggle-row">
                  <input
                    type="checkbox"
                    checked={settingsDraft.aiClipMatchingEnabled}
                    onChange={(event) =>
                      setSettingsDraft({
                        ...settingsDraft,
                        aiClipMatchingEnabled: event.target.checked,
                      })
                    }
                  />
                  <span>
                    {settingText.aiClipMatching}
                    <small>{settingText.aiClipMatchingHelp}</small>
                  </span>
                </label>

                <div className="setting-field">
                  <span>{settingText.lastAiIndex}</span>
                  <small>
                    {aiIndexSummary
                      ? `处理 ${aiIndexSummary.processed}，跳过 ${aiIndexSummary.skipped}，抽帧不足 ${aiIndexSummary.insufficientFrames ?? 0}，失败 ${aiIndexSummary.failed}`
                      : aiIndexProgress
                        ? `${aiIndexProgress.phase}：处理 ${aiIndexProgress.processed}，抽帧不足 ${aiIndexProgress.insufficientFrames ?? 0}，失败 ${aiIndexProgress.failed}`
                        : settingText.noAiIndex}
                  </small>
                  {aiIndexSummary?.recentErrors?.length ? (
                    <div className="plan-list">
                      {aiIndexSummary.recentErrors.map((error) => (
                        <span key={error}>{shortPath(error)}</span>
                      ))}
                    </div>
                  ) : null}
                </div>

              </div>
            )}
            </div>
            <div className={`settings-save-bar ${isSettingsSaveBarVisible ? "visible" : "hidden"}`}>
              <span>
                {settingsDirty
                  ? settingsLanguage === "en" ? "Unsaved changes" : "有未保存的更改"
                  : settingsLanguage === "en" ? "All changes saved" : "更改已保存"}
              </span>
              <button
                className="primary-button"
                onClick={() => void saveSettingsDraft()}
                disabled={!settingsDirty}
              >
                <Save size={18} />
                {settingText.saveAiSettings}
              </button>
            </div>
          </section>
        )}

        {view === "settings" && (
          <section className="page-panel settings-page">
            <div
              className="settings-page-scroll"
              onScroll={(event) => handleSettingsScroll(event.currentTarget.scrollTop)}
            >
            <div className="detail-head">
              <div>
                <span className="eyebrow">settings</span>
                <h1>{settingText.settingsTitle}</h1>
              </div>
            </div>

            {settingsDraft && (
              <div className="settings-grid">
                <label className="setting-field compact-setting">
                  <span>{settingText.language}</span>
                  <select
                    value={normalizeSettingsLanguage(settingsDraft.uiLanguage)}
                    onChange={(event) =>
                      setSettingsDraft({
                        ...settingsDraft,
                        uiLanguage: normalizeSettingsLanguage(event.target.value),
                      })
                    }
                  >
                    <option value="zh">{settingText.chinese}</option>
                    <option value="en">{settingText.english}</option>
                  </select>
                  <small>{settingText.languageHelp}</small>
                </label>

                <label className="setting-field">
                  <span>{settingText.backupDir}</span>
                  <div className="path-input-row">
                    <input
                      value={settingsDraft.backupDir}
                      onChange={(event) =>
                        setSettingsDraft({ ...settingsDraft, backupDir: event.target.value })
                      }
                    />
                    <button
                      className="secondary-button"
                      onClick={chooseBackupFolder}
                      disabled={isPickingFolder}
                    >
                      <FolderOpen size={17} />
                      {settingText.select}
                    </button>
                  </div>
                  <small>{settingText.backupDirHelp}</small>
                </label>

                <label className="setting-field compact-setting">
                  <span>{settingText.keeperWindow}</span>
                  <NumberSettingInput
                    value={Math.round(settingsDraft.keeperSizePriorityDurationSeconds / 60)}
                    min={0}
                    max={1440}
                    step={1}
                    fallback={5}
                    onCommit={(value) =>
                      setSettingsDraft({
                        ...settingsDraft,
                        keeperSizePriorityDurationSeconds: value * 60,
                      })
                    }
                  />
                  <small>{settingText.keeperWindowHelp}</small>
                </label>

                <label className="toggle-row">
                  <input
                    type="checkbox"
                    checked={settingsDraft.ramDiskEnabled}
                    onChange={(event) =>
                      setSettingsDraft({ ...settingsDraft, ramDiskEnabled: event.target.checked })
                    }
                  />
                  <span>
                    {settingText.ramDisk}
                    <small>{ramDiskStatus?.message ?? settingText.ramDiskHelp}</small>
                  </span>
                </label>

                {settingsDraft.ramDiskEnabled && (
                  <div className="setting-field ram-disk-setting">
                    <span>{settingText.ramDiskSize}</span>
                    <NumberSettingInput
                      value={settingsDraft.ramDiskSizeMb}
                      min={512}
                      max={1048576}
                      step={512}
                      fallback={16384}
                      onCommit={(value) => {
                        setSettingsDraft({ ...settingsDraft, ramDiskSizeMb: value });
                        setRamDiskSizeMb(value);
                      }}
                    />
                    <button
                      className="secondary-button"
                      onClick={handleConfigureRamDisk}
                      disabled={isConfiguringRamDisk}
                    >
                      <MemoryStick size={17} />
                      {isConfiguringRamDisk ? "正在配置" : settingText.configureRamDisk}
                    </button>
                    <small>{settingText.ramDiskSizeHelp}</small>
                  </div>
                )}

                <label className="setting-field">
                  <span>{settingText.primaryCache}</span>
                  <div className="path-input-row">
                    <input
                      value={settingsDraft.localPreprocessTempDir}
                      readOnly={settingsDraft.ramDiskEnabled}
                      onChange={(event) =>
                        setSettingsDraft({ ...settingsDraft, localPreprocessTempDir: event.target.value })
                      }
                      placeholder="Z:\\TEMP"
                    />
                    {!settingsDraft.ramDiskEnabled && (
                      <button
                        className="secondary-button"
                        onClick={chooseLocalPreprocessTempFolder}
                        disabled={isPickingFolder}
                      >
                        <FolderOpen size={17} />
                        {settingText.select}
                      </button>
                    )}
                  </div>
                  <small>{settingText.primaryCacheHelp}</small>
                </label>

                <label className="setting-field">
                  <span>{settingText.secondaryCache}</span>
                  <div className="path-input-row">
                    <input
                      value={settingsDraft.localPreprocessSecondaryTempDir}
                      onChange={(event) =>
                        setSettingsDraft({
                          ...settingsDraft,
                          localPreprocessSecondaryTempDir: event.target.value,
                        })
                      }
                      placeholder="D:\\TEMP"
                    />
                    <button
                      className="secondary-button"
                      onClick={chooseLocalPreprocessSecondaryTempFolder}
                      disabled={isPickingFolder}
                    >
                      <FolderOpen size={17} />
                      {settingText.select}
                    </button>
                  </div>
                  <small>{settingText.secondaryCacheHelp}</small>
                </label>

                <div className="setting-field">
                  <span>{settingText.namingDirs}</span>
                  <div className="path-input-row">
                    <input
                      value={namingFolderInput}
                      onChange={(event) => setNamingFolderInput(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter") addNamingFolder();
                      }}
                      placeholder={settingText.namingDirsPlaceholder}
                    />
                    <button
                      className="secondary-button"
                      onClick={chooseNamingFolder}
                      disabled={isPickingFolder}
                    >
                      <FolderOpen size={17} />
                      {settingText.select}
                    </button>
                    <button
                      className="secondary-button"
                      onClick={() => addNamingFolder()}
                      disabled={!namingFolderInput.trim()}
                    >
                      <FolderPlus size={17} />
                      {settingText.add}
                    </button>
                  </div>
                  <small>{settingText.namingDirsHelp}</small>
                  {(settingsDraft.namingSourceDirs ?? []).length > 0 && (
                    <div className="settings-path-list">
                      {(settingsDraft.namingSourceDirs ?? []).map((path, index) => (
                        <div className="settings-path-row" key={path}>
                          <span>{index + 1}</span>
                          <strong>{path}</strong>
                          <button className="mini-danger" onClick={() => removeNamingFolder(path)}>
                            {settingText.delete}
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                <label className="toggle-row">
                  <input
                    type="checkbox"
                    checked={settingsDraft.allowDirectDelete}
                    onChange={(event) =>
                      setSettingsDraft({
                        ...settingsDraft,
                        allowDirectDelete: event.target.checked,
                      })
                    }
                  />
                  <span>
                    {settingText.allowDirectDelete}
                    <small>{settingText.allowDirectDeleteHelp}</small>
                  </span>
                </label>

                <label className="toggle-row">
                  <input
                    type="checkbox"
                    checked={settingsDraft.restrictScanToTestPath}
                    onChange={(event) =>
                      setSettingsDraft({
                        ...settingsDraft,
                        restrictScanToTestPath: event.target.checked,
                      })
                    }
                  />
                  <span>
                    {settingText.restrictTestPath}
                    <small>{settingText.restrictTestPathHelp}</small>
                  </span>
                </label>

                <div className="storage-panel">
                  <div className="storage-head">
                    <div>
                      <span>{settingText.storageTitle}</span>
                      <small>{settingText.storageHelp}</small>
                    </div>
                    <div className="storage-actions">
                      <button
                        className="secondary-button"
                        onClick={refreshStoragePanel}
                        disabled={isMeasuringStorage || isCleaningStorage || isVacuumingDatabase}
                      >
                        <RefreshCw size={17} />
                        {isMeasuringStorage ? "..." : settingText.refreshStorage}
                      </button>
                      <button
                        className="secondary-button"
                        onClick={handleCleanupStorage}
                        disabled={isMeasuringStorage || isCleaningStorage || isVacuumingDatabase}
                      >
                        <Database size={17} />
                        {isCleaningStorage ? "..." : settingText.cleanupStorage}
                      </button>
                      <button
                        className="secondary-button"
                        onClick={handleCleanupCompletedFrameCache}
                        disabled={isMeasuringStorage || isCleaningStorage || isVacuumingDatabase}
                      >
                        <Database size={17} />
                        {isCleaningStorage ? "..." : settingText.cleanupCompletedFrameCache}
                      </button>
                      <button
                        className="secondary-button"
                        onClick={handleVacuumDatabase}
                        disabled={isMeasuringStorage || isCleaningStorage || isVacuumingDatabase}
                      >
                        <Database size={17} />
                        {isVacuumingDatabase ? "..." : settingText.vacuumDatabase}
                      </button>
                    </div>
                  </div>

                  {storageUsage ? (
                    <>
                      <div className="storage-total">
                        <strong>{settingText.storageTotal}</strong>
                        <span>{formatBytes(storageUsage.totalBytes)}</span>
                        <small>{storageUsage.dataDir}</small>
                      </div>
                      <div className="storage-grid">
                        {storageUsage.items.map((item) => (
                          <div className="storage-row" key={item.key}>
                            <strong>{storageItemLabel(item.key, settingsLanguage)}</strong>
                            <span>{formatBytes(item.bytes)}</span>
                            <small>
                              {fileCountText(item.fileCount, settingsLanguage)} · {shortPath(item.path)}
                            </small>
                          </div>
                        ))}
                      </div>
                      <div className="storage-db-stats">
                        <strong>{settingText.storageRows}</strong>
                        <span>{storageDbLabel("videos", settingsLanguage)} {storageUsage.databaseStats.videos.toLocaleString()}</span>
                        <span>{storageDbLabel("sessions", settingsLanguage)} {storageUsage.databaseStats.scanSessions.toLocaleString()}</span>
                        <span>{storageDbLabel("embeddings", settingsLanguage)} {storageUsage.databaseStats.frameEmbeddings.toLocaleString()}</span>
                        <span>{storageDbLabel("pairScores", settingsLanguage)} {storageUsage.databaseStats.aiPairScores.toLocaleString()}</span>
                        <span>{storageDbLabel("edges", settingsLanguage)} {storageUsage.databaseStats.aiMatchEdges.toLocaleString()}</span>
                        <span>{storageDbLabel("models", settingsLanguage)} {storageUsage.databaseStats.embeddingModels.toLocaleString()}</span>
                      </div>
                    </>
                  ) : (
                    <div className="empty-inline">{settingText.storageNotLoaded}</div>
                  )}

                  {storageCleanupSummary && (
                    <div className="storage-cleanup-summary">
                      <strong>{settingText.cleaningSummary}</strong>
                      <span>
                        {storageItemLabel("aiFrameCache", settingsLanguage)} {formatBytes(storageCleanupSummary.deletedAiFrameCacheBytes)}
                        {" / "}
                        {storageItemLabel("thumbnails", settingsLanguage)} {formatBytes(storageCleanupSummary.deletedThumbnailBytes)}
                        {" / "}
                        pair-score {storageCleanupSummary.deletedPairScores.toLocaleString()}
                      </span>
                      <small>
                        {fileCountText(
                          storageCleanupSummary.deletedAiFrameCacheFiles + storageCleanupSummary.deletedThumbnailFiles,
                          settingsLanguage,
                        )}
                        {" · "}
                        {storageDbLabel("videos", settingsLanguage)} {storageCleanupSummary.removedOrphanVideos.toLocaleString()}
                        {" · "}
                        {storageDbLabel("embeddings", settingsLanguage)} {storageCleanupSummary.deletedFrameEmbeddings.toLocaleString()}
                      </small>
                    </div>
                  )}
                  {completedFrameCacheCleanupSummary && (
                    <div className="storage-cleanup-summary">
                      <strong>{settingText.completedFrameCacheCleanupSummary}</strong>
                      <span>
                        {formatBytes(completedFrameCacheCleanupSummary.deletedBytes)}
                        {" / "}
                        {fileCountText(completedFrameCacheCleanupSummary.deletedFiles, settingsLanguage)}
                      </span>
                      <small>
                        {settingsLanguage === "en" ? "checked" : "已检查"}{" "}
                        {completedFrameCacheCleanupSummary.checkedVideos.toLocaleString()}
                        {" / "}
                        {settingsLanguage === "en" ? "eligible" : "可清理"}{" "}
                        {completedFrameCacheCleanupSummary.eligibleVideos.toLocaleString()}
                      </small>
                    </div>
                  )}
                  {vacuumSummary && (
                    <div className="storage-cleanup-summary">
                      <strong>{settingText.vacuumDatabase}</strong>
                      <span>{vacuumSummary}</span>
                    </div>
                  )}
                </div>
              </div>
            )}
            </div>
            <div className={`settings-save-bar ${isSettingsSaveBarVisible ? "visible" : "hidden"}`}>
              <span>
                {settingsDirty
                  ? settingsLanguage === "en" ? "Unsaved changes" : "有未保存的更改"
                  : settingsLanguage === "en" ? "All changes saved" : "更改已保存"}
              </span>
              <button
                className="primary-button"
                onClick={() => void saveSettingsDraft()}
                disabled={!settingsDirty}
              >
                <Save size={18} />
                {settingText.saveSettings}
              </button>
            </div>
          </section>
        )}
      </main>

      {manualCopyText && (
        <div className="modal-scrim" role="dialog" aria-modal="true">
          <div className="modal">
            <div className="detail-head">
              <div>
                <span className="eyebrow">clipboard</span>
                <h1>可复制文本</h1>
              </div>
              <button className="icon-button" onClick={() => setManualCopyText(null)} title="关闭">
                <X size={18} />
              </button>
            </div>
            <textarea
              className="manual-copy"
              readOnly
              value={manualCopyText}
              onFocus={(event) => event.currentTarget.select()}
            />
            <div className="modal-actions">
              <button className="primary-button" onClick={() => setManualCopyText(null)}>
                完成
              </button>
            </div>
          </div>
        </div>
      )}

      {settings?.ramDiskEnabled &&
        ramDiskStatus &&
        (!ramDiskStatus.setupCompleted || !ramDiskStatus.driverInstalled) &&
        !ramDiskSetupDismissed && (
          <div className="modal-scrim" role="dialog" aria-modal="true" aria-labelledby="ram-disk-title">
            <div className="modal ram-disk-setup-modal">
              <div className="detail-head">
                <div>
                  <span className="eyebrow">first run setup</span>
                  <h1 id="ram-disk-title">配置内存缓存盘</h1>
                </div>
                <MemoryStick size={24} />
              </div>
              <p>
                输入计划用于视频缓存的内存容量。软件会自动选择盘符、创建缓存目录，并在任务开始时挂载、任务结束后卸载。
              </p>
              <div className={`ram-disk-status ${ramDiskStatus.driverInstalled ? "ready" : "warning"}`}>
                {ramDiskStatus.driverInstalled ? <CheckCircle2 size={17} /> : <AlertTriangle size={17} />}
                <div>
                  <strong>{ramDiskStatus.driverInstalled ? "已检测到 ImDisk 驱动" : "未检测到 ImDisk 驱动"}</strong>
                  <small>{ramDiskStatus.message}</small>
                </div>
              </div>
              <label className="setting-field compact-setting">
                <span>内存缓存容量 MB</span>
                <NumberSettingInput
                  value={ramDiskSizeMb}
                  min={512}
                  max={1048576}
                  step={512}
                  fallback={16384}
                  onCommit={setRamDiskSizeMb}
                />
                <small>建议至少为 Windows、FFmpeg 和 AI 推理保留 8GB 可用内存。</small>
              </label>
              <div className="modal-actions ram-disk-actions">
                <button className="secondary-button" onClick={skipRamDiskSetup} disabled={isConfiguringRamDisk}>
                  暂时跳过
                </button>
                {!ramDiskStatus.driverInstalled && (
                  <>
                    <button
                      className="secondary-button"
                      onClick={() =>
                        openRamDiskDriverDownload().catch((error) => setNotice(String(error)))
                      }
                    >
                      打开驱动下载
                    </button>
                    <button className="secondary-button" onClick={recheckRamDisk} disabled={isConfiguringRamDisk}>
                      重新检查
                    </button>
                  </>
                )}
                <button
                  className="primary-button"
                  onClick={handleConfigureRamDisk}
                  disabled={!ramDiskStatus.driverInstalled || isConfiguringRamDisk}
                >
                  <ShieldCheck size={17} />
                  {isConfiguringRamDisk ? "正在申请权限并配置" : "申请权限并创建"}
                </button>
              </div>
            </div>
          </div>
        )}

      {confirmDialog && (
        <div className="modal-scrim" role="dialog" aria-modal="true">
          <div className="modal confirm-modal">
            <div className="detail-head">
              <div>
                <span className="eyebrow">confirm</span>
                <h1>{confirmDialog.title}</h1>
              </div>
              <button className="icon-button" onClick={() => settleConfirm(false)} title="关闭">
                <X size={18} />
              </button>
            </div>
            <p>{confirmDialog.body}</p>
            <div className="modal-actions">
              <button className="secondary-button" autoFocus onClick={() => settleConfirm(false)}>
                取消
              </button>
              <button
                className={confirmDialog.danger ? "danger-button" : "primary-button"}
                onClick={() => settleConfirm(true)}
              >
                {confirmDialog.confirmLabel}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
