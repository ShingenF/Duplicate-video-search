import { invoke } from "@tauri-apps/api/core";
import type {
  AppSettings,
  AiIndexSummary,
  AiModelStatus,
  BatchMergeTask,
  CompletedAiFrameCacheCleanupSummary,
  DeleteIndexSummary,
  IndexRefreshSummary,
  MatchGroup,
  OperationHistoryEntry,
  OperationResult,
  ReplacementPlan,
  RamDiskStatus,
  ScanSummary,
  ScanSession,
  StaleVideoPruneSummary,
  StorageCleanupSummary,
  StorageUsageSummary,
  ToolStatus,
  VacuumSummary,
  VideoRecord,
} from "./types";

const hasTauri = () => "__TAURI_INTERNALS__" in window;

const previewVideos: VideoRecord[] = [];
const previewGroups: MatchGroup[] = [];

export async function getToolStatus(): Promise<ToolStatus> {
  if (!hasTauri()) {
    return {
      workspaceRoot: "C:\\Example\\Duplicate-video-search",
      dataDir: "C:\\Example\\Duplicate-video-search\\data",
      databasePath: "C:\\Example\\Duplicate-video-search\\data\\index.sqlite",
      allowedSource: "\\\\EXAMPLE-NAS\\Test",
      ffmpegPath: "C:\\Tools\\ffmpeg\\bin\\ffmpeg.exe",
      ffprobePath: "C:\\Tools\\ffmpeg\\bin\\ffprobe.exe",
      ffmpegVersion: "ffmpeg version 8.1.1-full_build-www.gyan.dev",
      ffprobeVersion: "ffprobe version 8.1.1-full_build-www.gyan.dev",
    };
  }
  return invoke<ToolStatus>("get_tool_status");
}

export async function scanSource(source: string): Promise<ScanSummary> {
  if (!hasTauri()) {
    await new Promise((resolve) => window.setTimeout(resolve, 300));
    return {
      sessionId: null,
      source,
      databasePath: "C:\\Example\\Duplicate-video-search\\data\\index.sqlite",
      totalFiles: 0,
      scanned: 0,
      failed: 0,
      elapsedMs: 300,
    };
  }
  return invoke<ScanSummary>("scan_source", { source });
}

export async function scanSources(sources: string[]): Promise<ScanSummary[]> {
  if (!hasTauri()) {
    const summaries = [];
    for (const source of sources) {
      summaries.push(await scanSource(source));
    }
    return summaries;
  }
  return invoke<ScanSummary[]>("scan_sources", { sources });
}

export async function listVideos(sessionIds?: number[]): Promise<VideoRecord[]> {
  if (!hasTauri()) return previewVideos;
  return invoke<VideoRecord[]>("list_videos", { sessionIds });
}

export async function listScanSessions(): Promise<ScanSession[]> {
  if (!hasTauri()) return [];
  return invoke<ScanSession[]>("list_scan_sessions");
}

export async function deleteScanSession(sessionId: number): Promise<void> {
  if (!hasTauri()) return;
  return invoke<void>("delete_scan_session", { sessionId });
}

export async function deleteScanSessions(sessionIds: number[]): Promise<DeleteIndexSummary> {
  if (!hasTauri()) {
    return {
      requestedSessions: sessionIds.length,
      deletedSessions: sessionIds.length,
      affectedVideos: 0,
      deletedVideos: 0,
      deletedPairScores: 0,
      deletedMatchEdges: 0,
      deletedFrameEmbeddings: 0,
    };
  }
  return invoke<DeleteIndexSummary>("delete_scan_sessions", { sessionIds });
}

export async function refreshIndexSources(): Promise<IndexRefreshSummary> {
  if (!hasTauri()) return { removedMissingSources: 0, skippedUnavailableRoots: 0 };
  return invoke<IndexRefreshSummary>("refresh_index_sources");
}

export async function pruneStaleVideos(sessionIds?: number[]): Promise<StaleVideoPruneSummary> {
  if (!hasTauri()) {
    return {
      checkedVideos: 0,
      removedMissingVideos: 0,
      removedOrphanVideos: 0,
      deletedPairScores: 0,
      deletedMatchEdges: 0,
      deletedFrameEmbeddings: 0,
    };
  }
  return invoke<StaleVideoPruneSummary>("prune_stale_videos", { sessionIds });
}

export async function getStorageUsage(): Promise<StorageUsageSummary> {
  if (!hasTauri()) {
    return {
      dataDir: "C:\\Example\\Duplicate-video-search\\data",
      totalBytes: 0,
      items: [],
      databaseStats: {
        videos: 0,
        scanSessions: 0,
        frameEmbeddings: 0,
        aiPairScores: 0,
        aiMatchEdges: 0,
        embeddingModels: 0,
      },
    };
  }
  return invoke<StorageUsageSummary>("get_storage_usage");
}

export async function cleanupStorage(): Promise<StorageCleanupSummary> {
  if (!hasTauri()) {
    return {
      deletedAiFrameCacheFiles: 0,
      deletedAiFrameCacheBytes: 0,
      deletedThumbnailFiles: 0,
      deletedThumbnailBytes: 0,
      removedOrphanVideos: 0,
      deletedPairScores: 0,
      deletedMatchEdges: 0,
      deletedFrameEmbeddings: 0,
    };
  }
  return invoke<StorageCleanupSummary>("cleanup_storage");
}

export async function cleanupCompletedAiFrameCache(): Promise<CompletedAiFrameCacheCleanupSummary> {
  if (!hasTauri()) {
    return {
      checkedVideos: 0,
      eligibleVideos: 0,
      deletedFiles: 0,
      deletedBytes: 0,
    };
  }
  return invoke<CompletedAiFrameCacheCleanupSummary>("cleanup_completed_ai_frame_cache");
}

export async function vacuumDatabase(): Promise<VacuumSummary> {
  if (!hasTauri()) {
    return {
      beforeBytes: 0,
      afterBytes: 0,
      reclaimedBytes: 0,
    };
  }
  return invoke<VacuumSummary>("vacuum_database");
}

export async function listMatchGroups(
  sessionIds?: number[],
  minConfidence?: number,
  maxConfidence?: number,
): Promise<MatchGroup[]> {
  if (!hasTauri()) return previewGroups;
  return invoke<MatchGroup[]>("list_match_groups", { sessionIds, minConfidence, maxConfidence });
}

export async function createReplacementPlan(
  highQualityId: number,
  namingSourceId: number,
): Promise<ReplacementPlan> {
  if (!hasTauri()) {
    return {
      planId: "replace-plan-mock",
      highQualityId,
      namingSourceId,
      highQualityPath: "",
      targetPath: "",
      backupPath: "",
      planFile: "C:\\Example\\Duplicate-video-search\\data\\operations\\replace-plan-mock.json",
      steps: [],
      warnings: ["预览模式没有可替换文件"],
    };
  }
  return invoke<ReplacementPlan>("create_replacement_plan", {
    highQualityId,
    namingSourceId,
  });
}

export async function executeReplacement(
  highQualityId: number,
  namingSourceId: number,
  confirmation: string,
): Promise<OperationResult> {
  if (!hasTauri()) {
    return {
      operationId: "replace-operation-mock",
      status: confirmation === "REPLACE" ? "planned-mock" : "rejected-mock",
      logFile: "C:\\Example\\Duplicate-video-search\\data\\operations\\operation-log.jsonl",
      messages: ["浏览器预览模式不会改动文件"],
    };
  }
  return invoke<OperationResult>("execute_replacement", {
    highQualityId,
    namingSourceId,
    confirmation,
  });
}

export async function executeMergeSelection(
  highQualityId: number,
  namingSourceId: number,
  extraVideoIds: number[],
  disposal: "backup" | "delete",
  confirmation: string,
  filenameSourceId?: number | null,
): Promise<OperationResult> {
  if (!hasTauri()) {
    return {
      operationId: `merge-${disposal}-mock`,
      status: "planned-mock",
      logFile: "C:\\Example\\Duplicate-video-search\\data\\operations\\operation-log.jsonl",
      messages: ["浏览器预览模式不会改动文件"],
    };
  }
  return invoke<OperationResult>("execute_merge_selection", {
    highQualityId,
    namingSourceId,
    extraVideoIds,
    disposal,
    confirmation,
    filenameSourceId,
  });
}

export async function executeBatchMergeSelection(
  tasks: BatchMergeTask[],
  disposal: "backup" | "delete",
  confirmation: string,
): Promise<OperationResult> {
  if (!hasTauri()) {
    return {
      operationId: `batch-merge-${disposal}-mock`,
      status: "planned-mock",
      logFile: "C:\\Example\\Duplicate-video-search\\data\\operations\\operation-log.jsonl",
      messages: ["浏览器预览模式不会改动文件"],
    };
  }
  return invoke<OperationResult>("execute_batch_merge_selection", {
    tasks,
    disposal,
    confirmation,
  });
}

export async function openVideo(path: string): Promise<void> {
  if (!hasTauri()) return;
  return invoke<void>("open_video", { path });
}

export async function getAppSettings(): Promise<AppSettings> {
  if (!hasTauri()) {
    return {
      uiLanguage: "zh",
      backupDir: "C:\\Example\\Duplicate-video-search\\data\\backups",
      namingSourceDirs: [],
      keeperSizePriorityDurationSeconds: 300,
      scanWorkerCount: 1,
      sampleHashCount: 11,
      temporalHashThreshold: 8,
      minTemporalMatchPoints: 3,
      allowedUnmatchedSampleFrames: 14,
      allowDirectDelete: false,
      restrictScanToTestPath: true,
      compareWithinSameFolder: false,
      aiVisionEnabled: true,
      aiModelPath: "models\\dinov2-small-dynamic\\model.onnx",
      aiDevice: "auto",
      aiFrameCount: 32,
      aiBatchSize: 32,
      aiSimilarityThreshold: 0.86,
      aiMinMatchedFrames: 8,
      aiClipMatchingEnabled: true,
      aiIndexAfterScan: true,
      aiExtractWorkerCount: 4,
      aiGpuWorkerCount: 4,
      aiMatchWorkerCount: 8,
      aiFrameCacheEnabled: true,
      deleteAiFrameCacheAfterIndex: false,
      ramDiskEnabled: true,
      ramDiskSizeMb: 16 * 1024,
      ramDiskSetupCompleted: false,
      localPreprocessEnabled: true,
      localPreprocessVideoWorkers: 1,
      localPreprocessOverlapStartPercent: 95,
      localPreprocessProcessWorkers: 2,
      localPreprocessFrameWorkers: 16,
      localPreprocessTempDir: "Z:\\TEMP",
      localPreprocessSecondaryTempDir: "D:\\TEMP",
      localPreprocessSecondaryThresholdMb: 16 * 1024,
      nasSshPreprocessEnabled: false,
      nasSshHost: "",
      nasSshUser: "",
      nasSshPassword: "",
      nasSshHostKey: "",
      nasSshPort: 22,
      nasRemoteRoot: "",
      nasSmbRoot: "\\\\EXAMPLE-NAS\\Test",
      nasSshFfmpegWorkers: 1,
    };
  }
  return invoke<AppSettings>("get_app_settings");
}

export async function saveAppSettings(settings: AppSettings): Promise<AppSettings> {
  if (!hasTauri()) return settings;
  return invoke<AppSettings>("save_app_settings", { settings });
}

export async function getRamDiskStatus(): Promise<RamDiskStatus> {
  if (!hasTauri()) {
    return {
      supported: true,
      driverInstalled: true,
      mounted: true,
      ready: true,
      setupCompleted: true,
      configuredSizeMb: 16 * 1024,
      actualCapacityMb: 16 * 1024,
      driveLetter: "Z",
      cachePath: "Z:\\TEMP",
      message: "浏览器预览模式",
    };
  }
  return invoke<RamDiskStatus>("get_ram_disk_status");
}

export async function ensureRamDisk(): Promise<RamDiskStatus> {
  if (!hasTauri()) return getRamDiskStatus();
  return invoke<RamDiskStatus>("ensure_ram_disk");
}

export async function releaseRamDisk(): Promise<RamDiskStatus> {
  if (!hasTauri()) return getRamDiskStatus();
  return invoke<RamDiskStatus>("release_ram_disk");
}

export async function configureRamDisk(sizeMb: number): Promise<RamDiskStatus> {
  if (!hasTauri()) return getRamDiskStatus();
  return invoke<RamDiskStatus>("configure_ram_disk", { sizeMb });
}

export async function openRamDiskDriverDownload(): Promise<void> {
  if (!hasTauri()) return;
  return invoke<void>("open_ram_disk_driver_download");
}

export async function getAiModelStatus(): Promise<AiModelStatus> {
  if (!hasTauri()) {
    return {
      enabled: false,
      ready: false,
      modelId: null,
      modelPath: "",
      device: "auto",
      message: "browser preview",
    };
  }
  return invoke<AiModelStatus>("get_ai_model_status");
}

export async function buildAiIndex(
  sessionIds?: number[],
  forceRebuild = false,
): Promise<AiIndexSummary> {
  if (!hasTauri()) {
    return {
      modelId: "mock",
      modelPath: "models\\mock.onnx",
      totalVideos: 0,
      processed: 0,
      skipped: 0,
      insufficientFrames: 0,
      failed: 0,
      elapsedMs: 0,
      recentErrors: [],
    };
  }
  return invoke<AiIndexSummary>("build_ai_index", { sessionIds, forceRebuild });
}

export async function cancelCurrentWork(): Promise<void> {
  if (!hasTauri()) return;
  return invoke<void>("cancel_current_work");
}

export async function pickFolder(initialPath?: string): Promise<string | null> {
  if (!hasTauri()) return null;
  return invoke<string | null>("pick_folder", { initialPath });
}

export async function pickFolders(initialPath?: string): Promise<string[]> {
  if (!hasTauri()) return [];
  return invoke<string[]>("pick_folders", { initialPath });
}

export async function executeFileAction(
  videoIds: number[],
  action: "backup" | "delete",
  confirmation: string,
  deferIndexUpdate = false,
): Promise<OperationResult> {
  if (!hasTauri()) {
    return {
      operationId: `${action}-mock`,
      status: "planned-mock",
      logFile: "C:\\Example\\Duplicate-video-search\\data\\operations\\operation-log.jsonl",
      messages: ["浏览器预览模式不会改动文件"],
    };
  }
  return invoke<OperationResult>("execute_file_action", {
    videoIds,
    action,
    confirmation,
    deferIndexUpdate,
  });
}

export async function listOperationHistory(): Promise<OperationHistoryEntry[]> {
  if (!hasTauri()) return [];
  return invoke<OperationHistoryEntry[]>("list_operation_history");
}

export async function rollbackOperation(operationId: string): Promise<OperationResult> {
  if (!hasTauri()) {
    return {
      operationId: `rollback-${operationId}`,
      status: "planned-mock",
      logFile: "C:\\Example\\Duplicate-video-search\\data\\operations\\operation-log.jsonl",
      messages: ["浏览器预览模式不会改动文件"],
    };
  }
  return invoke<OperationResult>("rollback_operation", { operationId });
}
