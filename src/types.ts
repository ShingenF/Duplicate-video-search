export type ToolStatus = {
  workspaceRoot: string;
  dataDir: string;
  databasePath: string;
  allowedSource: string;
  ffmpegPath: string | null;
  ffprobePath: string | null;
  ffmpegVersion: string | null;
  ffprobeVersion: string | null;
};

export type ScanSummary = {
  sessionId: number | null;
  source: string;
  databasePath: string;
  totalFiles: number;
  scanned: number;
  failed: number;
  elapsedMs: number;
};

export type ScanSession = {
  id: number;
  source: string;
  startedUnixMs: number;
  completedUnixMs: number | null;
  totalFiles: number;
  scanned: number;
  failed: number;
};

export type ScanProgress = {
  source: string;
  phase: string;
  totalFiles: number;
  scanned: number;
  failed: number;
  currentPath: string | null;
};

export type IndexRefreshSummary = {
  removedMissingSources: number;
  skippedUnavailableRoots: number;
};

export type DeleteIndexSummary = {
  requestedSessions: number;
  deletedSessions: number;
  affectedVideos: number;
  deletedVideos: number;
  deletedPairScores: number;
  deletedMatchEdges: number;
  deletedFrameEmbeddings: number;
};

export type StaleVideoPruneSummary = {
  checkedVideos: number;
  removedMissingVideos: number;
  removedOrphanVideos: number;
  deletedPairScores: number;
  deletedMatchEdges: number;
  deletedFrameEmbeddings: number;
};

export type StorageUsageItem = {
  key: string;
  path: string;
  bytes: number;
  fileCount: number;
};

export type StorageDatabaseStats = {
  videos: number;
  scanSessions: number;
  frameEmbeddings: number;
  aiPairScores: number;
  aiMatchEdges: number;
  embeddingModels: number;
};

export type StorageUsageSummary = {
  dataDir: string;
  totalBytes: number;
  items: StorageUsageItem[];
  databaseStats: StorageDatabaseStats;
};

export type StorageCleanupSummary = {
  deletedAiFrameCacheFiles: number;
  deletedAiFrameCacheBytes: number;
  deletedThumbnailFiles: number;
  deletedThumbnailBytes: number;
  removedOrphanVideos: number;
  deletedPairScores: number;
  deletedMatchEdges: number;
  deletedFrameEmbeddings: number;
};

export type CompletedAiFrameCacheCleanupSummary = {
  checkedVideos: number;
  eligibleVideos: number;
  deletedFiles: number;
  deletedBytes: number;
};

export type VacuumSummary = {
  beforeBytes: number;
  afterBytes: number;
  reclaimedBytes: number;
};

export type DeleteIndexProgress = {
  phase: string;
  phaseProcessed: number;
  phaseTotal: number;
  requestedSessions: number;
  deletedSessions: number;
  affectedVideos: number;
  deletedVideos: number;
};

export type AiIndexProgress = {
  phase: string;
  totalVideos: number;
  processed: number;
  skipped: number;
  insufficientFrames: number;
  failed: number;
  started: number;
  prepared: number;
  currentPath: string | null;
};

export type RamDiskStatus = {
  supported: boolean;
  driverInstalled: boolean;
  mounted: boolean;
  ready: boolean;
  setupCompleted: boolean;
  configuredSizeMb: number;
  actualCapacityMb: number | null;
  driveLetter: string | null;
  cachePath: string;
  message: string;
};

export type AiIndexSummary = {
  modelId: string;
  modelPath: string;
  totalVideos: number;
  processed: number;
  skipped: number;
  insufficientFrames: number;
  failed: number;
  elapsedMs: number;
  recentErrors: string[];
};

export type AiModelStatus = {
  enabled: boolean;
  ready: boolean;
  modelId: string | null;
  modelPath: string;
  device: string;
  message: string;
};

export type VideoRecord = {
  id: number | null;
  path: string;
  fileName: string;
  parentPath: string;
  sizeBytes: number;
  modifiedUnixMs: number;
  extension: string;
  containerFormat: string | null;
  durationSeconds: number | null;
  width: number | null;
  height: number | null;
  bitrate: number | null;
  codec: string | null;
  frameRate: number | null;
  audioCodec: string | null;
  partialHash: string | null;
  sampleHashes: string[];
  previewImages: string[];
  scanStatus: string;
  error: string | null;
  qualityScore: number;
  scannedAtUnixMs: number;
};

export type MatchRange = {
  startSeconds: number;
  endSeconds: number;
  startFraction: number;
  endFraction: number;
  matchedFrames: number;
  averageSimilarity: number;
};

export type MatchHitPoint = {
  seconds: number;
  fraction: number;
  similarity: number;
};

export type MatchDetail = {
  peerVideoId: number | null;
  peerFileName: string;
  points: MatchHitPoint[];
  startSeconds: number;
  endSeconds: number;
  hitCount: number;
  displayedHitCount: number;
  averageSimilarity: number;
  confidence: number;
  shortCoverage: number;
  longCompactness: number;
  longSpanSeconds: number;
  longSpanFraction: number;
  relationType: string;
  relationLabel: string;
  relationExplanation: string;
};

export type MatchItem = {
  video: VideoRecord;
  role: string;
  qualityRank: number;
  matchRanges?: MatchRange[];
  matchDetail?: MatchDetail | null;
};

export type MatchGroup = {
  id: string;
  title: string;
  kind: string;
  confidence: number;
  recommendedVideoId: number | null;
  reclaimableBytes: number;
  itemCount: number;
  evidence: string[];
  report: string;
  items: MatchItem[];
};

export type ReplacementPlan = {
  planId: string;
  highQualityId: number;
  namingSourceId: number;
  highQualityPath: string;
  targetPath: string;
  backupPath: string;
  planFile: string;
  steps: string[];
  warnings: string[];
};

export type OperationResult = {
  operationId: string;
  status: string;
  logFile: string;
  messages: string[];
};

export type BatchMergeTask = {
  keeperId: number;
  namingSourceId: number;
  filenameSourceId?: number | null;
  extraVideoIds: number[];
};

export type OperationHistoryEntry = {
  operationId: string;
  action: string;
  summary: string;
  createdAtUnixMs: number;
  reversible: boolean;
  rolledBack: boolean;
  messages: string[];
};

export type MatchRefreshProgress = {
  phase: string;
  totalVideos: number;
  totalPairs: number;
  processedPairs: number;
  cachedPairs: number;
  computedPairs: number;
  groups: number;
  pairsPerSecond: number;
  phaseProcessed: number;
  phaseTotal: number;
  phasePairsPerSecond: number;
};

export type AppSettings = {
  uiLanguage: "zh" | "en" | string;
  backupDir: string;
  namingSourceDirs: string[];
  keeperSizePriorityDurationSeconds: number;
  scanWorkerCount: number;
  sampleHashCount: number;
  temporalHashThreshold: number;
  minTemporalMatchPoints: number;
  allowedUnmatchedSampleFrames: number;
  allowDirectDelete: boolean;
  restrictScanToTestPath: boolean;
  compareWithinSameFolder: boolean;
  aiVisionEnabled: boolean;
  aiModelPath: string;
  aiDevice: "auto" | "gpu" | "cpu" | string;
  aiFrameCount: number;
  aiBatchSize: number;
  aiSimilarityThreshold: number;
  aiMinMatchedFrames: number;
  aiClipMatchingEnabled: boolean;
  aiIndexAfterScan: boolean;
  aiExtractWorkerCount: number;
  aiGpuWorkerCount: number;
  aiMatchWorkerCount: number;
  aiFrameCacheEnabled: boolean;
  deleteAiFrameCacheAfterIndex: boolean;
  ramDiskEnabled: boolean;
  ramDiskSizeMb: number;
  ramDiskSetupCompleted: boolean;
  localPreprocessEnabled: boolean;
  localPreprocessVideoWorkers: number;
  localPreprocessOverlapStartPercent: number;
  localPreprocessProcessWorkers: number;
  localPreprocessFrameWorkers: number;
  localPreprocessTempDir: string;
  localPreprocessSecondaryTempDir: string;
  localPreprocessSecondaryThresholdMb: number;
  nasSshPreprocessEnabled: boolean;
  nasSshHost: string;
  nasSshUser: string;
  nasSshPassword: string;
  nasSshHostKey: string;
  nasSshPort: number;
  nasRemoteRoot: string;
  nasSmbRoot: string;
  nasSshFfmpegWorkers: number;
};
