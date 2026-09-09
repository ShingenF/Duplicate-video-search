# 配置说明 / Configuration

首次运行在 GUI 的 Settings 页面设置并保存。应用把设置写入 `data/settings.json`，使用 camelCase 字段；不要把该文件加入 Git。

| 字段 | 用途 |
| --- | --- |
| `compareWithinSameFolder` | 默认 `false`；只匹配直接父文件夹相同的路径，子文件夹独立。Rust 字段为 `compare_within_same_folder` |
| `restrictScanToTestPath` | 默认 `true`；关闭后才能扫描测试范围外的本地或 SMB 路径 |
| `backupDir` | 备份目录，默认应用 `data/backups`；扫描时排除该目录 |
| `allowDirectDelete` | 默认 `false`；开启后的直接删除不可回滚 |
| `aiModelPath` | ONNX 文件绝对路径或相对应用工作目录的路径 |
| `aiDevice` | `auto`、`gpu` 或 `cpu` |
| `ramDiskEnabled` | 默认 `true`；普通磁盘缓存模式设为 `false` |
| `ramDiskSetupCompleted` | 由 GUI 配置流程维护，请勿手动设为 `true` 绕过检查 |
| `localPreprocessTempDir` | 一级缓存目录；默认 `Z:\TEMP`，可改为用户选定目录 |
| `localPreprocessSecondaryTempDir` | 大文件二级缓存目录；默认 `D:\TEMP`，请按磁盘情况修改 |

配置文件有必填字段，不要只写入上表部分字段来替换整个文件。先由 GUI 生成，再按需调整。`localPreprocessEnabled` 与帧缓存当前会在加载时强制开启，`nasSshPreprocessEnabled` 会强制关闭。

`DVS_HOME` 指定工作目录；`DVS_FFMPEG` 和 `DVS_FFPROBE` 指定工具完整路径。这些都是用户本地环境变量，不需要提交 `.env` 或真实路径样例。

旧 NAS 字段（主机、用户、密码、主机密钥、远程目录、SMB 根目录）仅保留兼容性，没有预置真实账号。设置以明文 JSON 保存；即使 NAS 功能关闭，已有字段仍可能包含私人数据。

Configure and save settings through the GUI first. The file contains required fields, so do not replace it with a partial JSON snippet. Same-folder matching defaults to false and affects traditional and AI matching, including cached results. Configure actual writable cache directories before work; never manually bypass RAM disk setup checks. Runtime settings are plain JSON and must remain private.
