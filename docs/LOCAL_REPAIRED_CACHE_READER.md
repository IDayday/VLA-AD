# 本机完整教师特征缓存读取

`scripts/student_distillation/read_teacher_cache.py` 直接读取本轮新拟合、冻结的本机教师缓存。每例完整接口为原始 FP32 Base mmap 加上实际评分时保存的 FP32 修复量，包含原始 2,800 个位置及 padding。读取过程核验完整接口的字节 SHA，与同一次 GPU 评分执行保存的教师接口一致。

五条轨迹及七项原生评分直接来自该例的 `SCORES.npz`。读取器只使用 CPU/NumPy；模型权重和动作头不参与读取。

在代码仓库根目录运行：

```bash
python scripts/student_distillation/read_teacher_cache.py \
  --paths /mnt/project/recogdrive-no-routing-round1-results-20261026/protocol/PATHS.json \
  --population train --token <实际 token> --target-version full \
  --out /mnt/project/my_research/teacher_scene.npz
```

`--paths` 也接受 `{"paths": {...}}` 形式。它引用本机新教师的 `teacher_bundle` 和冻结的 `feature_manifest`。不带 `--out` 时，命令验证该例并打印身份、分支、噪声与 SHA 等元数据。`--verify-index` 额外核验完整索引和选择文件，适合全局审计。

## 返回内容

| 字段 | 类型与形状 |
|---|---|
| `teacher_interface`、`Base_interface` | FP32 `[2800,384]`，保留原始布局 |
| `actual_shift`、`actual_role_shifts` | FP32 `[384]`、`[2,384]` |
| `image_mask`、`text_mask`、`valid_mask`、`padding_mask` | bool `[2800]` |
| `token_ids`、`attention_mask`、`position_ids` | 原始 int64 `[2800]` |
| `state24` | 原始 FP32 `[24]`：history12、status8、command4 |
| `teacher_trajectories`、`Base_trajectories` | FP32 `[5,8,3]`，按 S0–S4 排列 |
| `teacher_quality`、`Base_quality` | FP64 `[5,7]`，原生 0–1 单位 |
| `noise_indices`、`integer_seeds` | 五噪声身份；实际噪声带路径与 SHA 在元数据中 |
| `metadata_json` | NPZ 中的 JSON 字符串，包含教师身份、token/log/global index、分支、实际 RMS/cap、能量、原始评分文件与 mmap 来源 |

评分列顺序为 `score`、`no_at_fault_collisions`、`drivable_area_compliance`、`time_to_collision_within_bound`、`ego_progress`、`comfort`、`driving_direction_compliance`。PDMS 点数为 `100 × score`。

`full` 返回冻结 full 教师的场景选择。`gain1` 使用同一份预先固定的严格 `gain_points > 1` 清单；增益未超过 1 点的场景返回分支 0 的 Base 接口和同次执行的 Base 五轨迹/评分，即使原 full 选择为非 Base 专家。字段 `gain1_keep=true` 表示保留 full 修复，`gain1_uses_Base=true` 表示返回 Base。两种版本的身份和原 full 分支都会保留。Validation/Test 的事后教师输出带有 `diagnostic_only=true`。

能量字段分别记录 Base 全 token/有效 token 的 FP64 平方均值，以及教师相对 Base 的有效 token/全 token 残差能量。全 token 残差能量采用 `有效 token 残差能量 × 有效 token 数 / 2800`。这些字段用于审计与研究记录。

## Python 批量读取

在仓库根目录启动 Python，可复用一次打开的 SQLite token 索引：

```python
from scripts.student_distillation.read_teacher_cache import TeacherCacheReader

paths = "/mnt/project/recogdrive-no-routing-round1-results-20261026/protocol/PATHS.json"
with TeacherCacheReader(paths, "train") as cache:
    scene = cache.read(token, target_version="gain1")
    teacher = scene["teacher_interface"]  # numpy.float32 [2800,384]
    trajectories = scene["teacher_trajectories"]
    provenance = scene["metadata"]
```

每次读取通过 token 直接定位一行索引，随后读取该例的原始 mmap 切片及已保存评分文件。它验证实际修复量、mask/layout/state、原始 padding 字节、完整教师接口 SHA 和 native RMS/cap。

## 索引构建与自动等待

每个人群完成真实全量候选评分及选择后，单独构建：

```bash
python scripts/student_distillation/read_teacher_cache.py \
  --paths <PATHS.json> --population train --build-index --workers 4
```

新增文件位于本机教师实验的 `requests/<population>/`：

- `REPAIRED_CACHE_INDEX.jsonl`：token/global index、完整 Base mmap 来源、实际 δ、分支、噪声和同次评分文件 SHA。
- `REPAIRED_CACHE_INDEX.sqlite3`：直接 token 查询所需的字节偏移与行 SHA。
- `REPAIRED_CACHE_INDEX.COMPLETE.json`：全量覆盖与身份/产物 SHA；最后写入。

这些是追加的新索引产物。已有教师参数、选择表、评分和 Base mmap 保持原状。

可用独立 CPU watcher 在各人群就绪后立即构建：

```bash
python scripts/student_distillation/read_teacher_cache.py \
  --paths <提前登记的 teacher_bundle/feature_manifest 路径 JSON> \
  --watch-build-index --populations train test validation --workers 4
```

watcher 等待真实人群完成标志，然后逐个人群构建索引。全部三个索引完成才写 `repaired_cache_index_watcher/COMPLETE.json`。
