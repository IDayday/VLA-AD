# PSI-Drive Stage2 APSD Clean Run Status

更新时间：2026-06-25 15:16 UTC

## 运行入口

- run root：`/mnt/project/VLA-AD/outputs/psi_drive_stage2_apsd_clean_8gpu_20260624T214438Z`
- launcher：`scripts/psi_drive/run_stage2_apsd_8gpu.sh`
- 训练脚本：`navsim/planning/script/run_training_recogdrive.py`
- checkpoint layout：`psi`
- raw checkpoint dir：`checkpoints/raw`
- immutable store：`checkpoint_store/objects`
- val6000 ranking：`rankings/val6000/current_top5.{json,tsv}`
- navtest ranking：`rankings/navtest/current_top5.{json,tsv}`

## 数据与权重

- Stage2 初始化权重：`/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt`
- Stage1 VLM 权重：`/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B`
- official hidden cache：`/mnt/project/VLA-AD/cache/recogdrive_official_stage1_hidden_navtrain_2b`
- Stage2 Pareto support index：`/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_pareto_support_clean_full.pt`
- fixed val6000 token file：`artifacts/splits/navtrain_val6000_seed260306049.txt`
- navtrain metric cache：`/mnt/project/VLA-AD/cache/metric_cache_train_full`
- navtest metric cache：`/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1`
- navtest fast metric cache：`/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1_fast_pickle`

## 训练配置核对

来自 `data_report.json`：

- loader mode：`official-cache-loader`
- train records：`85109`
- validation records：`18179`
- train target source：`pareto_support`
- validation target source：`gt`
- support index records：`103288`
- `stage2_pareto_require_index=true`

来自 `precision_report.json`：

- requested precision：`16-mixed`
- trainable params：`34329219`
- total params：`46157972`
- optimizer LR：`5e-5`
- expert/JEP A/VGGT/Last-VLA/two-expert paths：全部关闭

## 当前训练状态

- Stage2 APSD Clean 训练已完成。
- `logs/train.log` 记录：`Trainer.fit stopped: max_epochs=60 reached.`
- 最后一轮进度：`Epoch 59/59`，`train/loss_epoch≈0.016`，`val/loss_epoch≈0.016`。
- 训练过程中 APSD 诊断正常：`stage2_pareto_used_ratio=1.0`，`stage2_pareto_missing_ratio=0.0`。

## Checkpoint 状态

- raw checkpoint：`epoch_001.ckpt` 到 `epoch_060.ckpt`，以及 `latest.ckpt`，共 `61` 个候选文件。
- immutable checkpoint store：`checkpoint_store/objects`
- inventory：`checkpoint_store/inventory.tsv`
- inventory rows：`61`
- unique objects：`61`
- navtest state：`61/61 done`
- val6000 state：`4 done`，`1 started`，`56 pending`
- 当前 val6000 正在评估 `epoch_005`。

checkpoint store 验证：

- `scripts/checkpoints/verify_checkpoint_store.py --run-root ...` 通过
- inventory rows：`61`
- unique objects：`61`
- failures：`[]`

## 自动评估与 Top-5

新增并已 dry-run 通过：

- `scripts/evaluation/run_psi_stage2_pdm_eval_8gpu_exact_pool.sh`
  - `EVAL_SPLIT=val6000`：使用 `train_test_split=navtrain`、fixed val6000 token file、train metric cache。
  - `EVAL_SPLIT=navtest`：使用 `train_test_split=navtest`、navtest metric cache、navtest fast metric cache。
- `scripts/evaluation/watch_psi_stage2_checkpoint_eval_8gpu.sh`
  - 监控 raw checkpoint；
  - 写入 immutable checkpoint store；
  - 等待 GPU 空闲；
  - 分别运行 val6000/navtest PDM；
  - 追加 `eval/<split>/summary.tsv`；
  - 刷新 `rankings/<split>/current_top5.{json,tsv}`。
  - 回写 `checkpoint_store/inventory.tsv` 中对应 split 的状态和指标列。

已启动 watcher：

- val6000 watcher PID：`3894141`
- navtest watcher PID：`3894153`

当前 watcher 策略：

- `WAIT_FOR_FREE_GPUS=1`
- `GPU_MAX_MEM_USED_MB=2000`
- `GPU_MAX_UTIL=10`
- `STABLE_SECONDS=120`
- `POLL_SECONDS=300`

当前评估状态：

- navtest full PDM：`61/61` 完成，`rankings/navtest/current_top5.tsv` 已刷新。
- val6000 fixed-token PDM：`epoch_001`、`epoch_002`、`epoch_003`、`epoch_004` 完成，`epoch_005` 正在运行，其余候选排队。
- Stage3 初始化 checkpoint 必须等待 val6000 Top-5 完成后按 val6000 选择；navtest 只用于报告和最终泛化验证，不能反向用于训练选择。

当前 val6000 临时结果仅用于进度跟踪，不作为 Stage3 初始化 checkpoint 的最终选择依据：

| rank | checkpoint | epoch | PDMS | NC | DAC | TTC | EP |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `epoch_004` | 004 | `0.9006617565300609` | `0.9846666666666667` | `0.9636666666666667` | `0.9273333333333333` | `0.8892172667499721` |
| 2 | `epoch_002` | 002 | `0.8994676237911241` | `0.98825` | `0.9603333333333334` | `0.9308333333333333` | `0.8862873589864733` |
| 3 | `epoch_001` | 001 | `0.8982863878815402` | `0.9871666666666666` | `0.9623333333333334` | `0.9183333333333333` | `0.891469549629211` |
| 4 | `epoch_003` | 003 | `0.8769757201525776` | `0.9830833333333333` | `0.9451666666666667` | `0.9118333333333334` | `0.8709863415492709` |

## Stage3 启动准备

已新增 `scripts/psi_drive/run_stage3_after_val6000_top1.sh`，用于自动衔接 Stage2 与 Stage3：

- 轮询 `eval/val6000/checkpoint_eval_status.tsv`；
- 要求 `61/61` 个候选 checkpoint 的 val6000 评估全部完成；
- 从 `rankings/val6000/current_top5.tsv` 读取最终 Top-1 的 immutable object checkpoint；
- 将选择结果写入 Stage3 run root 的 `stage2_val6000_selected_top1.tsv`；
- 启动 `scripts/psi_drive/run_stage3_sr_pgrpo_8gpu.sh`；
- 同时启动 `stage3` 的 val6000/navtest checkpoint eval watcher。

后台等待进程：

- PID：`1359605`
- output dir：`/mnt/project/VLA-AD/outputs/psi_drive_stage3_sr_pgrpo_from_stage2_val6000_20260625T151340Z`
- log：`logs/stage3_after_val6000_orchestrator.log`
- 当前状态：只在 `sleep 300` 轮询，不占用 GPU；会等 val6000 全部完成后再启动 Stage3。

相关脚本校验：

- `bash -n scripts/psi_drive/run_stage3_after_val6000_top1.sh scripts/evaluation/watch_psi_stage2_checkpoint_eval_8gpu.sh scripts/evaluation/watch_psi_stage3_checkpoints.sh scripts/evaluation/run_recogdrive_stage3_safe_diffgrpo_eval_8gpu_exact_pool_pdm.sh scripts/evaluation/run_recogdrive_stage3_safe_diffgrpo_eval_8gpu.sh`：通过
- `STATUS_ONLY=1 scripts/psi_drive/run_stage3_after_val6000_top1.sh`：正确读到当前 val6000 完成数和临时 Top-1。
- `DRY_RUN=1 scripts/evaluation/run_recogdrive_stage3_safe_diffgrpo_eval_8gpu_exact_pool_pdm.sh`：沙箱外通过，确认 8 张 GPU 可见、navtest metric cache/fast cache 可读，命令显式关闭 JEPA/VGGT/Last-VLA/two-expert。

当前 navtest Top-5：

| rank | checkpoint | epoch | PDMS | NC | DAC | TTC | EP |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `epoch_011` | 011 | `0.8767243811626906` | `0.9799802273850716` | `0.9498269896193772` | `0.9391992090954029` | `0.8385466421598063` |
| 2 | `epoch_010` | 010 | `0.8748095169153595` | `0.9814219805569286` | `0.9487559729774263` | `0.9401054539462844` | `0.8338191131261046` |
| 3 | `epoch_048` | 048 | `0.8700656805038407` | `0.9770555280935904` | `0.9480968858131488` | `0.929230515735706` | `0.8351164948477742` |
| 4 | `epoch_046` | 046 | `0.8655468698222394` | `0.9705882352941176` | `0.9465315537979898` | `0.9173669467787114` | `0.8410678350409626` |
| 5 | `epoch_018` | 018 | `0.8644278204750405` | `0.974007249958807` | `0.9408469270060965` | `0.9221453287197232` | `0.8385389345640343` |

2026-06-24 22:20 UTC 已修正 watcher 的锁继承问题：

- `sleep` 子进程不再继承 watcher lock fd；
- global eval lock 的提前返回路径会显式关闭 fd；
- watcher 优先从 inventory 命中已归档 checkpoint，避免 GPU 忙时反复 hash 旧 checkpoint；
- `lsof` 核对显示只有两个 watcher bash 进程持有 split lock，sleep 不再持锁。

## 本轮新增验证

- `pytest -q tests/test_psi_checkpoint_tools.py`：`4 passed`
- `python -m py_compile scripts/checkpoints/update_checkpoint_inventory_eval_metrics.py scripts/checkpoints/archive_checkpoint_immutable.py scripts/checkpoints/rank_eval_checkpoints.py scripts/checkpoints/verify_checkpoint_store.py`：通过
- `bash -n scripts/evaluation/run_psi_stage2_pdm_eval_8gpu_exact_pool.sh scripts/evaluation/watch_psi_stage2_checkpoint_eval_8gpu.sh scripts/psi_drive/run_stage2_apsd_8gpu.sh scripts/psi_drive/run_stage3_sr_pgrpo_8gpu.sh`：通过
- `DRY_RUN=1 scripts/psi_drive/run_stage3_sr_pgrpo_8gpu.sh`：通过，命令显式设置官方 VLM，并关闭 JEPA/VGGT/Last-VLA/two-expert。

## 待完成

1. val6000 watcher 完成全部 `61` 个候选 checkpoint 的固定 val6000 PDM 评估并刷新 Top-5。
2. 按 val6000 Top-5/Top-1 规则选定 Stage2 APSD checkpoint；不得用 navtest Top-5 反向选择。
3. 用选定 Stage2 checkpoint 启动 Stage3 SR-PGRPO。
4. Stage3 完成训练、固定 val6000 评估、full navtest 评估和 Top-5 安全保存。
