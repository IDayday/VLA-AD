# Stage2 DPSI-FS-x0-AnchorBank 改进审查记录

本文档记录 `stage2_dpsi_fs_x0_anchorbank_plan.md` 与当前代码实现的对齐情况，以及本轮准备启动的 Stage2 训练配置。

## 目标

本轮目标是改进 ReCogDrive Stage2 DiT 训练，不再使用单 GT 作为唯一模仿目标，而是使用已经构建完成的 SG-FPS Pareto 支持集。训练必须从 Stage2 DiT 随机初始化开始，不从已有 IL / Stage2 checkpoint 微调。

## 当前开发分支

当前工作分支：

- `feature/recogdrive-last-vla-v2`

该分支已经包含 SG-FPS、DPSI、FS-Norm、x0/geometry auxiliary、Feasible Pareto-GRPO 等相关改动。本轮继续基于该分支修改，而不是另起分支重做。

## 数据与资产

最终 Pareto 支持集：

- `/mnt/project/VLA-AD_last_vla_dev/outputs/sg_fps_support_semantic_full_20260705T032521Z/reselect_ref_relative_feasrelax_gtimprover_20260705T185120Z/support_v3`

支持集审计报告：

- `reports/sg_fps/support_audit_20260705/full_audit.md`
- `reports/sg_fps/support_audit_20260705/full_audit.json`
- `docs/sg_fps_pareto_support_set_analysis.md`

FS-Norm stats：

- `/mnt/project/VLA-AD_last_vla_dev/outputs/sg_fps_dpsi_clean_support_gtimprover_20260705T192117Z/fs_norm_stats.pt`

已核查：

- `support_archive_path.txt` 指向最终 `support_v3`。
- stats shape 为 `[8, 3]`。
- `use_robust=True`。
- `clip=5.0`。

## 设计对齐情况

## 方案取舍：不机械照搬文档

用户文档提供的是优化方向和设计思想，不是必须逐字落地的固定实现。本轮最终采纳的原则是：

1. Stage2 DiT 从随机初始化开始训练，不能依赖已有 IL checkpoint。
2. 训练目标要让模型学到高质量、可行、合理、多样的轨迹策略。
3. Stage2 的输出应能作为 Stage3 Pareto-GRPO 的好初始化，而不是只追求 Stage2 imitation loss 最低。
4. 支持集不均匀是事实，设计要吸收不均匀性，而不是要求每个 scene 有完整 tag 或完整数量。

### 对 single-target DPSI 的判断

文档中的 single-target sampling 优点是最干净：

- 每个 scene 每步只贡献一个 target。
- 不会因为 support 数量多而让某些 scene 重复出现。
- 实现简单，训练语义清楚。

但它对随机初始化 DiT 有明显缺点：

- 梯度方差较大，同一个 scene 每步可能采到差异较大的 target。
- 如果某次采到较偏的 diversity support，早期模型可能更难收敛到稳定轨迹形态。
- 不能在同一步同时看到 GT/IL anchor、best support 和 Pareto/diversity support 之间的相对结构。

因此 single-target 更适合作为 ablation，而不是本轮主训练方案。

### 对 ASMI-style weighted multi-target 的判断

当前采用 ASMI/DPSI weighted multi-target：

- batch 维度仍然是 scene-level，不把 supports 展平成 dataset items。
- 每个 scene 每步选择少量 target，当前固定为 4 个。
- 强制包含 anchor 和当前 best，保证随机初始化早期有稳定目标。
- 其余 target 按 ASMI 权重从 Pareto/diversity supports 中采样，提供多样性。

这个方案更适合当前目标：

- 保持 scene-level uniform，避免 support-count bias。
- 比 single-target 更低方差。
- 能让模型同时学习“安全稳定基准”和“Pareto 改进方向”。
- 更适合作为 Stage3 Pareto-GRPO 前的初始化策略。

当前采纳结论：

- 主训练采用 ASMI-style weighted multi-target。
- single-target DPSI 保留为后续 ablation。
- AnchorBank 暂不进入主训练，等 A3 稳定后再作为 A4 ablation。

### 1. Ragged Support Target Sampling / ASMI 多目标场景均衡

设计要求：

- 不把 support 展平成 dataset item。
- 每个 scene 在 batch 维度仍只出现一次，避免 support 多的场景被重复放大。
- support 数量不均匀时，通过 mask / 权重处理，不让高 support-count scene 获得更多训练次数。

当前实现：

- 通过 `offline_rl_support_archive_path` 从 support archive 按 token 加载候选。
- 通过 `support_indices` 过滤到最终 selected supports。
- 在 planner 内部构建 ASMI/DPSI 权重。
- 本轮训练采用更适合优化稳定性的 ASMI-style 多目标方案：
  - `offline_rl_dpsi_target_sample_m: 4`
  - `offline_rl_dpsi_target_sample_m_after_warmup: 4`
  - `offline_rl_dpsi_force_anchor_target: true`
  - `offline_rl_dpsi_force_best_target: true`
  - `offline_rl_dpsi_beta_warmup_epochs: 0`
  - `offline_rl_dpsi_empty_tag_zero: false`
  - `offline_rl_dpsi_high_gt_reward: 1.1`
  - `offline_rl_dpsi_weight_unknown: 0.55`

结论：

- 当前训练配置没有把 support 展平成 dataset item，因此仍保持 scene-level uniform。
- 每个 scene 内部使用 4 个加权目标，目的是降低随机单 target 的梯度方差，并让模型在同一步内同时看到 GT/IL anchor、当前 best 和 Pareto/diversity supports。
- 这不是严格照搬文档中的 single-target 版本，而是更适合随机初始化 DiT 稳定训练的版本。

2026-07-09 复核发现：

- 初始正式训练的 event 标量显示 `dpsi_sampled_target_count_mean=1`、`dpsi_pareto_weight_mean=0`、`dpsi_gt_weight_ratio≈0.98`。
- 原因不是 support archive 缺失，而是 `dpsi_beta_warmup_epochs=10` 让 epoch 0 的 Pareto 分支 beta 为 0；同时 `dpsi_empty_tag_zero=true` 会把 selected support 中未打 tag 的高质量轨迹置零。
- 该训练已停止，不能作为最终 ASMI/DPSI 训练结果解读。
- 已改为从第 0 epoch 起启用非 GT support，同时保留 `force_anchor_target` 和 `force_best_target` 作为随机初始化早期稳定项。
- 修正后 1-step smoke 的 event 标量为：`dpsi_sampled_target_count_mean=3.9375`、`dpsi_pareto_weight_mean=0.321965`、`dpsi_beta_mean=0.296985`、`dpsi_gt_weight_ratio=0.713312`、`dpsi_external_weight_ratio=0.220397`。

### 2. FS-Norm Action Representation

设计要求：

- 将 raw ego-local absolute trajectory 转为 step-wise delta。
- 对 delta 做 robust normalization。
- diffusion 学习 normalized delta representation。
- 推理 / 评价前 decode 回 raw trajectory。

当前实现：

- `navsim/agents/recogdrive/fs_norm.py` 提供 `FSNormTransform`。
- planner 中 `_encode_action_target()` / `_decode_action_target()` 在 `use_fs_norm=True` 时走 FS-Norm。
- `scripts/tools/build_fs_norm_stats.py` 可从支持集构建 stats。
- `sg_fps_asmi_stage2_103k_fs_norm.yaml` 启用 `use_fs_norm: true`。

已验证：

- FS-Norm roundtrip 单元测试通过。
- 手动 torch roundtrip 检查 heading 误差约 `2.98e-08`。

### 3. x0 / delta / geometry Auxiliary Loss

设计要求：

- 主损失仍为 epsilon/noise prediction。
- 从 predicted noise 还原 `x0`。
- 在 clean trajectory 空间增加 x0、delta、geometry 辅助损失。
- 只在低噪声 timestep 上启用辅助约束。

本轮修改：

- 新增 `delta_aux_weight` 配置，默认 `0.0`，保持旧行为兼容。
- 在 `_compute_x0_geo_aux_per_sample_losses()` 中加入 step-wise delta SmoothL1。
- 在普通 forward、AWAC/DPSI weighted target loss、Last-VLA 分支中接入 `delta_aux_loss`。
- Lightning optional metrics 中加入 `delta_aux_loss`。

本轮训练配置：

- `x0_aux_weight: 0.10`
- `delta_aux_weight: 0.05`
- `geo_aux_weight: 0.05`
- `x0_aux_low_noise_frac: 0.5`

结论：

- 当前训练目标是 `L_eps + 0.10 L_x0 + 0.05 L_delta + 0.05 L_geo`。

### 4. Global AnchorBank Condition

设计文档中 AnchorBank 是 A4 ablation：

- 从全量 Pareto supports 构建全局轨迹 anchor bank。
- 训练时选择最近 anchor，以 residual condition 注入 DiT。
- 推理 / Stage3 可用于多模态候选采样。

当前状态：

- 本轮尚未实现 AnchorBank condition。
- 当前准备启动的是 A3：`DPSI + FS-Norm + x0/delta/geo`。

原因：

- 设计文档第 17 节明确将前 5 项列为“必须做”，AnchorBank 是“建议做 / A4 ablation”。
- 当前最需要先验证的是：已有 Pareto 支持集作为 Stage2 imitation target 后，随机初始化 DiT 是否能稳定训练并改善轨迹质量。
- AnchorBank 会改变 DiT conditioning 路径，引入额外 train/test mismatch 风险，适合作为 A4 后续对照实验。

后续建议：

- A3 训练稳定后，再实现 `scripts/build_pareto_anchor_bank.py` 和 planner residual condition。
- 如果 A4 unanchored 单轨评测不低于 A3，再把 A4 用作 Stage3 初始化或候选生成器。

## 随机初始化核查

旧训练记录中存在一个重要问题：

- `/mnt/project/VLA-AD_last_vla_dev/outputs/sg_fps_dpsi_clean_support_gtimprover_stepckpt_20260705T193744Z/dpsi_fs_norm_200ep/resolved_command.txt`
- 该命令包含：
  - `agent.checkpoint_path=/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-2B-IL/ReCogDrive_Diffusion_Planner_2B_IL.ckpt`

这说明旧训练是从 IL checkpoint 加载，不符合当前“随机初始化 DiT”的要求。

本轮新配置：

- `navsim/planning/script/config/experiment/sg_fps_asmi_stage2_103k.yaml`
- 设置：
  - `agent.checkpoint_path: ''`
  - `agent.allow_random_init: true`

训练脚本：

- `scripts/training/run_sg_fps_asmi_stage2_103k.sh`
- 默认 `HYDRA_EXPERIMENT=sg_fps_asmi_stage2_103k_fs_norm`
- command log 写入 `RANDOM_INIT_DIT=1`

结论：

- 新训练会从随机初始化 Stage2 DiT 开始。
- `vlm_path` 仍指向 ReCogDrive VLM 基座，这是 hidden/cache 读取和模型构造所需，不等于加载 DiT checkpoint。

## 本轮代码修改

修改文件：

- `navsim/agents/recogdrive/recogdrive_diffusion_planner.py`
- `navsim/agents/recogdrive/recogdrive_agent.py`
- `navsim/planning/script/config/common/agent/recogdrive_agent.yaml`
- `navsim/planning/script/config/experiment/sg_fps_asmi_stage2_103k.yaml`
- `navsim/planning/script/config/experiment/sg_fps_asmi_stage2_103k_fs_norm.yaml`
- `navsim/planning/training/agent_lightning_module.py`
- `scripts/training/run_sg_fps_asmi_stage2_103k.sh`
- `tests/test_sg_fps_asmi.py`

新增 / 更新文档：

- `docs/stage2_dpsi_fs_x0_anchorbank_review.md`

## 训练超参数定稿

### Batch size

原版 / 本地官方对齐的 ReCogDrive Stage2 配置使用：

- per-GPU batch size: `16`
- GPU 数量: `8`
- gradient accumulation: `1`
- effective scene batch size: `128`

本轮训练为了让 optimizer step 数、epoch-based scheduler 行为和原版 Stage2 对齐，也固定 effective batch size 为 `128`。

执行策略：

- 默认：8 卡，每卡 micro batch `16`，`accumulate_grad_batches=1`。
- 如果显存不够：8 卡，每卡 micro batch `8`，`accumulate_grad_batches=2`。
- 两种方式的 effective batch 都是 `128`，学习率和 scheduler 不随 micro batch 改变。

2026-07-09 的 1-GPU micro-batch 16 smoke 和正式 8-GPU 启动都显示显存安全；正式训练阶段约 `10GB/GPU`。

### Optimizer / LR / scheduler

当前采用官方 Stage2 风格配置：

- optimizer: `AdamW`
- learning rate: `1e-4`
- weight decay: `1e-4`
- betas: `(0.9, 0.95)`
- scheduler: `WarmupCosLR`
- max epochs: `200`
- warmup epochs: `3`
- min lr: `1e-6`

学习率不按旧的 effective batch `64` 方案缩放。原因是当前已恢复到原版 effective batch `128`，每 epoch optimizer step 数约为原版尺度。

### DataLoader

SG-FPS Stage2 配置中显式设置：

- `num_workers: 8`
- `prefetch_factor: 2`
- `pin_memory: false`

`pin_memory=false` 是本轮 8-GPU DDP 启动时的必要修复。开启 pin memory 会在第一个 batch 取数阶段触发 `CUDA error: invalid argument`，关闭后训练正常进入实际 step。

## 已运行验证

Python 编译检查：

```bash
python -m py_compile \
  navsim/agents/recogdrive/recogdrive_diffusion_planner.py \
  navsim/agents/recogdrive/recogdrive_agent.py \
  navsim/agents/recogdrive/fs_norm.py \
  navsim/planning/training/agent_lightning_module.py \
  scripts/tools/build_fs_norm_stats.py
```

结果：通过。

Shell 语法检查：

```bash
bash -n scripts/training/run_sg_fps_asmi_stage2_103k.sh
```

结果：通过。

Pytest：

```bash
pytest -q tests/test_sg_fps_asmi.py
```

结果：

- `11 passed`

新增覆盖：

- `test_asmi_production_settings_keep_support_active_from_epoch0` 确认当前生产设置下 epoch 0 不会退化成单 GT：`beta>0`、Pareto weight 非零、采样目标数为 4。

1-step training smoke：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
  navsim/planning/script/run_training_recogdrive.py \
  +experiment=sg_fps_asmi_stage2_103k_fs_norm \
  trainer.params.devices=1 \
  dataloader.params.batch_size=16 \
  trainer.params.max_epochs=1 \
  trainer.params.limit_train_batches=1 \
  trainer.params.limit_val_batches=0 \
  trainer.params.num_sanity_val_steps=0
```

结果：通过。完成一个实际训练 step，未出现 OOM。

修正后的关键 event 标量：

- `train/loss_epoch=1.135656`
- `train/diffusion_loss_epoch=1.046333`
- `train/dpsi_sampled_target_count_mean_epoch=3.937500`
- `train/dpsi_pareto_weight_mean_epoch=0.321965`
- `train/dpsi_gt_weight_ratio_epoch=0.713312`
- `train/dpsi_external_weight_ratio_epoch=0.220397`
- `train/dpsi_beta_mean_epoch=0.296985`
- `train/dpsi_support_count_mean_epoch=10.937500`
- `train/x0_aux_loss_epoch=0.786159`
- `train/delta_aux_loss_epoch=0.139212`
- `train/geo_aux_loss_epoch=0.074923`

## 正式训练状态

废弃训练记录：

- `/mnt/project/VLA-AD_last_vla_dev/outputs/stage2_sg_fps_asmi_fsx0_micro16_eff128_random_init_20260709T152600Z`
  - 问题：实际训练样本数为 `85109`，不是全量 `103288`。
- `/mnt/project/VLA-AD_last_vla_dev/outputs/stage2_sg_fps_asmi_fsx0_full103k_micro16_eff128_random_init_20260709T154500Z`
  - 问题：虽然训练样本数为 `103288`，但 ASMI warmup 使 epoch 0 退化成单 GT/anchor 训练，`dpsi_pareto_weight_mean=0`。
- `/mnt/project/VLA-AD_last_vla_dev/outputs/stage2_sg_fps_asmi_fsx0_full103k_supportactive_micro16_eff128_random_init_20260709T155839Z`
  - 问题：训练和 DPSI support 指标正常，但 launcher 默认 step checkpoint 列表没有覆盖每 `10000` step 保存，已在 epoch 0 早期停止并重启。

上述 run 均已停止，不能作为本轮最终实验结果解读。

已启动的正式训练设置：

- GPUs: 8 x A800
- per-GPU micro batch: `16`
- gradient accumulation: `1`
- effective batch: `128`
- epochs: `200`
- train samples: `103288`
- validation inside Lightning: disabled, 外部 val6000 / navtest 单独评估
- run dir: `/mnt/project/VLA-AD_last_vla_dev/outputs/stage2_sg_fps_asmi_fsx0_full103k_supportactive_micro16_eff128_step10k_random_init_20260709T160705Z`
- launcher PID: `1383066`
- step checkpoints: `10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,110000,120000,130000,140000,150000,160000`

正式启动后已确认：

- 8 个 DDP rank 均进入训练。
- cache 读取完成，训练集样本数为 `103288`。
- 训练使用 random-init DiT，未加载 IL checkpoint。
- 第 99 step event 标量：
  - `train/loss_step=0.441634`
  - `train/diffusion_loss_step=0.401087`
  - `train/dpsi_sampled_target_count_mean_step=3.789062`
  - `train/dpsi_effective_target_count_mean_step=1.832510`
  - `train/dpsi_pareto_weight_mean_step=0.285469`
  - `train/dpsi_gt_weight_ratio_step=0.743634`
  - `train/dpsi_external_weight_ratio_step=0.197663`
  - `train/dpsi_beta_mean_step=0.273713`
  - `train/dpsi_support_count_mean_step=9.804688`
  - `train/x0_aux_loss_step=0.383896`
  - `train/delta_aux_loss_step=0.036409`
  - `train/geo_aux_loss_step=0.006733`
- micro-batch 16 当前显存约 `10GB/GPU`，无需切到 micro-batch 8。若后续出现 OOM，则切换为 micro-batch 8 + grad accumulation 2，effective batch 仍保持 `128`。

## 可复现实验命令

后续如需重启，使用新的 run id，避免覆盖旧训练：

```bash
export RECOGDRIVE_VLM_PATH=/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B
export CACHE_PATH=/mnt/project/VLA-AD/cache/recogdrive_official_stage1_hidden_navtrain_2b
export TRAIN_TEST_SPLIT=navtrain
export SUPPORT_ARCHIVE_PATH=/mnt/project/VLA-AD_last_vla_dev/outputs/sg_fps_support_semantic_full_20260705T032521Z/reselect_ref_relative_feasrelax_gtimprover_20260705T185120Z/support_v3
export FS_NORM_STATS_PATH=/mnt/project/VLA-AD_last_vla_dev/outputs/sg_fps_dpsi_clean_support_gtimprover_20260705T192117Z/fs_norm_stats.pt
export OUTPUT_DIR=/mnt/project/VLA-AD_last_vla_dev/outputs/stage2_dpsi_fs_x0_delta_geo_random_init_$(date -u +%Y%m%dT%H%M%SZ)
export MASTER_PORT=49xxx
scripts/training/run_sg_fps_asmi_stage2_103k.sh
```

训练关键设置：

- random-init DiT: `agent.checkpoint_path=''`
- stage objective: `dpsi`
- support archive: final `support_v3`
- target sampling: scene-level ASMI/DPSI weighted multi-target, 4 targets per scene
- FS-Norm: enabled
- x0 / delta / geo auxiliary: enabled
- checkpoint: every epoch and every 10000 train steps

## 剩余风险

1. AnchorBank A4 尚未实现，本轮先启动更稳的 A3。
2. 当前 DPSI 权重实现是 ASMI-style adaptive mixture，不是文档中的简单 temperature sampling；这是有意选择，用更低方差的 scene-level weighted multi-target 训练替代随机单 target 训练。
3. 旧训练曾从 IL checkpoint 加载，不能作为“随机初始化 DiT”结果解读。
4. 内部 validation 被关闭，模型选择仍应依赖外部 val6000 / navtest 评估。
5. 后续可做 6-target ablation；本轮先固定 4-target，避免训练时间和显存开销进一步放大。
