# Cache Inventory

最后更新：2026-06-25 17:33 UTC

这份文档是当前项目 cache 的长期维护入口。以后新增、删除、迁移、重生成、复用策略变化，或者某个进行中 cache 完成时，都需要同步更新这里。

## 维护规则

1. 新增 cache 时补充：路径、split、样本数、生成脚本/日志、模型权重、是否包含外部特征、主要用途。
2. cache 完成时补充：最终样本数、大小、manifest/audit 结果、是否可直接用于训练或评估。
3. cache 被替换或废弃时不要只删除条目，要标明废弃原因和替代路径。
4. 对 ReCogDrive stage2 官方原版训练，只有“官方 stage1 VLM hidden cache”可作为默认输入。带 JEPA/VGGT/two-expert/LoRA 的 cache 需要明确标注，不能混用。
5. `torchinductor*`、Python venv/conda 环境目录不算数据 cache，除非调试编译缓存问题，否则不进入正式清单。

## Cache 类型

| 类型 | 典型格式 | 主要用途 | 注意事项 |
|---|---|---|---|
| 官方 stage1 VLM hidden gzip cache | `log_name/token/internvl_feature.gz` + `trajectory_target.gz` | 官方原版 stage2 训练 | 应只包含 `history_trajectory`、`high_command_one_hot`、`last_hidden_state`、`status_feature`，不注入外部知识 |
| Indexed/chunk hidden cache | `index.jsonl` + `samples/*.pt` | 快速读取 hidden 或外部专家特征 | 可能同时包含 JEPA/VGGT/two-expert 字段，不能默认视为官方纯 hidden |
| PDM metric cache | log/token 目录或 fast pickle | navtest/navtrain PDM 评估和 stage3 reward | 不是 VLM hidden cache，不用于官方 stage2 hidden 训练 |
| Offline RL / elite buffer | `*.pkl.xz` 或 target index `.pt` | stage3 AWAC/GRPO、自举 better-than-GT target | 需要记录选择规则，避免和 GT target 混淆 |
| Pareto support index | compact `.pt` + summary/audit | PSI-Drive Stage2 APSD 和 Stage3 SR-PGRPO | 只含 navtrain support，不是 VLM hidden cache，也不应包含 navtest |

## 当前正式可复用 Cache

### 0. Fixed Val6000 split artifact

- token 文件：`artifacts/splits/navtrain_val6000_seed260306049.txt`
- 规范 checksum：`artifacts/splits/navtrain_val6000_seed260306049.sha256`
- 兼容旧 checksum：`artifacts/splits/navtrain_val6000_seed260306049.txt.sha256`
- token 数：`6000`
- SHA256：`1b6355bfd1f1fbf9438897d34c64320d3f46d07c62437fe8da7df5e42c5cbc54`
- audit：
  - JSON：`artifacts/splits/navtrain_val6000_seed260306049_audit.json`
  - Markdown：`artifacts/splits/navtrain_val6000_seed260306049_audit.md`
- train complement：`artifacts/splits/navtrain_val6000_seed260306049_train_complement.txt`
- audit 摘要：
  - source navtrain tokens：`103288`
  - train complement tokens：`97288`
  - train/val overlap against complement：`0`
  - navtest overlap：`0`
- 用途：Stage2/Stage3 固定 val6000 exact PDM 评估；评估脚本通过 `+eval_token_file` 使用此文件，不使用 `--max-samples 6000` 替代。
- 注意：当前用户指定的 Stage2 random-init full103k run 使用完整 `103288` train cache，因此训练数据包含该 val6000 文件中的 token；这是本轮 full-cache 训练设置，不改变 split artifact 自身的 heldout complement audit。

### 1. Navtrain 官方 stage1 VLM hidden cache

- 路径：`/mnt/project/VLA-AD/cache/recogdrive_official_stage1_hidden_navtrain_2b`
- 状态：已完成
- 完成时间：2026-06-24 20:17 UTC
- 记录数：`103288`
- 目标 split：`navtrain`
- 目标数据规模：约 `103k` trainval 场景 token
- 权重：`/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B`
- 生成脚本：`scripts/cache_dataset/run_caching_recogdrive_official_stage1_hidden_state.sh`
- supervisor PID：`3402752`
- 生成日志：`/mnt/project/VLA-AD/outputs/recogdrive_official_stage1_hidden_navtrain_103k_20260624T175041Z.log`
- supervisor 日志：`/mnt/project/VLA-AD/outputs/recogdrive_official_stage1_hidden_navtrain_navtest_103k_20260624T175041Z.supervisor.log`
- 关键配置：
  - `agent.cache_hidden_state=True`
  - `agent.cache_mode=True`
  - `agent.train_backbone=false`
  - `agent.use_expert_features=false`
  - `agent.use_jepa=false`
  - `agent.use_vggt=false`
  - `agent.use_last_rd=false`
  - `agent.use_last_vla=false`
  - `agent.use_two_expert_slots=false`
- 用途：官方原版 ReCogDrive stage2 训练输入。
- 完成校验：
  - 已写入 `.recogdrive_official_stage1_hidden_cache.json`
  - `audit_recogdrive_official_hidden_cache.py --require-official` 通过
  - `official_source_proven=true`
  - `schema_ok=true`
  - sample hidden shape：`last_hidden_state=(2800, 1536)`
  - feature keys：`history_trajectory`、`high_command_one_hot`、`last_hidden_state`、`status_feature`

### 2. Navtrain 官方 hidden + elite target overlay

- 目标路径：`/mnt/project/VLA-AD/cache/recogdrive_official_stage1_hidden_navtrain_2b_elite_targets`
- 状态：已完成
- 完成时间：2026-06-24 20:32 UTC
- 生成脚本：`scripts/cache_dataset/supplement_recogdrive_hidden_cache_with_elite_targets.py`
- elite target index：
  `/mnt/project/VLA-AD/outputs/two_expert_slot_stage2_prefuse_v2_elite_full103k_random_8gpu_minlr5e6_20260616T222308Z/artifacts/stage2_elite_best_valid_above_gt_targets.pt`
- 总记录数：`103288`
- elite target 数：`49977`
- GT fallback target 数：`53311`
- elite 覆盖率：`48.39%`
- 选择规模：`49977` 条 better-than-GT target
- 用途：官方 hidden 不变，只将 target 从 GT 替换为“better-than-GT 或 GT fallback”，用于跑官方 stage2 训练时复现实验中的增强 target 设置。
- 注意：这不是额外知识注入到 VLM hidden，只是 target overlay。
- 完成校验：
  - 已写入 `.recogdrive_official_stage1_hidden_cache.json`
  - `audit_recogdrive_official_hidden_cache.py --require-official` 通过
  - `official_source_proven=true`
  - `schema_ok=true`
  - hidden feature 通过 symlink 复用 base official cache

### 3. PSI-Drive Stage2 Pareto support index, 原始 clean 版

- 路径：`/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_pareto_support_clean_full.pt`
- 状态：已完成，独立 audit 通过；已被 2026-06-25 GT-only 补充版替代为 Stage2 APSD 默认 support
- 完成时间：2026-06-24 21:16 UTC
- 生成脚本：`scripts/build_stage2_pareto_support_index.py`
- audit 脚本：`scripts/audit_stage2_pareto_support_index.py`
- summary：`/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_pareto_support_clean_full_summary.json`
- 生成报告：`/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_pareto_support_clean_full_audit.md`
- 独立 audit：
  - JSON：`/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_pareto_support_clean_full_independent_audit.json`
  - Markdown：`/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_pareto_support_clean_full_independent_audit.md`
- SHA256：`b3009a61d93a0760288685660c59d950788fe98f1dab62a8de59a68ba3570498`
- source mode：`clean`
- 记录数：`103288`
- unique tokens：`103288`
- duplicate tokens：`0`
- support 数量范围：`1` 到 `3`
- support 数量分布：
  - `1`: `65017`
  - `2`: `31232`
  - `3`: `7039`
- 选择参数：
  - quality band：`0.02`
  - minimum normalized descriptor distance：`0.75`
  - target weights：best `0.5`，GT `0.2`，other `0.3`
- descriptor pass 统计：
  - buffer records：`85109`
  - descriptor count：`812663`
  - GT descriptor fallback：`18179`
- fallback 统计：
  - `gt_fallback`: `69`
  - `gt_fallback_error:FileNotFoundError`: `18179`
- source histogram：
  - `gt`: `35056`
  - `gt_fallback`: `18179`
  - `il`: `25448`
  - `policy`: `31648`
  - `progress_endpoint`: `22063`
  - `endpoint_lateral`: `6045`
  - `progress_speed`: `5332`
  - `progress_gamma`: `1762`
  - `timing_delay`: `1120`
  - `timing_slow_first`: `1143`
  - `lateral_offset`: `802`
- 用途：
  - Stage2 APSD 训练读取 support target；validation/inference 仍使用原始 GT 目标，不读 support。
  - Stage3 SR-PGRPO 训练用于 support-relative advantage 分桶；inference 不读 support。
- 完成校验：
  - `passed=true`
  - `finite_trajectories=true`
  - `finite_scores=true`
  - `finite_weights=true`
  - `weights_sum_min/max=1.0/1.0`
  - `overlap_navtest_tokens=None`
- 注意：
  - 该原始版的 `85109` 条 elite buffer 可用，但有 `18179` 个 navtrain token 缺候选 buffer，只能落到 `gt_fallback_error:FileNotFoundError`。
  - 后续 Stage2 APSD full103k 训练默认改用下一条 GT-only 补充版 index。

### 4. PSI-Drive Stage2 GT-only structured supplement + Pareto support index

- GT-only 缺候选 token 列表：
  `/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_support_gt_only_missing_candidate_tokens_18179.tsv`
- 补充 buffer 根目录：
  `/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_gt_only_structured_elite_buffer_20260625T161025Z`
- 补充 buffer 记录目录：
  `/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_gt_only_structured_elite_buffer_20260625T161025Z/buffer`
- 补充记录数：`18179`
- 状态：已完成，错误数 `0`
- 生成脚本：`scripts/generate_stage2_gt_only_structured_elite_buffer.py`
- 生成设置：
  - 只处理原始 clean support index 中缺候选的 `18179` 个 GT-only token。
  - 读取官方 stage1 VLM hidden cache 的 GT target：`/mnt/project/VLA-AD/cache/recogdrive_official_stage1_hidden_navtrain_2b`
  - 使用 navtrain PDM metric cache 做 exact PDM 评分：`/mnt/project/VLA-AD/cache/metric_cache_train_full`
  - 候选来源为 GT anchoring 的 structured perturbations：`progress_endpoint`、`progress_speed`、`progress_gamma`、`lateral_offset`、`endpoint_lateral`、`timing_delay`、`timing_slow_first`
  - 不使用 JEPA/VGGT，不使用外部知识注入，不为这 `18179` 条补官方 Stage2/Stage3 policy proposal。
- 补充 buffer 生成摘要：
  - shard 数：`8`
  - `skip_done` 保留重启前已写记录：`3516`
  - 重启后写入记录：`14663`
  - 重启后 `valid_better_than_gt`: `11024`
  - 重启后 `best_valid_delta_mean`: `0.01640556747702145`
  - 重启后 best valid source 直方图：`gt=3639`，`progress_endpoint=7204`，`progress_speed=3656`，`timing_delay=158`，`timing_slow_first=4`，`lateral_offset=2`
- union buffer：
  `/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_elite_buffer_union_clean_gt_supplemented_20260625T162351Z`
- union manifest：
  `/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_elite_buffer_union_clean_gt_supplemented_20260625T162351Z/union_manifest.json`
- union 组成：
  - 原始 AWAC/elite buffer：`85109`
  - GT-only structured supplement：`18179`
  - union records：`103288`
  - precedence：后一个 source root 覆盖前一个 source root
- 新 support index：
  `/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_pareto_support_clean_full_gt_supplemented_20260625T162351Z.pt`
- summary：
  `/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_pareto_support_clean_full_gt_supplemented_20260625T162351Z_summary.json`
- audit：
  `/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_pareto_support_clean_full_gt_supplemented_20260625T162351Z_audit.json`
- SHA256：`817ada27e18a92062e2fb4bd5fddacaa3f75cbd30a0f8d903119bc04ab4bbf27`
- source mode：`clean`
- 记录数：`103288`
- unique tokens：`103288`
- duplicate tokens：`0`
- support 数量范围：`1` 到 `3`
- support 数量分布：
  - `1`: `58287`
  - `2`: `34054`
  - `3`: `10947`
- 选择参数：
  - quality band：`0.02`
  - minimum normalized descriptor distance：`0.75`
  - target weights：best `0.5`，GT `0.2`，other `0.3`
- descriptor pass 统计：
  - buffer records：`103288`
  - descriptor count：`1394391`
- fallback 统计：
  - `gt_fallback`: `69`
- source histogram：
  - `gt`: `40407`
  - `il`: `26639`
  - `policy`: `32015`
  - `progress_endpoint`: `32694`
  - `progress_speed`: `10268`
  - `endpoint_lateral`: `7404`
  - `progress_gamma`: `5109`
  - `timing_delay`: `1909`
  - `timing_slow_first`: `1643`
  - `lateral_offset`: `1148`
- 用途：
  - 当前 Stage2 APSD full103k 随机初始化训练的默认 support target。
  - validation/inference 不读 support，仍使用 GT target。
- 完成校验：
  - `passed=true`
  - `finite_trajectories=true`
  - `finite_scores=true`
  - `finite_weights=true`
  - `weights_sum_min/max=1.0/1.0`
  - `failures=[]`
- 当前使用记录：
  - `/mnt/project/VLA-AD/outputs/psi_drive_stage2_apsd_random_init_full103k_gt_supp_20260625T162149Z`
  - 随机初始化：`agent.checkpoint_path=`，`agent.allow_random_init=true`
  - full cache train：`cache_train_all_records=true`
  - train samples：`103288`
  - val samples：`18179`

### 5. Navtest PDM metric cache

- 路径：`/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1`
- 大小：约 `3.1G`
- metadata 行数：`12139` 行
- 配套 fast pickle：`/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1_fast_pickle`
- fast pickle 大小：约 `8.4G`
- split：`navtest`
- 原始数据：
  - `/mnt/navsim/test_navsim_logs/test`
  - `/mnt/navsim/test_sensor_blobs/test`
- 用途：stage3/navtest PDM、best-of-N、sensitivity 评估。
- 已验证使用记录：
  - `stage3_best_of6_step21600_navtest_full_20260624T0345Z`
  - `recogdrive_stage3_train_subset_sota_validation_20260624/navtest_sensitivity_bon8_v3`
- 关键区别：stage3/navtest 评估使用 `agent.cache_hidden_state=False`，依赖 raw navtest 数据和 PDM metric cache，不依赖 VLM hidden gzip cache。

### 6. Navtrain PDM metric cache

- 路径：`/mnt/project/VLA-AD/cache/metric_cache_train_full`
- 大小：约 `35G`
- metadata 行数：`103289` 行
- 用途：stage3 训练 reward、train-side PDM 分析、policy diversity 或 failure mode 分析。
- 注意：这是 PDM metric cache，不是 VLM hidden cache。

## 已存在但不可直接当官方 stage2 输入的 Cache

### 7. Navtest ReCogDrive expert chunk cache, VLM hidden + JEPA + VGGT

- 路径：`/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1`
- 大小：约 `197G`
- split：`navtest`
- chunk 文件：
  - `navtest_full_chunk_000000`
  - `navtest_full_chunk_000001`
  - `navtest_full_chunk_000002`
  - `navtest_full_chunk_000003`
- index 总记录数：`12146`
- metadata 关键信息：
  - `recogdrive_vlm_path=/mnt/project/VLA-AD/checkpoints/recogdrive/ReCogDrive-VLM-2B`
  - `contains_vlm_hidden=true`
  - `contains_jepa=true`
  - `contains_vggt=true`
  - `jepa_model_path=/mnt/project/VLA-AD/checkpoints/teachers/vjepa2-vitl-fpc64-256`
  - `vggt_model_path=/mnt/project/VLA-AD/checkpoints/teachers/VGGT-1B`
- 样本格式：`index.jsonl` 指向 `samples/*.pt`
- payload 字段示例：
  - `last_hidden_state`: `(2800, 1536)`
  - `history_trajectory`: `(4, 3)`
  - `high_command_one_hot`: `(4,)`
  - `status_feature`: `(8,)`
  - `trajectory`: `(8, 3)`
  - `jepa_context_tokens`, `jepa_target_tokens`
  - `vggt_context_tokens`, `vggt_target_tokens`
- 结论：
  - VLM hidden 来源是官方 ReCogDrive-VLM-2B。
  - 但该 cache 同时含 JEPA/VGGT，因此不是“纯官方 stage1 hidden gzip cache”。
  - 它是 navtest，不是 navtrain，不能替代正在生成的 navtrain 103k official hidden cache。

### 8. Navtest official hidden gzip skip marker

- 路径：`/mnt/project/VLA-AD/cache/recogdrive_official_stage1_hidden_navtest_2b.skip_generation`
- 状态：已创建
- 作用：让 `run_caching_recogdrive_official_stage1_hidden_state.sh` 在 navtrain 完成后跳过 navtest official hidden gzip 生成。
- 原因：
  - 当前 stage3/navtest 评估不使用 VLM hidden gzip cache。
  - 已有 navtest metric cache 可支撑 PDM 评估。
  - 已有 navtest expert chunk 可提供官方 VLM hidden，但它不是纯官方 gzip 格式。
- 如果未来确实需要纯官方 navtest hidden gzip cache，先删除该 marker，再运行 navtest 生成命令。

### 9. Two-expert slot historical caches, 2026-06-12

- 根路径：`/mnt/project/VLA-AD/cache/two_expert_slot/real_20260612_172446`
- 子 cache：
  - `hidden_navtrain_stage1_lora_bf16`
  - `jepa_dynamic`
  - `vggt_feature23`
- 记录数：
  - hidden navtrain：`103036`
  - JEPA dynamic：`103036`
  - VGGT feature23：`103036`
- 关键信息：
  - hidden dtype：`bf16`
  - hidden cache version：`two_expert_slot_hidden_cache_v1`
  - JEPA teacher：`vjepa2-vitl-fpc64-256`
  - VGGT teacher：feature dim `1024`
- 用途：two-expert slot 相关实验和旧 stage1/teacher 特征分析。
- 注意：这是 LoRA/two-expert 体系 cache，不是官方原版 stage2 hidden 输入。

### 10. Two-expert slot stage1_v2 caches, 2026-06-15/17

- 根路径：`/mnt/project/VLA-AD/cache/two_expert_slot/stage1_v2_20260615_174154`
- 子 cache：
  - `hidden_navtrain_stage1_v2_clean_bf16_20260616T164153Z`
  - `hidden_navtest_stage1_v2_clean_bf16_20260617T020404Z`
- 记录数：
  - navtrain：`103036` 个 `.pt` sample
  - navtest：`12146` 个 `.pt` sample
- metadata：
  - `merged=true`
  - `output_dtypes=["bf16"]`
  - `teacher_targets_included=false`
  - `version=two_expert_slot_hidden_cache_v1`
- 用途：
  - two-expert stage2 prefuse/cross-attention 训练和评估。
  - 旧 SOTA stage2 设置曾使用 navtrain cache：
    `/mnt/project/VLA-AD/cache/two_expert_slot/stage1_v2_20260615_174154/hidden_navtrain_stage1_v2_clean_bf16_20260616T164153Z`
- 注意：
  - 之前 audit 发现该 cache 不是官方纯 stage1 hidden：包含 two-expert metadata、非官方 stage1 训练痕迹或外部槽位。
  - 不应用于“官方原版 stage2”训练。

### 11. Stage3 AWAC elite buffer

- 路径：`/mnt/project/VLA-AD/cache/recogdrive_stage3_awac_elite_buffer_train_v2_stage3_awac_iql_dualhost_20260612T182947Z`
- 文件数：`85109` 个 `*.pkl.xz`
- 用途：stage3 AWAC / elite buffer 训练和后续构造 better-than-GT target。
- 下游产物：
  `/mnt/project/VLA-AD/outputs/two_expert_slot_stage2_prefuse_v2_elite_full103k_random_8gpu_minlr5e6_20260616T222308Z/artifacts/stage2_elite_best_valid_above_gt_targets.pt`
- 下游选择规模：`49977` 条 better-than-GT target。
- 注意：该 buffer 是 RL/selection 数据，不是 hidden cache。

## Smoke / 验证用 Cache

### 12. Official stage1 hidden smoke cache

- 路径：`/mnt/project/VLA-AD/cache/smoke_recogdrive_official_stage1_hidden_navtrain_2b`
- 记录数：`1`
- 状态：完成，audit 通过
- manifest：
  `.recogdrive_official_stage1_hidden_cache.json`
- 用途：验证官方权重、cache schema、stage2 fast-dev-run。
- 不用于正式训练。

### 13. Official stage1 hidden smoke elite overlay

- 路径：`/mnt/project/VLA-AD/cache/smoke_recogdrive_official_stage1_hidden_navtrain_2b_elite_targets`
- 记录数：`1`
- 状态：完成，audit 通过
- target supplement：`elite_best_valid_above_gt_or_gt`
- 用途：验证 elite target overlay 逻辑。
- 不用于正式训练。

## 临时或非数据 Cache

以下目录存在于 `/mnt/project/VLA-AD/cache`，但默认不作为训练/评估数据 cache 管理：

- `torchinductor*`：PyTorch/TorchInductor 编译缓存。
- `vggt_py310_conda`、`vggt_py310_venv`：环境目录。
- `recogdrive_official_stage1_hidden_cache` 和 `recogdrive_official_stage1_hidden_cache.log`：早期默认路径/日志，当前正式 navtrain official hidden 生成使用 `recogdrive_official_stage1_hidden_navtrain_2b`。
- `last_vla_v2`：当前仅看到小规模 manifests/teacher_traj 入口，若后续恢复 LaST-VLA cache 生成，需要单独补完整条目。

## 当前可执行判断

| 任务 | 应使用的 cache | 不应使用的 cache |
|---|---|---|
| 官方原版 stage2 训练 | `recogdrive_official_stage1_hidden_navtrain_2b`，完成后可用 elite overlay | two-expert/stage1_v2、JEPA/VGGT chunk、PDM metric cache |
| stage3/navtest PDM 或 best-of-N 评估 | `metric_cache_navtest_full_v1` + `metric_cache_navtest_full_v1_fast_pickle` | 纯 hidden gzip 不是必要输入 |
| PSI-Drive Stage2 APSD 训练 | `recogdrive_official_stage1_hidden_navtrain_2b` + `stage2_pareto_support_clean_full_gt_supplemented_20260625T162351Z.pt` | navtest、PDM metric cache、JEPA/VGGT chunk、原始缺候选版 `stage2_pareto_support_clean_full.pt` |
| PSI-Drive Stage3 SR-PGRPO 训练 | Stage2 初始化 checkpoint + `stage2_pareto_support_clean_full_gt_supplemented_20260625T162351Z.pt` + navtrain PDM metric cache | navtest support 或由 navtest 结果筛出的 support |
| 复查 navtest VLM hidden 来源 | `recogdrive_expert_chunks/full_v1` 可证明 VLM hidden 来自官方权重 | 不能据此替代 navtrain 103k official hidden |
| 训练集 PDM 分析或 reward 统计 | `metric_cache_train_full` | hidden cache |
| better-than-GT target 复用 | `stage2_elite_best_valid_above_gt_targets.pt` | 直接把 AWAC pkl.xz 当 stage2 target |

## 下一次更新 Checklist

- [x] `recogdrive_official_stage1_hidden_navtrain_2b` 完成后更新最终记录数、manifest 内容、audit 摘要。
- [x] `recogdrive_official_stage1_hidden_navtrain_2b_elite_targets` 生成后更新记录数、symlink/target 文件数、audit 摘要。
- [x] `stage2_pareto_support_clean_full.pt` 生成后更新 support 数量、hash、audit 摘要。
- [x] `stage2_pareto_support_clean_full_gt_supplemented_20260625T162351Z.pt` 生成后更新 GT-only 补充、union、support 数量、hash、audit 摘要。
- [ ] 如果删除 `recogdrive_official_stage1_hidden_navtest_2b.skip_generation` 并生成 navtest official hidden，新增正式条目。
- [ ] 如果 stage2 官方训练改用新路径，更新“当前可执行判断”。
- [ ] 如果 metric cache 重建，更新 metadata 行数和相关评估 run。
