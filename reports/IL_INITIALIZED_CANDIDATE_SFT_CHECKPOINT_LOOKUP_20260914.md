# 官方 IL 初始化的候选轨迹 SFT：历史权重核查

核查日期：2026-09-14。此次只读检查原始训练配置、日志、数据档案和权重；没有训练、重新评测或修改历史文件。

## 结论与前次说明的范围

**本地确实做过“官方 IL 权重初始化 → 候选轨迹 SFT”，并且有保存下来的权重。** 前次五 checkpoint 分析中，86.92（V6）和87.51（A5）这两个指定 run 的配置为随机初始化；这个结论不能推广成“历史上没有 IL 初始化的候选 SFT”。

## 1. 2026-06-24 PSI-Drive / Pareto-support 正式训练

运行目录：`outputs/psi_drive_stage2_apsd_clean_8gpu_20260624T214438Z/`。

证据链：

- `resolved_command.sh` 与 `train_args.json` 指定官方 `ReCogDrive_Diffusion_Planner_2B_IL.ckpt`，`grpo=False`，`stage2_target_source=pareto_support`，学习率 `5e-5`，60 epochs。
- `logs/train.log:53` 确认实际加载官方 IL：347 keys loaded，`strict=False`；同时报告208个 missing non-expert keys，包括新增 CoT 模块。因此能确认 IL 初始化，但不能未经架构审计就称其为“与原版结构完全相同、只替换监督目标”的严格因果对照。
- `data_report.json`：训练85,109条记录；验证18,179条记录使用GT。训练使用的 Pareto support 索引覆盖103,288个 token。
- 数据文件 `outputs/psi_drive/support_index/stage2_pareto_support_clean_full.pt` 实存。`support_trajectories` 为 `[103288, 3, 8, 3]`；每场景1–3个有效 targets，support count 分布为1条65,017场景、2条31,232场景、3条7,039场景。该计数包括验证GT fallback，不能把103,288全部说成训练场景。
- Support 原料包括GT、IL/policy轨迹和多类结构化扰动；不是只有GT的普通监督档案。数据内记录的构建方法为 `psi_drive_stage2_pareto_support`。

### 保存的权重实际在哪里

历史 `checkpoint_store/inventory.tsv` 有61行索引，但索引里的 `.ckpt` 路径当前不存在。目录内另有30个隐藏 `.tmp` 对象。**这30个对象全部通过了完整SHA256校验，与对应历史checkpoint hash完全一致，并能用 `torch.load` 读取 `state_dict`。** 它们不是仅凭大小猜测的残缺临时文件。

没有重命名、覆盖或删除这些原始对象。全部真实路径、hash、epoch、step和历史PDMS已保存在：

`outputs/five_checkpoint_training_distribution/il_initialized_sft_lookup/historical_psi_checkpoints.csv`

其中可直接定位的三个例子如下；PDMS为历史Navtest记录，乘100转为point，未在此次重新评测。

| checkpoint | global step | 历史 Navtest PDMS | SHA256 |
|---|---:|---:|---|
| epoch_010 | 6650 | 87.480952 | a648022fb8fc3126dec5b37e2f01d7ee058d1c258e1d73642c98d3a5f7e28d7a |
| epoch_011 | 7315 | 87.672438 | 051ac3312a848b36eeeaccc3935e03582f9acc89737a99672349a5887faa0aff |
| epoch_059 | 39235 | 84.434191 | d78df93ee82ddeeffe3a9054772b79e119fb63f4e2bd98a298095d14d29406e9 |

epoch_011实际文件：

`outputs/psi_drive_stage2_apsd_clean_8gpu_20260624T214438Z/checkpoint_store/objects/.051ac3312a848b36eeeaccc3935e03582f9acc89737a99672349a5887faa0aff.3941396.tmp`

epoch_011是上述历史索引中的最高Navtest分数记录；它可以作为定位示例，但若按此选择后续比较权重，应公开“历史Navtest选checkpoint”的选择依据，不能称为未触碰测试集的选择。

## 2. V3 controlled micro-SFT 权重仍在

目录：`outputs/pc_mts_diagnostics_v3/checkpoints/sft/`。

五种候选监督：`score`、`pareto`、`gt_distance`、`old_pc`、`conditional_pc`。每种有1701/2903两个seed，保存step0/20/100/200。另有GT-only对照。

- 五种候选方法共40个snapshot，其中30个是更新后的snapshot、10个是step200最终权重；包括GT-only后共48个snapshot。
- 从相同官方IL初始化，700 train / 300 diagnostic holdout，200 optimizer updates。这是小预算机制实验，不能冒充历史大规模SFT性能实验。
- 官方IL初始化SHA256：`4569221da6962d4e26d21a463cd70c06a7ad4d76654943fb3870086f43bdb443`。
- `manifests/INITIALIZATION_TENSOR_AUDIT.json` 和各 `train_sft_<method>_<seed>.json` 记录初始化及训练来源。
- 此次检查所有48个snapshot实体及对应metadata，并额外完整校验 `score_seed1701/step0200.pt` 和 `conditional_pc_seed1701/step0200.pt` 的SHA256。
- `.pt`保存action-head `state_dict`、optimizer及metadata，需要配合原官方模型及V3加载代码使用；不是整套VLM权重打包。

可用例子：

`outputs/pc_mts_diagnostics_v3/checkpoints/sft/score_seed1701/step0200.pt`

`outputs/pc_mts_diagnostics_v3/checkpoints/sft/conditional_pc_seed1701/step0200.pt`

## 3. 7月的其他DPSI尝试

在 `VLA-AD_last_vla_dev/outputs/sg_fps_*` 中另外找到7月3–5日官方IL初始化、`offline_rl_use_dpsi=true`、候选support路径明确的配置与历史评测记录。例如 `sg_fps_dpsi_clean_support_gtimprover_stepckpt_20260705T193744Z` 使用了后来A5也使用的GT-improver support档案。

本次在已搜索的这些run目录中尚未核验到保留的权重实体，不能把配置文件或历史checkpoint路径当成当前可用checkpoint。这不影响上面6月PSI与V3权重已确认存在的结论。

## 后续比较的意义

要回答“候选SFT相对官方IL究竟改变了什么”，6月PSI权重是值得补入的历史IL初始化候选SFT链，V3则提供更受控但预算很小的对照。86.92/87.51仍应保留为既定模型的描述性分布比较，不应把它们与IL的差异全部归因为在IL上的SFT微调。

机器可读核查记录：`outputs/five_checkpoint_training_distribution/il_initialized_sft_lookup/checkpoint_inventory.json`。此次未复跑推理；文件可读和hash一致不等于已完成对应运行环境的推理兼容性测试。
