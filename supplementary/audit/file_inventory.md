# Repository and Paper Evidence Inventory

审计日期：2026-08-01（UTC）  
工作树：`feature/curious-vla-audit`，`ba5908d`  
范围：论文、AMPT 三阶段相关实现、NAVSIM v1/v2 评价、配置/启动器、checkpoint 元数据、日志和结果资产。原始结果仅作只读检查。

## 1. 审计方法与证据等级

使用了 `git grep`、`rg`、`find`、`git log --all`、`git ls-tree`、`git show <commit>:<path>` 和 `git branch --contains`。由于最终论文代码分布在未合并历史分支中，本清单区分：

- **W**：当前工作树可直接读取；
- **H**：仅可从 Git 历史对象读取，必须带 commit/ref；
- **R**：原始或逐场景结果；
- **P**：仅论文 PDF 报告，尚无对应原始实验资产；
- **D**：派生/诊断结果，不可替代原始运行。

当前仓库存在大量未跟踪结果与论文资产；本次未改写任何 `outputs/`、`checkpoints/`、`docs/` 或源码文件。

## 2. 论文资产

| 资产 | 状态 | 证据 |
|---|---|---|
| 最新正文 PDF | W | `docs/AuthorKit27 (7).pdf`，9 页，2,184,673 bytes，SHA256 `25edcd6399b6520ce863292941a5322da2bf22ce4e8ba8ca61e952b59e414524` |
| 主论文 LaTeX | 缺失 | 当前工作树和全部已检查 Git refs 均未定位到与该 PDF 对应的 `.tex` |
| BibTeX | 缺失 | 未定位到与该 PDF 对应的 `.bib` |
| 主论文编译记录 | 缺失 | 未定位到 latexmk/TeX 构建日志或源包 manifest |

PDF 可证实正文中的方法名、算法叙述与全部主表数字；但 PDF 本身不是训练配置、逐场景统计或多 seed 证据。

## 3. 方法实现的 Git 谱系

### 3.1 PC-MTS 与 FF-PGRPO 的可追溯实现

最完整的实现位于历史提交 `41a86139794c684ff7ddc45e6647275186ac3974`（2026-07-25，`feat: implement FIRST-Drive V7 training and data contracts`），可通过 ref `codex/pareto-full-action-head-contract` 读取。该提交未合入当前 HEAD。

关键目录：

- `navsim/agents/recogdrive/curriculum/`：候选加载、去重、可达性、局部扰动稳健性、安全门、真实 coherent reference、quota/role 选择；
- `navsim/agents/recogdrive/stage3_ff_pareto.py`：feasibility-first Pareto credit assignment；
- `navsim/agents/recogdrive/stage3_positive_credit_contract.py`：正信用安全契约及诊断；
- `navsim/agents/recogdrive/stage3_reference_selector.py`：完整真实行 reference 选择；
- `navsim/agents/recogdrive/stage3_runtime.py`、`stage3_metric_adapter.py`、`stage3_reference_cache.py`、`stage3_policy_geometry.py`：运行时桥接；
- `navsim/agents/recogdrive/first_drive_v7_agent.py`、`first_drive_v7_planner.py`、`v7_curriculum_runtime.py`：训练/采样接入。

包含该实现的主要 refs：`feature/first-drive-v7`、`codex/pareto-full-action-head-contract`、`codex/safe-core-stage3-contract` 及多个 V7 contract 分支。附录引用实现时必须固定 commit，不可写成当前 HEAD 路径已经存在。

### 3.2 APR 相关实现

APR 所需机制分散在另一条历史谱系，主要通过 ref `codex/a5-epdms-stage3-20260726`（tip `ab9124c08100d3f14e6ca45da19e40e3d5a5c3fb`）读取，并未发现单一类或脚本直接命名为 `APR`：

- teacher 构建与相对审计：`scripts/stage3/build_pdms_*teacher*.py`；
- teacher/retention 日程：`scripts/stage3/build_pdms_teacher_refinement_schedule.py`；
- 轨迹插值与精确重新评分：`build_pdms_blended_teacher_reference.py`、`filter_pdms_teacher_reference_by_safety_tube.py`；
- checkpoint 插值：`interpolate_lightning_checkpoints.py`；
- task-vector merge：`merge_ties_base_relative_checkpoints.py`、`merge_conflict_free_anchored_task_vector.py`；
- 训练/监控：`scripts/stage3/launch_pdms91_*`、`watch_*`、`wait_*`。

相关历史提交：

| Commit | 日期 | 作用 |
|---|---|---|
| `58cab66e18499a1726f836f0fcba08b7ef27229c` | 2026-07-26 | 安全 checkpoint 插值 |
| `fb96a037839428217b603637bcc55116dd322fb2` | 2026-07-26 | sparse teacher refinement schedule |
| `d24b7c7388f20dd7bc0a8fac334c346e75c5a1ba` | 2026-07-27 | TIES merge |
| `64f553174dc9e1531f67625cdbdfa52a9b5d379f` | 2026-07-27 | anchored orthogonal residual graft |

这些脚本构成 APR 的实现证据，但未发现“3 轮动态重建、teacher activated/expired 状态、每轮统一 manifest”这一完整 orchestrator，因此只能逐机制引用，不能声称存在一个一键 APR 训练程序。

## 4. 训练配置与 launch

### 4.1 PC-MTS/FF-PGRPO 谱系（H，commit `41a8613`）

- `navsim/planning/script/config/experiment/first_drive_v7_phase_a.yaml`
- `navsim/planning/script/config/experiment/first_drive_v7_phase_b.yaml`
- `navsim/planning/script/config/experiment/first_drive_v7_phase_c.yaml`
- `navsim/planning/script/config/experiment/first_drive_v7_native_public_phase_b.yaml`
- `navsim/planning/script/config/common/agent/first_drive_v7_agent.yaml`
- `configs/first_drive_v7/native_public_il_g64.json`
- `configs/first_drive_v7/recogdrive_2b_public_il_n1_archive.json`
- `configs/first_drive_v7/recogdrive_2b_pareto_grpo_v2_step21600_n1_archive.json`
- `scripts/curriculum/build_v7_curriculum.py`
- `scripts/curriculum/audit_v7_curriculum.py`
- `scripts/curriculum/audit_v7_policy_gate.py`
- `scripts/curriculum/fit_stage3_geometry_calibration.py`
- `scripts/curriculum/seal_v7_source_bundle_v2.py`
- `scripts/training/materialize_first_drive_v7_stage2.py`
- `scripts/training/launch_first_drive_v7_stage2.py`
- `scripts/training/run_first_drive_v7_phase_a.sh`
- `scripts/training/run_recogdrive_stage3_core_pareto_2b_local.sh`

注意：Phase B/C YAML 仍含 `???`，不能直接作为最终复现配置；Phase A 同时声明 scheduler 200 epochs 与 trainer `max_epochs: 100`，必须在缺失证据中保留冲突。历史 `A/B/C` 是 V7 curriculum 的内部阶段，并非论文 AMPT Stage 1/2/3。

### 4.2 APR 谱系（H，ref `codex/a5-epdms-stage3-20260726`）

- `scripts/stage3/launch_pdms91_external_teacher_retention_3ep.sh`
- `scripts/stage3/launch_pdms91_rap_dual_branches_vla.sh`
- `scripts/stage3/launch_pdms91_safety_awac_resume.sh`
- `scripts/stage3/launch_pdms91_lfp_vla_zt.sh`
- `scripts/stage3/watch_ddv2_structured_then_build_pdms_teachers.sh`
- `scripts/stage3/watch_stage3_interpolations_and_eval_fold0.sh`
- `scripts/training/run_a5_epdms_lfp_grpo_v2.sh`
- `scripts/training/run_pdms91_lfp_grpo_v1.sh`

这些是实验族 launchers，不是论文版 APR 的单一可复现入口。`RAP` 在该谱系中是外部模型/候选源名，不能改名成 APR。

## 5. NAVSIM 评价实现

### 5.1 NAVSIM v1（W）

- `navsim/evaluate/pdm_score.py:pdm_score`：从 scorer 读取 NC、DAC、EP、TTC、Comfort、DDC 和 PDMS；
- `navsim/planning/script/run_pdm_score.py:run_pdm_score/main`：模型逐 token 评价与 CSV 汇总；
- `navsim/planning/script/run_pdm_score_from_submission.py:run_pdm_score/main`：单 submission 评价；源码明确提示此入口不支持 multi-seed；
- `navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py:PDMScorer`：v1 指标计算与聚合；
- `navsim/planning/script/config/pdm_scoring/*.yaml`：scorer/simulator/提交评价配置；
- `scripts/evaluation/run_*_pdm_score_evaluation*.sh`：现有评价启动器。

### 5.2 NAVSIM v2（H + R）

- `scripts/evaluation/score_recogdrive_predictions_navsim_v2.py`（历史 ref `codex/a5-epdms-stage3-20260726`）：调用 official NAVSIM v2 checkout，生成逐场景 EPDMS CSV/summary，记录 official revision；
- `navsim/agents/recogdrive/stage3_v2_official_evaluator.py:OfficialNAVSIMV2MetricEvaluator`（同 ref）：训练期与官方 v2 scorer 的隔离桥接；
- `scripts/evaluation/archive_recogdrive_predictions_navsim_v2.py`（ref `codex/pareto-full-action-head-contract`）：v2 预测归档。

当前 HEAD 不含上述 v2 evaluator 源码，但原始 v2 输出保留在工作区；最终复现必须恢复/固定相应历史 commit。

## 6. Checkpoint 与元数据

| 资产 | 证据与限制 |
|---|---|
| 公开 IL policy contract | H：`configs/first_drive_v7/native_public_il_g64.json`，checkpoint SHA256 `4569221d…`, epoch 199/global step 44400，64-sample cloud，DDIM 5 steps，base seed 0 |
| Core-Pareto GRPO v2 | W：`checkpoints/recogdrive/stage3_sota_backups/core_pareto_grpo_v2_step21600_pdms0.910274_20260621/`，checkpoint SHA256 `b8f74771…`，manifest 含 run/step/metrics |
| 最终 v1 91.45 checkpoint | R：`outputs/pdms914497_reverse_conflict_residual_trust105_scale14p3_20260727/reverse_conflict_residual_scale14p3.ckpt`，232,243,418 bytes，SHA256 `aeea278c…` |
| 最终 submission provenance | R：`outputs/ampt_pdms914505_navtest_submission_20260730/submission_provenance.json`，记录 checkpoint hash、seed 20260726、每 token 单轨迹、无 test-time selection |

最终 91.45 checkpoint 是 anchored/residual merge 产物；当前资产没有把它唯一映射到论文中三个 APR round 的统一训练 lineage。

## 7. 可直接复用的原始/逐场景结果

### 7.1 NAVSIM v1 主结果（R，强证据）

目录：`outputs/ampt_pdms914505_navtest_submission_20260730/`

- `full_navtest_pdms.csv`：12,138 scored rows，SHA256 `86687acc…`；
- `full_navtest_pdms_summary.json`：12,146 predictions、12,138 scored、8 个 metric-cache token 缺失、0 failures，SHA256 `bf09ec42…`；
- exact mean：PDMS 0.9145046521、NC 0.9852941176、DAC 0.9798978415、TTC 0.9600428407、Comfort 1.0、EP 0.8674798204、DDC 0.9764788268；
- `submission_provenance.json`：单轨迹推理、无 reranking/test-time scorer；同时明确 `navtest_used_for_checkpoint_selection: true`，应在公平比较说明中披露。

这些值精确支持正文 Table 1 的四舍五入值。

### 7.2 NAVSIM v2 主结果（R，强证据）

目录：`outputs/fair_ec_pdms914153_navtest_20260727/raw_path_temporal_cap2_final_score/`

- `navsim_v2_navtest_epdms.csv`：12,146 rows，SHA256 `9caa0a51…`；
- `summary.json`：official NAVSIM revision `0a380a9063d7162ec93d0f51e9990ebac585f720`，0 failed；
- exact means：EPDMS 0.8911677436、NC 0.9842334925、DAC 0.9769471431、DDC 0.9899555409、TLC 0.9971183929、TTC 0.9766178166、EP 0.8944757726、LK 0.9173390417、HC 0.9709369340、EC 0.8773904382；
- EC 在 10,040/12,146 行可用，summary 中给出聚合值；需要在评价协议中解释缺失 EC 的官方处理方式。

这些值精确支持正文 Table 3 的四舍五入值。

### 7.3 固定 658 hard scenes（R + D，存在口径差异）

目录：`outputs/navtest_hard_stable_zero658_recogdrive_stage3_vs_pdms914505_20260727/`

- `hard_stable_zero_658_tokens.txt`：定义为 5/5 历史运行 PDMS 恰为 0 的交集；
- `per_token_comparison.csv`、`cause_breakdown.csv`、`component_summary.csv`、`summary.json`：逐 token/原因/组件/汇总；
- baseline ReCogDrive Stage3：367 positive、291 zero；
- old Core-Pareto v2：440 positive、218 zero；相对 baseline 为 93 zero→positive、20 positive→zero，net zero repair = 73；
- 最终 91.45：415 positive、243 zero；相对 baseline 为 80 zero→positive、32 positive→zero，net zero repair = 48。

因此正文“AMPT recovers 440；比 scalar GRPO 多 73”混用了旧 Pareto v2 的 positive count 与 transition 净变化，不能归因给最终 91.45 AMPT。详见 `repo_evidence_notes.md`。

### 7.4 成对统计（D，可复用但非多 seed）

`outputs/pdms914497_multisource_ttc_consensus_reverse_graft_full_navtest_seed20260726_20260727/epoch_000_step_3200/paired_analysis.json` 提供 12,138 paired scenes、136 log clusters、20,000 次 scene/log-cluster bootstrap、单轨迹推理。它比较的候选相对 base 为负，且仅一个 inference seed，不能作为关键方法的多 seed 证据。

### 7.5 定性案例（D，有选择偏差）

`outputs/ampt9145_vs_recogdrive_rl_ddv2_navtest_cases_20260728/`：

- 319 个候选中按分数/距离阈值选出 30 个（left/right/straight 各 10）；
- `selected_cases.csv`、`candidate_pool_with_ampt_diagnostics.csv`、`all_source_metrics.csv`；
- `visualizations/manifest.csv` 与 provenance；
- 该集合是显式筛选的 qualitative set，所报均值不能用作无偏 benchmark 统计。

## 8. 仅论文报告、尚未定位原始证据的结果

下列值在 `docs/AuthorKit27 (7).pdf` 可见，但本次未找到对应逐场景输出、完整 run config、seed manifest 或 checkpoint lineage：

- GT/Score/Pareto/PC-MTS：`86.4→90.3`、`87.2→85.8`、`87.5→86.7`、`86.9→91.1`；
- stage ablation：`86.5, 86.9, 91.0, 91.1, 91.4`；
- APR rounds：`91.10, 91.21, 91.37, 91.45`；
- matched candidate count/acceptance rate、rollout feasibility、zero-rate、CVaR、early-GRPO gain；
- PC-MTS/FF-PGRPO/APR 内部机制消融与关键多 seed 对照。

这些只能标记 `reported`，不得由目录名或正文数字反向合成 raw data。

## 9. 结果与日志格式

仓库中已观察到的相关格式包括：

- 表格：CSV、TSV；
- 结构化记录：JSON、JSONL、YAML/Hydra resolved config；
- 模型/数据：CKPT、PT、NPZ、PKL/submission；
- 运行状态：LOG、TXT command/manifests、PID、LOCK、exit-status；
- 定性图：PNG、JPG；
- 统计：`paired_analysis.json`、per-token CSV、summary JSON/MD。

未正向定位到与论文关键对照对应的 TensorBoard event、W&B export 或 Parquet。不能把“仓库其他实验存在同格式文件”当作 AMPT 实验证据。

## 10. 优先缺口

1. 锁定 AMPT 三阶段的单一 commit/tag 与最终 checkpoint lineage；
2. 恢复与 PDF 一致的主稿 `.tex`/`.bib`；
3. 提供 PC-MTS 的真实候选源统计、KNN 参数与筛选 manifest；
4. 提供 GT/Score/Pareto/PC-MTS 及 stage/APR ablation 的逐场景、多 seed 原始输出；
5. 明确 658 场景正文应报告 old Core-Pareto v2（440/net 73）还是最终 AMPT（415/net 48）；
6. 解释 v1 的 8 个未评分 token、v2 EC 仅 10,040 行可用、以及 navtest checkpoint selection 对公平比较的影响。
