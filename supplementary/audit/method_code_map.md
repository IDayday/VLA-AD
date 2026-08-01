# AMPT Method-to-Code Map

本映射以论文术语为主键，并明确给出 `commit:path:function/class`。代码没有统一采用论文命名，且 PC-MTS/FF-PGRPO 与 APR 分处不同历史谱系；因此“对应”表示机制级实现证据，不自动证明最终论文 checkpoint 完整执行了该代码。

## 1. Stage 1 — PC-MTS

主要 commit：`41a86139794c684ff7ddc45e6647275186ac3974`（可由 `codex/pareto-full-action-head-contract` 读取）。源码内部名为 FIRST-Drive V7 curriculum。

| 论文机制 | commit:path:function/class | 对齐状态与备注 |
|---|---|---|
| 候选源声明、manifest/provenance | `41a8613:navsim/agents/recogdrive/curriculum/proposals.py:ProposalSourceSpec`, `load_source_specs`, `load_scene_proposals` | 对齐；source config 支持 hash/manifest 校验 |
| 候选去重 | `41a8613:.../curriculum/proposals.py:deduplicate_proposals`, `deduplicate_proposals_with_immutable_baseline` | 对齐；GT/immutable baseline 被保护 |
| scene 构建总入口 | `41a8613:.../curriculum/builder.py:prepare_build_artifacts`, `build_scene_intermediate`, `finalize_scene_record`, `build_v7_curriculum` | 对齐；完整 gates/quota/record 构建 |
| policy rollout cloud | `41a8613:.../curriculum/builder.py:_policy_cloud_for_scene` | 对齐机制；实际 cloud 大小由 source contract 决定 |
| 轨迹距离分量 | `41a8613:.../curriculum/reachability.py:physical_distance_components` | ADE/FDE/heading/velocity 四分量 |
| command/cluster 校准 | `41a8613:.../curriculum/reachability.py:fit_reachability_calibration`, `ReachabilityCalibration.group_parameters` | 对齐；默认 now/later quantile 0.90/0.975、最少 16 个 group observations |
| compatibility 判定 | `41a8613:.../curriculum/reachability.py:evaluate_reachability` | **不完全对齐**：代码 line 319 使用单个最近样本 `argmin`；论文写平均 K-nearest，且代码无 K 参数 |
| rollout/query 独立性 | `41a8613:.../curriculum/reachability.py:fit_reachability_calibration` | calibration 对 cloud 样本采用 leave-one-out；但论文所称完整 held-out query/cloud 数据独立性仍缺运行 manifest |
| 轨迹/命令意图 gate | `41a8613:.../curriculum/builder.py:_intent_compatible` | 对齐；阈值在函数中，不是论文可直接引用的实验 manifest |
| hard feasibility | `41a8613:.../curriculum/safety.py:absolute_feasibility` | NC/DAC、DDC absolute、physical gate |
| protected non-regression guards | `41a8613:.../curriculum/safety.py:relative_nonregression`, `positive_safety_contract` | TTC/DDC/comfort/EP protected deltas |
| metric/geometry gate | `41a8613:.../curriculum/safety.py:point_and_geometry_gate`, `geometry_metrics` | 对齐；包含 curvature/jerk 等 |
| 时间相关局部扰动 | `41a8613:.../curriculum/robustness.py:CorrelatedErrorModel.sample_perturbations` | 对齐；trajectory residual + timing/curvature，默认 16，正式允许 16/32 |
| 扰动的 exact score 与通过统计 | `41a8613:.../curriculum/robustness.py:evaluate_robustness_tube`, `summarize_robustness_metrics`, `finalize_robustness_against_reference` | 对齐；含 Q10、safe probability 等 |
| component-wise Pareto/role quota | `41a8613:.../curriculum/quota_selector.py:role_aware_quota_select`; `builder.py:finalize_scene_record` | 机制存在；需运行 artifact 才能给出候选数/接受率 |
| coherent real reference | `41a8613:.../curriculum/coherent_reference.py:select_coherent_reference` | 对齐；docstring 明确禁止 componentwise 合成，选现有完整 row |
| 单 scene 单 step 一个 target | `41a8613:.../stage2_role_sampler.py`; `.../stage2_bridge.py`; `.../curriculum/builder.py:_stage2_target_shortlist`; config `first_drive_v7_phase_{a,b,c}.yaml:target_sampler_contract` | 对齐；配置显式 `one_target_per_scene` / `one_retention_target_per_scene_per_update` |
| runtime 数据载入 | `41a8613:.../v7_curriculum_runtime.py:SealedV7CurriculumIndex` | 对齐；要求 sealed manifest |
| agent/planner 接入 | `41a8613:.../first_drive_v7_agent.py:FirstDriveV7Agent`; `.../first_drive_v7_planner.py` | 对齐历史实现，当前 HEAD 不含 |
| fallback | `41a8613:.../curriculum/builder.py:finalize_scene_record` | fail-closed/GT retention 逻辑可见；空候选运行频数缺 manifest |

可引用配置：

- `41a8613:navsim/planning/script/config/experiment/first_drive_v7_phase_a.yaml`
- `41a8613:navsim/planning/script/config/experiment/first_drive_v7_phase_b.yaml`
- `41a8613:navsim/planning/script/config/experiment/first_drive_v7_phase_c.yaml`
- `41a8613:configs/first_drive_v7/native_public_il_g64.json`

注意 `native_public_il_g64.json` 的 `layout.k: 1` 是 archive layout 字段，不能据此声称论文 KNN 的 K=1；实际 `evaluate_reachability` 本身没有 KNN average。

## 2. Stage 2 — FF-PGRPO

主要 commit 同为 `41a8613`。源码内部名 `Feasibility-First Anchor Pareto v3` / FF-Pareto，历史实验名含 Core-Pareto GRPO v2。

| 论文机制 | commit:path:function/class | 对齐状态与备注 |
|---|---|---|
| 配置与 fingerprint | `41a8613:navsim/agents/recogdrive/stage3_ff_pareto.py:FeasibilityFirstParetoConfig` | `FF_PARETO_VERSION="feasibility-first-anchor-pareto-v3"`；阈值和 loss weights 有默认值，但须用运行 resolved config 证实实际值 |
| 三类 group | `41a8613:.../stage3_ff_pareto.py:GroupState`, `classify_group_state` | `NO_SAFE` / `MIXED` / `ALL_SAFE` 对应 all-infeasible / mixed / all-feasible |
| credit 主逻辑 | `41a8613:.../stage3_ff_pareto.py:FeasibilityFirstCreditAssigner.compute` | 完整实现 hard feasibility、reference guards、Pareto positive gate、no-safe recovery ranking |
| anchor-local Pareto | `41a8613:.../stage3_ff_pareto.py:anchor_local_pareto` | 代码按 anchor 局部 Pareto，默认 `rollouts_per_anchor=2`；与论文“每 scene 16 rollouts 的一个 group”表述需解释 layout |
| sign-preserving normalization/clipping | `41a8613:.../stage3_ff_pareto.py:rms_scale_without_mean_centering`, `CreditOutput.normalized` | RMS without mean centering，默认 clip 3.0 |
| coherent reference 数据结构/选择 | `41a8613:.../stage3_reference_selector.py:ReferenceCandidateBatch`, `CoherentReferenceSelector.select`, `gather_reference_rows` | 对齐；API 使逐指标 synthetic envelope 无法表达 |
| positive-credit violation contract | `41a8613:.../stage3_positive_credit_contract.py:evaluate_positive_credit_contract`, `assert_positive_credit_contract` | 直接产生 illegal/dominated/no-safe/safety-regression positive counts，可实现附录诊断率 |
| optional SDR | `41a8613:.../stage3_positive_credit_contract.py:apply_safe_only_sdr` | 默认关闭；不应作为论文 full FF-PGRPO 必选组件 |
| metric adaptation | `41a8613:.../stage3_metric_adapter.py` | exact evaluator 指标到 credit tensors |
| reference cache | `41a8613:.../stage3_reference_cache.py` | coherent rows/runtime cache |
| geometry/trust region | `41a8613:.../stage3_policy_geometry.py` | physical/trust masks |
| planner loss 接入 | `41a8613:.../first_drive_v7_planner.py`; `.../stage3_runtime.py` | policy、KL/BC/recovery loss 接入的历史实现 |

代码默认可见但不能未经 resolved run config 写成“最终训练参数”的值包括：DDC/TTC absolute 0.95、protected tolerances TTC 0.02/DDC 0.01/comfort 0.01/EP 0.02、KL 0.02、BC 0.10→0.05（5 epochs）、recovery coeff 0.25、advantage clip 3.0。历史结果报告 `core_pareto_grpo_v2_results_summary_20260621.md` 支持一部分 Core-Pareto 运行设置，但不等同于最终 AMPT 的 FF-PGRPO run。

关键不确定性：论文称 group size 16；历史 `stage3_ff_pareto.py` 的默认 layout 是多个 anchors × 每 anchor 2 rollouts，必须从实际 run config/shape manifest 说明 16 如何分解。未定位最终 FF-PGRPO 多 seed 日志。

## 3. Stage 3 — APR

主要 ref：`codex/a5-epdms-stage3-20260726`，tip `ab9124c08100d3f14e6ca45da19e40e3d5a5c3fb`。下表按机制列出具体历史 commit/path/function；未发现统一 `APR` 类。

| 论文机制 | commit:path:function/class | 对齐状态与备注 |
|---|---|---|
| strict teacher 相对当前 reference 的审计 | `fb96a03:scripts/stage3/build_pdms_strict_teacher_buffer.py:_strict_teacher_upgrade`, `build_teacher_record` | scalar gain + component deltas；explicit teacher mask |
| actual-policy safety repair teacher | `fb96a03` 后续该 ref 中 `scripts/stage3/build_pdms_safety_repair_teacher_buffer.py:safety_repair_eligibility`, `choose_safety_teacher`, `build_safety_teacher_record` | 对齐“相对 current policy + protected safety”机制 |
| 外部/多源 safety teacher | `ab9124c:scripts/stage3/build_pdms_external_safety_repair_teacher_buffer.py:choose_external_safety_teachers` | source-aware selection；不提供论文所需真实来源占比，需 manifest |
| teacher/retention schedule | `fb96a03:scripts/stage3/build_pdms_teacher_refinement_schedule.py:make_refinement_schedule`, `_teacher_audit` | 默认 CLI teacher repeats 8、retention per teacher 4.0；是脚本默认，不足以证明论文 run |
| 轨迹插值 | `ab9124c:scripts/stage3/build_pdms_blended_teacher_reference.py:blend_trajectory`, `_build_task` | `anchor + alpha(teacher-anchor)` |
| 插值后重新评价与接受 | `ab9124c:.../build_pdms_blended_teacher_reference.py:_score_one`, `accept_scored_teacher`, `main` | 对齐 exact post-interpolation rescoring；单次 `--alpha`，descending alpha 列表由外层 launch 管理 |
| 局部 safety tube | `ab9124c:scripts/stage3/filter_pdms_teacher_reference_by_safety_tube.py:build_safety_tube_variants`, `safety_tube_variant_passes`, `choose_best_componentwise_safe_variant` | lateral/longitudinal variants + exact score；未见论文所称 curvature/jerk audit 的统一 APR config |
| checkpoint 插值 | `58cab66:scripts/stage3/interpolate_lightning_checkpoints.py:interpolate_state_dicts` | base→tuned 线性插值、hash/provenance、单 policy 输出 |
| TIES-style merge | `d24b7c7:scripts/stage3/merge_ties_base_relative_checkpoints.py:ties_merge_base_relative_deltas` | trim/elect/merge；仅正文保留 task-vector 时引用 |
| anchored conflict-free merge | `64f5531:scripts/stage3/merge_conflict_free_anchored_task_vector.py:conflict_free_anchored_graft` | salient coordinate、方向一致、graft norm、total norm、anchor 约束 |
| train/retention launch | `ab9124c:scripts/stage3/launch_pdms91_external_teacher_retention_3ep.sh`; `scripts/stage3/launch_pdms91_safety_awac_resume.sh` | 实验族启动器；不是完整 3-round APR orchestration |

未在代码/manifest 中闭合的论文机制：

- accepted → expired / newly activated 的显式 teacher lifecycle state machine；
- 固定三轮、每轮步数、晋级/停止条件的统一配置；
- advantage weight/temperature/clip 的论文版最终值；
- teacher/retention 的最终采样比例与各 teacher source 占比；
- descending interpolation alpha 列表及每轮选择结果；
- 最终 91.45 checkpoint 与三轮 APR teacher pools 的内容哈希链。

## 4. 评价代码映射

| Benchmark | commit:path:function/class | 备注 |
|---|---|---|
| NAVSIM v1 | `ba5908d:navsim/evaluate/pdm_score.py:pdm_score` | current HEAD 可用 |
| NAVSIM v1 | `ba5908d:navsim/planning/script/run_pdm_score.py:run_pdm_score`, `main` | agent inference + CSV |
| NAVSIM v1 submission | `ba5908d:navsim/planning/script/run_pdm_score_from_submission.py:run_pdm_score`, `main` | 入口明确不支持 multi-seed submission list |
| NAVSIM v1 scorer | `ba5908d:navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py:PDMScorer.score_proposals`, `_aggregate_scores` | scorer formula |
| NAVSIM v2 offline | `ab9124c:scripts/evaluation/score_recogdrive_predictions_navsim_v2.py:_score_one`, `_finalize_epdms`, `main` | official v2 checkout + EPDMS/EC |
| NAVSIM v2 training bridge | `ab9124c:navsim/agents/recogdrive/stage3_v2_official_evaluator.py:OfficialNAVSIMV2MetricEvaluator.score` | isolated official scorer process |

## 5. 与论文伪代码写作有关的强制限定

1. Algorithm 1 可按 `builder → reachability/safety/robustness → quota → one-target sampler` 写，但必须把当前实现描述为 nearest rollout，除非先修代码或提供另一实现证据；不得凭 PDF 写 KNN average 为“代码一致”。
2. Algorithm 2 应反映 `NO_SAFE/MIXED/ALL_SAFE`、coherent row、positive contract、sign-preserving RMS；需解释 anchor-local layout 与 16 rollouts 的关系。
3. Algorithm 3 应拆成 teacher audit、interpolate/rescore、retention train、pool rebuild 的论文逻辑，并在代码注释中分别指向上述脚本；不得声称仓库存在一个 `APR.run(rounds=3)`。
4. task-vector merge 是可选实现族，最终 91.45 checkpoint 的确为 anchored residual graft 产物；若正文仍称它属于 APR，附录应明确它是 round checkpoint 的保守 consolidation，而非 teacher 选择本身。
