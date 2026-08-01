# Repository Evidence Notes

本文件汇总审计中对补充材料最重要的“可证实 / 缺失 / 不一致”结论。数值均来自现有 PDF、CSV、JSON 或 checkpoint manifest；未用正文目标值反向生成数据。

## A. 已由原始资产证实

### A.1 NAVSIM v1 AMPT 91.4

来源：

- `outputs/ampt_pdms914505_navtest_submission_20260730/full_navtest_pdms.csv`
- `outputs/ampt_pdms914505_navtest_submission_20260730/full_navtest_pdms_summary.json`
- `outputs/ampt_pdms914505_navtest_submission_20260730/submission_provenance.json`

12,146 predictions 中 12,138 有 metric cache 并成功评分；8 个未评分。PDMS = 0.9145046521，组件为 NC 0.9852941176、DAC 0.9798978415、TTC 0.9600428407、Comfort 1.0、EP 0.8674798204、DDC 0.9764788268。正文 Table 1 的 91.4/98.5/98.0/96.0/100/86.7 与原始值一致。

provenance 支持：每 token 1 条 `[8,3]` 轨迹；seed 20260726；无 test-time candidate selection；checkpoint SHA256 `aeea278c…`。同时 provenance 写明 navtest 用于 checkpoint selection，公平比较章节应披露，不能表述为完全 held-out model selection。

### A.2 NAVSIM v2 AMPT 89.1

来源：

- `outputs/fair_ec_pdms914153_navtest_20260727/raw_path_temporal_cap2_final_score/navsim_v2_navtest_epdms.csv`
- 同目录 `summary.json`

12,146/12,146 successful，EPDMS = 0.8911677436；NC 0.9842334925、DAC 0.9769471431、DDC 0.9899555409、TLC 0.9971183929、TTC 0.9766178166、EP 0.8944757726、LK 0.9173390417、HC 0.9709369340、EC 0.8773904382。与正文 Table 3 四舍五入完全一致。官方 NAVSIM revision 是 `0a380a9063d7162ec93d0f51e9990ebac585f720`。

EC 只有 10,040 行有直接值；附录需按实际 evaluator/official protocol 解释 aggregation，不能简单写“12,146 场景均有 EC”。

### A.3 单轨迹推理

v1 submission provenance 和 matched full-navtest paired analysis 均记录：online candidate count/inference trajectory count = 1，无 test-time selection。该部分可写成 reproduced protocol，而非仅论文宣称。

### A.4 可用的成对 bootstrap 框架

`outputs/pdms914497_multisource_ttc_consensus_reverse_graft_full_navtest_seed20260726_20260727/epoch_000_step_3200/paired_analysis.json`：12,138 paired scenes、136 log clusters、scene-pair 与 log-cluster 各 20,000 次 bootstrap，seed 分别 20260726/20260727。该例的候选分数下降，且是单 inference seed；只能证明统计管线可复用，不能证明 AMPT 增益稳定。

## B. 必须显式记录的结果差异

### B.1 658 hard scenes：440 与最终 AMPT 415 不同

唯一完整场景级证据：`outputs/navtest_hard_stable_zero658_recogdrive_stage3_vs_pdms914505_20260727/summary.json` 与 `per_token_comparison.csv`。

固定集合定义为“5/5 历史运行 PDMS 恰为 0”的 658-token intersection。结果为：

| 模型 | positive | zero | zero→positive | positive→zero | net zero repair |
|---|---:|---:|---:|---:|---:|
| ReCogDrive Stage3 baseline | 367 | 291 | — | — | — |
| old Core-Pareto v2 | 440 | 218 | 93 | 20 | 73 |
| final PDMS 91.45 / AMPT submission checkpoint | 415 | 243 | 80 | 32 | 48 |

故正文句式“scalar GRPO recovers 367；AMPT recovers 440；difference 73”至少有两处口径问题：

1. 440 是 old Core-Pareto v2 的 positive count，不是 final AMPT 91.45 的 415；
2. 73 是 `93 - 20` 的净 zero repair，不是简单的 gross positive-count 差 `440 - 367` 所能独立解释的同义量（数值碰巧相等）。

建议二选一，不能混写：

- 若论文坚持最终 AMPT：报告 415 positive、80 gross recovered initial zeros、32 new zeros、net 48；
- 若论文分析旧 FF-Pareto/Core-Pareto v2：明确模型名，报告 440 positive，同时给 93/20/net 73 transition。

最终模型在该 hard subset 上 PDMS 0.584898，低于 old v2 的 0.614451，尽管全 navtest 分数更高；这也应在失败案例/局限性中诚实说明。

### B.2 PC-MTS compatibility 的 KNN 描述与实现不一致

论文写“average distance to K nearest policy rollouts”。`41a8613:navsim/agents/recogdrive/curriculum/reachability.py:evaluate_reachability` 实际计算所有归一化距离后使用 `nearest_index = argmin(...)` 和单个最近样本的 distance；未找到 K 参数或平均 KNN 实现。`configs/.../native_public_il_g64.json` 中的 `layout.k=1` 属于 archive layout，不足以证明论文公式的 K。

处理建议：附录不要虚构 K。要么将正文/附录改成 nearest-rollout calibrated distance；要么补交真正执行 KNN-average 的代码和 sealed run manifest，并说明 K。

### B.3 FF-PGRPO group layout 与论文 G=16 需解释

论文报告每 scene 16 rollouts。`stage3_ff_pareto.py` 默认 `rollouts_per_anchor=2`，credit 计算输入维度是 `[batch, anchors, rollouts]` 并做 anchor-local Pareto。可能通过 8 anchors × 2 rollouts 合成 16，但本次未定位最终 resolved run config。附录必须给实际 layout，不可只把 default 2 或正文 16 任意写入。

### B.4 训练 epoch 配置冲突

`41a8613:.../first_drive_v7_phase_a.yaml` 同时含 scheduler `epochs: 200` 和 trainer `max_epochs: 100`；Phase B/C 同样 scheduler 200 / trainer 100，且含 `???`。PDF 称 IL 200 epochs。没有实际 resolved config/log 时，训练参数表只能写 PDF-reported 200，并把历史 config 冲突标记未解决，不能称 reproduced。

## C. 仅 PDF 可证实、缺少运行证据

下列主表值在 PDF 中存在，但未找到逐场景 CSV、run-level seed manifest 或完整 resolved config：

- supervision mismatch：GT、Score、Pareto、PC-MTS 的 IL→GRPO 四组值；
- stage ablation 五行值；
- APR 4 个 round 值；
- PC-MTS candidate source/count/acceptance 与 policy-compatibility diagnostics；
- FF-PGRPO positive-credit violation、all-infeasible rate、首次 feasible rollout；
- APR teacher accepted/expired/newly activated、alpha 分布、每轮覆盖率；
- Score/Pareto/PC-MTS、Scalar vs FF-PGRPO、No APR vs Full APR 的 ≥3 seeds。

目录名中出现 `pdms913707`、`pdms914153`、`pdms914497` 等不能替代运行证据，也不能自动建立 APR round 顺序。

## D. 方法实现层面的证据边界

- PC-MTS/FF-PGRPO 的最完整代码来自 `41a8613` 历史分支；当前 HEAD 不含这些 modules。
- APR 是多个 teacher/interpolation/retention/merge 脚本的机制组合；没有直接命名 APR 的统一 orchestrator。
- 最终 91.45 checkpoint 是 reverse-conflict residual anchored graft 产物。其 provenance 可追踪 checkpoint hash，但缺少从三轮 teacher pool 到最终 checkpoint 的完整 hash chain。
- `RAP` 是外部模型/候选源名，非 APR 拼写错误。
- A/B/C、A1/A2/A3 在仓库内对应多个不相关实验族，不能映射成 AMPT 三阶段。

## E. 可用于附录、但需加限定的资产

1. v1/v2 主结果表：可标 `reproduced from scene-level outputs`；v1 注明 12,138 scored / 12,146 predictions，v2 12,146 successful。
2. 30 个 qualitative cases：可用图与 manifest，但 caption 必须写“selected cases”，不可报告为总体均值。
3. hard-658 transition/cause breakdown：可自动生成 2×2 matrix 和 failure-type 图，但须选定 old v2 或 final AMPT 的一致模型口径。
4. Core-Pareto v2 checkpoint/report：可作 historical FF mechanism evidence，不能直接替代 final FF-PGRPO multi-seed ablation。

## F. 应进入 missing evidence 的事项

- 主稿 `.tex`/`.bib`；
- 最终三阶段 code tag/commit；
- PC-MTS K 或正确 nearest-neighbor 公式、rollout/calibration 独立性 manifest；
- candidate pool/source/acceptance/fallback counts；
- GT/Score/Pareto/PC-MTS 的逐场景 matched-control 数据；
- 最终 FF-PGRPO G=16 layout 和 resolved hyperparameters；
- APR 三轮 teacher lifecycle 和 alpha list；
- 关键对照多 seed；
- 软件环境 lockfile/容器 hash、训练耗时/存储/offline scoring cost；
- baseline reported/reproduced protocol provenance。
