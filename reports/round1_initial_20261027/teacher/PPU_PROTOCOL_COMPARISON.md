# 本机完整教师分数与原PPU口径核对

本机为 A800 新重拟合 D / C_to_P / K8 / double，offline 固定场景 Oracle 上限；不是学生成绩。

| 条件 | Population | Base | Teacher | Gain |
|---|---|---:|---:|---:|
| 本机新重拟合 | train(85813) | 95.0720890580 | 97.5298020752 | +2.4577130172 |
| 本机新重拟合 | test(12146) | 90.6792243763 | 96.5544360231 | +5.8752116468 |
| 历史PPU6bb7119 | train(85813) | 94.9163155149 | 97.3053145346 | +2.3889990197 |
| 历史PPU6bb7119 | test(12146) | 90.4793124745 | 96.2389009126 | +5.7595884380 |

原 f85 full_six_statistics_v1.py:68–73 在 fraction 五噪声均值上 argmax，随后乘100；本机 collector 在乘100后 argmax。全量 Train/Test 实际比较均为 **0 个 choice 差异**，同一固定分支服务全部五组噪声。整体归约顺序仅造成约1e-14末位变化，完整原顺序数值保存在 ORIGINAL_SELECTION_CHECK.json。

已确认差异与条件：
- fraction-vs-point argmax：No selected branch or trajectory changes in either full population. No Native/scorer rerun warranted by this difference alone.
- overall reduction order：Same selected observations and equal scene weights; full score changes only ~1e-14 floating reduction rounding. Exact alternate values retained in ORIGINAL_SELECTION_CHECK.json.
- 实验条件 device/software and Native Base：Experimental condition and actual Native numerical difference, not a population/weighting/selection change. Base must be reported separately.
- 实验条件 teacher fitted parameters：Requested new teacher; historical parameters/choices never substituted for local labels.
- 实验条件 metric cache serialization/transport：Cache bytes are intentionally not claimed byte-identical to PPU. Same frozen scoring source/config; existing registry reports independent70 source trajectories exact. This review verifies all metadata/SHA joins, does not rerun those scorers or assert exhaustive semantic object equality.

| 口径项 | 核对结果 |
|---|---|
| Full population | CONFIRMED SAME full scopes: Train85813/130sessions; Val17475/32heldout; Test12146/44sessions. Frozen token/index/log/metric joins exact. |
| Candidates | CONFIRMED SAME family definition: this single D/C_to_P/K8/double bank, Base0+8 slots. No cross-library union. |
| Noise/sampler | CONFIRMED explicit originalPPU tapes S0-S4 and token joins; Native FP32/DDIM5 [80,60,40,20,0], actualeta1/floor1e-4/noiseclip1; original seed formula token|20261008|j. Five noises are not five training seeds. |
| Selection | CONFIRMED one branch per scene across all5; Base first then slot order on exact ties, no70 eligibility threshold. Fraction/point difference has0changed choices on both full populations. |
| Scoring | CONFIRMED frozen6bb score_group/pdm_score/pdm_scorer/config byte-identical locally; proposal40×0.1s, trajectory8×0.5s, weights5/5/2/0, fixed seven-field order. |
| Aggregation | CONFIRMED each full scene equal weight; each scene has5draws. Log tables descriptive, not log-equal reweighting. No natural-panel weights used. |
| Display precision | Raw FP64 arrays not rounded before selection. Historical README/report display4decimals; exact EXPECTED retained. Alternative floating reduction orders shown separately. |

PPU历史权重、choice不充当本机标签。未发现当前闭合原分数的采样/范围/评分配置偏离证据；保留已确认的实现细节差异和metric缓存语义验证范围，不重跑、不删除、不改canonical。详细文件SHA/行号见PPU_PROTOCOL_COMPARISON.json，逐场景与逐log诊断见REPORT.json和CSV。
