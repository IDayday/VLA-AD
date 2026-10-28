# 本机 ReCogDrive 蒸馏：已停止并归档

用户已于 2026-10-28 取消本轮剩余训练与评测。所有本项目训练、评测、自动重试及控制器已在五台服务器实际退出，自动恢复已禁止；后续恢复需要新的用户指令。A/H_full 已完整训练 40 epoch；B/C/D 最后完整训练 epoch 为 32/23/21，原定 40 epoch 预算没有全部完成。

**最新结果：[完整 Train/Test 表格、相关性、曲线和停止清理报告](reports/round1_full_train_stopped_20261028/README.md)。**

## 完整 Train 与 Test 的经验结果

固定 A/H_full、seed20261027 的 epoch4/8/16/24/32。每个权重覆盖完整 Train85,813 和完整 Test12,146；各场景 S0–S4 先平均再按场景等权，H 每场景只计一次。每点 Train 复用已完成 8,192 场景，新增 77,621 场景；Test 全部复用原完整结果。

| 人群 | epoch4 H | epoch32 H | epoch4 PDMS | epoch32 PDMS | epoch32 学生−自身Base |
|---|---:|---:|---:|---:|---:|
| 完整 Train85,813 | 0.301131 | 0.059741 | 94.9994 | 96.6635 | +1.5915 |
| 完整 Test12,146 | 0.397157 | 0.551774 | 88.8716 | 87.7419 | -2.9373 |

完整 Train 上 H 下降伴随行为均分提高，五点均值 Pearson=-0.999245、Spearman=-1。Test H 上升、行为下降，Pearson=-0.977338、Spearman=-1：当前训练数据上的收益未迁移到 Test，不能笼统归结为 loss 与 PDMS 无关。均值关系不能推出逐场景保证或因果，也不能当作跨 seed 规律。Test 已参与模型选择，不是独立 holdout。

[完整五点表](reports/round1_full_train_stopped_20261028/A_full_train/CHECKPOINT_CURVES.csv) · [相关性](reports/round1_full_train_stopped_20261028/A_full_train/CORRELATIONS.csv) · [相邻差分](reports/round1_full_train_stopped_20261028/A_full_train/ADJACENT_DIFFERENCES.csv) · [端点 log 聚类区间](reports/round1_full_train_stopped_20261028/A_full_train/ENDPOINT_EVIDENCE.json) · [保持/修复组](reports/round1_full_train_stopped_20261028/A_full_train/KEEP_REPAIR_CURVES.csv)。

## 四组方法与实际完成边界

冻结 VLM、feature_encoder、状态编码器和 RL DiT。学生共享连续适配器使用 4 learned queries、hidden256、两层 cross-attention、4 heads、FFN expansion4、dropout0；从有效 post384 token 与 24维普通状态预测48维连续系数，经固定 D48 解码为384维修复量。全场景同一网络，无路由或候选挑选。

| 组别 | 表征目标 | 功能损失状态 | 最后完整训练 epoch | 已完成正式 Test epoch |
|---|---|---|---:|---|
| A / H_full | full | 无 | 40 | 8/16/24/32；另有独立诊断点 |
| B / H_gain1 | gain1 | 无 | 32 | 8/16/24 |
| C / H_mu_teacher | gain1 | 冻结教师链 | 23 | 8/16 |
| D / H_mu_student | gain1 | 当前学生接口 detach 后刷新 | 21 | 8/16 |

全部保留训练使用 seed20261027；seed20261028 的重复 run 此前已取消。B32 Test 在停止时只有 partial，没有 COMPLETE，不报完整分数。A 完成40轮训练不等于 epoch40 Test 已完成；剩余评价被用户取消。

[B/C/D 已完成完整 Test 与在线/固定 loss 表](reports/round1_full_train_stopped_20261028/BCD_snapshot/RESULTS.md)。所有已完成 B/C/D Test 均低于同条件 Base；缺失的 Test H 没有补算。B/C/D 使用 gain1 目标，A 使用 full；H 绝对值不能跨目标直接排名。在线 epoch loss 与固定 checkpoint H 是不同统计量；C/D 在线总损失包含功能项，Test H 不含功能项。B/C/D 没有做本次 A 那样的完整 Train 行为/H 联合分析。

## 本机教师与源码身份

教师采用本机重新筛选、GT 接口反传配对、子空间/聚类/专家重新拟合的 D／full384先聚类／共享D48／K8／增强double；原 PPU 资产只作冻结方法证据。

| 完整人群 | 场景数 | 本机 Base PDMS | 本机 full 教师 PDMS |
|---|---:|---:|---:|
| Train | 85,813 | 95.07208906 | 97.52980208 |
| Test | 12,146 | 90.67922438 | 96.55443602 |

教师是离线场景选择上限，不是单学生成绩。Train gain1 教师为97.47006368，严格增益>1点保留36,359个修复，其余49,454个目标为Base。学生目标始终来自独立冻结 Base。

原运行训练源码提交 `0a58bd8e6271e26103512add2c5cb7816d7eb8f1`；冻结交付基线 `6bb7119583513094a169b4c2f2dc2a05ac293aa3`，完整原研究源码证据 `f85d4e557b12a7d05f5deb8d1640834cacd94d32`。本分支发布结果，不改运行代码 HEAD。

[学生源码](student_distillation/) · [基础配置](configs/student_distillation/) · [本机教师重建](local_teacher_rebuild/native_candidates_20261027/) · [原指南](docs/recogdrive_no_routing_distillation_round1_guide.md) · [实际有效配置和身份](reports/round1_initial_20261027/active_seed20261027/)。原指南中完整40轮/全部评测要求已被本次用户停止指令覆盖，不能直接据旧控制器自动续跑。

## 实际停止与保留

15:06:51 UTC 完成停止，15:12:10 UTC 完成清理。A 组全部 1,106 个文件均保留且元信息核验未变，其中 97 份 checkpoint 经 SHA256 核验未变。B/C/D 保留已完成 Test 关联权重和每个 run 最后完整 epoch，共12份；删除177个其他中间 `.pt`，4,409,115,022逻辑字节（4.106308GiB）。所有数据、教师、Base、缓存、原始日志、metadata、逐场景评价和报告保留。

[实际停止收据](reports/round1_full_train_stopped_20261028/stop_cleanup/STOP_COMPLETE.json) · [实际清理收据](reports/round1_full_train_stopped_20261028/stop_cleanup/CLEANUP_COMPLETE.json) · [A保护核查](reports/round1_full_train_stopped_20261028/stop_cleanup/A_PROTECTED_AFTER.json) · [保留路径/SHA与last替代](reports/round1_full_train_stopped_20261028/stop_cleanup/inventory/KEEP.json)。本仓库只发布聚合结果和compact证据；大权重、原始轨迹与数据未上传。

历史开训和早期结果：[初期快照](reports/round1_initial_20261027/) · [先前成绩与loss表](reports/navtest_epoch8_20261028/SCORES_AND_LOSS.md)。这些文件反映当时状态；当前停止状态和最新全量分析以上述新归档为准。
