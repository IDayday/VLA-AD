# 本机 D_C8_double 教师重建与无路由蒸馏

本分支以原交付 `6bb7119583513094a169b4c2f2dc2a05ac293aa3` 为证据基线，新增代码与原资产分开保存。只重建差分、先聚类、K8、增强这一套本地理想修复教师，然后按用户指南执行 A/B/C/D × 两个训练 seed 的八个完整学生实验。旧九组实验保持暂停。

- [本地教师构建代码与执行链](local_teacher_rebuild/README.md)
- [固定允许资源上的完整GT执行与蒸馏接续](local_teacher_rebuild/permanent_fullflow/README.md)
- [GT反传配对与D48教师的关系](docs/GT_PAIRING_AND_D48.md)
- [具体教师方法、数据与数值协议](local_teacher_rebuild/METHOD_PROTOCOL.md)
- [本地教师替代历史 PPU 标签的协议补充](docs/LOCAL_PROTOCOL_ADDENDUM.md)
- [原始 Round1 完整实验指南](docs/recogdrive_no_routing_distillation_round1_guide.md)
- [模块集成约定](docs/LOCAL_ROUND1_INTEGRATION.md)

`student_distillation/` 包含连续共享 D48 适配器、原生 DDIM 桥接、数据与目标、定标、真实梯度/多卡/恢复核验、训练、评价、权重选择、完整执行控制器及报告汇总。`configs/student_distillation/` 保存四个实验配方。`scripts/student_distillation/` 提供启动与读取接口。

完整训练严格使用 85,813 场景、40 epochs、global batch64；每 run 53,640 次更新。完整 Validation 为 17,475 场景，Test 为 12,146 场景。Test 必须等待全部八个 run 的规定验证完成并锁定 best/last SHA 后才执行。

原 `assets/`、`decisions/`、`reference/`、`manifests/` 与 `vendor/` 保留冻结证据。新教师参数、表征、检查点、逐场景评分和报告写独立实验目录。实际资产路径由本机校验过的配置提供，不在代码里下载替代模型或旧教师大分片。

源码和工程核验通过不等于实验完成。实际阶段以各 owner 的注册表、完整产物和进程退出状态为准；最终须交付全部正式训练与评价及 `ROUND1_REPORT.md`，才标记 `REPORT_COMPLETE`。

当前要求保留完整 GT 反传配对流程；临时服务器授权已撤回。仅在固定允许资源上继续未完成配置，复用已完成结果。短版配对探索不作为正式教师输入。
