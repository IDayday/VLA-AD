# 官方基线前向契约修正

首轮 v1 为 INVALID，不能用于标准 ReCogDrive 的 Q1–Q3 结论。固定官方源码版本 `5831ee83c7a6a40bf6919715b96b0e9ba2b48335`，真实8场景证明：当前仓库动作头在相同特征下与官方逐元素一致，但历史系统提示词 trim 和动态 padding 改变了轨迹；最大单坐标差为 0.4966259002685547。

新增兼容层仅由 OPD 显式启用，恢复系统提示词边界换行、左侧 fixed max_length=2800 padding，保留全部 hidden states。8场景修正特征与官方逐元素一致，官方/当前动作头输出误差0。未改变原仓库默认路径或 JEPA/VGGT 实验。

全体受影响观察、教师轨迹/chain、训练 gate、E0–E5、选参和评分统一重做；场景/log/父片段成员、方法矩阵、种子、原始损失、固定训练预算和选择规则保持一致。辅助权重按原先登记的同一 fit 梯度比例规则重算。旧权重不用于新初始化。复用的物理 metric cache 与GT不含VLM特征。

旧批次的代码、配置、日志、结果、图和 protocol 保留在外部 `outputs/recogdrive_opd_20261020/`，状态见 INVALID.json；旧报告快照在 reports_invalid_snapshot。修正批次在 `outputs/recogdrive_opd_20261020/official_v2/`。v1测试集基线均值曾被查看；v1方法测试集聚合未被查看，不用于修正选参。工程修正按源码一致性决定，测试集不保证对开发者完全未接触，结论需披露此限制。
