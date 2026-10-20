# 源码核验与方法边界

本轮基于 `560bb257cd493613130f4b3643a0a2e4a88a7115`，规范原文及 SHA256 保存在报告的 specification 目录。

| 来源 | 实际处理 | 没有移植的部分 |
|---|---|---|
| 本仓库 OPD checkpoint_io / conditioning | 直接复用严格 active-policy 加载和官方固定 2800 token 输入；核心层必须齐全 | old_policy 不能作为 RL 教师；禁用专家与 action-aware 辅助参数明确登记 |
| 本仓库原生 DDIM p_mean_variance | 包装调用，统一 actual std、截断噪声、末步 clamp、norm/denorm | 不重写转移公式，不使用 logprob 下限替代实际采样 std |
| 本仓库 scoring.Scoring | 直接复用逐候选 reference 进度归一化等价批量包装；原 scorer 不变 | 不用候选集合共同归一化的 naive 分数作为标签 |
| DiffusionOPD | 数学适配：当前学生 detached chain，冻结教师逐步查询，学生重新计算转移均值 | 图像模型、LoRA、多教师路由和高斯 KL 不直接移植；截断核的 MSE 不称精确 KL |
| VIDD | 核验 roll-in 分布和 reward-guided 更新的分离、迭代 reference 思路 | DNA/protein categorical token loss、argmax Tweedie 终态近似、奖励网络均未移植 |
| SVDD-image | 核验候选转移与后续价值之间的关系，采用真实完整后缀蒙特卡洛评价 | 不复制图像 VAE、图像 scorer、x0 reward shortcut 或部署时搜索 |
| MPO | 数学参照：指数倾斜权重、有效样本量约束、受限投影 | 本方法是中心化有限候选均值启发式；不宣称得到原生截断核的精确 KL 投影 |
| DPPO | 核验原生 DiffGRPO 沿用的去噪链、eta、方差下限、概率裁剪与 BC | 不另换 PPO/奖励定义；5/5/2 对照仅改变 scorer 权重 |

`configs/process_improvement/references.lock.json` 固定实际下载的代码 commit、许可证检查、阅读文件及哈希。VIDD 和 SVDD-image 未发现仓库许可证文件，因此仅使用数学思想，不复制其代码或导入其运行依赖。

本方法估计的是参考后缀下的普通期望终态 PDMS，即 Q。它不是 `eta * log E exp(reward/eta)` 形式的 soft value。训练用 fit、接受判据用 guard、效果检验用 test，三者使用独立命名空间。候选选择的测试收益、原生均值投影的测试收益、部署学生的收益分别报告。

观察编码缓存只保存原始共享 VLM 输出；每个 planner 重新计算自己的 feature_encoder、历史与自车状态条件。学生前向图绝不复用长期冻结的投影。临时 K/V 公共子表达式优化有单独的真实前向/梯度/参数更新门禁，主实验默认关闭。

原生 GRPO 的 clipped-normal 采样配 Gaussian logprob surrogate、std floor 和 logprob clamp 全部保留。其 sample_chain 原始计算图行为也保留；这是一项被审计的原实现限制，并非在本轮悄悄修订的算法。
