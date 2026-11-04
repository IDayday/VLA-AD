# 11．从现在开始怎样组织论文

## 可直接采用的正文骨架

| 章节 | 回答的问题 | 优先证据 |
| --- | --- | --- |
| Introduction | 现有策略的部分失败可被内部干预改善，但逐场景特权解与可部署收益之间有距离。 | GT反事实、Oracle/普通差距、当前主表；避免将故障根因先验归给VLM。 |
| Method 3.1：内部反事实 | 同一输入怎样得到行为有效的配对；GT引导与实际评分各做什么。 | GT源报告、TRAIN配对manifest、初始化/隔离审计。 |
| Method 3.2：共享修复专家 | 归一化差分SVD、全维方向聚类、六中心/两MLP及统一力度。 | GEO/OPS、EXPERT_SOURCE_AUDIT、S2。 |
| Method 3.3：普通选择 | A预估候选行为，F/H联合评分，原G执行；训练/冻结界限。 | WORKFLOW_SPEC、A/FH训练和部署实现。 |
| Experiments | 人口/会话、固定场景五噪声、三训练seed、TUNE选模、主结果。 | S0/S1/S2和独立数值回执。 |
| Mechanism analysis | 空间、配对、分组和条件算子分别贡献什么；为什么Oracle不能替代部署。 | S3匹配对照、各库自身重训、两类随机性。 |
| Transfer & cost | IL/v2完整迁移、独立两阶段与普通输入成本。 | S4/S7；不同基模和分数不跨行直接比绝对大小。 |
| Discussion / Appendix | 历史蒸馏、合成RL负结果、当前测试使用及外推边界。 | D4/S5/S6；保留负结果，不塞进主贡献叙事。 |

## 建议先做的图表

- **图1：完整框架。**上半离线GT优化/差分构库，下半普通输入候选预估/选择；冻结与训练模块明确着色。

- **表1：普通主结果。**PDMS主线、IL/v2迁移分块，各自Base、三seed均值±SD和选模规则。

- **表2：核心消融。**匹配库E1−E2/E3、R6−E4、主库−R6，普通成绩与Oracle并列但分列，附会话区间。

- **图2：修复与损伤。**低/中/高分组贡献，避免总体均值遮住高质量样本退化。

- **表3或附录表：课程、力度与成本。**保留全部预定设置，不只展示最高TEST。

- **补充材料：**合成RL足额曲线、新Base重建、历史蒸馏局限和完整来源谱系。

## 一段可作为摘要事实骨架的文字

我们研究如何在冻结驾驶策略的条件下，通过内部表征干预改善驾驶行为。利用训练场景的GT引导优化，我们构造相同输入下的失败状态与行为有效反事实，提炼共享差分子空间及少量修复专家。推理时，独立辅助模型预估各修复条件的候选行为，单前视评分器选择条件，原动作头据此完成一次完整生成。在项目固定NAVSIM评价人口上，纯真实新增训练的主方法将RL Base的PDMS从90.4793提高到92.4074；在IL/v2完整重建中，ePDMS从86.3003提高到90.5778。进一步的匹配消融、独立两阶段迁移与端到端开销测量，分别检验共享修复结构、实际选择与迁移表现。

这是写作草案，应在最终论文中补充实验划分、三seed统计与测试集历史使用声明；没有文献比对前不加入“首次”“SOTA”或普遍架构无关性等主张。

## 写作前仍需锁定的项目

- **确定论文范围。**已定稿的S0/S1/S2/S3/S4/S5/S6可作为当前版本；已读IL/v1.1与大K报告尚未给出完成三seed的普通最终结果，不提前将Oracle或计划写入主表。

- **如要强调更广泛泛化，另行取得相应证据。**本批未完成DDP跨架构迁移，TEST也曾被历史观察；当前IL/v2支持的是另一基模与评价协议上的完整重建。

- **保留机制结论强度。**正确配对/方向分组有库能力证据和普通均值优势，当前会话不确定性应随结果保留；更强主张需要新的确认设计。

- **汇总离线准备代价。**本次已核实在线开销，尚未从全历史优化/构库搜索账本汇总统一离线GPU时长；投稿若强调总成本，应另行按明确范围汇总。

- **完成Related Work定位。**本次只核查本地研究材料；表征编辑、策略修复、反事实监督与候选选择的外部文献尚未系统比对。

这些是证据边界和写作取舍，本次未据此恢复实验、提交训练、改动队列或修改共享目录。

# 12．推荐阅读顺序与复查入口

- **研究导航：当前主线与全部历史入口** [NAV](../sources/source_index.html#ref-NAV)

- **第十三批计划：研究假设、固定取舍与分支边界** [PLAN](../sources/source_index.html#ref-PLAN)

- **S0：来源、初始化与协议** [S0](../sources/source_index.html#ref-S0)

- **GT辅助逐场景表征优化** [GT](../sources/source_index.html#ref-GT)

- **第五批：同场景配对、空间与专家库** [LIB](../sources/source_index.html#ref-LIB)

- **S1：训练课程比较** [S1](../sources/source_index.html#ref-S1)

- **S2：力度与正式主结果** [S2](../sources/source_index.html#ref-S2)

- **S3：前端消融及必要重复** [S3](../sources/source_index.html#ref-S3)

- **S4：IL／NAVSIM v2完整迁移** [S4](../sources/source_index.html#ref-S4)

- **S5：合成RL续训与最终Base** [S5](../sources/source_index.html#ref-S5)

- **S6：续训后完整重建修复** [S6](../sources/source_index.html#ref-S6)

- **S7：正式代表原图部署成本** [S7](../sources/source_index.html#ref-S7)

精读方法时再看GEO/OPS、CORE/ATRAIN/FTRAIN；核对主表时看MAIN/RLNUM、V2MAIN/HARD、MIXED，避免仅凭HTML转述数字。[GEO](../sources/source_index.html#ref-GEO) [OPS](../sources/source_index.html#ref-OPS) [CORE](../sources/source_index.html#ref-CORE) [ATRAIN](../sources/source_index.html#ref-ATRAIN) [FTRAIN](../sources/source_index.html#ref-FTRAIN) [MAIN](../sources/source_index.html#ref-MAIN) [RLNUM](../sources/source_index.html#ref-RLNUM) [V2MAIN](../sources/source_index.html#ref-V2MAIN) [HARD](../sources/source_index.html#ref-HARD) [MIXED](../sources/source_index.html#ref-MIXED)

**这次独立核对实际做到了哪一层**

读取项目指南、研究导航、用户指定五报告及关联机制/迁移/成本报告；从保存的逐场景五噪声结果复算了**156组不同的模型×人口**（共158次检查），检查选择与预测argmax一致、人口唯一性、均值及训练seed SD，并完成**374次来源哈希校验**。v2官方EC复评分以既有逐场景误差为0的重放回执交叉核对；本次没有重新生成轨迹，也没有重新运行NAVSIM评分器或GPU实验。

09:44旧交付回执的HTML哈希早于09:48报告再生成；当前resources/REPORT_FINALIZATION.json与五份正式HTML均匹配。以当前登记和原始数值证据为准，不能把旧回执的版式哈希差异误判成结果不一致。
