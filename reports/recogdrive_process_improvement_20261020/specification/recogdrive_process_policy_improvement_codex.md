# ReCogDrive 过程策略改进与蒸馏：Codex 完整执行指令

你负责在真实 ReCogDrive/NAVSIM 工程中完成下一轮算法开发、规模实验、过程级策略改进、强 RL 对照及独立审计。目标是提高单个学生 DiT 的实际规划性能，不是为了得到正向结论而堆叠模块。

## 0. 固定任务范围与研究目标

仓库：https://github.com/IDayday/VLA-AD
基准提交：560bb257cd493613130f4b3643a0a2e4a88a7115
基准分支：research/recogdrive-opd-20261020
新分支建议：research/recogdrive-process-improvement-20261020
本地优先检查：/mnt/project/VLA-AD；实际路径以环境审计为准。

当前模型设定：
- Student 从 ReCogDrive-2B-IL 初始化。
- Teacher 使用 ReCogDrive-2B-RL 的 active action_head，绝不误用 old_policy。
- VLM 完全冻结，师生使用相同的原始 VLM hidden states。
- 训练范围是整个基础动作规划器，包括其可训练投影、状态编码器、DiT、动作解码器；不是只训练最后一层，也不是默认仅训练 LoRA。
- 原始 VLM 特征相同，不代表动作头投影后特征或 DiT 内部特征相同。
- 保留五步原生 DDIM、8×3 轨迹、原始归一化、原始输入与推理协议。
- 部署仅保留一个学生，输出一条轨迹；不增加搜索、教师、评分器、未来信息或额外推理模块。

本轮必须回答：
H1：扩大独立场景覆盖与训练曝光后，普通 OPD 能迁移多少教师能力？
H2：在学生访问的去噪状态上，真实回报是否能支持优于原教师的局部转移？
H3：局部改进投影回原 DiT 所使用的转移均值后，收益是否仍然存在？
H4：学生能否学到这种过程级改进，并在训练外提高 PDMS、控制严重退化？
H5：相同物理评分预算和总计算预算下，它与原生 GRPO 谁更有效？

本轮不默认加入：VLM/BEV 特征蒸馏、DiT 层间对齐、异构轨迹教师、GT 伪标签训练、教师方向外推、PCGrad、价值网络、BranchGRPO、世界模型或新的动作头。
这些不属于当前主路线。不得因结果不理想临时全部加入。

最终高分轨迹只用于评价去噪转移，不能作为本轮新方法的伪 GT 训练目标。

## 1. 工作区、资源与历史保护

1. 阅读当前 AGENT.md/AGENTS.md、已有资源调度策略及 OPD 文档。
2. 记录当前分支、HEAD、dirty 状态、远端、环境、GPU、磁盘与数据。
3. 从指定基准提交建立独立 worktree；原工作区、旧协议、旧报告和旧实验产物只读。
4. 本指令明确覆盖旧项目“必须训练 JEPA/VGGT”等无关算法目标，但不覆盖其安全、数据保护和资源权限约束。
5. 本轮建立新协议，不能修改旧协议后声称原实验已采用新设置。
6. 不自动接入商业付费算力，不终止或暂停其他人的任务，不自行运行填卡/压力程序。
7. 只使用已授权的空闲资源；多机地址、GPU 数量、路径必须现场核验。
8. 权重、数据、全量缓存、大型轨迹文件不进入 git；提交配置、代码、测试、精简统计、图表与产物索引。
9. 不向我反复确认已知信息。能够本地探测的路径、依赖、配置自行查清；确实缺失则明确 BLOCKED，并继续完成其他独立任务。
10. 不得停在设计文档或 dummy smoke。环境允许时运行真实训练、NAVSIM 评分和完整评价。

## 2. 先读源码与参考方法，再实现适配

必须检查基准中的：
navsim/agents/recogdrive/opd/checkpoint_io.py
navsim/agents/recogdrive/opd/conditioning.py
navsim/agents/recogdrive/opd/ddim_adapter.py
navsim/agents/recogdrive/opd/losses.py
navsim/agents/recogdrive/opd/trainer.py
scripts/opd/train.py
scripts/opd/scoring.py
scripts/opd/splits.py
configs/opd/base.yaml
reports/recogdrive_opd_20261020/07_research_conclusions.md

必须核验以下原始来源与实际公开代码：
- DiffusionOPD：https://arxiv.org/abs/2605.15055
  https://github.com/ali-vilab/DiffusionOPD
- VIDD：https://arxiv.org/abs/2507.00445
  https://github.com/divelab/VIDD
- SVDD：https://arxiv.org/abs/2408.08252
  https://github.com/masa-ue/SVDD-image
- MPO：https://arxiv.org/abs/1806.06920
- DPPO：https://arxiv.org/abs/2409.00588
  https://github.com/irom-princeton/dppo

VIDD 的公开部分涉及 DNA/蛋白离散扩散；SVDD-image 涉及图像模型。
复用它们的 roll-in/roll-out、候选组织、价值加权及更新思想，不把离散 categorical loss 或图像 posterior-mean 奖励近似直接搬到 ReCogDrive。
DPPO 仅作为扩散策略优化参考，不假定换用它就自动获得精确的原生采样密度。

建立 references.lock.json，保存 URL、实际 commit、许可证、读取文件与本地源码 hash。
建立 SOURCE_REUSE.md，逐项标明：直接复用、仅包装、数学适配、未采用。
保留版权；不要引入整套生物模型、Stable Diffusion 或第二套 NAVSIM 依赖。
若参考源码不可访问，记录失败；数学适配必须有独立测试，不能写“已复现官方算法”。

## 3. 重建训练总体，不再继承 4115 场景上限

旧 4115 场景保留为 legacy_4k 对照，不作为新正式训练总体。

从官方 navtrain token/log 定义与原始日志重建可用场景，逐项统计：
官方成员、重复 token、观测缺失、历史长度不足、真实模型/评分所需未来日志不足、地图缺失、metric cache 缺失、其他排除。
不要把 103288 或 85000 写成断言；实际总数取决于本地数据与合法过滤。
不要因为旧缓存不存在就永久删除一个本应合法的场景；允许重建缓存。
不继承其他世界模型实验额外未来帧需求造成的不必要筛选。

划分规则：
1. 保留旧 dev 的1696场景及其父驾驶记录作为训练外审计组。
2. 保留旧 cal，并从剩余非 legacy_4k 父记录中，按固定哈希增加校准数据。
3. 新 cal 目标至少约3000场景、20条父驾驶记录；不足时如实记录实际覆盖。
4. 新 train 排除所有 cal/dev 父驾驶记录。
5. navtest 不参与划分、教师目标构建、超参数选择或训练筛选。
6. 建立新 train_full；按父驾驶记录固定哈希构建嵌套 train_25k、train_50k，达到目标规模时允许整条父记录带来的轻微超额。
7. legacy_4k 尽量完整保留在新 train 内；若发现真实跨用途重叠，优先保证隔离，记录排除，不能隐藏改变。
8. 数据选取不使用 PDMS、失败标签或教师优势；主训练不做困难场景筛除。

生成可版本化的 membership、来源 hash、排除清单和可重建脚本。
旧 navtest 已经被查看，后续仍可做统一最终确认，但不能称为首次完全盲测。
新增数据均可能被基础 IL/RL 预训练见过；只能称“未参与本轮新增训练”。

## 4. VLM 缓存与输入契约

复用旧 official_v2 输入兼容与 parity，不改变系统提示词、padding、token 顺序、图像预处理。
不能裁掉原始动作头实际使用的 padded hidden states，不能缓存学生可训练 feature_encoder 的输出。

全量 hidden states 可能很大，采用分块、只读、可淘汰的原始 VLM 特征缓存：
- 所有实验共享相同观测缓存。
- cache key 包含场景 token、VLM 权重 hash、输入契约、dtype 与预处理版本。
- 当前代码未知的新字段先审计再允许进入模型。
- 教师/学生动作头的 conditioning 分别计算；不能错误共享可训练投影后的结果。
- 数据缓存缺失时按需重建，支持断点与校验。
- 缓存容量根据已有授权磁盘余量登记，不要求一次生成 TB 级全量缓存。

对真实场景执行在线 VLM 与缓存输入、动作头输出逐元素或明确容差内的一致性测试。

## 5. 算法定义与理论边界

使用正向执行索引 i=0,...,K-1，K=5；实际 diffusion time 从 head.ddim_t 读取。
不能把执行索引 i 当成物理轨迹时间，也不能把 i 当成 diffusion training timestep。

记：
S_theta：当前可训练学生。
T0：不可变的官方 RL 教师。
R：冻结的局部评价/教学参考策略，第一版 R=T0。
A0：原始 IL anchor，仅用于明确的对照。
c：可见观测的原始 VLM hidden states、ego状态和历史。
s_i=(c,z_i,i)：学生访问的当前去噪状态。
u：执行当前一步后、最终动作 clamp 之前的下一噪声状态。

定义普通后续价值：
Q_R(s_i,u)=E[R_PDMS(c,tau) | 当前下一状态为u，余下去噪均由R完成]。

理想化局部改进：
q*(u|s_i) ∝ p_R(u|s_i) exp(Q_R(s_i,u)/eta)。
它对应 max_q E_q Q_R - eta KL(q||p_R) 的解。

以下边界必须写入代码文档：
- Q_R 是期望最终 PDMS，不是 VIDD 的 eta*log E exp(reward/eta) soft value。
- 默认不混用两种价值定义；soft-value 变体第一轮关闭。
- 原生 clipped-normal 噪声含边界原子质量，不是截断后重新归一化的 truncated Gaussian。
- 原生最终动作也有 clamp；转移均值 MSE 是代理目标，不声称精确有限 Gaussian KL。
- 理想分布局部改进的性质，不保证有限采样、均值投影、学生训练或未见场景必然改进。
- 本研究是 generation-level on-policy；固定日志并不构成车辆持续环境交互。

## 6. 拓展原 DDIMAdapter，不重写数值核心

保留已验证的 p_mean_variance、normalize/denormalize、sigma floor、noise clamp、最终 clamp。
增加以下能力：

transition_stats(head, context, z_i, i, kernel_contract)
  返回 raw proposal mean、实际 sampling std、x0 prediction、所有裁剪元数据。

sample_next(stats, raw_noise)
  严格按当前 kernel_contract 执行噪声裁剪与 mean+std*noise。

rollout_suffix(head, context, start_state, next_index, raw_suffix_noises)
  从已完成当前一步的状态开始，执行 next_index,...,K-1；不重复当前一步。
  next_index==K 时只执行原生最终 clamp/denorm，不再调用 DiT。

sample_rollin(student, raw_context, noise_plan)
  无梯度生成当前学生完整 chain，并带上 student_version。

要求：
- Chain.states 保留原生定义。当前 adapter 的最后 raw state 与最终 trajectory 的 clamp 后状态必须区分。
- 中间步骤不随意 clamp 到[-1,1]，最终 clamp 不提前，也不重复改变语义。
- sigma 使用真实采样 std，不使用 get_logprobs 的替代标准差。
- 训练与推理 kernel 明确标识 native_inference / native_grpo，不静默互换。
- 默认过程目标使用 native_inference，与旧 E3 一致。
- 保存未经裁剪的随机噪声，并由同一函数实施裁剪，避免双重裁剪或重采样。
- teacher、reference 全部 no_grad；student roll-in no_grad；重新计算学生预测时启用梯度。
- 不能复用 no_grad 下计算的学生可训练 conditioning 做梯度更新；必须重新计算。

实测 full rollout 与任意位置 split+suffix 在相同随机数下完全一致。

## 7. 候选转移及真实后续回报

对每个选中的 s_i：
1. 用 R 得到 mu_R、sigma_R。
2. 采样 M 个下一状态候选，默认 M=8。
3. 每个候选从下一执行步 i+1 开始，由同一个 R 完成余下生成。
4. 最终恢复物理坐标后，调用统一 NAVSIM scorer。

默认使用 antithetic 候选：
- 采样 M/2 组标准高斯 eps。
- 使用 eps 与 -eps，并执行相同的对称原生 noise clamp。
- 每个候选的边际分布仍匹配原生噪声；候选之间相关，不冒称 iid。
- 同时实现 iid 采样用于数值对照。

数组契约：
raw_context: 原始可见条件；不包含GT、score或未来对象。
z: [B,H,D]，H=8,D=3。
mu/std: [B,H,D] 或可明确广播到该形状。
u_candidates: [B,M,H,D]，存 raw next states。
fit_suffix_noise: [B,L_fit,K,H,D]，绝对执行索引，当前步之前不使用。
fit_scores: [B,M,L_fit,F]，F使用明确的评分字段表。

默认 L_fit=2；局部诊断用 L_fit=8。
同一个状态的所有 M 候选共享每一组后续噪声，即 common random numbers。
不同 suffix replicate 相互独立；不同场景不无意共享随机数。
最后一步没有随机后续路径，重复 suffix 不能伪装成独立证据；应检测并去重评分，但保留逻辑查询计数。

第一版不使用：奖励梯度、预测价值网络、scorer网络、单次x0预测代替完整续采样、最终轨迹伪GT。

## 8. 随机数隔离与评分正确性

建立独立的 stateless RNG key：
(study_id, train_seed, student_version, reference_version, scene_token,
 occurrence_id, denoise_index, namespace, draw_id)。

至少隔离以下 namespace：
rollin / candidate / fit_suffix / target_guard / diagnostic_test /
reference_promotion / final_eval。

不得把 rank、worker 数或 shard 编号作为决定场景随机噪声的必要语义字段。
用显式 Generator/noise tensors，不依赖 Python hash()，避免线程中修改全局随机数。

三个随机数层次必须分开：
A. fit：计算候选价值及目标权重。
B. guard：检查由 A 得到的投影均值，允许决定是否回退。
C. diagnostic_test：仅用于独立评价 A+B 整体程序，不参与筛选、权重或回退。

不得用挑选候选的同一批回报报告“真实提升”。
同一状态的多个分支不能当成多个独立场景做显著性检验。

评分必须复用 scripts/opd/scoring.py 的候选独立 reference normalization：
- scalar 与 batch 等价。
- 候选顺序变化、增加无关候选、分片方式变化不改变原候选评分。
- 不共享可变 scorer 对象进行不受控多线程调用；使用独立进程/实例。
- PDMS范围[0,1]，统一正式分项权重；错误、缺失与合法零分区分。
- 外部评分结果返回后核验 scene/token/candidate/request ID，不能错配异步结果。
- 模型产生非法轨迹，正式评测按预先定义的失败规则计入；工程评分异常不能静默当作0继续训练。

## 9. 构造改进权重与中心化均值目标

对每个状态：
Q_j = mean_l fit_score[j,l].PDMS
w_j = softmax((Q_j-max_j Q_j)/eta)

默认 eta_base=0.05；校准候选为[0.02,0.05,0.10]。
PDMS不除以场景回报标准差，避免把几乎没有价值差异的状态人为放大。

控制有效样本量：
ESS=1/sum_j(w_j^2)。
默认 ESS_min=M/2。
从 eta_base 开始，只允许增大 eta，直到 ESS>=ESS_min；使用确定性的二分/扩张求解。
ESS约束属于本项目有限样本稳定化处理，不宣称来自原始论文的固定配置。

当所有Q相等或差异在明确数值容差内时，直接返回 mu_R，并标记 FLAT_VALUE。
不因奖励全0或全1删除场景，该场景仍接受普通过程蒸馏。

计算：
delta = rho * sum_j[(w_j-1/M)*(u_j-mu_R)]
默认rho=0.5；校准rho仅比较[0.25,0.5]。

性质与边界：
- 等权时delta严格为0。
- antithetic且数值对称时，样本均值等于mu_R；rho=1时与经验加权均值一致。
- iid情况下中心化是控制项，不是精确的有限样本KL投影；必须如实标注。
- M=1没有相对改进信息，直接退回mu_R。

对delta施加标准差尺度的幅度限制：
a = sqrt(mean_{H,D}[(delta/std_actual)^2])
delta <- delta * min(1, radius/(a+epsilon))
默认radius=0.25。
std_actual为真实采样std；为0的维度delta必须为0，不能靠替代floor制造探索。
mu_proposed=mu_R+delta。
目标计算用FP32；guard注入目标时必须遵循原生采样器的dtype提升/舍入路径。记录requested_delta与实际realized_delta，不能在FP32中验证收益，却让学生在不同精度路径上无法实现该目标。

这里的radius是局部位移约束，不是原生裁剪核的精确KL半径。
不能把mu_proposed直接clamp到[-1,1]，因为它是raw proposal mean，不是最终物理轨迹。

## 10. 用独立噪声检查“均值投影后”是否仍然有效

guard不评价之前被加权的候选本身，而是比较两个实际转移核：
U_base = mu_R + std_actual * clip(eps_guard)
U_plus = mu_proposed + std_actual * clip(eps_guard)

两组共享当前步噪声与后续R噪声，完整续采样并评分。
默认 L_guard=2，诊断时可取8；这只是有限样本工程检查，不是逐状态95%安全保证。

默认接受规则：
mean(PDMS_plus-PDMS_base) > 0.002；
mean(NC_plus-NC_base)>=-1e-8；
mean(DAC_plus-DAC_base)>=-1e-8；
mean(TTC_plus-TTC_base)>=-1e-8。

通过：mu_target=mu_proposed。
未通过：mu_target=mu_R。

只检查一组预先配置的rho，不针对同一guard噪声反复试探多种rho直到通过。
关闭guard的变体只作为明确消融；主方法保留该检查。
若guard过于保守导致几乎没有修正，记录原因，并用预登记的无guard对照和更大独立诊断样本分析，不能静默降低门槛。

重要：回退的是“改进量”，不是教师监督本身。
所有正常场景、所有去噪步骤仍有参考策略目标，绝不恢复旧E5的稀疏OPD mask。

## 11. 分开评价三类收益，禁止把搜索成绩冒充学生成绩

局部诊断必须分别评估：
P0：原参考转移核 + 参考suffix。
P1：按fit权重重采样候选u_j，再用独立suffix评价。
P2：中心化均值投影、guard后的目标核 + 独立suffix。
P3：训练后学生自己的完整单链、无搜索输出。

P1/P2都是训练期教学程序/局部分析，不是部署结果。

diagnostic_test默认16组独立随机数；遇到无后续随机性的步骤，不伪造重复独立样本。
使用多个独立candidate banks复核有限候选选择波动。
P1可以直接计算sum_j w_j*Q_test(u_j)，不需要为了统计再引入一次分类重采样噪声。
P0/P2使用新当前步噪声；P0与P2严格配对。

记录：
- P1-P0：价值选择的机会。
- P2-P0：可部署转移族的局部机会。
- P1-P2：投影损失，明确它是不同局部过程的配对统计，不宣称严格KL误差。
- 学生相对T0、普通OPD与P2教学程序的差距。
- 同一候选分别用R和冻结学生snapshot续采样的回报差异，用于诊断suffix mismatch，默认不改变训练定义。

在动态生成的新状态上逐步构造P2教学链的全程搜索评价，只对少量固定诊断场景执行；不能用一次静态chain上算好的所有目标冒充一个有效闭环生成程序。

## 12. 训练目标与一个完整optimizer step

主损失保持过程级转移均值MSE：
L = mean_{b,i,H,D}[(mu_student - stopgrad(mu_target))^2]。

默认每个场景每次更新只选一个去噪执行索引，均匀轮换/随机选取；其余步骤使用原参考目标。
- 所有五步都计算学生和参考预测。
- 选中一步执行第7–10节的回报改进。
- 未选中或改进不通过的步骤都回到普通OPD。
- 明确默认每次20%的去噪位置有机会被改进，不将其说成每次都改进全部五步。
- 不除以“通过guard的数量”，不因通过少就放大梯度。
- 不做未声明的1/selection_probability重要性放大。
- all_steps变体允许单独配置，但必须报告约5倍的目标构建预算变化。

伪代码：

for optimizer_step in training_plan:
    raw_context = next_native_observation_batch()
    student_version = version(student)

    with no_grad():
        chain = student_adapter.sample(raw_context, explicit_rollin_noise)
        reference_means = reference_adapter.evaluate_chain(chain, raw_context)
        targets = reference_means.clone()
        selected = select_one_step_per_scene_independent_of_reward()
        for state_group in selected:
            candidates = sample_reference_next_states(state_group)
            Q_fit = score_complete_reference_suffixes(candidates, fit_noises)
            proposed = centered_projected_target(Q_fit, candidates, reference_stats)
            accepted = independent_guard(proposed, guard_noises)
            targets[state_group] = where(accepted, proposed, reference_means[state_group])

    assert version(student) == student_version
    student_means = student_adapter.evaluate_chain(
        chain, raw_context, require_current=True, grad_enabled=True)
    loss = mean_square_float32(student_means, targets.detach())
    backward(loss)
    aggregate_gradients_correctly()
    clip_global_grad_norm()
    optimizer.step()
    record_and_increment_student_version()

不通过完整roll-in反传，不通过物理模拟器反传，不把终局轨迹当action target。
同一个optimizer step的梯度累积期间student/reference版本不变；更新后不能复用旧学生chain冒充新on-policy数据。

默认不缓存过程目标跨多个optimizer更新。允许缓存不可变raw observation和物理metric cache。
若日后实验采用off-policy replay，必须单列方法，不能沿用“严格当前学生roll-in”的标签。

## 13. Teacher/Reference更新与可选能力保持

第一版固定R=T0，先完成受控结果。

reference promotion只在ITERATIVE变体启用：
- 每个预先登记的完整epoch结束后最多申请一次；同一epoch不反复挑checkpoint。
- student、旧R、T0在固定cal上用配对噪声比较。
- 使用单独的cal-promotion日志分组，避免用同一小组既反复调参又验收参考更新。
- 只有学生相对旧R平均PDMS提升达到0.002且预定父记录bootstrap下界>0，同时NC/DAC/TTC和高分退化满足非劣约束，才复制active head为新R。
- 未通过则保持旧R，继续训练，不终止本轮研究。
- 新R的哈希、版本、所有目标来源写入manifest；过期目标失效。
- 原始T0始终保留，不被覆盖；不做默认EMA或权重混合。

能力保持第一轮只保留E4-A作为单独对照。
若主方法确有平均增益但严重退化明显，再运行预先登记的REPLAY变体：
- 从train独立筛选噪声确定稳定高分池；不使用dev/navtest场景形成保护池。
- 固定10% replay曝光，替换而非额外偷偷增加每批总曝光。
- 参考行为使用固定IL或T0，须按实验配置单选；不得根据测试逐场景挑最好参考。
- loss权重在train/cal校准后锁定。
- 所有额外样本和前向计入预算。

ITERATIVE和REPLAY不是初始主方法必需模块，不在首轮混合全部开启。

## 14. 训练规模与基线计划

### 14.1 先恢复可信的规模基线

必须完成旧E0-IL、E0-RL、E0-COPY加载与推理parity。
旧E3/E4-A用旧batch8/1000steps做历史复现检查，不拿其直接当新方法唯一对照。

新正式训练以场景曝光、有效epoch和optimizer updates三者共同记录。
默认global_batch=32，显存不足用micro-batch与梯度累积保持global batch；只有在资源资格检查阶段统一决定后才能改变。
历史复现继续用global_batch=8，不混淆两个协议。

数据规模诊断首先对E3运行：
A. legacy_4k：2、8 epoch端点。
B. train_25k：1、2、4 epoch端点。
C. train_full：1、2 epoch端点；若尚未收敛且预登记预算允许，继续到4 epoch。
D. 固定64000次场景曝光：比较legacy_4k、train_25k、train_full；明确最后一项不等于遍历完整数据。

不要为每个端点重新随机训练；同一预登记训练run保存对应端点。
规模诊断先用seed42，最终完整数据E3和选中的E4-A配置做42/43/44三个训练种子。
样本量不足的先导结果不得阻止完整数据基线训练。

若实际full约100000场景、batch32：2 epoch约6250次更新；这是预算示例，实际步数按manifest计算。
不能沿用train_steps=1000限制全量训练。

### 14.2 优化器

新OPD基线及主方法使用相同AdamW、weight_decay=1e-4、梯度裁剪1.0。
学习率仅比较1e-5与3e-5，统一校准预算；不要自动按GPU数放大学习率。
固定短warmup后恒定学习率作为初始协议，便于不同预算端点比较；若采用cosine，则各对照共享完全相同且预登记的schedule定义。
混合精度根据原parity决定；loss/权重/梯度统计至少FP32，评分FP64。
不能根据前100步得分就断定整个学习率不合适；新校准至少覆盖train_25k的一次完整曝光，并记录完整曲线。

## 15. 主实验矩阵与预算公平

统一方法ID：
B0_IL：原IL。
B0_RL：原RL。
B0_COPY：RL权重复制。
B1_OPD_FULL：全量普通E3。
B2_ANCHOR_FULL：全量E4-A。
P1_COLD：IL初始化、固定R=T0的过程改进蒸馏。
P2_WARM：充分训练的OPD初始化、固定R=T0的过程改进蒸馏。
C1_CONTINUE_OPD：与P2同一warm checkpoint，继续普通OPD。
R1_IL_GRPO：IL初始化的原生GRPO。
R2_WARM_GRPO：与P2同一warm checkpoint的原生GRPO。
R3_RL_GRPO：原RL初始化继续GRPO。
P3_ITERATIVE：P2基础上的可选reference promotion。

“充分训练OPD”的warm recipe在cal上按预先规则从B1/B2中统一选择，所有种子使用同一个recipe和相同epoch端点，不能每个种子挑最好版本。
不能只将seed42的最好warm模型用于全部后续种子。

必须优先完成：B0、B1、B2、P2、C1、R1、R2、R3。
P1至少做真实中等规模比较；资源允许时做完整确认。
P3仅在固定参考版本得到可信提升后扩大正式训练。
某个P方法局部诊断失败，不得阻止B/R方法继续运行。

三种预算分别报告：
1. 同场景曝光/更新预算：比较过程监督本身的作用。
2. 同训练期物理评分预算：比较P与GRPO的回报利用效率。
3. 总计算预算：含VLM提取、warmup、教师查询、候选续采样、guard、调参、评价及原IL/RL共同前置成本说明。

评分预算初始端点：每个种子250000、1000000、4000000次实际candidate评分请求。
以完整optimizer update为原子停止点，允许至多一个update的明确预算越界；所有方法记录真实数而非偷偷抹平。
同时记录逻辑请求数、去重/cache命中后物理执行数；不能靠重复调用或缓存命中伪装公平。

默认P一个选中状态需要M*L_fit+2*L_guard=20次逻辑评分，额外诊断不混入此式；最后一步去重节省另计。
GRPO通常每场景G条候选，G默认8；若baseline内部还有额外评分，按实际计入。
P2/R2的warm阶段必须计入总成本，不得只报告后续成本。
C1除相同曝光对照外，再给出不超过P2总GPU预算的普通OPD延长训练对照，避免新方法仅靠更多计算取胜。

原生GRPO先保持代码与默认reward的复现。
公平主比较另运行canonical-PDMS reward版本，明确只改分项聚合权重，不改采样/advantage/logprob/BC；其余条件相同。
10/5/2原生训练reward与5/5/2正式PDMS分别命名，不混算结果。
保留原生概率近似并披露，不以本轮为由重写整个RL算法。
旧E6的E5 admission约束不得用于本轮新实验。

## 16. 局部诊断、校准与科学判断

先在固定512个train场景上诊断，按父记录哈希取样而不是只取已知可修复场景。
同时使用IL、T0以及已有OPD学生的roll-in状态，分开报告，不混为一种状态分布。

默认诊断：M=16，L_fit=8，L_guard=8，L_test=16。
每个场景覆盖全部5个去噪位置；不足或退化位置明确标注。
随后在独立cal子集上用同样程序验证，参数不依据dev/navtest调整。

初始只比较6个(eta_base,rho)组合：
eta_base∈{0.02,0.05,0.10}，rho∈{0.25,0.5}。
共用候选库和随机数，不因为多配方重复产生不同候选而失去配对性。
每个组合只运行一次guard，并在独立test噪声上评价完整程序。
参数选择使用cal-tune，正式诊断结论使用保留cal-confirm子组；dev仅在训练方案锁定后审计。cal-tune、cal-confirm、cal-promotion按父驾驶记录划分，不能按相邻场景随机拆分。

至少实施以下消融：
- UNIFORM_TARGET：不构造改进，等于普通OPD。
- SHUFFLED_VALUE：状态内打乱candidate与Q的对应，保留同样数值分布和计算成本。
- NO_GUARD：相同目标但不做guard，仅作为消融。
- IID_VS_ANTITHETIC：只做局部估计稳定性诊断，避免不必要的完整训练矩阵。

如果P1局部选择有收益但P2均值投影无收益，结论是当前投影方案不成立；不靠扩大训练掩盖。
如果P2有收益但小规模学生学不到目标，检查mean可实现范围、clamp饱和、梯度和训练覆盖。
如果学生训练内学得到但训练外失败，归入泛化/状态分布问题。

科学门槛不是“第一次小样本CI不显著就否定方向”：
- 工程错误：阻断受影响正式训练，修复后重跑。
- 证据不足：仅允许一次预登记的诊断增样，例如1024场景或L_test=32，所有比较同步增样。
- 明确负向：停止该投影变体的昂贵扩展，继续独立基线和RL对照。
- 正向：进入中等规模训练，然后正式规模。

最终性能优先；假设失败可以保留负结果，不为论文叙事调整有利阈值。

## 17. 训练外评价与成功标准

正式部署评价只用native单条学生轨迹，K=5，禁止oracle候选筛选、过程搜索或额外teacher/scorer。
对T0也使用同一评价协议。

三种独立训练种子42/43/44；五次配对推理种子与旧协议保持可追踪关系。
稳定高分集合的筛选噪声与最终评价噪声分离。

报告：
PDMS、NC、DAC、TTC、EP、Comfort、DDC；
每次输出零分率、五次均值零分率；
M_plus、M_minus、原高分→零分、正分→零分；
teacher正分/student零分、双方失败后的修复；
单次推理时间、峰值显存、参数量；
全部训练/评分/查询成本。

M_plus和M_minus按每个独立训练策略的场景均值变化先截正负再平均，不先把多个策略平均成不存在的ensemble。
统计以父驾驶记录为主要cluster，同时保留完整log敏感性分析；训练seed和配对推理seed的结构保持一致。
不能把相邻场景、同一candidate bank或五个推理种子当作大量独立训练实验。

锁定主比较：
P2 vs C1；P2 vs R2；P2 vs T0。
其余为次要分析，不通过选择性展示制造显著性。
对多个主比较说明多重检验处理；至少给出效应量、配对CI和未经/经过调整的判断。

科学成功分开定义：
- 过程改进：P2局部均值核在独立噪声下有正向收益。
- 迁移改进：学生优于同预算普通OPD，实质性门槛0.002。
- 超教师：最终学生相对T0 PDMS均值提高至少0.002且主配对95%CI下界>0。
- 安全保持：不能仅凭总分宣布成功；相对对照的NC/DAC/TTC下降、严重零分退化必须单独判定。

初始平均安全分项非劣容差0.001，高分→零分比例允许增加上界0.001；这些是研究判据，不是真实道路安全保证。
若PDMS提升但安全性恶化，明确写“性能—安全交易”，不称无代价改善。
若CI不足以确认，不写已经突破教师。

navtest只对锁定的最终候选与对应强对照执行完整12146场景确认，实际token清单现场核验。
不循环利用navtest逐轮调参；旧测试已被查看的历史在报告披露。

## 18. 工程接口、存储与分布式

建议新增独立包，不覆盖旧opd：
navsim/agents/recogdrive/process_improvement/
  contracts.py
  suffix_rollout.py
  rng_plan.py
  candidate_value.py
  improvement_target.py
  projection_guard.py
  objective.py
  reference_manager.py
  budget_counter.py

scripts/process_improvement/
  audit.py
  build_splits.py
  prepare_cache.py
  diagnose.py
  calibrate.py
  train.py
  train_grpo.py
  evaluate.py
  analyze.py
  export_student.py
  run_suite.py

configs/process_improvement/
  base.yaml
  scaling.yaml
  local_diagnostics.yaml
  opd_full.yaml
  anchor_full.yaml
  process_cold.yaml
  process_warm.yaml
  process_iterative.yaml
  grpo_controls.yaml
  protocol.yaml

tests/process_improvement/
reports/recogdrive_process_improvement_20261020/

TargetRecord至少保存：
scene_token、student/reference版本、执行索引与实际timestep、raw z、mu/std、
candidate IDs、fit评分、温度、ESS、rho、radius clipping、guard评分与接受原因、
最终mu_target、各随机数namespace、评分/输入契约hash。

全量训练仅保留紧凑统计与可重建noise keys；完整候选链按固定hash抽样保留，并完整保留所有错误案例。
不要重复保存海量原始hidden states或所有suffix激活。

分布式：
- 独立进程持有自己的可变simulator/scorer。
- B、M、L、K维度flatten/unflatten必须有对应单元测试。
- 全局loss按真实样本/有效时间步总数归约，不按每卡局部通过guard数平均。
- 一个rank没有接受目标，也正常贡献普通OPD梯度；避免collective不一致。
- 禁止跨线程修改同一个head的临时forward方法。
- 可用单场景GRPO微批累积兼容旧接口；若扩展到多场景，必须和逐场景结果对齐。
- 保存optimizer、scheduler、AMP scaler、每rank RNG、epoch/sampler游标、reference版本与评分预算游标。
- exact resume首先要求同world_size；弹性恢复如未验证则明确不支持，不能宣称exact。
- 原子写checkpoint/结果；重复作业需拒绝覆盖或使用新run_id。

## 19. 必须通过的测试

A. 模型与旧协议：
- T0 active权重严格加载；old_policy排除；IL/RL-COPY parity。
- 关闭新方法完全恢复旧E3 loss、梯度及更新。
- teacher/reference/anchor权重与buffer保持不变。
- raw VLM共享，但student可训练投影有梯度。

B. 采样与续采样：
- full vs prefix+suffix覆盖每个切分位置。
- 最后一步不多执行、最终clamp仅在正确位置。
- 原生sigma/noise clipping、dtype与归一化一致。
- candidate边际、antithetic对称性、零std和M=1行为。
- 改变worker/shard数不改变同场景随机计划。

C. 目标与统计：
- Q全相等时target==mu_R。
- rho=0时target==mu_R。
- eta变大时权重趋于均匀、ESS非降。
- ESS约束、softmax溢出保护和reward尺度正确。
- 中心化delta对candidate排列不变。
- guard与fit/test噪声严格隔离。
- accepted==False时仍保留普通教师监督。
- iid与antithetic的有限样本性质不能混同。

D. 可解析toy问题：
- 一维高斯/裁剪高斯、已知二次回报，检查权重与均值改进方向。
- 两个高回报模式、均值位于低回报区域的反例：程序必须能够显示“分支选择更好但均值投影更差”，不得通过平均伪造改进。
- 具体投影反例可用离散位置族：原均值1，等概率偏移[-3,-1,1,3]，候选[-2,0,2,4]；奖励仅在±2附近为1。原期望奖励0.5，温度0.2的价值加权奖励约0.993，但投影均值约0.013，在相同偏移核下奖励为0。此例仅是数学单元测试，不是NAVSIM结果。
- 纯随机无信息回报：fit上的赢家优势不能被test当作真实收益。
- 有限离散支持上直接求q*与归一化权重，对照数值实现。

E. 评分与训练：
- scalar/batch、候选增删/置换、分片独立评分parity。
- 正常零分与评分异常的处理不同。
- 学生梯度有限且真实更新；selected/unselected timestep损失分解正确。
- no_grad学生context不能导致投影参数静默不训练。
- DDP归约与单卡global batch一致。
- 连续训练和中断恢复一致。
- 导出后学生不加载teacher、metric cache、未来标签或改进模块也能推理。

先CPU数学与接口测试，再真实checkpoint、真实场景，再真实小规模训练。
不得伪造通过；skipped、failed、blocked分别列明。

## 20. 默认配置和可执行入口

至少实现以下配置语义，可按仓库规范组织：

study_id: recogdrive_process_improvement_20261020
base_commit: 560bb257cd493613130f4b3643a0a2e4a88a7115
model:
  student_init: IL
  reference_init: RL_ACTIVE
  freeze_vlm: true
  use_expert_features: false
  train_action_head_full: true
sampler:
  kernel: native_inference
  deterministic: false
  steps: 5
improvement:
  enabled: true
  selected_steps_per_scene: 1
  candidate_count: 8
  antithetic: true
  fit_suffix_samples: 2
  eta_base: 0.05
  ess_fraction_min: 0.5
  rho: 0.5
  normalized_shift_radius: 0.25
  value_definition: expected_canonical_PDMS
  guard_samples: 2
  guard_gain_margin: 0.002
  guard_safety_fields: [NC, DAC, TTC]
  fallback: original_reference_mean
  cache_across_student_updates: false
training:
  global_batch: 32
  optimizer: AdamW
  learning_rates: [1.0e-5, 3.0e-5]
  weight_decay: 1.0e-4
  gradient_clip_norm: 1.0
  seeds: [42, 43, 44]
  target_full_epochs: 2
  registered_max_full_epochs: 4
  loss: all_step_mean_matching
  extra_GT_loss: 0.0
  extra_feature_loss: 0.0
reference:
  mode: fixed
  promotion_enabled: false

必须提供统一入口，例如：
python scripts/process_improvement/audit.py
python scripts/process_improvement/build_splits.py
python scripts/process_improvement/run_suite.py --phase parity
python scripts/process_improvement/run_suite.py --phase scaling
python scripts/process_improvement/run_suite.py --phase local_diagnostics
python scripts/process_improvement/run_suite.py --phase calibration
python scripts/process_improvement/run_suite.py --phase main
python scripts/process_improvement/run_suite.py --phase rl_controls
python scripts/process_improvement/run_suite.py --phase final_eval
python scripts/process_improvement/analyze.py

这些是需要实现并真实验证的接口，不是声称仓库里已经存在的命令。
支持--dry-run、--resume、--run-root和--resource-policy；dry-run不算完成实验。
整个suite不能因为一个独立方法未通过科学门槛就终止所有其他方法。

## 21. 执行顺序与最终交付

执行顺序：
1. 审计、reference源码核验、原生parity、数据总体重建。
2. 扩大普通OPD训练与局部改进诊断并行准备。
3. 实现完整candidate→suffix→value→projection→guard→process-loss路径。
4. 通过toy、真实checkpoint、真实NAVSIM与训练集成测试。
5. 运行中等规模过程方法和原生GRPO对照。
6. 完成全量基线与有依据的新方法正式训练。
7. 锁定最终候选，执行完整dev/navtest评价。
8. 独立审查统计、数据泄漏、预算、导出和结论。

报告必须包含：
- 旧4k/1000步设置与新规模结果的差别。
- 普通OPD是否收敛，教师迁移差距随数据/曝光如何变化。
- 原参考、价值重采样、均值投影、guard后目标、最终学生五者的分离结果。
- 目标接受率、改进位置覆盖、ESS、clamp饱和、后续价值方差、suffix mismatch。
- P2相对同预算普通OPD、GRPO与T0的结果。
- 安全分项、高分→零分与正向收益共同分析。
- 训练评分预算、总算力、推理成本和缓存成本。
- 源码复用与新适配范围，不虚称原论文复现。
- 所有失败、无效运行和NOT_RUN，保留不利结果。

结论状态分别使用SUPPORTED / NOT_SUPPORTED / INCONCLUSIVE / NOT_RUN。
区分工程完成与科学成功；局部目标更好不等于学生已经超过教师。

最后检查git diff、秘密信息、权重/数据误提交、测试与报告溯源。
提交独立分支，具备权限时正常push并核验远端SHA，不force push。
最终回复给出真实完成阶段、结果表、关键结论、未完成原因、报告位置和直接复现命令。

请开始检查、开发、测试和执行可用实验，不要只复述本指令。
