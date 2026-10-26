# ReCogDrive RL 第一轮无路由表征蒸馏实验指南

版本：1.0  
日期：2026-10-27（Asia/Tokyo）  
任务性质：代码开发、完整蒸馏训练、完整验证与测试；不能以冒烟、预实验或提交训练作业作为结束。  
实验规模：4 组 × 2 个独立训练种子 × 40 epochs；每个 run 为 53,640 次 optimizer updates。

## 0. 执行摘要与不可变要求

本轮直接学习连续修复表征。学生为原 ReCogDrive-2B-RL 加一个可训练的全上下文低秩适配器，所有场景执行同一网络：

\[
H_S=H_0+M\delta_\theta(H_0,s),\qquad
\delta_\theta\in\operatorname{span}(U_{D48}).
\]

原 VLM、1536→384 投影、RL DiT 和状态编码器全部冻结。新网络每场景仅输出一个 48 维连续系数，经固定 basis 解码成一个共享 384 维修复量。它作用于有效 image/text token，原 padding 不变。

**用户明确排除路由。本轮禁止开发以下部署逻辑：**

- expert ID 分类器、softmax 专家权重、MoE、argmax 选专家；
- 是否修复的预测门、置信度阈值、输入相关的启停分支；
- 多候选轨迹、选轨 scorer、测试评分后择优；
- 根据测试选择表进行推理。

统一范数限幅只是数值/幅度约束，不是专家路由。零修复由网络在相应监督样本上直接学出。原 expert ID 仅能用于离线构造教师和事后分组统计，不能进入学生输入、损失分类头或部署状态。

“增益 > 1 PDMS 点”的规则只用于提前生成一个固定的训练目标版本；部署时没有 PDMS、没有该阈值、没有 Base/修复二选一。

### 本轮四组

| 组 | 配置名 | 教师目标 | 损失 | 功能监督查询状态 |
|---|---|---|---|---|
| A | H_full | 原始 D_C8_double 选定教师 | H | 无 |
| B | H_gain1 | 原教师相对 Base 增益严格 >1 点才保留修复，否则目标为 Base | H | 无 |
| C | H_mu_teacher | 与 B 相同 | H + DDIM transition-mean | 冻结教师去噪链 |
| D | H_mu_student | 与 B 相同 | H + DDIM transition-mean | 当前学生去噪链 |

A/B 比较目标构造；B/C 比较功能监督；C/D 比较去噪状态分布。四组采用相同学生结构、初始化规则、有效 batch、优化器和总训练长度。

每组训练种子为 20261027、20261028，所有 8 个 run 都必须完成。两个训练种子不等于 S0–S4 五个动作采样种子。

本指南给出足量且可比较的优化预算，不承诺特定 PDMS 增益。若完整训练后效果为负，也必须完成规定评测、保存参数和解释结果。不能通过提前停组、替换成 Base 或仅展示最优组来隐藏负结果。

## 1. 固定资产、已核对事实和证据边界

### 1.1 源仓库与模型

- 仓库：https://github.com/sjc786526-coder/-recogdrive-representation-teacher-2
- 固定提交：6bb7119583513094a169b4c2f2dc2a05ac293aa3
- Base：本仓库实际使用的 ReCogDrive-2B-RL。
- checkpoint SHA256：23fb949b10387efeff87e093703d5faf41b61b9e5ead16d4076661e3c7c08a20。
- 以 configs/repair_final_baseline.json 核对 VLM、prompt/tokenizer、采样器和评分器身份。
- 保留 assets/、decisions/、reference/、manifests/ 及现有 vendor/ 冻结。新增代码置于独立目录，以导入/包装方式接入；不为方便覆盖原教师实现。

当前环境若只有文档和代码、没有真实权重/数据/计算资源，不能声称已经训练。负责执行的 Codex 应定位当地已有资产并继续运行，只有确实缺少无法自行定位的必要资产时才报告具体阻塞。

### 1.2 数据

| 数据 | 场景数 | 会话数/用途 |
|---|---:|---|
| Train | 85,813 | 130 logs；全部用于正式训练 |
| Validation | 17,475 | 32 logs；与 Train 会话隔离 |
| Test | 12,146 | 独立测试 |

Train 与 Validation 合计 103,288。用 token 连接所有资产，global index 不是 CSV 行号。禁止将相邻场景随机拆成新的训练/验证集替代原 log 切分。

当前教师库在训练侧重拟合，Train 选择为 in-sample 教师标签，不是 OOF 教师。Validation 已被历史研究使用过，应称“会话留出验证”，不能声称首次盲测。

### 1.3 教师结构

- VLM hidden 宽1536，feature_encoder 输出宽384。
- 修复发生在 feature_encoder 之后、DiT 之前。
- 当前库八个专家共用 D48；六个固定中心方向、两个小 MLP。
- 同场景所有有效 image/text token 使用同一个 shift；padding 不变。
- 同场景 S0–S4 使用同一教师接口。
- 当前 D_C8_double 的 alpha=2、raw_shift_factor=2、cap_D0=2、cap_factor=2，因此当前统一 RMS cap 为 4D0。运行时从实际资产读取并断言，不把 4 当作适用于其他库的常量。
- basis 的 center 只用于专家输入坐标。把系数解码为“修复量”时不能再加 center。
- 精确教师以 Library.selected_target / Library.targets 的输出和 actual_role_shifts 为准；requested shift 不能冒充最终实际目标。

### 1.4 已重算的训练统计

原始教师 Train PDMS=97.3053145，Base=94.9163155，非 Base 选择45,659/85,813=53.2076%。

严格仅保留增益 >1 PDMS 点：

- 修复31,511/85,813=36.7205%；
- 零修复54,302例；
- 新教师 Train oracle=97.2233742；
- 保留原教师总增益96.5701%；
- 比原教师少0.0819404个总体 PDMS 点。

这是保存的训练评分矩阵查表重算，不是新学生成绩，不说明测试集也保留同样比例。程序必须从全部11个 pdms_*.npz 自行重算并与这些数字核验。

原始 Train choice 计数：
[40154,1059,1174,2000,3215,2667,17265,14976,3303]。

NPZ mean_PDMS_score 单位0–1，CSV分数单位0–100。1 PDMS点等于NPZ中的0.01，必须严格区分。为了固定边界实现，以 float64 的100倍分数构造 gain_points 并比较 >1.0，保存最终布尔清单；其余代码只读取清单，不重复采用不同精度比较。

### 1.5 最重要的代码事实

正式教师导出通过 agent.forward → action_head.get_action 执行，不能用 sample_chain 替代。

| 项 | get_action 正式路径 | sample_chain |
|---|---|---|
| DDIM std 下限 | 默认1e-4 | min_sampling_denoising_std，默认0.04 |
| 每步噪声裁剪 | 默认±1 | randn_clip_value，默认±5 |
| 返回梯度 | 原函数本身不强制detach | 返回detach |
| 本轮用途 | 忠实包装并重放 | 不作为正式过程监督来源 |

其他事实：

- DDIM五步[80,60,40,20,0]，epsilon prediction，eta=1，deterministic=False。
- 初始Gaussian不裁剪；每步新噪声裁剪；整条链最后对normalized action做最终裁剪，再denorm。
- 正式DDIM最后一步仍保留最小std=1e-4，不能按理论零方差自行关掉噪声。
- p_mean_variance 已接受post384条件，不能再调用feature_encoder把post384投影一次。
- 原prepare()左padding到max_length=2800；实际token布局以每例原输入为准。
- DiT存在未mask的vl_features.mean(1)，不能删除原padding或仅凭loss mask认定padding无影响。
- history编码器输入12维；status编码器输入8维。适配器额外可读历史4×3、status8、command4，共24维。实际字段从现有builder取，不猜顺序。

## 2. 第一轮研究问题与完成标准

本轮回答：

1. 当前最终教师在受D48约束的连续学生中，能吸收多少表征修复？
2. 减少低收益修复是否有助于完整自然分布上的保持与泛化？
3. 对齐冻结DiT的去噪作用是否改善表征误差向PDMS收益的转化？
4. 学生去噪状态监督与教师状态监督有何差别？

本轮不同时引入VLM解冻、改变采样步数、额外BEV/OCC任务、RL奖励优化或不同教师库合并。它们不是本次8个正式任务的替代品。

完整完成状态：
ASSET_READY → TARGETS_READY → CODE_CHECKS_PASS → TRAINING → TRAIN_COMPLETE → FULL_VAL_COMPLETE → SELECTION_LOCKED → TEST_COMPLETE → REPORT_COMPLETE。

只有最后状态算完成。代码可运行、loss下降、某个小集涨分、训练已提交或进程仍在后台，都不是任务完成。

## 3. 建议新增代码布局

以下为需要Codex实现的文件和CLI接口，不是声称仓库已经包含这些命令。

~~~text
student_distillation/
  __init__.py
  asset_identity.py
  build_targets.py
  data.py
  feature_cache.py
  adapter.py
  native_bridge.py
  noise.py
  losses.py
  calibration.py
  distributed.py
  train.py
  evaluate.py
  select_checkpoint.py
  aggregate.py
  run_suite.py
  checkpointing.py
  experiment_registry.py
configs/student_distillation/
  common.yaml
  H_full.yaml
  H_gain1.yaml
  H_mu_teacher.yaml
  H_mu_student.yaml
tests/student_distillation/
  test_target_units_and_split.py
  test_adapter_identity_and_cap.py
  test_native_bridge_replay.py
  test_input_gradients.py
  test_ddp_objective.py
  test_checkpoint_resume.py
scripts/student_distillation/
  run_round1.sh
docs/recogdrive_no_routing_distillation_round1_guide.md
outputs/student_distillation_round1/  # 不提交大缓存/权重到Git
~~~

不要修改原 Library 的数学定义。扩展 validation 数据读取时，在新模块实现只读population reader；使用原冻结库对Validation评分的结果另存输出目录，不覆盖decisions/。

每个run必须保存完整effective_config，不能只保存有继承引用但无法恢复展开值的YAML。

## 4. 教师目标构造和数据契约

### 4.1 建立两份冻结目标清单

版本名建议：

- teacher_full_D_C8_double_6bb7119_v1；
- teacher_gain1_D_C8_double_6bb7119_v1。

两版共享冻结H0和原selected shift。gain1版仅把不满足gain_points>1的actual shift置零；不重新选其他专家，不改变原力度，不把六库合并。

实际API的actual_role_shifts形状为[selected_count,2,384]。当前库必须逐值断言image/text两个role的shift完全相同，再取单个384维向量；不把requested_shifts、池化表征之差或两个role的平均值冒充该向量。重建H_T必须与Library.targets返回的interfaces逐值核对。若未来库的两role不再相同，本轮协议应报资产不匹配，而非静默平均。

伪代码：

~~~python
q_points = 100.0 * np.asarray(mean_PDMS_score, dtype=np.float64)
choice = q_points.argmax(axis=1)  # 仅离线教师标签，列0为Base，精确并列优先Base
gain_points = q_points[np.arange(len(choice)), choice] - q_points[:, 0]
repair_gain1 = gain_points > 1.0
# 与保存choice逐行核对，按global index连接token；不能把index当行号。
# teacher_full使用原selected_target；teacher_gain1在repair_gain1=False时取原H0。
~~~

为每个token保存：

~~~text
token, log_id, original_global_index, population
teacher_version, original_choice, gain_points, target_is_zero
source_checkpoint_sha256, library_sha256, basis_sha256
prompt/tokenizer/token_layout_hash, actual_Base_interface_sha256
actual_shift_fp32[384], image_mask[L], valid_text_mask[L], valid_mask[L]
token_ids[L]或可校验的等价布局记录
history_trajectory[4,3], status_feature[8], high_command_one_hot[4]
native_sequence_length, actual_target_RMS, cap_RMS
~~~

original_choice只用于来源与分组；模型dataset输出给网络时剔除choice、分数、token字符串和其他可能泄漏标签的字段。token仅由数据管理和stateless RNG使用。

不要把场景索引、文件顺序、log_id、GT未来轨迹、PDMS或专家编号作为适配器输入。

### 4.2 缓存策略

首选一次性构建冻结post384的FP32缓存，复用现有一致的VLM hidden缓存也可。缓存必须恢复原2800布局及padding值；不要只存valid token后在训练时补零，因为原padding hidden不是任意零值。

典型2800×384×4bytes约4.3MB/场景，Train post缓存约0.37TB（十进制估算，不含额外元数据，实际取决于长度）。导出前预估磁盘并检查可用空间。使用分片连续数组/mmap、独立offset索引及校验；不要每次训练随机解压成千上万个大pickle。

如没有足够磁盘：

1. 优先复用已有冻结hidden缓存，在线做原投影；
2. 或由冻结VLM按需生成H0，缓存热分片；
3. 保持相同目标和训练曝光，不能改成只训练可缓存子集。

不需要为每个场景保存九份完整表征，也不需要为S0–S4存五份相同教师接口。actual shift约384个FP32即可重建选定教师接口。

第一轮不进行改变输入图像、视角或prompt而不重算教师目标的随机增强。

### 4.3 原导出器的开销优化

现有export_teacher.py会生成Base和教师各五条轨迹。H-only数据构建应新增representation_only模式，只生成H0、mask、实际shift和元数据；不要为A/B无端运行十条轨迹。

C可缓存教师去噪状态和均值；D必须使用当前学生产生状态。两者共享同一H0缓存和目标清单。

### 4.4 Validation

现有DecisionTable不支持validation，Validation也无当前六库选择标签。

区分两个任务：

- **学生PDMS验证**只需要原观测与metric cache，不需要任何oracle目标。
- 本轮最终报告需要Validation表征误差与教师上限，因此必须在新验证入口用冻结Base和冻结八算子，对全部17,475个Validation场景独立评分，固定每场景S0–S4均分选择并生成full和gain1两版诊断教师。gain1仍采用同一预注册的严格>1点规则，不能用Validation另选阈值。结果严禁进入训练梯度、适配器输入或训练归一化。

主训练流程不能因为全量Validation教师构造耗时而停在导出阶段。先可完整训练与评价学生，Validation oracle诊断作为同轮独立任务调度；必须在REPORT_COMPLETE之前完成并清晰注明事后教师身份。注册表单列VAL_TEACHER_DIAGNOSTICS状态，不能漏掉后仍宣称整轮完成。

Train的gain1清单只由Train标签生成。验证、测试的选择表不参与该清单构造。

## 5. 连续D48适配器的实现

### 5.1 固定结构

- 4个learned queries，hidden=256。
- 2个cross-attention block，每个4 heads。
- FFN expansion=4，GELU，dropout=0。
- token输入为原post384，固定训练统计标准化；image/text role embedding。
- 24维普通状态经小MLP投到256，加入queries。
- queries读有效token；读出拼接4×256，经MLP输出48维。
- 末层Linear的weight和bias为0，其余正常初始化。
- 不增加gate、classifier、expert embedding或softmax分支。
- 输出广播到有效token，原padding保留。
- 新模块FP32，原动作头FP32且eval；原VLM保留正式权重与精度。

token标准化使用Train固定样本统计，按channel记录mean/std；不是动态依赖teacher的标准化。统计可用确定性抽取4096个Train场景计算，记录token清单和SHA。state统计也只来自Train。方差接近零的维度使用预定义std下限。

### 5.2 系数与幅度

用无量纲q：

\[
\delta_\text{raw}=D_0\sqrt{384}\,(qU^\top).
\]

先检查实际FP64 basis的UᵀU接近I，再加载固定FP32副本用于学生。若U正交：

\[
\operatorname{RMS}(\delta)=D_0\|q\|_2.
\]

当前库cap=4D0，所以将q的L2范数限制到4即可。运行时仍复核原生FP32加法之后的actual RMS，容差与教师一致：
max(1e-7, cap_RMS*1e-4)。

**零初始化处必须保留梯度。** 原native_cap在nominal==0时把scale设0；直接照搬到学生会让零输出陷入零梯度。学生零点的缩放系数应为1。

### 5.3 参考实现骨架

以下是供实现和当地验证的参考代码，所有数值行为仍要用真实资产完成第12节检查。

~~~python
import math
import torch
from torch import nn

class ReadBlock(nn.Module):
    def __init__(self, width=256, heads=4):
        super().__init__()
        self.q_norm = nn.LayerNorm(width)
        self.m_norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, dropout=0.0, batch_first=True)
        self.ff_norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, q, memory, valid):
        key_value = self.m_norm(memory)
        a, _ = self.attn(
            self.q_norm(q), key_value, key_value,
            key_padding_mask=~valid, need_weights=False,
        )
        q = q + a
        return q + self.ff(self.ff_norm(q))


class ContinuousD48Adapter(nn.Module):
    def __init__(self, basis, d0, token_mean, token_std,
                 state_mean, state_std, cap_in_d0=4.0):
        super().__init__()
        assert basis.shape == (384, 48)
        self.register_buffer("basis", basis.detach().float().clone())
        self.register_buffer("d0", torch.as_tensor(d0, dtype=torch.float32))
        self.register_buffer("token_mean", token_mean.detach().float().clone())
        self.register_buffer("token_std", token_std.detach().float().clamp_min(1e-3))
        self.register_buffer("state_mean", state_mean.detach().float().clone())
        self.register_buffer("state_std", state_std.detach().float().clamp_min(1e-3))
        self.cap_in_d0 = float(cap_in_d0)
        self.input_proj = nn.Linear(384, 256)
        self.role_embed = nn.Embedding(3, 256)  # padding/image/text；不是专家编号
        self.queries = nn.Parameter(torch.randn(1, 4, 256) * 0.02)
        self.state_encoder = nn.Sequential(
            nn.Linear(24, 256), nn.GELU(), nn.Linear(256, 256)
        )
        self.blocks = nn.ModuleList([ReadBlock(), ReadBlock()])
        self.output_norm = nn.LayerNorm(4 * 256)
        self.output_hidden = nn.Sequential(nn.Linear(4 * 256, 256), nn.GELU())
        self.output_projection = nn.Linear(256, 48)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, base_post, image_mask, text_mask, state24):
        assert base_post.dtype == torch.float32
        assert base_post.ndim == 3 and base_post.shape[-1] == 384
        assert image_mask.dtype == text_mask.dtype == torch.bool
        valid = image_mask | text_mask
        assert not bool((image_mask & text_mask).any())
        assert bool(valid.any(dim=1).all())

        roles = image_mask.long() + 2 * text_mask.long()
        normalized = (base_post - self.token_mean) / self.token_std
        memory = self.input_proj(normalized) + self.role_embed(roles)
        state = (state24.float() - self.state_mean) / self.state_std
        q = self.queries.expand(base_post.shape[0], -1, -1)
        q = q + self.state_encoder(state).unsqueeze(1)
        for block in self.blocks:
            q = block(q, memory, valid)
        features = self.output_hidden(self.output_norm(q.flatten(1)))
        coefficients = self.output_projection(features)  # [B,48]

        # clamp必须在sqrt之前，且零点scale=1。
        norm = coefficients.square().sum(-1, keepdim=True).clamp_min(1e-12).sqrt()
        scale = (self.cap_in_d0 / norm).clamp(max=1.0)
        coefficients_capped = coefficients * scale
        delta = (coefficients_capped @ self.basis.T) * (self.d0 * math.sqrt(384.0))

        cap = self.d0.double() * self.cap_in_d0
        tolerance = torch.maximum(cap * 1e-4, cap.new_tensor(1e-7))
        for _ in range(9):
            student_post = base_post + valid.unsqueeze(-1).to(base_post.dtype) * delta.unsqueeze(1)
            with torch.no_grad():
                difference = student_post.double() - base_post.double()
                numerator = (difference.square() * valid.unsqueeze(-1)).sum((1, 2))
                denominator = valid.sum(1).double() * 384.0
                actual_rms = (numerator / denominator).sqrt()
                over = actual_rms > cap + tolerance
                if not bool(over.any()):
                    break
                correction = torch.where(
                    over, (cap / actual_rms.clamp_min(1e-300)).clamp(max=1.0),
                    torch.ones_like(actual_rms),
                )
            # 数值舍入校正系数不求导；delta路径仍然可微。
            delta = delta * correction.to(delta.dtype).unsqueeze(1)
        else:
            raise RuntimeError("Post-rounding RMS cap did not converge")

        return {
            "post": student_post,
            "delta": delta,
            "coefficients_raw": coefficients,
            "actual_rms": actual_rms,
            "valid_mask": valid,
        }
~~~

实现时不要把base_post.detach()扩展成整个forward的no_grad。base_post本身可来自冻结缓存，适配器必须有梯度。不要调用Library.targets生成学生结果，它是教师API且含detach/no_grad。

记录cap触发率；大量样本长期饱和可能意味着输出尺度或优化异常，但这项统计不是科学早停门槛。

## 6. 从post384出发的原生DDIM桥接

### 6.1 建议接口

~~~python
class NativeDDIMBridge:
    def encode_context(self, history12, status8):
        """忠实复用冻结his_traj_encoder与ego_status_encoder。"""

    def mean_from_post(self, actions, step_index, post, context):
        """返回p_mean_variance的mean/logvar/x0；不再feature_encoder。"""

    def states_from_post(self, post, context, noise_tape, stop_before=None):
        """复现get_action，返回执行各步之前的state和最终trajectory。"""
~~~

桥接层应导入真实冻结head，保持head.eval()与requires_grad_(False)。不要使用agent.train()，避免进入原SFT/GRPO训练分支。

history编码采用原his_traj_encoder(history12.unsqueeze(1)).repeat(1,8,1)，status直接进入原ego_status_encoder。遵循真实builder的张量形状。

### 6.2 噪声带

正式评价使用原scene_seed(token,j)：
int(sha256(f"{token}|20261008|{j}".encode()).hexdigest()[:8],16)。

为避免训练只记住评价五种噪声，训练使用独立namespace的5个noise banks：

~~~python
import hashlib

def train_noise_seed(token: str, training_seed: int, bank: int) -> int:
    payload = f"NRKD_train_v1|{training_seed}|{token}|{bank}"
    return int(hashlib.sha256(payload.encode()).hexdigest()[:8], 16)
~~~

bank=0..4，两组C/D同训练seed同token同bank使用相同噪声带。评价时不得使用这个namespace冒充S0–S4。

为了原生CUDA RNG一致性，按原形状[1,8,3]连续抽6次：1次初始Gaussian、5次更新噪声。不要把它改成一次randn([6,1,8,3])。用fork_rng或独立可验证的generator隔离数据shuffle和训练随机状态；时间步采样不能消耗动作噪声RNG。

### 6.3 参考桥接骨架

~~~python
import torch

class NativeDDIMBridge:
    def __init__(self, frozen_head):
        self.head = frozen_head.eval().requires_grad_(False)
        assert self.head.config.sampling_method == "ddim"
        assert self.head.ddim_steps == 5
        assert self.head.ddim_t.tolist() == [80, 60, 40, 20, 0]

    def encode_context(self, history12, status8):
        with torch.no_grad():
            history = self.head.his_traj_encoder(history12.unsqueeze(1)).repeat(1, 8, 1)
            status = self.head.ego_status_encoder(status8)
        return history, status

    def mean_from_post(self, actions, step_index, post, context):
        history, status = context
        batch = actions.shape[0]
        t = self.head.make_timesteps(batch, self.head.ddim_t[step_index], actions.device)
        index = self.head.make_timesteps(batch, step_index, actions.device)
        # 不加no_grad：student post需要经冻结DiT获得输入梯度。
        return self.head.p_mean_variance(
            actions, t, index, post, history, status, deterministic=False
        )

    def states_from_post(self, post, context, noise_tape, stop_before=None):
        # noise_tape为6个原生形状tensor；noise_tape[0]不裁剪。
        current = noise_tape[0].clone()
        states = []
        means = []
        std_floor = float(getattr(self.head, "eval_min_sampling_denoising_std", 1e-4))
        noise_clip = float(getattr(self.head, "eval_randn_clip_value", 1.0))
        for step_index in range(5):
            if stop_before is not None and step_index == stop_before:
                return current
            states.append(current)
            mean, logvar, _ = self.mean_from_post(current, step_index, post, context)
            means.append(mean)
            std = torch.exp(0.5 * logvar).to(post.dtype).clamp(min=std_floor)
            noise = noise_tape[step_index + 1].clamp(-noise_clip, noise_clip)
            current = mean + std * noise
        clip_value = getattr(self.head, "final_action_clip_value", 1.0)
        if clip_value is not None:
            current = current.clamp(-clip_value, clip_value)
        trajectory = self.head.denorm_odo(current)
        return {"states": states, "means": means, "trajectory": trajectory}
~~~

该桥接没有修改scheduler，也没有将eta改成0；eta应从正式模型核验为1。每步的std下限包括最后一步。原代码的原地clamp若影响autograd，可使用数值等价的非原地实现，但要验收前向一致性，不能变更裁剪顺序。

### 6.4 C/D怎样生成查询状态

C：

1. 使用冻结gain1教师接口和训练noise bank生成教师链。
2. 使用第j步执行前的a_j。
3. 教师mean可缓存；student mean必须重新计算并保留对H_S的梯度。

D：

1. 使用当前H_S.detach()生成链，到第j步之前停止，不必跑剩余步骤。
2. 将该a_j固定，不反传状态生成过程。
3. 同一a_j、同一j，分别以H_T和有梯度的H_S查询均值。
4. 教师分支no_grad；学生分支有输入梯度。
5. 每次访问场景刷新状态，不能把长期陈旧student states当on-policy。

D是学生状态分布上的局部蒸馏/半梯度目标，不宣称完整轨迹KL的无偏梯度。当前采样含噪声/动作裁剪，本轮使用mean MSE，不把未裁剪高斯KL写成实现策略的精确KL。

## 7. 损失、定标和课程

### 7.1 表征损失

\[
\ell_H(i)=
\frac{\operatorname{Mean}_{\ell\in V_i,c}\!\left[(H^S_{i\ell c}-H^T_{i\ell c})^2\right]}
{s_H^2}.
\]

定义统一固定s_H²为原始full教师Train非零修复的mean(actual_shift²)，从全部训练compact shifts计算。四组共用；保存数值与SHA。若数值为0或非有限，属于资产错误；不能静默加一个很大的epsilon掩盖。

分母可使用max(s_H²,1e-12)保护，并报告保护是否生效。它是全训练固定尺度，不是逐样本、逐batch或随学生变化的归一化。

Base目标的H_T就是H0，分母仍同一固定值，不能除以零目标自身范数。

### 7.2 功能损失

\[
\ell_\mu(i,j)=
\frac{\operatorname{Mean}_a\!\left[
\left(\mu(a_j,j,H^S_i)-\operatorname{sg}\mu(a_j,j,H^T_i)\right)^2
\right]}{s_{\mu,j}^2}.
\]

C/D共用s_mu[j]，在正式训练前用Train固定校准集计算：

- 2048个gain1非零修复样本，按Train logs覆盖确定性抽取；
- 冻结Base链的状态；
- 训练noise banks 0、1；
- 五个时间步全部覆盖；
- 计算同点Base mean与gain1 teacher mean的MSE。

令raw_s_mu[j]为上述平均值，固定：
s_mu²[j]=max(raw_s_mu[j], 0.1*mean(raw_s_mu), 1e-6)。

单位是模型normalized action空间。报告每步原始值、最终值和下限触发情况。若五步全部被数值下限主导，先核对单位、目标是否全零、状态/条件是否错误；如确认真实修复影响就是很小，仍记录并按预定配置训练，不在查看验证结果后放大loss。

只从Train定标。不能用Validation/Test的教师反馈设定loss尺度。

### 7.3 总目标与保持

A/B：L=mean_i ell_H(i)。

C/D：L=mean_i [ell_H(i)+lambda_mu(u)*ell_mu(i,j)]。

本轮additional_keep_weight=0。所有明确零修复的场景已以Base表征和Base功能为目标；不再加入第三个相反的保持loss，也不把所有Base>90的场景强制拉回Base。

各组主目标按自然场景等权。每epoch全量遍历一次，不按expert ID等权重采样，不做基于测试比例的再平衡。分组统计用于理解失败，不用于本轮自适应改权重。

### 7.4 C/D功能项课程

更新计数u从0开始：

- u<5364：lambda_mu=0（4 epochs H warmup）。
- 5364≤u<8046：线性从0增长到0.5（2 epochs）。
- u≥8046：lambda_mu=0.5（后34 epochs）。

C/D各自从与A/B相同seed的初始适配器和空Adam开始，不从验证挑出的B checkpoint启动。本轮实现先不做跨实验共享warmup恢复，以简化审计。

### 7.5 时间步与训练噪声安排

功能监督开启后令r=epoch_index-4，使用稳定hash生成每场景偏移：

~~~text
time_index  = (r + hash_time(token, training_seed)) mod 5
noise_bank  = (floor(r / 5) + hash_noise(token, training_seed)) mod 5
~~~

在25次访问中覆盖5×5组合，场景间偏移避免同batch全在一个时间步。C/D安排相同。每次访问只监督一个时间步，以控制计算量；正式功能诊断覆盖全部五步。

Python内置hash受进程随机化影响，不能使用；用sha256派生整数。

### 7.6 单场景训练骨架

~~~python
def scene_loss(batch, adapter, bridge, cfg, update, calibration):
    h0 = batch["base_post"].float()       # frozen FP32, native layout
    valid = batch["image_mask"] | batch["text_mask"]
    delta_t = batch["teacher_actual_shift"].float().detach()
    h_t = (h0 + valid.unsqueeze(-1).to(h0.dtype) * delta_t.unsqueeze(1)).detach()

    result = adapter(h0, batch["image_mask"], batch["text_mask"], batch["state24"])
    h_s = result["post"]
    diff = (h_s - h_t).square()
    per_scene_h = (diff * valid.unsqueeze(-1)).sum((1, 2))
    per_scene_h = per_scene_h / (valid.sum(1) * 384.0)
    loss_h = per_scene_h.mean() / calibration["h_mse_scale"]
    weight_mu = cfg.mu_weight(update) if cfg.enable_transition_loss else 0.0
    if weight_mu == 0.0:
        return loss_h, {"h": loss_h.detach()}

    context = bridge.encode_context(batch["history12"], batch["status8"])
    j_value = batch["time_index"]
    if isinstance(j_value, torch.Tensor):
        assert j_value.numel() == 1
        j = int(j_value.item())
    else:
        j = int(j_value)
    assert 0 <= j < 5
    tape = batch["noise_tape"]
    with torch.no_grad():
        if cfg.state_source == "teacher":
            a_j = bridge.states_from_post(h_t, context, tape, stop_before=j)
        elif cfg.state_source == "student":
            a_j = bridge.states_from_post(h_s.detach(), context, tape, stop_before=j)
        else:
            raise ValueError(cfg.state_source)
        teacher_mean, _, _ = bridge.mean_from_post(a_j, j, h_t, context)

    student_mean, _, _ = bridge.mean_from_post(a_j.detach(), j, h_s, context)
    loss_mu = (student_mean - teacher_mean.detach()).square().mean()
    loss_mu = loss_mu / calibration["mu_mse_scale"][j]
    return loss_h + weight_mu * loss_mu, {
        "h": loss_h.detach(), "mu": loss_mu.detach()
    }
~~~

代码实现应将mask、shape、device和finite检查集中封装；不能在训练器里不断隐式CPU/GPU拷贝。collator优先把time_index、noise_bank保留为CPU Python整数，避免默认collate生成一维Tensor后传给原make_timesteps。若C采用缓存，仍须验证与上述在线教师查询逐值一致。A/B的enable_transition_loss必须为false；加入配置检查，确保其全部53,640次更新都不调用功能损失桥接，而非仅在最初4epoch保持关闭。

## 8. 正式超参数与完整预算

### 8.1 每个run固定预算

- 训练数据：85,813个真实场景/epoch。
- epochs=40。
- effective_global_batch=64。
- 每epoch optimizer updates=ceil(85,813/64)=1341。
- total_updates=53,640。
- 真实场景曝光=3,432,520/run。
- 4组×2seed=8个run，共27,460,160次真实场景曝光。
- 补齐global batch的零权重占位不计入真实曝光。

40epochs和5万余次更新用于避免将短期优化失败误判为不可学；不把固定步数当作性能保证。

### 8.2 优化器

| 项 | 默认值 |
|---|---|
| AdamW peak LR | 1e-4 |
| betas | [0.9,0.999] |
| eps | 1e-8 |
| weight_decay | 0.01 |
| no_decay | bias、LayerNorm、learned queries、最终output_projection |
| global grad clip | 1.0 |
| LR warmup | 1341 updates（1 epoch） |
| LR schedule | warmup + cosine |
| minimum LR | 1e-5 |
| dropout | 0 |
| EMA | 关闭，避免多一套未对照权重 |
| adapter/action head | FP32 |
| TF32 | 关闭 |
| mixed precision | 首轮关闭新增模块与动作头的autocast |

不对VLM整体float()改变正式Base的数值路径。训练无需更新VLM，可只保留已验证的冻结缓存与动作头。

每次梯度裁剪记录裁剪前范数、裁剪比例和非有限计数。默认配置不能根据某个验证点负分而静默改动。

### 8.3 DDP与梯度累积

DDP只包装适配器。每GPU原生scene microbatch=1，通过累积实现global batch64。

| world size | microbatch/GPU | accumulation |
|---:|---:|---:|
| 1 | 1 | 64 |
| 2 | 1 | 32 |
| 4 | 1 | 16 |
| 8 | 1 | 8 |
| 16 | 1 | 4 |

资源更多时可并行独立run。选择world size使64可整除，不自动增加global batch或线性放大学习率。

no_sync()应覆盖对应microstep的forward和backward。对于完整64场景batch，常规loss/accumulation配合DDP均值即可。

最后一个global batch只有53个真实场景：

- 给所有rank安排同样数量的microsteps；
- 多出的11个slot使用有效样本形状的零权重占位；
- 全局按53个真实样本平均，不能按64稀释，也不能把重复补齐样本当有效训练样本。

更一般地，如果一个global update有N_valid个真实场景，world size为W，每个有效scene的本地backward系数为W/N_valid；DDP再除以W，得到全局真实场景均值。不要在此基础上再除一次accumulation。

scheduler只在optimizer.step后更新，不能每个microstep更新。epoch步数、global_step和真实scene曝光分别记录。

### 8.4 完整common.yaml参考

~~~yaml
protocol:
  name: recogdrive_no_routing_round1_v1
  repo_commit: 6bb7119583513094a169b4c2f2dc2a05ac293aa3
  library: D_C8_double
  student_kind: continuous_shared_d48
  teacher_is_fixed: true
  route_or_gate: false

data:
  train_scenes: 85813
  validation_scenes: 17475
  test_scenes: 12146
  split_key: token
  preserve_native_token_layout: true
  source: verified_post_cache
  input_augmentation: none
  natural_scene_weights: true

model:
  interface_dim: 384
  subspace_rank: 48
  query_count: 4
  hidden_dim: 256
  cross_attention_layers: 2
  attention_heads: 4
  ffn_expansion: 4
  state_dim: 24
  dropout: 0.0
  zero_init_output: true
  coefficient_parameterization: l2_dimensionless
  cap_from_library: true
  freeze_vlm: true
  freeze_feature_encoder: true
  freeze_action_head: true

training:
  seeds: [20261027, 20261028]
  epochs: 40
  global_batch_size: 64
  microbatch_per_device: 1
  updates_per_epoch: 1341
  total_optimizer_updates: 53640
  last_batch: padded_slots_zero_weight_normalize_real_count
  early_stopping: false
  automatic_scientific_stop_on_negative_pdms: false

optimizer:
  name: adamw
  lr: 0.0001
  betas: [0.9, 0.999]
  eps: 0.00000001
  weight_decay: 0.01
  no_decay_bias_norm_queries_output: true
  grad_clip_norm: 1.0

scheduler:
  name: warmup_cosine
  warmup_updates: 1341
  total_updates: 53640
  min_lr: 0.00001

precision:
  adapter: float32
  action_head: float32
  loss: float32
  tf32: false
  autocast: false

loss:
  representation_weight: 1.0
  representation_scale: frozen_train_full_nonzero_actual_shift_mse
  transition_mean_weight_max: 0.5
  transition_start_update: 5364
  transition_ramp_updates: 2682
  transition_scale: frozen_train_calibration_per_step
  additional_keep_weight: 0.0

sampler:
  method: ddim
  timesteps: [80, 60, 40, 20, 0]
  deterministic: false
  eta: 1.0
  std_floor_source: get_action_eval_min_sampling_denoising_std
  noise_clip_source: get_action_eval_randn_clip_value
  native_scene_microbatch: 1
  train_noise_namespace: NRKD_train_v1
  train_noise_banks: 5
  evaluation_noise_indices: [0, 1, 2, 3, 4]

checkpoint:
  interval_updates: 1000
  save_every_epoch: true
  save_full_validation_epochs: [8, 16, 24, 32, 40]
  include_optimizer_scheduler_rng_sampler: true
  atomic_write: true

evaluation:
  monitor_every_epochs: 2
  monitor_scenes: 2048
  monitor_noise_indices: [0]
  full_validation_epochs: [8, 16, 24, 32, 40]
  full_validation_noise_indices: [0, 1, 2, 3, 4]
  test_checkpoints: [last, validation_best]
  new_noise_validation_indices: [5, 6, 7, 8, 9]
  bootstrap_unit: log
  bootstrap_replicates: 2000
~~~

四个覆盖配置：

~~~yaml
# H_full.yaml
experiment: H_full
teacher_target: full
state_source: none
enable_transition_loss: false
---
# H_gain1.yaml
experiment: H_gain1
teacher_target: gain1
state_source: none
enable_transition_loss: false
---
# H_mu_teacher.yaml
experiment: H_mu_teacher
teacher_target: gain1
state_source: teacher
enable_transition_loss: true
---
# H_mu_student.yaml
experiment: H_mu_student
teacher_target: gain1
state_source: student
enable_transition_loss: true
~~~

实际实现可用Hydra或普通YAML合并，但须展开并校验有效配置、总步数与冻结参数清单；不能留下未接入训练器的无效配置字段。

## 9. 性能与资源组织

### 9.1 优先优化开销，不减少实验

- A/B只需H0缓存和适配器，训练时不运行DiT。
- C教师链固定，可对每个训练seed、5个noise banks预计算state和teacher mean；只保存小型[5,8,3]状态/输出。
- D每个scene仅rollout到被监督的j步，平均不必执行完整五步；随后额外一次teacher mean和一次有梯度student mean。
- 原VLM与head无需optimizer state；DDP只同步小适配器。
- mask、state、H0分片预取到pinned memory；只使用已验证的等价缓存。
- 训练与评分分开队列，评分worker数根据CPU和metric cache吞吐决定；防止过量并发拖慢存储。
- 不用降低token数、改变DDIM步数/eta、BF16替换原生FP32或缩减数据来伪装加速。

C/D的场景曝光和更新次数相同，但FLOPs不同，应报告GPU hours、训练吞吐、数据准备时间和评分时间。不要编造尚未测量的小时数。

### 9.2 GPU未知时的调度

run_suite先检测当前可用设备、已有作业与现有队列/集群约束，只使用用户已授权且可用的资源，不终止无关作业，不创建未授权付费实例。

若仅一张卡，8个run串行；有多卡可按完整run或DDP并行。自动估计一次真实训练step及评分吞吐后给出预计剩余时间，但不把时间估计改成早停预算。

长任务交给当地已配置调度器、持久化作业系统或可恢复后台进程，并持续跟踪；不能仅提交作业就结束任务。因会话/平台硬限制必须交接时，应准确列出仍未完成任务，不写“实验已完成”。

## 10. 评价协议

### 10.1 训练过程

每epoch统计全部真实Train访问的：

- H loss，全体/零目标/非零目标分开；
- C/D的mu loss及五时间步分解；
- actual shift RMS、cap触发率、预测方向/幅度；
- 梯度范数、裁剪比例、LR、真实曝光数；
- 仅诊断的原choice来源分组，不训练类别头；
- train/validation差距。

每2epochs固定2048个Validation场景、S0做监控。监控集按log覆盖固定，不按新模型收益挑样本，不用监控小集取消正式训练。

### 10.2 完整Validation与checkpoint选择

在epoch8、16、24、32、40，对全部17,475场景执行S0–S4真实PDMS。

选择规则预先固定：

1. 每场景先平均五噪声；
2. 对全部场景等权平均；
3. 从该run五个完整训练端点中选平均PDMS最高者；
4. 精确并列取较早epoch；
5. 不把Base作为训练checkpoint候选，不因训练结果较差而替换成Base；
6. 同时保留epoch40 last。

每个run都必须训到40epoch，即使第8epoch成绩最好，也不能在那里停止。

正式评分使用configs/repair_final_baseline.json中的PDM配置，包括progress_weight=5、ttc_weight=5、comfortable_weight=2、driving_direction_weight=0，以及原proposal sampling；不要使用上游GRPOConfig里不同默认权重。

### 10.3 Test

全部8个run完成训练与规定Validation后，冻结每个run的last与validation_best SHA、epoch、选择依据和协议SHA，再进行Test。

每个run的last和validation_best均评价全部12,146场景×S0–S4。如果同一SHA，只评一次并记录共享结果。所有组都评，不只评赢家。

Test过程不读取测试choice作为学生输入，不修改模型/阈值/超参数。原Test oracle可以独立展示为参考，但须注明事后选择身份。

最后对每个run的validation_best补完整Validation S5–S9，检查未参与原教师选择的新噪声迁移。该结果在主checkpoint选择锁定后生成，不反向改变主选择。

### 10.4 Train拟合与收益吸收

每个run的last与best执行完整85,813 Train表征指标评估。

Train行为诊断使用固定8,192场景面板，S0–S4；按log×原Base五噪声均分桶（<70、[70,90]、>90）分层，配额按真实样本数比例用最大余数法分配，层内按sha256("NRKD_train_panel_v1|"+token)排序取前若干例。余数精确并列按(log_id,桶编号)排序。保存完整token清单和SHA并锁定。Base、对应教师、学生使用同一面板。明确标为面板，不写成全训练PDMS。

收益吸收率只能用同一人群、相同噪声、相同教师版本计算：

\[
\mathrm{capture}=
\frac{P_S-P_0}{P_T-P_0}.
\]

不能拿面板学生分数除以全训练97.3053的增益；gain1组也不能使用full教师作为其唯一监督吸收分母。可另外报告相对full oracle的距离，但字段名必须区分。

### 10.5 必报行为指标

- 完整总体PDMS及原生所有子项；
- 按Base五噪声均分分组：<70、[70,90]、>90；
- 场景均分改善/持平/下降数量及正收益、负损失分别合计；
- 每scene/noise的新增NC、DAC、TTC扣分、零分事件；
- 同一scene、同一noise下Base_PDMS>90且student_PDMS==0的配对事件；这与按五噪声场景均分分桶分开统计；
- 原始teacher非Base来源分组和目标keep/repair分组；
- 每log结果、最受损和最受益场景清单；
- 两个训练seed分别结果、均值与标准差。

S0–S4是同场景的动作采样重复，不是五次独立训练。两个训练seed的统计与五噪声统计分开展示。

置信区间采用log为单位的2000次成组bootstrap；对配对模型差值使用相同bootstrap抽样。只有两个训练seed时，不夸大对训练随机性的估计精度。

平均PDMS与严重退化均需报告；本轮不设置“出现一个新增零分就取消完整实验”的科学早停规则。

## 11. 保存恢复与故障处理

### 11.1 checkpoint内容

每1000 updates、每epoch末以及完整Validation端点保存：

- adapter state_dict及固定basis/统计身份；
- optimizer完整状态、scheduler状态、global_step；
- epoch、global batch位置、已经访问的真实样本数；
- 全rank Python/NumPy/torch CPU/CUDA RNG；
- shuffle permutation或可恢复生成器状态；
- 当前配置、目标清单、缓存manifest、代码commit的SHA；
- 当前功能项课程、校准值；
- optimizer参数组名和冻结/可训练参数清单。

checkpoint使用临时文件+原子rename，COMPLETE标志最后写。保存Adam/RNG等嵌套状态要deepcopy/clone，不能让恢复检查修改原快照或活训练状态。

### 11.2 恢复

中断后先核对协议和缓存SHA，再从最后完整checkpoint继续，保留累计步数，不把余下epochs重新解释成一个短run。

须有一个工程恢复检查：从同一完整checkpoint两次独立加载，使用同一下一批和噪声，核对下一次更新的参数/optimizer/scheduler及loss；不只比较模型权重文件能加载。

### 11.3 工程故障与科学负结果

工程故障包括NaN/Inf、梯度断开、teacher/student身份错误、mask错位、OOM、资源抢占、数据损坏。修复根因后继续完整计划。

科学负结果包括loss下降慢、PDMS没涨、正常场景退化、训练验证差距大。它们必须完成40epochs和正式评测，不作为停止借口。

若修复工程问题需要改变影响数学结果的配置，建立新run ID及recovery记录，保留旧异常；不能静默降低LR后把结果说成原配置。纯抢占恢复不建立新科学配置。

本轮不进行无限追加训练直到涨分。主预算固定40epochs。如40epoch末损失仍下降，报告该事实和后续候选方案；不得用它替代8个run的完成。本指南不预设额外LoRA或RL来“救活”负结果。

## 12. 必要工程检查

这些检查用于排除实现错误，通过后自动进入完整实验。不得把真实目标短期拟合不佳当作工程失败而永久停留于预实验。

1. **资产与数据检查**：verify_package.py、checkpoint/VLM指纹、token/log隔离、全部训练矩阵SHA与gain1计数。
2. **identity**：zero-init适配器输出严格零修复；原布局/padding一致；Base与学生在相同噪声下重放一致。
3. **实际教师注入**：用exported actual shift重建H_T，与Library输出逐值核对，再核对S0–S4轨迹和真实评分。
4. **bridge一致性**：get_action与新增from_post桥接在Base/教师条件下比较；检查初始噪声、每步状态、mean/std/noise clip与最终轨迹。
5. **零点梯度**：非零已知可实现目标时，第一步最终输出层梯度非零；不能因cap零点分支把梯度切断。第一步前层梯度为零可能由最后层zero-init导致，应核对第二步恢复。
6. **已知可实现正控制**：用相同适配器结构的一份小幅非零权重生成合成目标，32–64例、约200–400更新检验H拟合和保存恢复。它不是正式教师实验；通过后重新从正式初始化开始。
7. **DiT输入梯度**：C/D teacher分支无梯度；student mean对H_S和适配器有梯度；冻结head无参数梯度；状态a_j已detach且师生查询同一点。
8. **DDP目标**：同一组真实scene在单卡等效batch和多卡累积下，loss/梯度归一化一致，尤其最后53例batch。
9. **评测身份**：评分配置、轨迹单位/形状、S0–S4种子和恢复协议正确。

真实教师的128例短拟合可作为记录，但不能设置“达不到某个MSE就取消全部正式训练”的科学门槛。短检查完成不是本任务交付。

## 13. 建议CLI与执行顺序

以下命令是需要Codex实现的目标接口。路径由当地已核实资产配置给出，不能编造权重/数据位置。

~~~bash
python -B verify_package.py
python -B check_assets.py --paths paths.local.json

python -m student_distillation.build_targets \
  --paths paths.local.json \
  --library D_C8_double \
  --population train \
  --target-versions full gain1 \
  --out outputs/student_distillation_round1/targets

python -m student_distillation.feature_cache \
  --paths paths.local.json \
  --population train validation test \
  --preserve-native-layout \
  --out outputs/student_distillation_round1/features

python -m student_distillation.calibration \
  --config configs/student_distillation/common.yaml \
  --train-only \
  --out outputs/student_distillation_round1/calibration

# 由run_suite调度的独立诊断任务；可以与训练错峰/并行，不作为训练启动依赖。
python -m student_distillation.build_targets \
  --paths paths.local.json \
  --library D_C8_double \
  --population validation \
  --score-frozen-candidates \
  --noise-indices 0 1 2 3 4 \
  --target-versions full gain1 \
  --diagnostic-only \
  --out outputs/student_distillation_round1/validation_teacher

python -m student_distillation.run_suite \
  --config-dir configs/student_distillation \
  --experiments H_full H_gain1 H_mu_teacher H_mu_student \
  --training-seeds 20261027 20261028 \
  --epochs 40 \
  --global-batch-size 64 \
  --resume auto \
  --complete-through report \
  --out outputs/student_distillation_round1
~~~

run_suite需串起代码检查、8个训练任务、规定完整Validation、完整Validation教师诊断、checkpoint选择锁定、所有模型Test、新噪声验证和最终报告。Validation教师命令展示的是任务接口，不要求在shell中串行等它完成才启动训练；run_suite应自动调度并复用已完成的诊断结果。它不是只生成shell脚本的工具。

如使用torchrun，每个run的启动命令和world size写入注册表。日志必须实时记录run状态、完成epoch/update、预计剩余时间、GPU/CPU/IO瓶颈及失败恢复。

## 14. 实验注册表与产物

必须生成machine-readable registry，列明8个run，不能只记录成功项。

每个run目录：

~~~text
effective_config.yaml
PROTOCOL.json
ASSET_IDENTITIES.json
TRAINABLE_PARAMETERS.json
TARGET_IDENTITY.json
CALIBRATION.json
checkpoints/
  epoch_008.pt
  epoch_016.pt
  epoch_024.pt
  epoch_032.pt
  epoch_040.pt
  last.pt
  validation_best.json
logs/
  train_metrics.jsonl
  runtime.jsonl
validation/
  epoch_008_scene_scores.*
  ...
test/
  last_scene_scores.*
  best_scene_scores.*
diagnostics/
  full_train_representation_metrics.*
  train_panel_scores.*
  val_new_noise_scores.*
  regressions.*
COMPLETE.json
~~~

suite级别：

~~~text
EXPERIMENT_REGISTRY.json
VAL_TEACHER_DIAGNOSTICS.json
CHECKPOINT_SELECTION_LOCK.json
ROUND1_RESULTS.csv
ROUND1_RESULTS.md
ROUND1_REPORT.md
figures/
  learning_curves.*
  representation_vs_pdms.*
  repair_gain_vs_preservation_loss.*
  per_log_deltas.*
  per_step_function_error.*
reproduce_commands.sh
INCOMPLETE_OR_FAILED_TASKS.json
~~~

大权重/缓存保存在当地已授权实验存储，不提交到Git。代码、有效配置、小型统计、实验报告与复现命令可提交独立研究分支。不得覆盖main或原冻结证据。

COMPLETE.json只有在真实完成规定任务后写。若确有无法解决的资源/资产阻塞，写INCOMPLETE和具体剩余工作，不生成虚假完成标志。

## 15. 最终报告必须回答的问题

1. A的Train/Validation表征吸收是否显著不同？差距来自方向、幅度、零目标外溢还是特定来源？
2. B是否用很小教师上限损失换来更好的学生总体行为？两seed是否一致？
3. C是否在相似H误差下改善去噪功能与PDMS？还是仅降低某个代理loss？
4. D相对C的学生状态监督是否有效？额外训练开销是否值得？
5. 零修复目标是否被准确保持？没有路由时，连续网络是否在正常场景产生系统性非零shift？
6. 改善失败和损伤正常场景分别贡献多少总体分数？
7. 训练误差下降但PDMS无提升时，是否有明确的功能误差/行为阈值证据？
8. 原oracle分数与学生实测分数是否区分？新目标、Train面板和不同噪声是否使用正确分母？
9. 是否存在剩余可学习性或输入信息不足的证据？不得将一种适配器失败推广成所有表征蒸馏不可行。
10. 下一轮是否需要上游LoRA或更强输入表征？只作研究结论，不用额外方向替代本轮未完成工作。

主结果表必须列出全部组、全部seed、last及best、对应epoch、Train表征误差、Val/Test PDMS与各子项、成本和异常。

## 16. 代码开发分工

可由一个主agent协调三个并行开发agent：

- Agent 1：资产身份、Train两版目标、缓存、dataset与split防泄漏。
- Agent 2：连续D48适配器、学生cap、get_action一致的native bridge、H/mean losses。
- Agent 3：DDP训练器、固定40epoch日程、checkpoint恢复、评测/汇总。
- 主agent：先锁定公共接口和配置schema，集成、审核、真实检查、调度全部run并跟踪到报告完成。

各agent不得同时修改同一核心文件；不能各自开一个重复GPU任务。训练任务由统一run_suite/registry分配。

## 17. 可直接交给Codex的任务指令

下面整段可复制。请同时提供本Markdown文件。

~~~text
请依据《recogdrive_no_routing_distillation_round1_guide.md》完成代码修改和第一轮完整蒸馏实验。目标是把冻结接口修复教师吸收到ReCogDrive RL学生的连续表征修复网络中。本任务必须完成开发、全部正式训练、完整验证/测试和报告，不能以预实验、smoke通过、loss下降或作业已提交为结尾。

一、立即读取指南、仓库AGENTS.md、README、三份docs和两份报告。以sjc786526-coder/-recogdrive-representation-teacher-2的6bb7119583513094a169b4c2f2dc2a05ac293aa3为冻结证据基线。若当地是后续提交，保留用户改动，在独立工作树/分支中接入并记录差异；不要覆盖原冻结资产。先定位已有RL checkpoint、配套VLM、NAVSIM logs/sensor/maps/metric cache和可用设备，不猜路径，不覆盖公共环境，不终止无关作业。只有必要资产确实无法找到时才说明具体缺项。

二、严格无路由：只允许一个共享连续适配器，4 queries、hidden256、2层cross-attention、4 heads，输出48维连续系数，经固定D48解码为有效token共享384维修复。冻结原VLM、feature_encoder、RL DiT及其状态编码器。禁止expert分类、gate、MoE、argmax、scorer、多候选和测试选择表推理。expert ID只做离线教师来源记录和事后诊断，不能作为输入。gain>1点只是固定训练标签构造，部署不使用阈值。

三、实现指南全部必要模块和CLI，不只生成计划。重点核验：
1. 教师来源始终为独立冻结Base，actual shift与mask正确，所有teacher目标detach。
2. 学生zero-init残差在q=0处cap scale必须为1，不能沿用teacher nominal==0→scale0导致死梯度。
3. 正式过程桥接忠实复用get_action，不用sample_chain替代；DDIM五步、eta1、std下限1e-4、noise clip±1、最后一步仍保留噪声，完整原生padding不裁剪。
4. p_mean_variance直接读取post384，不能二次feature_encoder；冻结DiT的student条件前向必须保留对输入的autograd。A/B显式关闭enable_transition_loss，全程不继承C/D功能损失课程。
5. C/D师生查询同一个a_t,t；D的state由当前H_S.detach()产生，并每次刷新。
6. DDP只包adapter，原生microbatch1加梯度累积实现global batch64；最后不足64的batch按真实scene数归一化。
7. checkpoint包含Adam、scheduler、所有RNG和采样进度，独立deepcopy，真实验证恢复后的下一次更新。

四、正式必跑四组H_full、H_gain1、H_mu_teacher、H_mu_student；每组两个独立training seeds 20261027与20261028，共8个run。每run完整Train85813例，40epochs，global batch64，每epoch1341次更新，共53640次optimizer updates。AdamW LR1e-4、betas[0.9,0.999]、eps1e-8、weight decay0.01，bias/norm/query/最终输出层不衰减，clip norm1；warmup1341次，cosine降到1e-5。新增adapter和action head FP32，TF32关闭。C/D前4epoch只H，接着2epoch将固定归一化的mean loss权重升到0.5，之后34epoch联合训练。按指南用Train固定统计归一化，不使用逐样本零目标范数，不额外添加冲突的Base保持loss。

五、工程检查通过后自动继续全量任务，不再等待确认。已知可实现正控制失败应修复代码；真实教师小集拟合慢、早期PDMS负值或新增失败不构成取消正式实验的理由。所有8个run都训满40epochs。资源抢占/OOM等工程问题修复后断点恢复，保持有效batch、真实曝光和累计总步数。优化IO、缓存和去噪状态生成开销，不能缩短训练、缩数据或改变采样器冒充优化。不要无限调参直到涨分。

六、每run在epoch8/16/24/32/40做完整Validation17475例×S0–S4，用完整PDMS预定规则选择best，同时保留epoch40 last。全部8个run完成后冻结各best/last SHA再跑所有run的完整Test12146例×S0–S4；同SHA可复用。每个best另做完整Validation S5–S9新噪声检查。完成完整Validation冻结教师诊断及两版目标生成，允许与训练独立调度，结果只用于报告表征误差和上限，不进入训练或归一化。完整Train表征误差必须报告；Train行为用固定8192面板，分母和教师版本一致，不能冒称全训练成绩。所有组包括负结果均完成，不只测赢家。测试不用于反向调参。

七、请组织并行agent分别负责数据/目标、模型/原生桥接、训练/评测；主agent统一公共接口、集成、GPU调度和实验注册表。真实运行训练和评分，持续监控并恢复中断；仅提交后台任务不算完成。如受无法跨越的会话/平台资源限制，准确标记未完成任务与恢复命令，不能写成已完成。

八、最终交付独立分支中的代码、有效配置、8个完整run的checkpoint位置与SHA、全部逐场景评分、last/best全结果、学习曲线、repair与keep损益、各log差异、真实训练/评测成本和ROUND1_REPORT.md。报告区分表征拟合、功能一致、PDMS收益及oracle上限；不要把S0–S4当成5个训练种子，也不要承诺或编造分数。不要向main覆盖写入，不上传大模型、数据或私密路径到公开仓库。

请从核对资产和建立独立开发分支开始，完成指南规定的全部工作。
~~~

## 18. 研究依据与可复核来源

### 当前仓库

1. [README](https://github.com/sjc786526-coder/-recogdrive-representation-teacher-2/blob/6bb7119/README.md)：资产、模型、六库和oracle身份。
2. [教师数据字典](https://github.com/sjc786526-coder/-recogdrive-representation-teacher-2/blob/6bb7119/docs/教师定义与数据字典.md)：D48、共享shift、选择规则和字段。
3. [bank.py](https://github.com/sjc786526-coder/-recogdrive-representation-teacher-2/blob/6bb7119/interface_teacher/bank.py)与[native_cap.py](https://github.com/sjc786526-coder/-recogdrive-representation-teacher-2/blob/6bb7119/interface_teacher/native_cap.py)：精确教师API。
4. [实际动作头](https://github.com/sjc786526-coder/-recogdrive-representation-teacher-2/blob/6bb7119/vendor/upstream/model/recogdrive_diffusion_planner.py)：get_action、sample_chain和p_mean_variance。
5. [正式配置](https://github.com/sjc786526-coder/-recogdrive-representation-teacher-2/blob/6bb7119/configs/repair_final_baseline.json)：模型指纹、采样和评分器。
6. [完整训练矩阵](https://github.com/sjc786526-coder/-recogdrive-representation-teacher-2/tree/6bb7119/decisions/train/D_C8_double)：gain1统计来源。
7. [蒸馏诊断报告](https://github.com/sjc786526-coder/-recogdrive-representation-teacher-2/blob/6bb7119/reports/representation_distillation_diagnosis.html)：旧目标、优化/可达性、修复与保持的证据。

### 方法依据

- [SimKD, CVPR 2022](https://arxiv.org/abs/2203.14001)：复用教师下游头并学习适配特征。本文将该思想适配到固定RL DiT，不声称原论文验证了NAVSIM。
- [RMA, RSS 2021](https://arxiv.org/abs/2107.04034)：从可部署观测预测策略需要的特权latent；不证明单帧观测一定能预测oracle修复。
- [CaT, CVPR 2023](https://arxiv.org/abs/2306.10014)：驾驶特权教师的表征对齐与渐进教学；本轮不直接使用未评分的特征插值。
- [TED, ICML 2023](https://proceedings.mlr.press/v202/liang23j.html)：面向下游任务的信息蒸馏；冻结DiT作为功能映射是本指南的适配设计。
- [DiffusionOPD, 2026](https://arxiv.org/abs/2605.15055)及[官方代码](https://github.com/ali-vilab/DiffusionOPD)：学生去噪状态上的单步监督。原实现为flow matching图像生成，本轮按真实DDIM桥接，不照搬velocity或其完整KL主张。
- [Learning without Forgetting, ECCV 2016](https://arxiv.org/abs/1606.09282)：旧模型作为保持监督；本轮通过零修复目标实现保持，不额外叠加不受控正则。

以上文献支持机制选择。本文40epoch、batch64、LR1e-4和具体loss权重是面向当前资产的首轮工程实验配置，并非文献给出的性能保证，也不是已经完成的训练结果。

