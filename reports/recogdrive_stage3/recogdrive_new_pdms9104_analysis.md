# `/mnt/project/tsinghua_code/recogdrive_new` PDMS 91.04 版本代码与训练改动分析

更新日期：2026-06-15

## 1. 结论先行

`/mnt/project/tsinghua_code/recogdrive_new` 中的 91.0/91.04 开发版本，不是只在原版 ReCogDrive Stage3 上调 GRPO 超参得到的结果。它的核心变化是：

1. **Stage2 先增强策略输入表征**：使用 `ReCogDrive-VLM-8B` 生成缓存，不只缓存 `last_hidden_state`，还缓存 VLM 进入 LLM 前的 `raw_visual_features`。
2. **扩散规划器增加视觉融合模块**：主结果使用 `visual_fusion_strategy=cross_attention`，将 VLM 语义 token 和 raw visual token 做 cross-attention 融合，再送入 DiT。
3. **Stage3 使用 Reinforce++ 风格的 PDM 直接优化**：从 cross-attention SFT checkpoint 初始化 actor/reference policy，在 navtrain metric cache 上用 PDM reward 训练，`sample_time=16`，带 denoising-step discount、reference KL 和 BC anneal。
4. **91.0 主结果不是 geometry/DepthAnythingV2 版本**：仓库里有 `geometry_cross_attention`、`dr_bev_prompt_cross_attention` 等后续尝试，但自带说明明确 91.0 指的是 `cross_attention + RL/Reinforce++` 的 `epoch=9-step=13300.ckpt`。

本地可验证的 navtest PDM 结果：

| 版本 | 结果文件 | PDMS |
|---|---|---:|
| Stage2 SFT cross-attention，旧 cache 评估 | `outputs/recogdrive_sft_cross_attention_2026-05-08_14-18-17/eval_navtest_epoch196/2026.05.12.12.09.46.csv` | `0.7789699045` |
| Stage2 SFT cross-attention，新 cache 评估 | `outputs/recogdrive_sft_cross_attention_2026-05-08_14-18-17/eval_navtest_epoch196_newcache/2026.05.12.13.17.14.csv` | `0.8692214847` |
| Stage3 RL epoch9，原记录 | `outputs/grpo/cross_attention_rl_multinode_20260513_045331/eval_navtest_epoch9/2026.05.14.16.49.31.csv` | `0.9102931324` |
| Stage3 RL epoch9，2026-06-03 recheck | `.../eval_navtest_epoch9_recheck_wangsirui91_recheck_crossattn_fsmerge_20260603_071306/2026.06.03.07.45.53.csv` | `0.9100379803` |
| Stage3 RL epoch9，2026-06-12 rerun | `.../eval_navtest_epoch9_rerun/2026.06.12.07.44.18.csv` | `0.9107642578` |

所以用户提到的 “91.04” 可以理解为这个 checkpoint 在不同 navtest rerun 下稳定约 `91.0` 到 `91.08` 的版本。

## 2. 证据来源

主要查看了以下文件和结果：

| 类型 | 路径 |
|---|---|
| 91.0 版本说明 | `/mnt/project/tsinghua_code/recogdrive_new/docs/recogdrive_91_version_files_cn.md` |
| 模型设计说明 | `/mnt/project/tsinghua_code/recogdrive_new/model_design.md` |
| 融合设计说明 | `/mnt/project/tsinghua_code/recogdrive_new/model_fusion_design_cn.md` |
| Agent | `/mnt/project/tsinghua_code/recogdrive_new/navsim/agents/recogdrive/recogdrive_agent.py` |
| VLM backbone | `/mnt/project/tsinghua_code/recogdrive_new/navsim/agents/recogdrive/recogdrive_backbone.py` |
| FeatureBuilder | `/mnt/project/tsinghua_code/recogdrive_new/navsim/agents/recogdrive/recogdrive_features.py` |
| Diffusion planner | `/mnt/project/tsinghua_code/recogdrive_new/navsim/agents/recogdrive/recogdrive_diffusion_planner.py` |
| Cross-attention fusion | `/mnt/project/tsinghua_code/recogdrive_new/navsim/agents/recogdrive/fusion_cross_attention.py` |
| RL 算法 | `/mnt/project/tsinghua_code/recogdrive_new/navsim/agents/recogdrive/recogdrive_rl_algo.py` |
| RL 正则项 | `/mnt/project/tsinghua_code/recogdrive_new/navsim/agents/recogdrive/rl_regularizers.py` |
| SFT 启动脚本 | `/mnt/project/tsinghua_code/recogdrive_new/scripts/training/run_recogdrive_single_8gpu_cross_attention.sh` |
| RL 启动脚本 | `/mnt/project/tsinghua_code/recogdrive_new/scripts/training/run_reinforce.sh` |
| SFT Hydra overrides | `/mnt/project/tsinghua_code/recogdrive_new/outputs/recogdrive_sft_cross_attention_2026-05-08_14-18-17/code/hydra/overrides.yaml` |
| RL Hydra overrides | `/mnt/project/tsinghua_code/recogdrive_new/outputs/grpo/cross_attention_rl_multinode_20260513_045331/code/hydra/overrides.yaml` |
| navtest PDM CSV | `/mnt/project/tsinghua_code/recogdrive_new/outputs/grpo/cross_attention_rl_multinode_20260513_045331/eval_navtest_epoch9/*.csv` |
| NAVSIM v2 ePDMS summary | `/mnt/project/tsinghua_code/recogdrive_new/eval_outputs/navsimv2_epdms_91_epoch9_cross_attention_2026-05-21_03-14-13/plan/merged/summary.json` |

注意：该目录的 git 元信息不可直接作为可靠 diff 使用，因此本文基于文件内容、Hydra override、训练日志和评测输出进行分析。

## 3. 91.04 对 ReCogDrive 的主要代码修改

### 3.1 Backbone：暴露 raw visual token，并在 prompt 中强化 PDM 约束

文件：`navsim/agents/recogdrive/recogdrive_backbone.py`

关键变化：

1. `system_message` 中显式写入 PDM 公式和约束项：
   - `PDM Score = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12`
   - 同时描述 NC、DAC、TTC、EP、Comfort、DDC 等要求。
2. InternVL 前向时不只返回 `hidden_states[-1]`，还通过 `self.model.extract_feature(pixel_values.bfloat16())` 取出 raw visual features。
3. `_pack_visual_features()` 将每个样本由动态 patch 产生的视觉 token 拼接/填充成 batch 张量。
4. VLM forward 仍然带 `image_flags`，并将 `raw_visual_features` 挂到 `outputs.raw_visual_features`。

目的：

- `last_hidden_state` 已经过 LLM 和文本 prompt 混合，语义强但可能丢失局部视觉/空间细节。
- `raw_visual_features` 更接近 vision encoder 输出，保留局部纹理、车道、边界、障碍物等 token 信息。
- 后续 planner 可以在不训练 VLM 的情况下重新利用这部分视觉细节。

### 3.2 Feature cache：训练 cache 增加 `raw_visual_features`

文件：`navsim/agents/recogdrive/recogdrive_features.py`

训练 cache 里保存的核心字段：

| 字段 | 作用 |
|---|---|
| `history_trajectory` | 历史 ego pose，通常 `[4, 3]` |
| `high_command_one_hot` | 导航命令 one-hot |
| `status_feature` | ego 状态、速度、加速度等 |
| `last_hidden_state` | VLM 最后一层 hidden state |
| `raw_visual_features` | InternVL vision feature token |
| `trajectory` | 未来轨迹监督目标，由 target builder 保存 |
| `camera_intrinsic` / `camera_distortion` | 给 geometry 后续尝试使用 |
| `visual_depth_tokens` / `camera_xyz_tokens` | 仅当 `enable_geometry_features=True` 时出现，不是 91.0 主链路 |

`run_training_recogdrive.py` 和 `run_training_recogdrive_rl.py` 的 collate function 也同步做了修改，对 `last_hidden_state` 和 `raw_visual_features` 做 variable-length padding。

目的：

- Stage2/Stage3 训练时不用重复跑 8B VLM，节省显存和时间。
- 让 cross-attention fusion 可以读取 raw visual token。
- 评估时 cache 必须和 `visual_fusion_strategy` 对齐；从本地结果看，同一个 SFT checkpoint 在旧 cache 和新 cache 下 PDMS 差异很大，说明 cache 正确性会直接影响结果。

### 3.3 Planner：增加多种视觉融合策略，主结果用 cross-attention

文件：

- `navsim/agents/recogdrive/recogdrive_diffusion_planner.py`
- `navsim/agents/recogdrive/fusion_cross_attention.py`
- `navsim/agents/recogdrive/fusion_film.py`
- `navsim/agents/recogdrive/fusion_visual_guidance.py`
- `navsim/agents/recogdrive/fusion_geometry_cross_attention.py`
- `navsim/agents/recogdrive/fusion_dr_bev_prompt.py`

配置新增：

```text
visual_fusion_strategy:
  none
  cross_attention
  film
  visual_guidance
  geometry_cross_attention
  dr_bev_prompt_cross_attention
```

91.04 主结果使用：

```text
agent.visual_fusion_strategy=cross_attention
agent.dit_type=small
agent.sampling_method=ddim
```

实际 `fusion_cross_attention.py` 中的实现是：

```python
attended, _ = self.cross_attn(
    query=self.vlm_norm(vlm_features),
    key=self.visual_norm(visual_features),
    value=self.visual_norm(visual_features),
    key_padding_mask=visual_padding_mask,
    need_weights=False,
)
fused = vlm_features + attended
return fused + self.ffn(self.out_norm(fused))
```

即实际代码是：

```text
semantic VLM tokens query raw visual tokens
H_fused = H + CrossAttention(Q=LN(H), K=LN(V), V=LN(V))
H_out = H_fused + FFN(LN(H_fused))
```

这里需要特别注意：仓库里的 `model_fusion_design_cn.md` 有些段落讨论了 “action token query raw visual token” 或 gate 设计，但 91.04 主链路的实际代码不是 action-query，也没有显式 learnable gate；它是 VLM semantic token 对 raw visual token 的 residual cross-attention。

目的：

- 让 DiT 不只依赖 LLM 末层语义 token，也能看到更细粒度的视觉 token。
- 改善可行驶区域、碰撞/TTC、局部车道方向等需要视觉细节的指标。
- 在不改 VLM backbone、不训练 VLM 的情况下，提高 planner 可用信息量。

### 3.4 Agent：RL 分支从 planner 内部 GRPO 改为独立算法类

文件：`navsim/agents/recogdrive/recogdrive_agent.py`

关键变化：

1. Agent 新增 `rl_algo_type`，支持：
   - `reinforce`
   - `reinforce_plus_plus`
2. 当 `agent.grpo=True` 时，不直接走 planner 的 `forward_grpo`，而是初始化：

```python
ALGO_MAP = {
    "reinforce": ReinforceAlgorithm,
    "reinforce_plus_plus": ReinforcePlusPlusAlgorithm,
}
self.rl_algo = algo_class(cfg.grpo_cfg, self.action_head)
```

3. RL forward 会把以下输入一起传给算法：
   - `last_hidden_state`
   - `raw_visual_features`
   - `history_trajectory`
   - `status_feature`
   - `tokens_list`

目的：

- 把 RL 损失和 planner 主体解耦，便于切换 Reinforce / Reinforce++。
- 让 RL 采样、logprob、reference KL、BC loss 全部使用同一套 cross-attention planner 输入。
- 用 `tokens_list` 从 navtrain metric cache 中查 PDM reward，避免使用 navtest reward。

### 3.5 RL 算法：Reinforce++、reference KL、BC anneal 和 denoising-step discount

文件：

- `navsim/agents/recogdrive/recogdrive_rl_algo.py`
- `navsim/agents/recogdrive/rl_regularizers.py`

Stage3 主结果使用：

```text
+agent.rl_algo_type=reinforce_plus_plus
+agent.grpo_cfg.sample_time=16
+agent.grpo_cfg.gamma_denoising=0.6
+agent.grpo_cfg.reference_kl_coeff=0.02
+agent.grpo_cfg.use_bc_loss=true
+agent.grpo_cfg.bc_coeff_start=0.10
+agent.grpo_cfg.bc_coeff_end=0.05
+agent.grpo_cfg.bc_anneal_epochs=5
+agent.grpo_cfg.min_sampling_denoising_std=0.04
+agent.grpo_cfg.min_logprob_denoising_std=0.1
+agent.grpo_cfg.randn_clip_value=5.0
+agent.grpo_cfg.denoised_clip_value=1.0
```

Reinforce++ 的优势计算流程：

1. 对每个 scene/prompt 采样 `G=16` 条轨迹。
2. 用 navtrain metric cache 计算每条轨迹的 PDM reward。
3. 将 reward reshape 为 `[B, G]`。
4. 组内去中心化：

```text
adv = reward - mean(reward_group)
```

5. 再在整个 batch 的 `B*G` 样本上做标准化：

```text
adv = (adv - batch_mean) / batch_std
```

6. 按 quantile 裁剪 advantage。
7. 将 advantage 广播到扩散去噪链的每个 step，并用：

```text
gamma_denoising ** (num_denoising_steps - step - 1)
```

对早期 denoising step 降权，对更接近最终动作的 step 加权更高。
8. 计算 actor 的 chain logprob，经过 `clamp(-5, 2)` 和均值 reduce 后做 policy gradient：

```text
policy_loss = -mean(logprob * discounted_advantage)
```

同时加入：

- **reference KL**：reference policy 是 Stage2 SFT checkpoint 的 frozen copy，系数 `0.02`。
- **BC loss**：teacher chain 也来自 reference policy，BC 系数前 5 个 epoch 从 `0.10` 线性退火到 `0.05`。

目的：

- 组内去中心化减少不同 scene 难度差异造成的 reward 方差。
- batch 标准化控制梯度尺度，避免少数高 reward/低 reward 样本主导。
- reference KL 和 BC anneal 共同限制策略漂移，避免直接用 PDM reward 把 diffusion policy 推到低似然或 reward hacking 区域。
- denoising-step discount 更符合 diffusion policy 的结构：最终低噪声 step 对输出轨迹更直接，所以权重更高。

## 4. 各 Stage 训练改动

### 4.1 Stage1：VLM 阶段

91.04 主链路没有看到重新训练 VLM backbone 的证据。它使用已有的：

```text
/mnt/code/ckpt/ReCogDrive-VLM-8B
```

代码上的变化主要是推理/缓存时从 VLM 中额外取 `raw_visual_features`，而不是修改或继续训练 VLM。

目的：

- 复用更强的 8B VLM 认知能力。
- Stage2/Stage3 只训练 diffusion planner 和新增 fusion 模块，降低训练成本和不稳定性。

### 4.2 Stage2：SFT / IL 训练

Stage2 训练入口：

```text
scripts/training/run_recogdrive_single_8gpu_cross_attention.sh
```

实际 Hydra override：

```text
agent.lr=1e-4
agent.grpo=False
agent.vlm_path=/mnt/code/ckpt/ReCogDrive-VLM-8B
agent.cache_hidden_state=True
agent.vlm_type=internvl
agent.dit_type=small
agent.sampling_method=ddim
agent.visual_fusion_strategy=cross_attention
trainer.params.max_epochs=200
trainer.params.num_nodes=1
trainer.params.devices=8
cache_path=/mnt/code/wangsirui/recogdrive_new/exp/recogdrive_agent_cache_dir_train
use_cache_without_dataset=True
```

注意：SFT 脚本默认 `LR=2e-4`，但 91.04 使用的输出目录里实际 Hydra override 是 `agent.lr=1e-4`，应以 override 为准。

Stage2 训练目标仍然是 imitation / diffusion regression，即学习 GT 轨迹分布；它没有使用 PDM reward。

Stage2 的主要改动不是训练目标，而是**输入表征和可训练模块**：

1. VLM hidden state 和 raw visual token 从 cache 读取。
2. Planner 的 `feature_encoder` 将 3584 维 VLM token 投影到 DiT embedding dim 384。
3. Cross-attention fusion 融合 `last_hidden_state` 与 `raw_visual_features`。
4. DiT 在 fused VLM tokens 条件下做扩散轨迹回归。

目的：

- 让 IL 阶段先学会如何使用 raw visual token。
- 为 Stage3 RL 提供更好的初始策略，否则 RL 只是在较弱表征上微调，容易采样不到好轨迹。
- 缓存 8B VLM 输出，让大模型表征参与训练但不引入 VLM 训练成本。

Stage2 navtest PDM：

| 评估 | PDMS | NC | DAC | EP | TTC | DDC |
|---|---:|---:|---:|---:|---:|---:|
| epoch196 旧 cache | `0.7789699045` | `0.9597398320` | `0.8828420879` | `0.7375076909` | `0.8971677919` | `0.9609748065` |
| epoch196 新 cache | `0.8692214847` | `0.9823398650` | `0.9515066689` | `0.8126178447` | `0.9446731434` | `0.9814342170` |

这里说明两个问题：

- cross-attention SFT 本身已经明显强于错误/旧 cache 下的结果。
- 对此版本来说，feature cache 与 `visual_fusion_strategy` 的一致性是结果复现的前提。

### 4.3 Stage3：RL / Reinforce++ 训练

Stage3 训练入口：

```text
scripts/training/run_reinforce.sh
```

91.04 实际 Hydra override：

```text
agent.lr=1e-4
agent.grpo=True
agent.visual_fusion_strategy=cross_attention
+agent.rl_algo_type=reinforce_plus_plus
+agent.grpo_cfg.gamma_denoising=0.6
+agent.grpo_cfg.clip_advantage_lower_quantile=0.00
+agent.grpo_cfg.clip_advantage_upper_quantile=1.00
+agent.grpo_cfg.randn_clip_value=5.0
+agent.grpo_cfg.denoised_clip_value=1.0
+agent.grpo_cfg.min_sampling_denoising_std=0.04
+agent.grpo_cfg.min_logprob_denoising_std=0.1
+agent.grpo_cfg.sample_time=16
+agent.grpo_cfg.use_bc_loss=true
+agent.grpo_cfg.bc_coeff=0.10
+agent.grpo_cfg.bc_anneal=true
+agent.grpo_cfg.bc_coeff_start=0.10
+agent.grpo_cfg.bc_coeff_end=0.05
+agent.grpo_cfg.bc_anneal_epochs=5
+agent.grpo_cfg.reference_kl_coeff=0.02
+agent.grpo_cfg.scorer_config.progress_weight=10.0
+agent.grpo_cfg.scorer_config.ttc_weight=5.0
+agent.grpo_cfg.scorer_config.comfortable_weight=2.0
agent.metric_cache_path=/mnt/code/dataset/navsim/metric_cache/navtrain
agent.reference_policy_checkpoint=<Stage2 SFT checkpoint>
trainer.params.max_epochs=10
dataloader.params.batch_size=4
trainer.params.num_nodes=2
trainer.params.devices=8
```

Stage3 数据使用：

- 训练 split：`navtrain`
- feature cache：`exp/recogdrive_agent_cache_dir_train`
- reward cache：`/mnt/code/dataset/navsim/metric_cache/navtrain`
- navtest 只用于评估，不参与训练 reward。

Stage3 相比 Stage2 的指标变化，以新 cache SFT vs RL epoch9 原记录为准：

| 指标 | Stage2 SFT | Stage3 RL epoch9 | 变化 |
|---|---:|---:|---:|
| PDMS | `0.8692214847` | `0.9102931324` | `+0.0410716477` |
| NC | `0.9823398650` | `0.9804874033` | `-0.0018524617` |
| DAC | `0.9515066689` | `0.9843569900` | `+0.0328503211` |
| EP | `0.8126178447` | `0.8603775870` | `+0.0477597423` |
| TTC | `0.9446731434` | `0.9533179648` | `+0.0086448213` |
| Comfort | `0.9999176684` | `1.0000000000` | `+0.0000823316` |
| DDC | `0.9814342170` | `0.9662851968` | `-0.0151490203` |

主要收益来自：

- `EP` 明显提升；
- `DAC` 明显提升；
- `TTC` 小幅提升；
- `Comfort` 基本饱和；
- `NC` 略降但仍高；
- `DDC` 有明显下降，这是后续 Stage3 设计需要重点保护的指标。

目的：

- Stage2 IL 学 GT，但 GT 不一定是 PDM 最优。
- Stage3 通过当前策略采样多条候选轨迹，用 PDM reward 做直接优化，鼓励更高 progress、更高 DAC/TTC 的轨迹。
- reference KL + BC anneal 约束策略不要偏离 SFT 太远。

### 4.4 Evaluation / ePDMS

主 PDM 评估使用 navtest：

```text
outputs/grpo/cross_attention_rl_multinode_20260513_045331/eval_navtest_epoch9/2026.05.14.16.49.31.csv
```

NAVSIM v2 ePDMS 入口：

```text
scripts/evaluation/run_navtest_epdms_91_epoch9_cross_attention.sh
```

ePDMS summary：

```text
eval_outputs/navsimv2_epdms_91_epoch9_cross_attention_2026-05-21_03-14-13/plan/merged/summary.json
score_mean = 0.8883033609851688
successful = 12146
failed = 0
invalid_sum = 0
```

另一个 `navsimv2_epdms_91_epoch9_open_filter` summary 为 `0.8033434802`，更像是 open-filter/配置路径不一致的评估，不应和主 91.0 cross-attention ePDMS 混淆。

## 5. 其它实验分支和主结果关系

仓库里还有这些分支/脚本：

| 分支 | 文件/脚本 | 与 91.04 的关系 |
|---|---|---|
| FiLM fusion | `fusion_film.py`, `run_recogdrive_single_8gpu_film.sh` | 备选视觉融合，不是主结果 |
| Visual guidance | `fusion_visual_guidance.py` | 备选视觉融合，不是主结果 |
| Action-query cross attention | `outputs/recogdrive_sft_cross_attention_action_query_*` | 后续/并行实验，不是 91.0 主链路 |
| Geometry cross-attention | `fusion_geometry_cross_attention.py`, `run_recogdrive_single_8gpu_geometry.sh` | 后续 geometry/depth 尝试，不是 91.0 主链路 |
| DR-BEV prompt | `fusion_dr_bev_prompt.py`, `run_recogdrive_single_8gpu_dr_bev_prompt.sh` | 后续 geometry prompt 尝试，不是 91.0 主链路 |
| DepthAnythingV2 | `depth_geometry.py` | 后续版本，不是 91.0 主链路 |

主链路应认定为：

```text
8B VLM cache with raw_visual_features
-> Stage2 cross_attention SFT
-> Stage3 Reinforce++ / PDM reward
-> epoch=9-step=13300.ckpt
```

## 6. 这种修改的目的和机制解释

### 6.1 为什么先改 Stage2 表征

我们当前多轮 Stage3 实验表明，单纯依赖 2B IL checkpoint 的 Stage3 RL，在早期很难超过原版 90.6，甚至 AWAC/DPO 形式吸收高分 buffer 也不稳定。91.04 版本给出的经验是：

- 好的 Stage3 不是只靠 reward objective；
- RL 前的 policy representation 和采样分布非常关键；
- 如果 diffusion planner 条件信息不足，RL 采样出来的候选本身质量有限，reward 只能在低质量分布内做排序，提升空间受限。

91.04 的 cross-attention 让 planner 在 Stage2 就学习如何利用 raw visual token，因此 Stage3 采样分布更可能包含 DAC/TTC/EP 更好的候选。

### 6.2 为什么使用 Reinforce++ 而不是简单 GRPO/AWAC

该实现没有使用离线 elite buffer，也没有 AWAC/IQL。它直接使用当前 policy samples 计算 PDM reward，然后做 on-policy policy gradient。

相对简单 GRPO，它的关键稳定化点是：

- `sample_time=16`，每个 scene 有足够组内候选；
- group de-centering，降低 scene 难度差异；
- batch normalization，稳定 advantage 尺度；
- denoising-step discount，适配 diffusion 去噪链；
- logprob clamp，限制数值爆炸；
- min sampling/logprob std，避免概率退化；
- BC anneal 和 reference KL，保留 SFT trust region；
- progress/TTC/comfort scorer 权重定制，引导 PDM 子项。

这是一套比较完整的 diffusion-policy RL recipe，不只是 “把 reward 乘 logprob”。

### 6.3 为什么结果主要提升 EP/DAC/TTC

PDM 公式中：

```text
score = NC * DAC * (5*TTC + 5*EP + 2*Comfort + 0*DDC) / 12
```

Stage3 scorer 又把 `progress_weight=10`、`ttc_weight=5`、`comfortable_weight=2`。因此 reward 对 EP/TTC 的梯度信号更强；同时 DAC 是乘性门控，提升 DAC 也会显著提升最终分数。

从结果看：

- EP 从 `0.8126` 到 `0.8604`，这是最大收益；
- DAC 从 `0.9515` 到 `0.9844`，也非常明显；
- TTC 小幅提升；
- Comfort 已接近饱和；
- DDC 因为 PDM 公式权重为 0，但仍作为指标统计，所以 RL 后下降。

这说明 91.04 的 Stage3 确实在优化 PDM 主公式，但对 DDC 的保护不足。我们后续如果继续 GRPO，应显式加入 DDC gate / penalty / constrained reward，而不是只看总 PDMS。

## 7. 对当前 ReCogDrive Stage3 改进的启发

### 7.1 不能只在 2B 原输入上堆 Stage3 算法

91.04 的成功路径包含 Stage2 表征增强。它的 RL 起点是 cross-attention SFT，而不是普通 2B IL planner。若我们想复现或吸收这个版本，优先级应是：

1. 先确认是否能引入 `raw_visual_features` cache。
2. 训练 cross-attention SFT，让 planner 学会使用 raw visual token。
3. 再做 Stage3 RL。

如果只在当前 2B IL checkpoint 上改 AWAC/DPO/GRPO，可能会把问题误判为 RL objective 不行，实际是初始策略和条件信息不足。

### 7.2 GRPO / Reinforce++ 方向仍然值得重点做

本版本的最高结果来自 on-policy RL，而不是离线 buffer imitation。结合我们此前 AWAC/IQL 效果弱的经验，后续应更重视：

- on-policy 高质量采样；
- group-level advantage；
- diffusion chain logprob 的正确建模；
- trust-region/reference KL；
- BC 或 SFT anchor 的退火，而不是固定强 BC；
- 子指标约束，尤其 DDC 和 TTC。

### 7.3 高分 buffer 需要以“偏好/约束”方式吸收，而不是简单回归

91.04 没有依赖 elite buffer。它说明 diffusion policy 可以通过当前策略采样 + PDM reward 提升到 91 左右。我们已有高分 buffer 时，更合理的结合方式可能是：

- 将 buffer 用于 warm-start / self-imitation 的小权重辅助；
- 用 DPO/IPO/KTO 类偏好损失时，必须保证 logprob、reference logprob、chosen/rejected 的 timestep/noise 一致；
- buffer 不能替代 on-policy 探索，否则容易学到离当前策略分布较远、DiT 不易吸收的轨迹；
- buffer 候选必须通过 NC/DAC/TTC/DDC guard，否则会把 reward hacking 轨迹写进策略。

### 7.4 后续复现实验应优先对齐成熟 recipe

建议后续实验优先级：

1. 复现 91.04 的 Stage3 Reinforce++ 配置在我们的环境下是否能稳定达到相近趋势。
2. 如果当前 2B 路径继续推进，先把 Reinforce++ 的完整工程细节对齐：
   - `sample_time=16`
   - group de-centering + batch normalization advantage
   - `gamma_denoising=0.6`
   - logprob clamp `[-5, 2]`
   - min sampling std `0.04`
   - min logprob std `0.1`
   - BC anneal `0.10 -> 0.05`
   - reference KL `0.02`
3. 加入 DDC/TTC constrained reward，而不是只优化总 PDMS。
4. 再考虑把 high-PDMS buffer 作为 self-imitation/DPO 辅助项。
5. 若要追求 91+，需要把 Stage2 raw visual fusion 纳入主计划，而不是只做 Stage3 后训练。

## 8. 复现和使用时的注意事项

1. 文档路径中有 `/mnt/code/wangsirui/recogdrive_new`，当前实际可读路径是 `/mnt/project/tsinghua_code/recogdrive_new`；复现脚本前需要统一路径。
2. `run_reinforce.sh` 默认 checkpoint 写成 `epoch196_step131005.ckpt`，但实际说明和目录里的常见文件名是 `epoch=196-step=131005.ckpt`；启动前必须检查 checkpoint 是否真实存在。
3. SFT 脚本默认 LR 是 `2e-4`，但 91.04 所用实际 Hydra override 是 `1e-4`。
4. `model_fusion_design_cn.md` 的部分设计描述和实际 `fusion_cross_attention.py` 不一致，分析 91.04 应以代码和 Hydra override 为准。
5. Stage2 eval 的旧 cache 分数 `0.7789` 不应作为 cross-attention SFT 真实能力判断；同 checkpoint 新 cache 分数为 `0.8692`。
6. 91.04 主结果不是 geometry/depth 版本。不要把 `DepthAnythingV2`、`geometry_cross_attention` 或 `dr_bev_prompt_cross_attention` 的设计混入主结果归因。
7. RL 训练使用 navtrain metric cache，navtest 只用于评估；后续实验也应保持这个边界。

## 9. 可直接引用的启动配置摘要

### Stage2 SFT

```bash
agent=recogdrive_agent
agent.lr=1e-4
agent.grpo=False
agent.vlm_path=/mnt/code/ckpt/ReCogDrive-VLM-8B
agent.cache_hidden_state=True
agent.vlm_type=internvl
agent.dit_type=small
agent.sampling_method=ddim
agent.visual_fusion_strategy=cross_attention
trainer.params.max_epochs=200
trainer.params.num_nodes=1
trainer.params.devices=8
train_test_split=navtrain
cache_path=/mnt/code/wangsirui/recogdrive_new/exp/recogdrive_agent_cache_dir_train
use_cache_without_dataset=True
```

### Stage3 Reinforce++

```bash
agent=recogdrive_agent
agent.lr=1e-4
agent.grpo=True
agent.visual_fusion_strategy=cross_attention
+agent.rl_algo_type=reinforce_plus_plus
+agent.grpo_cfg.gamma_denoising=0.6
+agent.grpo_cfg.clip_advantage_lower_quantile=0.00
+agent.grpo_cfg.clip_advantage_upper_quantile=1.00
+agent.grpo_cfg.randn_clip_value=5.0
+agent.grpo_cfg.denoised_clip_value=1.0
+agent.grpo_cfg.min_sampling_denoising_std=0.04
+agent.grpo_cfg.min_logprob_denoising_std=0.1
+agent.grpo_cfg.sample_time=16
+agent.grpo_cfg.use_bc_loss=true
+agent.grpo_cfg.bc_coeff=0.10
+agent.grpo_cfg.bc_anneal=true
+agent.grpo_cfg.bc_coeff_start=0.10
+agent.grpo_cfg.bc_coeff_end=0.05
+agent.grpo_cfg.bc_anneal_epochs=5
+agent.grpo_cfg.reference_kl_coeff=0.02
+agent.grpo_cfg.scorer_config.progress_weight=10.0
+agent.grpo_cfg.scorer_config.ttc_weight=5.0
+agent.grpo_cfg.scorer_config.comfortable_weight=2.0
agent.metric_cache_path=/mnt/code/dataset/navsim/metric_cache/navtrain
agent.reference_policy_checkpoint=<cross_attention_sft_ckpt>
trainer.params.max_epochs=10
dataloader.params.batch_size=4
trainer.params.num_nodes=2
trainer.params.devices=8
```

## 10. 总体判断

91.04 的有效改动可以概括为：

```text
更强 VLM 表征缓存
+ raw visual token cross-attention SFT
+ 成熟的 Reinforce++ diffusion-policy RL recipe
= navtest PDMS 约 91.0
```

它对我们当前工作的最大启发是：如果目标是显著超过 90.6，不能只在 Stage3 上做轻量损失变体；需要同时保证 Stage2 policy 的信息输入、Stage3 on-policy reward 学习、reference trust region 和子指标约束都足够成熟。当前高分 buffer 的价值仍然存在，但应作为 GRPO/Reinforce++ 的辅助知识来源，而不是替代 on-policy RL 的主路径。
