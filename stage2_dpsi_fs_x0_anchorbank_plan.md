# VLA-AD / ReCogDrive Stage2 改进方案：DPSI-FS-x0-AnchorBank v1

> 目标：在不推翻 ReCogDrive 原始 DiT 框架的前提下，把 Stage2 从“单 GT、噪声预测式模仿”升级为“场景均衡、支持集驱动、轨迹可行性约束、可选全局锚点条件”的训练方案。该方案专门处理 Pareto support set 的不均匀问题：不同场景 support 数量不同、support tag 不全、候选来源不均匀。

---

## 0. 最终决策

采用一版 **DPSI-FS-x0-AnchorBank v1**，由四个部分组成：

1. **Ragged Support Target Sampling**：每个 scene 仍然只贡献一次训练样本，但 target 不再固定为 GT，而是在该 scene 的 selected support set 中随机采样一条高质量轨迹。selected support 数量可以是 1 到 12，不要求每个 scene 拥有完整 tag。
2. **FS-Norm Action Representation**：把未来绝对轨迹转成逐步增量轨迹，再做 step-wise normalization。扩散模型学习“每一步怎么走”，而不是直接学习绝对坐标。
3. **x0 Auxiliary + Geometry Auxiliary**：主损失仍然是 `epsilon/noise` prediction，采样逻辑尽量不动；同时从 `pred_noise` 还原 `pred_x0`，在 clean trajectory 空间加 `x0`、delta、曲率、jerk、reverse、heading jump 等辅助损失。
4. **Global AnchorBank Condition**：不要使用“每个 scene 必须有某些 tag”的硬条件。改为从全量 Pareto supports 中构建一个全局或 command-conditioned anchor bank。训练时把 sampled support target 映射到最近 anchor，并把 anchor trajectory 作为可选条件注入 DiT。推理 / Stage3 时可以从 anchor bank 采样多条候选。该设计天然支持不均匀 support 分布。

**结论**：你的 Pareto 支持集可以用于这个方案。它的不均匀性不会破坏 support / anchor condition，只要不要把 tag 设计成硬 slot 或硬 quota。

---

## 1. 设计依据与当前代码状态

### 1.1 当前 GitHub main 的 Stage2 路径

当前 `TrajectoryTargetBuilder` 只返回单条 GT `trajectory`，没有 support set 信息：

```python
class TrajectoryTargetBuilder(AbstractTargetBuilder):
    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)
        return {"trajectory": torch.tensor(future_trajectory.poses)}
```

当前 `ReCogDriveAgent.forward()` 在非 GRPO Stage2 下，只把 `targets["trajectory"]` 作为 action target 传给 `action_head`：

```python
action_inputs = BatchFeature(
    data={
        **action_input_data,
        "action": targets["trajectory"].to(device=action_device, dtype=model_dtype),
        "_allow_target_tokens_for_loss": True,
    }
)
return self.action_head(last_hidden_state, action_inputs)
```

当前 `ReCogDriveDiffusionPlanner.forward()` 的核心训练逻辑是：

```python
gt_actions = self.norm_odo(action_input.action)
noise = torch.randn_like(gt_actions)
t_discrete = self.sample_time(...)
noisy_actions = sqrt_alpha_t * gt_actions + sqrt_one_minus_alpha_t * noise
pred_noise = self._denoise_model_output(noisy_actions, t_discrete, dit_context, action_input)
diffusion_loss = F.mse_loss(pred_noise, noise, reduction='mean')
```

也就是说，GitHub main 上的可见路径仍然是 **single-GT epsilon-prediction**。如果你本地分支已经有 `use_fs_norm=true / x0_aux_weight / geo_aux_weight`，本方案可直接按下面模块对齐合并。

### 1.2 上传支持集文档的关键结论

当前最终版 support archive 覆盖 `103288` 个场景，selected valid ratio 为 `1.0`，非 GT 低分、起点错位、局部折线、驾驶语义错配等风险均为 `0`。主要剩余问题是 `12702` 个 scene 的高质量支持轨迹数量偏少，而不是入选轨迹明显无效。

支持集 selected_count 分布是：均值 `10.2792`，p50 `12`，p90 `12`，min/max 为 `1 / 12`。这说明可以用固定 `max_supports=12` 做 padding + mask。

support tag 分布并不均匀：`diversity_max`、`vector_pareto`、`best_pdms`、`gt_anchor`、`fallback_best` 数量差异很大。因此不要设计“每个 scene 必须有所有 tag”的模型输入。

---

## 2. 不均匀 support 分布会不会影响改进点 3？

### 2.1 会出问题的设计

下面这些设计不建议做：

1. **把 support 展平成 dataset item**：如果一个 scene 有 12 条 support，另一个 scene 只有 1 条 support，展平后前者在训练中出现 12 次，后者只出现 1 次。模型会被高 support-count scene 主导。
2. **按 support tag 建固定 slot**：例如强制输入 `[best_pdms, vector_pareto, diversity_max, gt_anchor, fallback]` 五个 slot。如果某 scene 缺少某 tag，就必须造假、补零或复制 GT，训练语义会混乱。
3. **按来源或类别强行 quota**：文档里已经说明 IL/current-policy anchor 质量不稳定，最终支持集选择目标是高质量、多样性和 Pareto 前沿，而不是来源丰富。因此不应为了 tag 平衡强保低质量轨迹。
4. **直接把 sampled support target 当 anchor condition**：这会造成 target leakage。模型可能学会复制 anchor，而不是根据 VLM 条件和 noisy action 去噪。推理时没有 scene-level support target，会产生 train/test mismatch。

### 2.2 推荐设计

采用两层解耦：

```text
scene-level support set  →  训练 target 采样
全局 AnchorBank          →  模型条件 / 推理候选模式
```

也就是：

- **support set 用来提供训练目标分布**，不直接要求每个 scene 有完整 tag；
- **anchor bank 用全量 selected supports 构建**，在训练和推理都可用；
- **support tag 只用于采样日志和轻微重权，不作为硬条件输入**；
- **每个 scene 每次训练只采一条 support target**，保证 scene-level uniform。

这样：

```text
selected_count = 1 的 scene：仍可训练，只是该 scene 不贡献多模态 target。
selected_count = 12 的 scene：每个 epoch 可采到不同 target，但不会在一个 epoch 中贡献 12 倍 loss。
缺少某 tag 的 scene：不受影响，因为模型不依赖 tag slot。
fallback_best 少量存在：可以采样，但权重不高，不破坏训练。
```

---

## 3. Stage2 v1 总体数据流

```text
Scene token
   ↓
ParetoSupportTargetBuilder
   ↓
返回 padded supports: [S=12, H=8, D=3] + mask + utility + tag/source metadata
   ↓
ReCogDriveAgent.forward
   ↓
把 support tensors 传入 BatchFeature
   ↓
ReCogDriveDiffusionPlanner.forward
   ↓
scene 内 masked sampling 一条 target trajectory
   ↓
FS-Norm encode target
   ↓
DDPM/DDIM 加噪
   ↓
DiT 预测 pred_noise
   ↓
从 pred_noise 还原 pred_x0
   ↓
L_eps + L_x0 + L_delta + L_geo
```

如果启用 AnchorBank：

```text
sampled support target
   ↓
nearest global anchor / command-conditioned anchor
   ↓
anchor trajectory encode
   ↓
anchor residual condition 注入 DiT fused_input
```

---

## 4. 代码修改总览

建议新增 / 修改以下文件：

```text
navsim/agents/recogdrive/support_archive.py          # 新增：读取 support_v3、padding、metadata 映射
navsim/agents/recogdrive/anchor_bank.py              # 新增：全局 anchor bank 加载、nearest assignment
navsim/agents/recogdrive/recogdrive_features.py      # 修改：新增 ParetoSupportTargetBuilder
navsim/agents/recogdrive/recogdrive_agent.py         # 修改：新增 config 参数，传 support target keys
navsim/agents/recogdrive/recogdrive_diffusion_planner.py  # 修改：FS-Norm、support sampling、x0/geo loss、anchor condition
scripts/build_fs_norm_stats.py                       # 新增：从 support archive 构建 FS-Norm 统计
scripts/build_pareto_anchor_bank.py                  # 新增：从 selected supports 构建 anchor bank
tests/test_fs_norm_roundtrip.py                      # 新增
tests/test_pareto_support_target_builder.py          # 新增
tests/test_dpsi_forward_ragged_support.py            # 新增
tests/test_anchor_bank_condition.py                  # 新增
```

---

## 5. Support Archive Loader 设计

### 5.1 新增文件

`navsim/agents/recogdrive/support_archive.py`

### 5.2 目标接口

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


TAG_TO_ID = {
    "gt_anchor": 0,
    "best_pdms": 1,
    "vector_pareto": 2,
    "diversity_max": 3,
    "fallback_best": 4,
    "unknown": 5,
}

SOURCE_TO_ID = {
    "gt": 0,
    "external": 1,
    "progress": 2,
    "lateral": 3,
    "timing": 4,
    "unknown": 5,
}


@dataclass
class SupportItem:
    trajectory: torch.Tensor      # [H, 3], raw absolute/ego-frame trajectory
    tag_id: int
    source_id: int
    utility: float
    pdms: float
    is_gt: bool
    is_improver: bool
    rank: int


class ParetoSupportArchive:
    def __init__(self, archive_path: str | Path, max_supports: int = 12):
        self.archive_path = Path(archive_path)
        self.max_supports = int(max_supports)
        self.records = self._load_index(self.archive_path)

    def _load_index(self, path: Path) -> Dict[str, List[SupportItem]]:
        """Load support_v3 into token -> List[SupportItem].

        这里需要按你 support_v3 的真实格式适配：
        - 如果每个 token 一个 json/pt/pkl，就遍历读取；
        - 如果有全局 index.json / shard files，就先读 index。
        """
        raise NotImplementedError

    def get(self, token: str) -> List[SupportItem]:
        return self.records.get(token, [])

    def padded(self, token: str, gt_trajectory: torch.Tensor) -> Dict[str, torch.Tensor]:
        items = self.get(token)
        H, D = gt_trajectory.shape
        S = self.max_supports

        trajs = gt_trajectory.new_zeros((S, H, D))
        mask = torch.zeros(S, dtype=torch.bool)
        tag_ids = torch.full((S,), TAG_TO_ID["unknown"], dtype=torch.long)
        source_ids = torch.full((S,), SOURCE_TO_ID["unknown"], dtype=torch.long)
        utility = torch.zeros(S, dtype=torch.float32)
        pdms = torch.zeros(S, dtype=torch.float32)
        is_gt = torch.zeros(S, dtype=torch.bool)
        is_improver = torch.zeros(S, dtype=torch.bool)
        rank = torch.full((S,), 999, dtype=torch.long)

        if not items:
            # 极端兜底：理论上当前审计 has_valid_candidate_ratio=1.0，不应触发。
            items = [SupportItem(
                trajectory=gt_trajectory.detach().cpu().float(),
                tag_id=TAG_TO_ID["gt_anchor"],
                source_id=SOURCE_TO_ID["gt"],
                utility=1.0,
                pdms=0.0,
                is_gt=True,
                is_improver=False,
                rank=0,
            )]

        for i, item in enumerate(items[:S]):
            trajs[i] = item.trajectory.to(dtype=gt_trajectory.dtype)
            mask[i] = True
            tag_ids[i] = int(item.tag_id)
            source_ids[i] = int(item.source_id)
            utility[i] = float(item.utility)
            pdms[i] = float(item.pdms)
            is_gt[i] = bool(item.is_gt)
            is_improver[i] = bool(item.is_improver)
            rank[i] = int(item.rank)

        return {
            "support_trajectories": trajs,
            "support_mask": mask,
            "support_tag_ids": tag_ids,
            "support_source_ids": source_ids,
            "support_utility": utility,
            "support_pdms": pdms,
            "support_is_gt": is_gt,
            "support_is_improver": is_improver,
            "support_rank": rank,
            "support_count": mask.sum().to(torch.long),
        }
```

### 5.3 关键规则

1. `support_trajectories` 固定 shape `[12, 8, 3]`。
2. `support_mask` 标记真实 support，不要把 padding 当 target。
3. `support_count` 只用于日志和采样逻辑，不用于放大 scene 权重。
4. `utility` 必须来自 evaluator-backed metrics 或 SG-FPS 最终选择 utility，不使用未验证 scorer 预测值作为最终 label。
5. 如果 archive 缺失某 token，fallback 到 GT，但必须记录 `support_fallback_missing_archive` 指标。

---

## 6. TargetBuilder 修改

### 6.1 当前问题

当前 `TrajectoryTargetBuilder` 只返回 GT，Stage2 无法接触 support archive。

### 6.2 新增 builder

在 `recogdrive_features.py` 中新增：

```python
class ParetoSupportTargetBuilder(AbstractTargetBuilder):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        support_archive_path: str,
        max_supports: int = 12,
    ):
        self._trajectory_sampling = trajectory_sampling
        self.archive = ParetoSupportArchive(support_archive_path, max_supports=max_supports)

    def get_unique_name(self) -> str:
        return "pareto_support_trajectory_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        future_trajectory = scene.get_future_trajectory(
            num_trajectory_frames=self._trajectory_sampling.num_poses
        )
        gt = torch.tensor(future_trajectory.poses, dtype=torch.float32)

        token = str(getattr(scene, "token", ""))
        if not token:
            # 如果 Scene 没有 token 字段，需要按项目实际接口改成 scene.scene_token / scene.get_token。
            raise ValueError("ParetoSupportTargetBuilder requires scene.token.")

        support = self.archive.padded(token, gt)
        return {
            "trajectory": gt,              # 保留 GT fallback / 兼容旧路径
            "gt_trajectory": gt,
            **support,
        }
```

### 6.3 Agent 中切换 target builder

在 `ReCogDriveAgent.__init__` 增加参数：

```python
use_pareto_support: bool = False
support_archive_path: Optional[str] = None
support_max_count: int = 12
```

保存到 self：

```python
self.use_pareto_support = use_pareto_support
self.support_archive_path = support_archive_path
self.support_max_count = support_max_count
```

修改 `get_target_builders()`：

```python
def get_target_builders(self) -> List[AbstractTargetBuilder]:
    if self.use_pareto_support:
        if not self.support_archive_path:
            raise ValueError("use_pareto_support=True requires support_archive_path.")
        return [ParetoSupportTargetBuilder(
            trajectory_sampling=self._trajectory_sampling,
            support_archive_path=self.support_archive_path,
            max_supports=self.support_max_count,
        )]
    return [TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]
```

### 6.4 Agent forward 传递 support keys

新增常量：

```python
SUPPORT_TARGET_KEYS = (
    "gt_trajectory",
    "support_trajectories",
    "support_mask",
    "support_tag_ids",
    "support_source_ids",
    "support_utility",
    "support_pdms",
    "support_is_gt",
    "support_is_improver",
    "support_rank",
    "support_count",
)
```

修改非 GRPO training 分支：

```python
if targets is not None and not self.grpo:
    target_data = {
        "action": targets["trajectory"].to(device=action_device, dtype=model_dtype),
        "_allow_target_tokens_for_loss": True,
    }
    for key in SUPPORT_TARGET_KEYS:
        if key in targets:
            value = targets[key]
            if isinstance(value, torch.Tensor):
                target_data[key] = value.to(device=action_device)
                if value.dtype.is_floating_point:
                    target_data[key] = target_data[key].to(dtype=model_dtype)

    action_inputs = BatchFeature(data={**action_input_data, **target_data})
    return self.action_head(last_hidden_state, action_inputs)
```

---

## 7. Planner Config 修改

在 `ReCogDriveDiffusionPlannerConfig` 增加：

```python
# --- DPSI / support training ---
use_pareto_support: bool = False
support_max_count: int = 12
support_sampling_temperature: float = 0.50
support_uniform_mix: float = 0.25
support_gt_keep_prob: float = 0.15
support_loss_weight_min: float = 0.50
support_loss_weight_max: float = 2.00

# --- FS-Norm ---
use_fs_norm: bool = True
fs_norm_stats_path: Optional[str] = None
fs_norm_clip_value: float = 1.0
fs_norm_min_scale_xy: float = 0.25
fs_norm_min_scale_heading: float = 0.03

# --- x0 / geometry aux ---
x0_aux_weight: float = 0.10
delta_aux_weight: float = 0.05
geo_aux_weight: float = 0.05
x0_t_max_ratio: float = 0.60
curvature_max: float = 0.35
heading_jump_max: float = 0.50
reverse_margin: float = 0.00

# --- AnchorBank condition ---
use_anchor_condition: bool = True
anchor_bank_path: Optional[str] = None
anchor_dropout: float = 0.50
anchor_residual_scale: float = 0.10
anchor_condition_train_only: bool = False
anchor_nearest_in_fs_space: bool = True
```

`ReCogDriveAgent.__init__` 也要加对应参数并传到 `cfg`。

---

## 8. Support Target Sampling：核心实现

### 8.1 原则

不要把 supports 展平成样本。每个 scene 一个 batch item，在 planner 内部从 `[S=12]` support slots 中采样一条 target。

### 8.2 Planner 中新增 `_sample_support_target`

```python
def _sample_support_target(self, action_input: BatchFeature) -> Dict[str, torch.Tensor]:
    """Return one training target per scene.

    Output:
        target_raw: [B, H, 3]
        target_index: [B]
        sample_weight: [B]
        tag_id/source_id/is_gt/...: [B]
    """
    if (
        not self.config.use_pareto_support
        or "support_trajectories" not in action_input
        or "support_mask" not in action_input
    ):
        B = action_input.action.shape[0]
        device = action_input.action.device
        return {
            "target_raw": action_input.action,
            "target_index": torch.zeros(B, dtype=torch.long, device=device),
            "sample_weight": torch.ones(B, dtype=action_input.action.dtype, device=device),
            "support_count": torch.ones(B, dtype=torch.long, device=device),
            "target_tag_id": torch.zeros(B, dtype=torch.long, device=device),
            "target_is_gt": torch.ones(B, dtype=torch.bool, device=device),
        }

    supports = action_input.support_trajectories      # [B, S, H, 3]
    mask = action_input.support_mask.bool()           # [B, S]
    utility = action_input.support_utility.float()    # [B, S]
    is_gt = action_input.support_is_gt.bool()         # [B, S]
    is_improver = action_input.support_is_improver.bool()
    tag_ids = action_input.support_tag_ids.long()
    support_count = mask.sum(dim=1).clamp_min(1)

    # masked logits
    tau = max(float(self.config.support_sampling_temperature), 1e-4)
    logits = utility / tau
    logits = logits.masked_fill(~mask, -1e9)
    probs_quality = torch.softmax(logits, dim=1)

    # uniform mixture to preserve diversity and prevent best_pdms collapse
    uniform = mask.float() / support_count.unsqueeze(1).float()
    mix = float(self.config.support_uniform_mix)
    probs = (1.0 - mix) * probs_quality + mix * uniform

    # optional GT keep probability
    B, S = mask.shape
    idx = torch.multinomial(probs, num_samples=1).squeeze(1)
    gt_exists = is_gt.any(dim=1)
    gt_idx = is_gt.float().argmax(dim=1)
    use_gt = (
        torch.rand(B, device=supports.device) < float(self.config.support_gt_keep_prob)
    ) & gt_exists
    idx = torch.where(use_gt, gt_idx, idx)

    batch_idx = torch.arange(B, device=supports.device)
    target_raw = supports[batch_idx, idx]

    # quality-aware but bounded loss weight
    chosen_utility = utility[batch_idx, idx].clamp(0.0, 1.0)
    chosen_improver = is_improver[batch_idx, idx].float()
    sample_weight = 0.75 + 0.75 * chosen_utility + 0.25 * chosen_improver
    sample_weight = sample_weight.clamp(
        float(self.config.support_loss_weight_min),
        float(self.config.support_loss_weight_max),
    )
    sample_weight = sample_weight / sample_weight.mean().clamp_min(1e-6)

    return {
        "target_raw": target_raw,
        "target_index": idx,
        "sample_weight": sample_weight.to(dtype=target_raw.dtype),
        "support_count": support_count,
        "target_tag_id": tag_ids[batch_idx, idx],
        "target_is_gt": is_gt[batch_idx, idx],
    }
```

### 8.3 为什么这样能处理不均匀数据

- batch 维度仍是 scene，不是 support item，因此高 support-count scene 不会贡献更多训练步。
- selected_count=1 的 scene 只有一个可选 target，但仍正常训练。
- selected_count=12 的 scene 在不同 epoch/step 能采不同 target，但每次仍只贡献一份 loss。
- tag 不完整不影响，因为采样只依赖 mask 和 utility，不依赖硬 tag slot。

---

## 9. FS-Norm 实现

### 9.1 构建统计

新增脚本：`scripts/build_fs_norm_stats.py`

输入：`support_v3` 中所有 selected supports + GT。
输出：`fs_norm_stats.pt`

推荐统计：

```python
# 对每条轨迹 traj [8,3]
delta_xy[0] = traj[0, :2] - [0, 0]
delta_xy[i] = traj[i, :2] - traj[i-1, :2]
delta_heading[0] = wrap_pi(traj[0, 2] - 0)
delta_heading[i] = wrap_pi(traj[i, 2] - traj[i-1, 2])

scale[h, d] = percentile(abs(delta[:, h, d]), 95)
scale[:, 0:2] = clamp_min(scale[:, 0:2], fs_norm_min_scale_xy)
scale[:, 2] = clamp_min(scale[:, 2], fs_norm_min_scale_heading)
```

保存：

```python
torch.save({
    "delta_scale": delta_scale.float(),  # [8, 3]
    "action_horizon": 8,
    "action_dim": 3,
}, output_path)
```

### 9.2 Planner 中注册 buffer

在 `__init__` 中：

```python
if config.use_fs_norm:
    if config.fs_norm_stats_path:
        stats = torch.load(config.fs_norm_stats_path, map_location="cpu")
        delta_scale = stats["delta_scale"].float()
    else:
        # fallback，仅用于 smoke test；真实训练必须用统计文件。
        delta_scale = torch.tensor([
            [2.0, 0.8, 0.20],
            [2.5, 0.9, 0.20],
            [3.0, 1.0, 0.22],
            [3.5, 1.1, 0.24],
            [4.0, 1.2, 0.26],
            [4.5, 1.3, 0.28],
            [5.0, 1.4, 0.30],
            [5.5, 1.5, 0.32],
        ], dtype=torch.float32)
    self.register_buffer("fs_delta_scale", delta_scale.view(1, config.action_horizon, config.action_dim))
```

### 9.3 encode / decode

```python
@staticmethod
def _wrap_pi(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def fs_norm_odo(self, trajectory: torch.Tensor) -> torch.Tensor:
    xy = trajectory[..., :2]
    hd = trajectory[..., 2:3]

    xy_prev = torch.cat([torch.zeros_like(xy[..., :1, :]), xy[..., :-1, :]], dim=-2)
    hd_prev = torch.cat([torch.zeros_like(hd[..., :1, :]), hd[..., :-1, :]], dim=-2)

    delta_xy = xy - xy_prev
    delta_hd = self._wrap_pi(hd - hd_prev)
    delta = torch.cat([delta_xy, delta_hd], dim=-1)
    z = delta / self.fs_delta_scale.to(device=trajectory.device, dtype=trajectory.dtype)
    return z.clamp(-float(self.config.fs_norm_clip_value), float(self.config.fs_norm_clip_value))


def fs_denorm_odo(self, normalized_delta: torch.Tensor) -> torch.Tensor:
    delta = normalized_delta * self.fs_delta_scale.to(device=normalized_delta.device, dtype=normalized_delta.dtype)
    xy = torch.cumsum(delta[..., :2], dim=-2)
    hd = self._wrap_pi(torch.cumsum(delta[..., 2:3], dim=-2))
    return torch.cat([xy, hd], dim=-1)


def encode_action(self, trajectory: torch.Tensor) -> torch.Tensor:
    if self.config.use_fs_norm:
        return self.fs_norm_odo(trajectory)
    return self.norm_odo_absolute(trajectory)


def decode_action(self, action_repr: torch.Tensor) -> torch.Tensor:
    if self.config.use_fs_norm:
        return self.fs_denorm_odo(action_repr)
    return self.denorm_odo_absolute(action_repr)
```

把原来的 `norm_odo` / `denorm_odo` 改成 wrapper，保持旧调用兼容：

```python
def norm_odo(self, trajectory: torch.Tensor) -> torch.Tensor:
    return self.encode_action(trajectory)

def denorm_odo(self, normalized_trajectory: torch.Tensor) -> torch.Tensor:
    return self.decode_action(normalized_trajectory)
```

原始 min-max 版本改名为：

```python
def norm_odo_absolute(...)
def denorm_odo_absolute(...)
```

---

## 10. x0 Auxiliary 与 Geometry Loss

### 10.1 修改 forward 主流程

当前代码：

```python
gt_actions = self.norm_odo(action_input.action)
...
pred_noise = self._denoise_model_output(...)
diffusion_loss = F.mse_loss(pred_noise, noise, reduction='mean')
```

改成：

```python
support_batch = self._sample_support_target(action_input)
target_raw = support_batch["target_raw"]
sample_weight = support_batch["sample_weight"]

gt_actions = self.encode_action(target_raw)
...
pred_noise = self._denoise_model_output(
    noisy_actions,
    t_discrete,
    dit_context,
    action_input,
    anchor_step_condition=anchor_step_condition,
)

eps_loss_per = ((pred_noise.float() - noise.float()) ** 2).mean(dim=(1, 2))
diffusion_loss = self._weighted_mean(eps_loss_per, sample_weight)

pred_x0 = self._x0_from_noise(noisy_actions, t_discrete, pred_noise).clamp(-1.0, 1.0)
aux = self._compute_stage2_aux_losses(
    pred_x0=pred_x0,
    target_repr=gt_actions,
    target_raw=target_raw,
    timesteps=t_discrete,
    sample_weight=sample_weight,
)

loss = (
    float(self.config.diffusion_loss_weight) * diffusion_loss
    + float(self.config.x0_aux_weight) * aux["x0_aux_loss"]
    + float(self.config.delta_aux_weight) * aux["delta_aux_loss"]
    + float(self.config.geo_aux_weight) * aux["geo_aux_loss"]
    + ... # existing expert/last_rd losses
)
```

### 10.2 Weighted mean

```python
def _weighted_mean(self, value_per_sample: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.to(device=value_per_sample.device, dtype=value_per_sample.dtype)
    return (value_per_sample * weight).sum() / weight.sum().clamp_min(1e-6)
```

### 10.3 Timestep gating

高噪声阶段不要强监督 x0/geometry：

```python
def _x0_aux_gate(self, timesteps: torch.Tensor) -> torch.Tensor:
    T = float(self.ddpm_num_train_timesteps)
    t_ratio = timesteps.float() / max(T - 1.0, 1.0)
    return (t_ratio <= float(self.config.x0_t_max_ratio)).float()
```

### 10.4 Aux losses

```python
def _compute_stage2_aux_losses(
    self,
    pred_x0: torch.Tensor,
    target_repr: torch.Tensor,
    target_raw: torch.Tensor,
    timesteps: torch.Tensor,
    sample_weight: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    gate = self._x0_aux_gate(timesteps).to(dtype=pred_x0.dtype)
    weight = sample_weight.to(pred_x0.dtype) * gate
    weight = weight / weight.mean().clamp_min(1e-6)

    x0_per = F.smooth_l1_loss(pred_x0.float(), target_repr.float(), reduction="none").mean(dim=(1, 2))
    x0_aux_loss = self._weighted_mean(x0_per, weight)

    pred_raw = self.decode_action(pred_x0)
    pred_delta = pred_raw[:, 1:, :] - pred_raw[:, :-1, :]
    target_delta = target_raw[:, 1:, :] - target_raw[:, :-1, :]
    target_delta = target_delta.to(pred_delta.dtype)
    delta_per = F.smooth_l1_loss(pred_delta.float(), target_delta.float(), reduction="none").mean(dim=(1, 2))
    delta_aux_loss = self._weighted_mean(delta_per, weight)

    geo = self._geometry_losses(pred_raw)
    geo_aux_loss = self._weighted_mean(geo["geo_per"], weight)
    return {
        "x0_aux_loss": x0_aux_loss,
        "delta_aux_loss": delta_aux_loss,
        "geo_aux_loss": geo_aux_loss,
        **geo,
    }
```

### 10.5 Geometry losses

```python
def _geometry_losses(self, traj: torch.Tensor) -> Dict[str, torch.Tensor]:
    xy = traj[..., :2]
    hd = traj[..., 2]
    seg = xy[:, 1:, :] - xy[:, :-1, :]
    seg_len = torch.linalg.norm(seg, dim=-1).clamp_min(1e-3)
    seg_yaw = torch.atan2(seg[..., 1], seg[..., 0])

    # curvature from segment yaw change
    yaw_change = self._wrap_pi(seg_yaw[:, 1:] - seg_yaw[:, :-1])
    avg_len = 0.5 * (seg_len[:, 1:] + seg_len[:, :-1]).clamp_min(1e-3)
    curvature = yaw_change.abs() / avg_len
    curv_per = F.relu(curvature - float(self.config.curvature_max)).pow(2).mean(dim=1)

    # jerk / second-order smoothness in xy
    if xy.shape[1] >= 4:
        vel = xy[:, 1:, :] - xy[:, :-1, :]
        acc = vel[:, 1:, :] - vel[:, :-1, :]
        jerk = acc[:, 1:, :] - acc[:, :-1, :]
        jerk_per = jerk.abs().mean(dim=(1, 2))
    else:
        jerk_per = xy.new_zeros(xy.shape[0])

    # reverse progress in ego frame
    reverse_per = F.relu(float(self.config.reverse_margin) - seg[..., 0]).mean(dim=1)

    # heading jump
    hd_diff = self._wrap_pi(hd[:, 1:] - hd[:, :-1]).abs()
    hd_jump_per = F.relu(hd_diff - float(self.config.heading_jump_max)).pow(2).mean(dim=1)

    geo_per = curv_per + 0.5 * jerk_per + 0.5 * reverse_per + 0.5 * hd_jump_per
    return {
        "geo_per": geo_per,
        "curvature_loss": curv_per.mean(),
        "jerk_loss": jerk_per.mean(),
        "reverse_loss": reverse_per.mean(),
        "heading_jump_loss": hd_jump_per.mean(),
    }
```

---

## 11. Global AnchorBank Condition

### 11.1 为什么不用 support tag condition

support tag 的存在是为了解释筛选原因，不适合作为强监督语义类别。比如 `diversity_max` 不是驾驶意图，`vector_pareto` 也不是动作类型。把这些 tag 当成 action mode 会误导模型。

更成熟的做法是：从全量支持轨迹中构造 **轨迹形状 anchor bank**。这类似 anchor-guided diffusion / truncated diffusion 的思想，但这里 anchor 来自你的 evaluator-verified Pareto supports。

### 11.2 构建 anchor bank

新增脚本：`scripts/build_pareto_anchor_bank.py`

输入：`support_v3` + `fs_norm_stats.pt`
输出：`pareto_anchor_bank.pt`

推荐参数：

```text
num_anchors_total = 24
如果 archive 中可读 command_id，则每个 command 8 个 anchor：left/straight/right × 8。
如果暂时读不到 command_id，则全局 24 个 anchor。
anchor selection = FPS / k-medoids in FS-Norm space，按 evaluator utility 加权。
```

输出结构：

```python
torch.save({
    "anchors": anchors,          # [C, K, H, 3] or [K, H, 3], raw trajectory
    "anchor_mask": anchor_mask,  # [C, K] or [K]
    "anchor_utility": utility,
    "anchor_tag_id": tag_id,
    "mode": "command" or "global",
}, output_path)
```

### 11.3 Planner 初始化

```python
if config.use_anchor_condition:
    if not config.anchor_bank_path:
        raise ValueError("use_anchor_condition=True requires anchor_bank_path.")
    bank = torch.load(config.anchor_bank_path, map_location="cpu")
    self.register_buffer("anchor_bank", bank["anchors"].float())
    self.anchor_step_encoder = Mlp(
        in_features=config.action_dim,
        hidden_features=config.hidden_size,
        out_features=config.input_embedding_dim,
        norm_layer=nn.LayerNorm,
    )
    self.anchor_residual_proj = nn.Linear(config.input_embedding_dim, config.input_embedding_dim)
    nn.init.zeros_(self.anchor_residual_proj.weight)
    nn.init.zeros_(self.anchor_residual_proj.bias)
```

为什么 `anchor_residual_proj` 零初始化：它不破坏已有 ReCogDrive checkpoint，训练初期等价于无 anchor，然后逐步学会利用 anchor。

### 11.4 Nearest anchor assignment

```python
def _select_nearest_anchor(self, target_raw: torch.Tensor, action_input: BatchFeature) -> torch.Tensor:
    # target_raw: [B,H,3]
    target_repr = self.encode_action(target_raw).detach() if self.config.anchor_nearest_in_fs_space else target_raw.detach()

    anchors = self.anchor_bank
    if anchors.ndim == 4:
        # command-conditioned: [C,K,H,3]
        cmd = action_input.high_command_one_hot.argmax(dim=-1).long().clamp(0, anchors.shape[0] - 1)
        bank_raw = anchors[cmd]  # [B,K,H,3]
    else:
        bank_raw = anchors.unsqueeze(0).expand(target_raw.shape[0], -1, -1, -1)

    B, K, H, D = bank_raw.shape
    bank_repr = self.encode_action(bank_raw.reshape(B * K, H, D)).reshape(B, K, H, D)
    dist = ((bank_repr.float() - target_repr[:, None].float()) ** 2).mean(dim=(2, 3))
    return dist.argmin(dim=1)
```

### 11.5 Anchor residual condition

```python
def _build_anchor_step_condition(
    self,
    anchor_raw: torch.Tensor,
    *,
    training: bool,
) -> torch.Tensor:
    # anchor_raw: [B,H,3]
    anchor_repr = self.encode_action(anchor_raw)
    anchor_feat = self.anchor_step_encoder(anchor_repr)
    anchor_feat = self.anchor_residual_proj(anchor_feat)
    anchor_feat = anchor_feat * anchor_feat.new_tensor(float(self.config.anchor_residual_scale))

    if training and float(self.config.anchor_dropout) > 0.0:
        keep = torch.rand(anchor_feat.shape[0], 1, 1, device=anchor_feat.device) >= float(self.config.anchor_dropout)
        anchor_feat = anchor_feat * keep.to(dtype=anchor_feat.dtype)
    return anchor_feat
```

### 11.6 注入位置

不要改 `fusion_projector` 的输入维度，避免 checkpoint shape mismatch。采用 residual 注入：

```python
fused_input = self.fusion_projector(
    torch.cat((his_traj_features, context_mean_features, action_features), dim=2)
)
if expert_step_condition is not None:
    fused_input = fused_input + expert_step_condition.to(...)
if anchor_step_condition is not None:
    fused_input = fused_input + anchor_step_condition.to(...)
```

需要修改：

```text
_denoise_model_output(..., anchor_step_condition=None)
p_mean_variance(..., anchor_step_condition=None)
get_action(...)
sample_chain(...)
get_logprobs(...)
```

### 11.7 训练时 anchor 使用

在 `forward()` 里：

```python
anchor_step_condition = None
if self.config.use_anchor_condition:
    anchor_idx = self._select_nearest_anchor(target_raw, action_input)
    anchor_raw = self._gather_anchor_by_index(anchor_idx, action_input)
    anchor_step_condition = self._build_anchor_step_condition(anchor_raw, training=self.training)
```

注意：anchor 是从全局 bank 选择的最近轨迹，不是直接把 target support 作为 anchor。因此没有直接 target leakage。

### 11.8 推理时 anchor 使用

默认 single-trajectory 评测可以先设：

```text
inference_anchor_mode = none
```

用于 Stage3 / best-of-N 时新增方法：

```python
def get_anchor_conditioned_actions(self, vl_features, action_input, anchor_indices=None, deterministic=False):
    # repeat B by K anchors, attach anchor_trajectory into action_input, call get_action
    ...
```

后续 GRPO 可以基于不同 anchors 采样 groups：

```text
同一 scene 下：
anchor 0 采 G0 条
anchor 1 采 G1 条
...
每个 anchor 内做 advantage normalization，anchor 间再做 Pareto/GRPO 选择。
```

---

## 12. 训练配置建议

### 12.1 推荐 YAML / Hydra override

```yaml
agent:
  _target_: navsim.agents.recogdrive.recogdrive_agent.ReCogDriveAgent
  dit_type: small
  sampling_method: ddim
  cache_hidden_state: true
  train_backbone: false
  grpo: false

  # DPSI support
  use_pareto_support: true
  support_archive_path: /mnt/project/VLA-AD_last_vla_dev/outputs/sg_fps_support_semantic_full_20260705T032521Z/reselect_ref_relative_feasrelax_gtimprover_20260705T185120Z/support_v3
  support_max_count: 12

  # FS/x0/geo
  use_fs_norm: true
  fs_norm_stats_path: /mnt/project/VLA-AD_last_vla_dev/outputs/stage2_assets/fs_norm_stats_support_v3.pt
  x0_aux_weight: 0.10
  delta_aux_weight: 0.05
  geo_aux_weight: 0.05
  x0_t_max_ratio: 0.60

  # support sampling
  support_sampling_temperature: 0.50
  support_uniform_mix: 0.25
  support_gt_keep_prob: 0.15

  # AnchorBank
  use_anchor_condition: true
  anchor_bank_path: /mnt/project/VLA-AD_last_vla_dev/outputs/stage2_assets/pareto_anchor_bank_support_v3.pt
  anchor_dropout: 0.50
  anchor_residual_scale: 0.10
```

### 12.2 推荐训练轮次

```text
max_epochs = 200
warmup_epochs = 3
lr_action_head = 1e-4
weight_decay = 1e-4
batch size = 沿用当前能跑满显存的设置
```

### 12.3 Loss 权重初始值

```text
L_eps:       1.00
L_x0:        0.10
L_delta:     0.05
L_geo:       0.05
anchor:      residual_scale 0.10, dropout 0.50
support:     uniform_mix 0.25, gt_keep_prob 0.15
```

如果训练前 10 epoch 不稳定：

```text
geo_aux_weight: 0.05 → 0.02
anchor_residual_scale: 0.10 → 0.05
anchor_dropout: 0.50 → 0.70
x0_t_max_ratio: 0.60 → 0.50
```

如果模型过度保守 / progress 下降：

```text
curvature_max: 0.35 → 0.45
reverse loss 权重减半
support_gt_keep_prob: 0.15 → 0.10
support_uniform_mix: 0.25 → 0.35
```

如果多样性不足：

```text
support_uniform_mix: 0.25 → 0.40
support_sampling_temperature: 0.50 → 0.80
anchor_dropout: 0.50 → 0.30
num inference anchors 增加
```

---

## 13. 日志与诊断指标

`_format_training_output()` 增加：

```python
output.update({
    "x0_aux_loss": aux["x0_aux_loss"],
    "delta_aux_loss": aux["delta_aux_loss"],
    "geo_aux_loss": aux["geo_aux_loss"],
    "curvature_loss": aux["curvature_loss"],
    "jerk_loss": aux["jerk_loss"],
    "reverse_loss": aux["reverse_loss"],
    "heading_jump_loss": aux["heading_jump_loss"],
    "support_count_mean": support_batch["support_count"].float().mean(),
    "support_target_is_gt_ratio": support_batch["target_is_gt"].float().mean(),
    "support_loss_weight_mean": sample_weight.mean(),
})
```

建议每个 epoch 额外统计：

```text
1. sampled support tag histogram
2. sampled source histogram
3. selected_count=1 scene ratio in sampled batches
4. x0_pred decoded curvature p95
5. early kink rate
6. reverse progress rate
7. best-of-N PDMS on val subset
8. unanchored vs anchored inference gap
```

---

## 14. Ablation 顺序

不要一次性全开后只给一个结果。建议按以下顺序：

### A0：GitHub main / Official-aligned baseline

```text
single GT
absolute norm_odo
L_eps only
no support
no FS
no x0/geo
no anchor
```

### A1：DPSI target sampling only

```text
support target sampling
absolute norm_odo
L_eps only
no FS
no x0/geo
no anchor
```

目的：验证 support archive 本身是否提升 imitation target 质量。

### A2：DPSI + FS-Norm

```text
support target sampling
FS-Norm
L_eps only
no x0/geo
no anchor
```

目的：确认 action representation 是否改善训练稳定性和轨迹平滑。

### A3：DPSI + FS-Norm + x0/geo

```text
support target sampling
FS-Norm
L_eps + L_x0 + L_delta + L_geo
no anchor
```

目的：这是最稳的主模型，应作为 Stage3 初始化的首选 checkpoint。

### A4：DPSI + FS-Norm + x0/geo + AnchorBank

```text
support target sampling
FS-Norm
L_eps + L_x0 + L_delta + L_geo
Global AnchorBank residual condition
anchor dropout
```

目的：为 Stage3 / best-of-N / anchor-conditioned GRPO 提供结构化多模态先验。

**推荐论文主结果**：A3 或 A4。
**推荐 Stage3 初始化**：如果 A4 unanchored 单轨评测不低于 A3，则用 A4；否则用 A3 作为 reference policy，用 A4 作为 candidate generator。

---

## 15. 测试用例

### 15.1 FS-Norm roundtrip

`tests/test_fs_norm_roundtrip.py`

```python
def test_fs_norm_roundtrip():
    planner = build_dummy_planner(use_fs_norm=True)
    traj = torch.randn(4, 8, 3)
    z = planner.encode_action(traj)
    rec = planner.decode_action(z)
    assert torch.allclose(traj[..., :2], rec[..., :2], atol=1e-4)
    assert torch.allclose(torch.sin(traj[..., 2]), torch.sin(rec[..., 2]), atol=1e-4)
```

### 15.2 Ragged support builder

```python
def test_support_padding_mask():
    builder = ParetoSupportTargetBuilder(... max_supports=12)
    targets = builder.compute_targets(scene_with_3_supports)
    assert targets["support_trajectories"].shape == (12, 8, 3)
    assert targets["support_mask"].sum().item() == 3
```

### 15.3 Scene-uniform sampling

```python
def test_no_flatten_support_bias():
    # batch contains scene A with 1 support and scene B with 12 supports
    # forward returns one loss per scene, not 13 losses
    out = planner(vl_features, action_input)
    assert out["diffusion_loss"].ndim == 0
```

### 15.4 Forward smoke

```python
def test_dpsi_forward_with_ragged_support():
    features, targets = make_dummy_batch_with_support_counts([1, 4, 12])
    out = agent.forward(features, targets)
    assert torch.isfinite(out["loss"])
    assert "x0_aux_loss" in out
    assert "geo_aux_loss" in out
```

### 15.5 Anchor no-dependency

```python
def test_anchor_dropout_no_inference_dependency():
    # get_action without anchor_trajectory should still work
    out = planner.get_action(vl_features, action_input_without_anchor)
    assert out["pred_traj"].shape[-2:] == (8, 3)
```

---

## 16. 主要风险与处理

### 风险 1：support target 分布太宽，Stage2 loss 上升

处理：

```text
support_uniform_mix 降低到 0.10
support_sampling_temperature 降低到 0.30
support_gt_keep_prob 提高到 0.25
只训练 A1/A2 观察 20 epoch
```

### 风险 2：x0/geo 让模型过度保守

处理：

```text
geo_aux_weight 0.05 → 0.02
curvature_max 0.35 → 0.45
reverse loss 系数减半
关注 EP / progress，不只看 comfort
```

### 风险 3：anchor condition 造成 train/test mismatch

处理：

```text
anchor_dropout ≥ 0.50
anchor_residual_proj 零初始化
官方 single-output eval 先用 no-anchor
Stage3/best-of-N 再启用 anchor-conditioned sampling
```

### 风险 4：low_support_count scene 多样性不足

处理：

```text
不要全量重建 archive。
只针对 low_support_count 与 poor_gt_few_candidates 做定向补充。
Stage2 训练中这些 scene 正常出现，但不会被错误地扩增或删除。
```

---

## 17. 最终建议

第一轮落地时，按下面优先级执行：

```text
必须做：
1. ParetoSupportTargetBuilder：padded supports + mask
2. planner 内 scene-level masked support sampling
3. FS-Norm encode/decode
4. pred_noise → pred_x0 auxiliary
5. geometry auxiliary

建议做：
6. AnchorBank 构建与 residual condition，但先作为 A4 ablation

暂不做：
7. support tag hard-slot conditioning
8. 按 tag/source 强制 quota
9. flatten supports as dataset items
10. test-time 使用 scene-level support archive
```

一句话版本：

> Stage2 v1 不把“support tag 不齐”当成问题去补齐，而是承认 support set 是 ragged 的；训练时 scene-uniform 地采样高质量 support target，模型侧用全局 AnchorBank 表示多模态先验，并用 FS-Norm + x0/geometry loss 把 DiT 学习目标拉回可行轨迹空间。
