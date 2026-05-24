# ReCogDrive 专家 Token 改造：训练、开发与验证指南

这份文档面向刚接触 ReCogDrive、NAVSIM 和专家模型注入方案的使用者。目标是让你能按步骤完成三件事：

1. 先在没有数据、没有真实权重的情况下确认代码能跑通。
2. 准备 NAVSIM/OpenScene 数据和 ReCogDrive 权重，跑原始基线和专家 token 版本。
3. 理解这次改造到底做了什么，以及后续怎样更稳妥地把 JEPA/VGGT 这类外部模型知识注入 ReCogDrive。

重要说明：当前工程已经支持“缓存专家特征并消费专家 token”。但是，真实 JEPA/VGGT 模型推理后端还没有接入，当前只有 dummy 后端用于计算流程自测。也就是说，现在可以验证代码、形状、训练流程和 checkpoint 兼容性；要验证真实性能提升，还需要把真实 JEPA/VGGT 的特征提取逻辑接到 `RealExpertFeatureBackend`。

## 一、先理解项目在做什么

ReCogDrive 可以粗略理解成两段：

1. 视觉语言模型看前视图和历史状态，输出一串隐藏特征。
2. 扩散规划器根据隐藏特征、历史轨迹和车辆状态，生成未来 8 个轨迹点。

原始训练为了省时间，会先把视觉语言模型的隐藏特征缓存到磁盘。之后训练扩散规划器时，直接读取缓存，不再每次跑大模型。

这次改造加了一条新支路：把外部教师模型的知识变成“专家 token”，和原来的 VLM token 一起送给扩散规划器。

当前支持的专家 token：

```text
jepa_tokens:         [B, Kj, Dj]   当前帧/当前观测 token，训练和推理都可以用
vggt_tokens:         [B, Kg, Dg]   当前帧/当前观测 token，训练和推理都可以用
jepa_target_tokens:  [B, Kj, Dj]   未来帧目标 token，只能训练时作为辅助监督
vggt_target_tokens:  [B, Kg, Dg]   未来帧目标 token，只能训练时作为辅助监督
```

默认小模型维度：

```text
last_hidden_state:     [B, Nv, 1536]
history_trajectory:   [B, 4, 3]，进入 planner 前会展平成 [B, 12]
status_feature:       [B, 8]
trajectory target:    [B, 8, 3]
planner token dim:    384
jepa_tokens:          [B, 4, 768]
vggt_tokens:          [B, 4, 2048]
```

大模型维度：

```text
last_hidden_state:     [B, Nv, 3584]
planner token dim:     1536
```

## 二、这次具体改了什么

核心实现文件：

| 文件 | 作用 |
| --- | --- |
| `navsim/agents/recogdrive/expert_backends.py` | 新增专家特征后端接口和 `DummyExpertBackend`。dummy 后端能生成确定性的假 JEPA/VGGT token，用于无数据自测。 |
| `navsim/agents/recogdrive/recogdrive_features.py` | `ReCogDriveFeatureBuilder` 支持读取专家 cache、dummy 专家特征、目标 token 泄漏保护、dummy cache 训练保护。 |
| `navsim/agents/recogdrive/recogdrive_agent.py` | `ReCogDriveAgent` 支持专家配置、无 checkpoint 随机初始化、安全 checkpoint 加载、把专家 token 传入 planner。 |
| `navsim/agents/recogdrive/recogdrive_diffusion_planner.py` | `ReCogDriveDiffusionPlanner` 支持 JEPA/VGGT token 投影、type embedding、gate、和 VLM token 拼接、alignment loss、GRPO 路径兼容。 |
| `navsim/planning/training/dataset.py` | cache-only 训练时，能根据隐藏状态 cache 的 token 路径补充加载专家 cache。 |
| `navsim/planning/training/agent_lightning_module.py` | 训练日志增加 diffusion loss、alignment loss、专家 gate 值等。 |
| `navsim/planning/script/run_training_recogdrive*.py` | 训练前检查 dummy cache，防止误把假特征用于正式训练。 |
| `navsim/planning/script/run_pdm_score_recogdrive.py` | 评估时如果检测到 dummy cache 会发出警告。 |
| `navsim/planning/script/run_recogdrive_expert_feature_caching.py` | 新增专家特征缓存生成脚本。当前 dummy 可跑，真实 JEPA/VGGT 后端预留。 |
| `scripts/create_dummy_expert_cache.py` | 生成扁平 dummy expert cache，用于本地 smoke test。 |
| `scripts/smoke_test_recogdrive_expert_dummy_flow.py` | 直接测试 planner 的 baseline 和专家路径。 |
| `scripts/smoke_test_recogdrive_agent_dummy_forward.py` | 测试 `ReCogDriveAgent.forward` 到 action head 的完整计算流。 |
| `scripts/debug_train_recogdrive_expert_on_dummy_data.py` | 在合成数据上跑几步优化，验证 loss、梯度和参数没有 NaN。 |
| `scripts/testing/smoke_recogdrive_checkpoint_loading.py` | 测试随机初始化模型的保存、加载，以及旧 baseline checkpoint 加载到专家模型时的兼容性。 |
| `scripts/evaluation/aggregate_recogdrive_expert_results.py` | 聚合多组 PDM 结果，输出 CSV 和 Markdown 表。 |

新增配置字段：

```yaml
use_expert_features: false
expert_feature_source: "none"  # none | dummy | cache | real
allow_random_init: true
allow_dummy_expert_cache: false
num_jepa_tokens: 4
num_vggt_tokens: 4
use_jepa: true
use_vggt: true
jepa_dim: 768
vggt_dim: 2048
expert_dropout: 0.0
expert_fusion_mode: "concat_context"
use_expert_type_embedding: true
use_expert_gates: true
expert_alignment_weight: 0.0
jepa_alignment_weight: 0.0
vggt_alignment_weight: 0.0
alignment_loss_type: "mse"
```

关键设计：

1. `use_expert_features=false` 时，不启用专家路径，保持原始基线行为。
2. 专家 token 不在 feature builder 里转半精度，只在模型 forward 时转到模型 dtype/device。
3. 当前 token 可以训练和推理使用。
4. 未来 target token 只能训练时用于辅助 alignment loss，推理和评估不读取。
5. checkpoint 加载默认 `strict=false`，新增专家参数缺失是预期情况；已有 baseline 参数如果 shape 不匹配会报错。
6. 核心模型文件不 import JEPA/VGGT 包，真实教师模型只应该放在缓存生成后端里。

## 三、无数据、无权重的第一轮自测

这一步强烈建议先做。它不需要 NAVSIM、不需要图片、不需要任何 checkpoint。

进入仓库：

```bash
cd /root/remote/recogdrive
pip install -e .
```

如果当前环境还没有完整依赖，至少先尝试下面的 smoke test。脚本里对部分重依赖做了轻量 stub，目的是先验证计算图。

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/smoke_test_recogdrive_expert_dummy_flow.py --device cpu
PYTHONDONTWRITEBYTECODE=1 python scripts/smoke_test_recogdrive_agent_dummy_forward.py
PYTHONDONTWRITEBYTECODE=1 python scripts/testing/smoke_recogdrive_dummy_expert_cache.py
PYTHONDONTWRITEBYTECODE=1 python scripts/testing/smoke_recogdrive_checkpoint_loading.py
PYTHONDONTWRITEBYTECODE=1 python scripts/debug_train_recogdrive_expert_on_dummy_data.py --device cpu --steps 1 --use-expert-features --use-alignment-loss
```

预期看到类似：

```text
baseline forward OK
expert forward OK
backward OK
get_action OK
no-future-leakage OK
Synthetic debug optimization loop completed successfully.
```

注意：dummy 和随机初始化只能证明“代码能算”，不能证明驾驶性能。

## 四、准备数据目录

正式训练和评估需要 NAVSIM/OpenScene 数据。仓库配置默认按下面目录寻找数据：

```text
$OPENSCENE_DATA_ROOT/
  maps/
  navsim_logs/
    trainval/
    test/
    mini/
  sensor_blobs/
    trainval/
    test/
    mini/

$NAVSIM_EXP_ROOT/
  用于保存 hidden cache、expert cache、metric cache、训练输出、评估输出
```

建议先设置环境变量：

```bash
export REPO_ROOT=/root/remote/recogdrive
export NAVSIM_DEVKIT_ROOT=$REPO_ROOT
export OPENSCENE_DATA_ROOT=/path/to/NAVSIM/dataset
export NAVSIM_EXP_ROOT=/path/to/NAVSIM/exp
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT=$OPENSCENE_DATA_ROOT/maps
export PYTHONPATH=$REPO_ROOT:$PYTHONPATH
```

检查目录：

```bash
ls $OPENSCENE_DATA_ROOT/maps
ls $OPENSCENE_DATA_ROOT/navsim_logs/trainval
ls $OPENSCENE_DATA_ROOT/sensor_blobs/trainval
ls $OPENSCENE_DATA_ROOT/navsim_logs/test
ls $OPENSCENE_DATA_ROOT/sensor_blobs/test
```

常用 split：

| 名称 | 数据目录 | 用途 |
| --- | --- | --- |
| `navtrain` | `trainval` | 训练 IL/RL |
| `navtest` | `test` | 评估 |
| `navmini` | `mini` | 小规模调试 |

## 五、准备预训练权重

至少需要：

1. ReCogDrive VLM 权重，用于生成隐藏状态 cache，或者在线评估时提取隐藏状态。
2. IL/RL 训练时可以从随机初始化开始，但真实实验建议从官方或已有 checkpoint 开始对比。
3. 真实 JEPA/VGGT 权重，用于生成真实专家 cache。当前代码还没有接入真实后端，先把路径预留好。

官方文档里给出的权重入口：

```text
InternVL3-2B: https://huggingface.co/OpenGVLab/InternVL3-2B
InternVL3-8B: https://huggingface.co/OpenGVLab/InternVL3-8B
ReCogDrive VLM/IL/RL: README 中的 HuggingFace collection 链接
```

设置路径：

```bash
export RECOGDRIVE_VLM_PATH=/path/to/ReCogDrive-VLM-2B
export RECOGDRIVE_IL_CHECKPOINT=/path/to/best_il.ckpt
export RECOGDRIVE_EVAL_CHECKPOINT=/path/to/checkpoint_to_eval.ckpt

export RECOGDRIVE_JEPA_MODEL_PATH=/path/to/jepa
export RECOGDRIVE_VGGT_MODEL_PATH=/path/to/vggt
```

如果只是跑 smoke test，`RECOGDRIVE_JEPA_MODEL_PATH` 和 `RECOGDRIVE_VGGT_MODEL_PATH` 可以填任意稳定字符串，因为 dummy 后端不会真的加载模型。

## 六、生成 VLM hidden-state cache

这是训练扩散规划器前最重要的一步。hidden cache 保存的是 ReCogDrive VLM 的隐藏状态，后续 IL 训练会直接读它。

建议用一个固定路径：

```bash
export RECOGDRIVE_HIDDEN_CACHE_DIR=$NAVSIM_EXP_ROOT/recogdrive_agent_cache_dir_train_2b
```

启动缓存：

```bash
GPUS=8
GPUS_PER_NODE=8
MASTER_PORT=63669

torchrun \
  --nproc_per_node=$GPUS_PER_NODE \
  --master_port=$MASTER_PORT \
  $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_dataset_caching_multi_node.py \
  agent=recogdrive_agent \
  experiment_name=recogdrive_agent_cache_train_2b \
  agent.cam_type=single \
  agent.cache_hidden_state=True \
  agent.cache_mode=True \
  agent.vlm_type=internvl \
  agent.vlm_size=small \
  agent.dit_type=small \
  train_test_split=navtrain \
  agent.vlm_path=$RECOGDRIVE_VLM_PATH \
  cache_path=$RECOGDRIVE_HIDDEN_CACHE_DIR
```

缓存完成后，目录大致像这样：

```text
$RECOGDRIVE_HIDDEN_CACHE_DIR/
  log_name/
    token/
      internvl_feature.gz
      trajectory_target.gz
```

检查：

```bash
find $RECOGDRIVE_HIDDEN_CACHE_DIR -name internvl_feature.gz | head
find $RECOGDRIVE_HIDDEN_CACHE_DIR -name trajectory_target.gz | head
```

如果磁盘不足，先用 `navmini` 或限制小样本调试；完整训练缓存可能需要很大空间。

## 七、生成专家 feature cache

专家 cache 的标准布局：

```text
$RECOGDRIVE_EXPERT_CACHE_DIR/
  metadata.json
  log_name/
    token/
      expert_features.pt
```

每个 `expert_features.pt` 里包含：

```python
{
  "jepa_tokens": Tensor[Kj, Dj],
  "vggt_tokens": Tensor[Kg, Dg],
  "jepa_target_tokens": Tensor[Kj, Dj],  # 可选，训练 alignment 用
  "vggt_target_tokens": Tensor[Kg, Dg],  # 可选，训练 alignment 用
}
```

### 7.1 先用 dummy 后端生成小样本

这只用于确认缓存脚本能跑，不用于真实训练。

```bash
export RECOGDRIVE_EXPERT_CACHE_DIR=$NAVSIM_EXP_ROOT/recogdrive_expert_cache_dummy
export RECOGDRIVE_JEPA_MODEL_PATH=dummy_jepa
export RECOGDRIVE_VGGT_MODEL_PATH=dummy_vggt

TEACHER_BACKEND=dummy \
TRAIN_TEST_SPLIT=navtrain \
SPLIT=trainval \
MAX_SAMPLES=32 \
COMPUTE_FUTURE_TARGETS=1 \
DEVICE=cuda \
PRECISION=fp16 \
sh scripts/cache_dataset/run_caching_recogdrive_expert_features.sh
```

检查：

```bash
cat $RECOGDRIVE_EXPERT_CACHE_DIR/metadata.json
find $RECOGDRIVE_EXPERT_CACHE_DIR -name expert_features.pt | head
```

如果你把 dummy cache 用于训练，训练脚本会拒绝，除非显式设置：

```bash
export ALLOW_DUMMY_EXPERT_CACHE=true
```

这个开关只允许用来调试计算流程，不要用它产出论文或报告结果。

### 7.2 真实 JEPA/VGGT cache

当前真实后端还没有实现。入口在：

```text
navsim/planning/script/run_recogdrive_expert_feature_caching.py
  class RealExpertFeatureBackend
```

你需要在这个类里完成：

1. 加载 JEPA 模型和 VGGT 模型。
2. 实现对应图像预处理。
3. `encode_current(batch)` 返回当前帧 dense 特征：

```python
{
  "jepa": Tensor[B, Nj_dense, Dj],
  "vggt": Tensor[B, Ng_dense, Dg],
}
```

4. `encode_future_targets(batch)` 返回未来帧 dense 特征，只在 `--compute-future-targets` 时使用。
5. 脚本会用 `adaptive_avg_pool1d` 把 dense 特征池化成固定 token 数：

```text
jepa_tokens: [B, num_jepa_tokens, jepa_dim]
vggt_tokens: [B, num_vggt_tokens, vggt_dim]
```

实现真实后端后，运行：

```bash
export RECOGDRIVE_EXPERT_CACHE_DIR=$NAVSIM_EXP_ROOT/recogdrive_expert_cache_real
export RECOGDRIVE_JEPA_MODEL_PATH=/path/to/real/jepa
export RECOGDRIVE_VGGT_MODEL_PATH=/path/to/real/vggt

TEACHER_BACKEND=real \
TRAIN_TEST_SPLIT=navtrain \
SPLIT=trainval \
COMPUTE_FUTURE_TARGETS=1 \
DEVICE=cuda \
PRECISION=fp16 \
sh scripts/cache_dataset/run_caching_recogdrive_expert_features.sh
```

评估集如果只需要当前 token，不要生成 future target：

```bash
export RECOGDRIVE_EXPERT_CACHE_DIR=$NAVSIM_EXP_ROOT/recogdrive_expert_cache_real_test

TEACHER_BACKEND=real \
TRAIN_TEST_SPLIT=navtest \
SPLIT=test \
COMPUTE_FUTURE_TARGETS=0 \
DEVICE=cuda \
PRECISION=fp16 \
sh scripts/cache_dataset/run_caching_recogdrive_expert_features.sh
```

## 八、跑 IL 训练

新增的专家训练脚本是：

```text
scripts/training/run_recogdrive_train_multi_node_2b_expert.sh
```

不要改官方原始脚本，先用 `_expert` 脚本做实验。

通用环境变量：

```bash
export OPENSCENE_DATA_ROOT=/path/to/NAVSIM/dataset
export NAVSIM_EXP_ROOT=/path/to/NAVSIM/exp
export RECOGDRIVE_VLM_PATH=/path/to/ReCogDrive-VLM-2B
export RECOGDRIVE_HIDDEN_CACHE_DIR=$NAVSIM_EXP_ROOT/recogdrive_agent_cache_dir_train_2b
export RECOGDRIVE_EXPERT_CACHE_DIR=$NAVSIM_EXP_ROOT/recogdrive_expert_cache_real

export GPUS=8
export GPUS_PER_NODE=8
export BATCH_SIZE=16
export MAX_EPOCHS=200
export LR=1e-4
```

可选实验分组：

| `EXPERT_VARIANT` | 含义 |
| --- | --- |
| `baseline` | 专家功能关闭，作为 sanity baseline |
| `jepa` | 只用 JEPA token |
| `vggt` | 只用 VGGT token |
| `jepa_vggt` | JEPA 和 VGGT 都用 |
| `jepa_vggt_alignment` | JEPA/VGGT 都用，并开启 alignment loss |

依次训练：

```bash
EXPERT_VARIANT=baseline sh scripts/training/run_recogdrive_train_multi_node_2b_expert.sh
EXPERT_VARIANT=jepa sh scripts/training/run_recogdrive_train_multi_node_2b_expert.sh
EXPERT_VARIANT=vggt sh scripts/training/run_recogdrive_train_multi_node_2b_expert.sh
EXPERT_VARIANT=jepa_vggt sh scripts/training/run_recogdrive_train_multi_node_2b_expert.sh
EXPERT_VARIANT=jepa_vggt_alignment sh scripts/training/run_recogdrive_train_multi_node_2b_expert.sh
```

如果显存不够，先降：

```bash
export GPUS=1
export GPUS_PER_NODE=1
export BATCH_SIZE=2
export MAX_EPOCHS=1
```

训练日志会保存到：

```text
$NAVSIM_EXP_ROOT/train_recogdrive_${EXPERT_VARIANT}_expert_il.txt
```

Hydra/Lightning 输出在：

```text
$NAVSIM_EXP_ROOT/training_recogdrive_${EXPERT_VARIANT}_expert_il/<时间戳>/
```

重点看这些日志项：

```text
loss
diffusion_loss
jepa_alignment_loss
vggt_alignment_loss
jepa_gate
vggt_gate
```

如果 alignment 权重为 0，alignment loss 可以为 0 或没有实际贡献，这是正常的。

## 九、跑 IL 评估

先准备 metric cache。官方脚本里有硬编码路径，建议按你的目录改脚本，或直接运行：

```bash
export RECOGDRIVE_METRIC_CACHE_DIR=$NAVSIM_EXP_ROOT/metric_cache

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
  train_test_split=navtest \
  cache.cache_path=$RECOGDRIVE_METRIC_CACHE_DIR
```

评估某个 IL checkpoint：

```bash
export RECOGDRIVE_EVAL_CHECKPOINT=/path/to/best_il.ckpt
export RECOGDRIVE_EXPERT_CACHE_DIR=$NAVSIM_EXP_ROOT/recogdrive_expert_cache_real_test
export RECOGDRIVE_METRIC_CACHE_DIR=$NAVSIM_EXP_ROOT/metric_cache

EXPERT_VARIANT=jepa_vggt_alignment \
sh scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_2b_expert.sh
```

baseline 评估：

```bash
export RECOGDRIVE_EVAL_CHECKPOINT=/path/to/baseline_or_official.ckpt
EXPERT_VARIANT=baseline \
sh scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_2b_expert.sh
```

评估会输出 NAVSIM PDM 相关指标。重点比较：

```text
PDMS, NC, DAC, TTC, CF, EP
```

## 十、跑 RL 训练和评估

RL 训练需要 train split 的 metric cache：

```bash
export RECOGDRIVE_METRIC_CACHE_DIR=$NAVSIM_EXP_ROOT/metric_cache_train

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py \
  train_test_split=navtrain \
  cache.cache_path=$RECOGDRIVE_METRIC_CACHE_DIR
```

从最好的 IL checkpoint 开始 RL：

```bash
export RECOGDRIVE_IL_CHECKPOINT=/path/to/best_il.ckpt
export RECOGDRIVE_EXPERT_CACHE_DIR=$NAVSIM_EXP_ROOT/recogdrive_expert_cache_real
export RECOGDRIVE_METRIC_CACHE_DIR=$NAVSIM_EXP_ROOT/metric_cache_train

EXPERT_VARIANT=jepa_vggt_alignment \
sh scripts/training/run_recogdrive_train_multi_node_rl_2b_expert.sh
```

评估 RL checkpoint：

```bash
export RECOGDRIVE_EVAL_CHECKPOINT=/path/to/best_rl.ckpt
export RECOGDRIVE_EXPERT_CACHE_DIR=$NAVSIM_EXP_ROOT/recogdrive_expert_cache_real_test
export RECOGDRIVE_METRIC_CACHE_DIR=$NAVSIM_EXP_ROOT/metric_cache

EXPERT_VARIANT=jepa_vggt_alignment \
sh scripts/evaluation/run_recogdrive_agent_pdm_score_evaluation_2b_expert.sh
```

## 十一、聚合对比实验结果

假设每个 variant 的评估输出目录如下：

```text
$NAVSIM_EXP_ROOT/recogdrive_baseline_expert_eval
$NAVSIM_EXP_ROOT/recogdrive_jepa_expert_eval
$NAVSIM_EXP_ROOT/recogdrive_vggt_expert_eval
$NAVSIM_EXP_ROOT/recogdrive_jepa_vggt_expert_eval
$NAVSIM_EXP_ROOT/recogdrive_jepa_vggt_alignment_expert_eval
```

运行：

```bash
python scripts/evaluation/aggregate_recogdrive_expert_results.py \
  baseline=$NAVSIM_EXP_ROOT/recogdrive_baseline_expert_eval \
  jepa=$NAVSIM_EXP_ROOT/recogdrive_jepa_expert_eval \
  vggt=$NAVSIM_EXP_ROOT/recogdrive_vggt_expert_eval \
  jepa_vggt=$NAVSIM_EXP_ROOT/recogdrive_jepa_vggt_expert_eval \
  jepa_vggt_alignment=$NAVSIM_EXP_ROOT/recogdrive_jepa_vggt_alignment_expert_eval \
  --output-csv $NAVSIM_EXP_ROOT/recogdrive_expert_ablation_results.csv \
  --output-md $NAVSIM_EXP_ROOT/recogdrive_expert_ablation_results.md
```

输出会包含：

```text
baseline
JEPA only
VGGT only
JEPA+VGGT
JEPA+VGGT+alignment
```

以及配置摘要：

```text
use_expert_features
jepa_dim/vggt_dim
token counts
alignment weights
checkpoint path
```

## 十二、常见问题

### 1. 训练报 expert cache 找不到

检查：

```bash
echo $RECOGDRIVE_EXPERT_CACHE_DIR
find $RECOGDRIVE_EXPERT_CACHE_DIR -name expert_features.pt | head
```

专家 cache 必须能按下面路径找到：

```text
$RECOGDRIVE_EXPERT_CACHE_DIR/log_name/token/expert_features.pt
```

其中 `log_name/token` 要和 hidden cache 里的路径一致。

### 2. 形状不匹配

检查 metadata：

```bash
cat $RECOGDRIVE_EXPERT_CACHE_DIR/metadata.json
```

确认这些值和训练配置一致：

```text
num_jepa_tokens
num_vggt_tokens
jepa_dim
vggt_dim
```

默认配置是：

```text
num_jepa_tokens=4
num_vggt_tokens=4
jepa_dim=768
vggt_dim=2048
```

### 3. accidentally 用了 dummy cache

训练默认会拒绝 dummy cache，这是保护机制。如果只是调试：

```bash
export ALLOW_DUMMY_EXPERT_CACHE=true
```

如果是正式训练，不要打开这个开关。

### 4. checkpoint 加载出现 missing expert keys

如果你把旧 baseline checkpoint 加载到专家模型，新增专家参数缺失是正常的。日志里会显示 expected missing expert keys。

如果 baseline 参数 shape mismatch，会直接报错。这个是故意的，因为这通常代表模型大小、VLM hidden 维度或 planner 配置错了。

### 5. 推理时出现 target token 警告

`jepa_target_tokens` 和 `vggt_target_tokens` 是未来帧监督，只能训练用。评估或推理时发现它们会警告并忽略。正式 eval cache 最好不要生成 target tokens。

## 十三、推荐的实验顺序

最稳妥的顺序：

1. 跑无数据 smoke test。
2. 用 `navmini` 或 `MAX_SAMPLES=32` 跑 dummy expert cache。
3. 生成一小段 hidden cache，跑 1 epoch baseline sanity。
4. 接真实 JEPA 后端，只跑 JEPA only。
5. 接真实 VGGT 后端，只跑 VGGT only。
6. 跑 JEPA+VGGT。
7. 跑 JEPA+VGGT+alignment。
8. 再进入完整 navtrain/navtest。
9. 最后从最好 IL checkpoint 做 RL。

不要一开始就直接开完整 JEPA+VGGT+alignment+RL。这样出问题很难定位。

## 十四、我对这个方向的想法

你的目标是用 CoWorld-VLA、LaST-VLA 那类思路，把外部模型的世界知识注入 ReCogDrive。这个方向是合理的，但要注意“注入方式”比“模型名”更关键。

我建议把它拆成三类知识：

1. 语义和场景规律：JEPA 这类自监督表征可能更擅长提供稳定的视觉语义、可通行区域、物体关系等隐变量。
2. 几何和三维结构：VGGT 这类模型更适合作为几何教师，提供深度、视角一致性、空间布局线索。
3. 未来一致性：target tokens 可以作为训练时辅助监督，让 planner 的内部表示更接近未来可预测的世界状态，但不能在推理时使用。

当前实现是最小可控版本：把 JEPA/VGGT token 投影到 planner 维度，然后拼到 VLM token 后面。优点是简单、容易做 ablation、baseline 风险小；缺点是专家 token 可能被 planner 忽略，或者和 VLM token 混在一起后语义不够清楚。

后续更好的实现路线：

1. 先做强 ablation，不要急着复杂化。JEPA only、VGGT only、JEPA+VGGT、alignment 分开跑，先确认哪个信号真的有用。
2. 给专家 token 加 LayerNorm 和小 adapter。不同老师的 feature 分布差异很大，直接 Linear 可能不稳定。
3. gate 初始值保持保守。现在 gate 从大约 0.1 开始，比较稳。等确认有效后再尝试更大的 gate 或动态 gate。
4. 把 JEPA 和 VGGT 分开 cross-attention。现在是 concat；后续可以做两个专家 cross-attention，再用门控融合，这样可解释性更好。
5. 对 VGGT 不只用 token，还可以蒸馏几何辅助任务。例如深度一致性、地面平面、可行驶区域边界、物体相对距离。
6. 对 JEPA 可以做时序一致性，而不是只取当前帧。比如当前帧 token 和历史帧 token 做轻量 temporal pooling。
7. alignment loss 不要太大。它是辅助项，主任务仍然是轨迹质量。建议从 0.01、0.05、0.1 三档扫。
8. 严格防止未来泄漏。任何 future target 只能训练用，eval cache 不要包含 target，get_action 绝不能读取 target。
9. 真实效果验证必须使用同一套 hidden cache、同一随机种子、同一训练步数和同一评估集。否则提升很可能来自训练噪声。

如果我是继续推进这个项目，我会先做下面三个版本：

1. JEPA current tokens + 小 gate，不加 alignment。
2. VGGT current tokens + 小 gate，不加 alignment。
3. JEPA+VGGT current tokens + 小 gate + 很小 alignment。

如果第 3 个版本没有明显提升，我不会立刻加更复杂结构，而是先检查：

1. 专家 cache 是否和样本 token 对齐。
2. 专家 feature 是否有稳定数值分布。
3. gate 是否一直接近 0，说明 planner 没有用专家 token。
4. alignment loss 是否压过 diffusion loss。
5. 专家模型输入的图像帧是否和 ReCogDrive 当前帧一致。

这类工作最怕“看起来接进去了，但模型完全没用上”。所以日志里的 gate、alignment loss、不同 variant 的 ablation，比单次最终 PDMS 更重要。

## 十五、一句话总结

现在这套改造已经具备完整工程骨架：baseline 可关闭、专家 token 可缓存、planner 可融合、alignment 可训练、dummy 模式可无数据自测、checkpoint 可兼容加载。下一步真正决定效果的是：把真实 JEPA/VGGT 后端接好，并用严格的 ablation 验证外部知识到底有没有帮助 ReCogDrive 的规划器。
