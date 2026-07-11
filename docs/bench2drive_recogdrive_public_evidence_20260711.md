# ReCogDrive Bench2Drive 同口径复现证据账本（2026-07-11）

## 当前结论

本轮只做 ReCogDrive 基线复现，不实现 Pareto GRPO，也不迁移已有研究代码。
可以由公开材料闭合的 Stage2 契约已经固化为 `closest-public`；Stage3 的
Bench2Drive reward 仍无公开证据，因此保持阻塞，不能用自定义 reward 冒充
复现。

机器可读的最终口径位于
`configs/bench2drive_recogdrive_reproduction_gate.json`。本文件记录证据来源和
推理过程，避免后续重复检索。

## 固定来源

| 来源 | 固定版本 | 用途 |
|---|---|---|
| [ReCogDrive 论文](https://arxiv.org/html/2506.08052v2) | arXiv v2 | 三阶段结构、DiT/优化器/GRPO 超参、NAVSIM PDMS reward |
| [ReCogDrive 代码](https://github.com/xiaomi-research/recogdrive) | `d54404796de7a44ca418b96057e3f8c3de3e8c0d` | 公共 DiT、DiffGRPO、NAVSIM metric-cache 实现；没有 B2D 训练代码 |
| ReCogDrive Bench2Drive Traj/QA | HF `f55bb18e0aca846bbedfb516760a1b9b7cfe3ebf` | 六视角 prompt、6 点文本目标、数据规模 |
| ReCogDrive-VLM-2B | HF `16873acca08e3c04ab229b3d973f39aeba9db68d` | Stage1 起点 |
| [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) | 本地 `2645714eb1f3a100217928dd113093cae0779f36` | 1,000 clip 原始数据、220 route 与官方评分 |
| [UniDriveVLA](https://github.com/xiaomi-research/unidrivevla) | `a93c175af893b35dc16618e659eca4d18bb1ec86` | 同团队公开 B2D 六相机、2 Hz、6 点 horizon 旁证 |
| [MindDrive](https://github.com/xiaomi-mlab/MindDrive) | `1a4085dab1c20895a0c8d2b67b4f8e65712fa8de` | B2D 在线 RL reward 参考，但不并入 ReCogDrive 基线 |
| [AutoDrive-R²](https://arxiv.org/html/2509.01944) | arXiv public revision | dense physics reward 参考；不是 B2D 同口径，不并入基线 |

## Stage2 契约是如何闭合的

### 可直接验证的事实

发布的 `Bench2drive_Traj.jsonl` 每条记录有六张图，顺序为 front、
front-left、front-right、back-left、back-right、back；human prompt 输入 ego
speed、二维 acceleration 和导航命令，要求六个 `[x,y,delta]` 点。

对首条记录
`InterurbanActorFlow_Town12_Route1296_Weather7/00010` 做原始数据回放：以当前
帧 10 为 anchor，未来帧为 15/20/25/30/35/40。令

```text
M = current_world2lidar @ inverse(future_world2lidar)
```

则文本目标恰好为

```text
[M[1,3], M[0,3], pi/2 - atan2(M[1,0], M[0,0])]
```

首条记录逐项四舍五入到两位后与发布答案完全一致；全量抽样中少数文本 heading
会相差 `±2π`，角度意义等价，planner target 统一归一化 yaw。human prompt 可由
raw speed、`[acceleration_x, -acceleration_y]` 和 `command_near` 逐字重建。

DiT 不能直接使用文本坐标。公共 `norm_odo` 的第一维范围明显是纵向距离，
因此 planner target 固定为：

```text
[M[0,3], M[1,3], atan2(M[1,0], M[0,0])]
```

即 `[forward, lateral, relative_yaw]`。

### 必须声明的 public proxy

| 未公开项 | 本次选择 | 依据 |
|---|---|---|
| clip split | 1,000 clip 全部训练 | 用户明确要求；不把训练内数据称为 held-out |
| anchor cadence | raw 连续帧，10 Hz | 发布 Traj 在同一 clip 中是 10、11、12… 连续 anchor；UniDriveVLA B2D 也使用 `sample_rate=1` |
| trajectory cadence | history/future 步长 5，即 0.5 秒 | 发布 Traj 的六个 target 与 raw matrix 逐点精确匹配 |
| history | 4 点、0.5 秒间隔 | ReCogDrive 公共 planner 固定 4×3 history |
| future | 6 点、0.5 秒间隔，共 3 秒 | 发布 Traj 精确匹配；UniDriveVLA `action_horizon=6` |
| visual input | 六视角，每视角最多 2 dynamic patch 加 thumbnail | 发布 JSONL 与 Stage1 实际 loader（每条 18 tiles） |
| hidden token | 最后一层全部非 padding token | 论文说 hidden states 作为 cognitive tokens，没有公开裁剪规则 |
| hidden dtype | BF16 | 不改变数值类型且将约 5k×1536 token/sample 的磁盘量减半 |
| checkpoint | epoch 200 final | 论文固定训练 200 epoch；禁止用完整 220 route 选 checkpoint |

按 4 history/6 future、anchor stride 1 扫描 1,000 clips，formal cache 必须
恰好为 202,656 条（247,656 raw frames，每个 clip 的首尾共排除 45 个不可形成
完整 history/future 的 anchor）。旧链路因标量 `theta=NaN` 丢弃的八个窗口，其官方
`world2lidar` 矩阵均为有限值，而且发布的 Stage1 JSONL 包含这些帧，因此新链路
按矩阵保留。cache validator 和 Stage2 launcher 会同时检查数量、六相机顺序、
坐标定义、VLM 来源和 contract ID。

cache 启动前还会固定间隔抽查 128 条发布记录，逐字验证 system/human prompt，
并验证六个未来点的坐标（heading 按 `2π` 等价）。任一条不匹配都会在加载八个
VLM worker 前终止。

## Stage2 训练固定值

- VLM：冻结，且必须是本轮完成的 B2D Stage1 checkpoint；
- DiT：随机初始化，384 hidden、8 heads、16 layers；
- DDPM train steps 100，DDIM inference steps 5；
- 200 epochs，global batch 512；
- AdamW，LR `1e-4`，weight decay `1e-4`；
- 前 1.5%（3 epochs）warmup，cosine decay 到 `1e-6`；
- 不启用 JEPA/VGGT、Stage2 Pareto supervision、BiT 或任何研究分支。

## 环境边界

Stage1 训练继续使用隔离的 `recogdrive_stage1_env`（Python 3.9、Torch 2.2.2、
Transformers 4.37.2）。Stage2 cache、DiT 训练和在线 serving 使用现有 `navsim`
环境（Python 3.9、Torch 2.5.1、Transformers 4.57.6）。后者已用于旧基线加载
同架构 ReCogDrive VLM，但其版本不是作者确认值，因此属于基础设施 proxy。

Hugging Face Trainer 的最终输出不自动复制 `trust_remote_code` Python 文件。
cache launcher 会从固定的官方 base VLM 目录复制并校验五个模型代码文件，写入
`recogdrive_remote_code_provenance.json`；只补充加载代码，不修改 Stage1 权重或
config。如果目标目录已有不同代码，launcher 会拒绝覆盖。

## Stage3 为什么现在不能实现 B2D reward

ReCogDrive 方法和 Figure 4 明确写的是：候选轨迹在 NAVSIM simulator 中
rollout，以 collision、drivable-area、TTC、comfort、progress 聚合得到的
NAVSIM PDMS 作为单一 reward。公共代码也只接受 NAVSIM metric cache。

论文对 Bench2Drive 只报告最终 Table 2 的 DS/SR/能力指标，未说明该模型是否
经过 Stage3，也未提供 B2D reward、rollout cache 或六点/八点 horizon 的转换。
因此以下做法都会污染基线，本轮禁止：

- 用 Bench2Drive DS 自行拼一个 trajectory-level reward；
- 引入 Pareto ranking、约束优势或动态权重；
- 把 MindDrive 的稀疏终局 reward 当作 ReCogDrive reward；
- 把 AutoDrive-R² 等非 B2D 工作的 dense physics reward 当作官方定义；
- 把我们已经实现的研究代码直接迁入 Stage3。

MindDrive 的公开 B2D 在线 RL 使用 route success `+1`、违规事件 `-1`、其余
`0` 的稀疏终局信号。它证明 B2D 上可以做 RL，但算法形态与 ReCogDrive 的
trajectory-group DiffGRPO 不同，只保留为复现后的研究参考。

## 执行顺序

1. 当前正式 Stage1 完成；
2. 用最终 Stage1 VLM 生成 1,000 clips、202,656 条六视角 BF16 hidden cache；
3. 随机初始化 Stage2，按论文超参训练 200 epochs；
4. 先做 cache 内 open-loop 诊断，再用 public-proxy wrapper 跑完整 220 routes；
5. 将结果与 DS 71.36、SR 45.45%、Efficiency 138.18、Comfort 17.45、
   Multi-Ability mean 42.03 对比；
6. 只有在基线结论明确后，才单独讨论 Stage3 reward 和现有研究代码迁移。

完整 220-route 结果只用于最终报告，不用于 reward、PID、频率或 checkpoint
调参。

## 自动衔接

```bash
STAGE1_RUN_DIR=outputs/bench2drive_recogdrive_stage1_official_multiview_<run> \
STAGE1_PID=<launcher-pid> \
bash scripts/bench2drive/watch_recogdrive_b2d_stage1_then_stage2.sh
```

watcher 只有在 Stage1 root checkpoint 的 `global_step` 达到 720、权重文件完整
后才启动 cache；cache 通过 1,000 clips/8 shards/202,656 records 校验后，立即
启动 scratch Stage2。它不会启动 Stage3。

## 资源量级（启动前估算）

六视角 prompt 约有 4,990 个 active tokens。按 BF16、hidden size 1,536 和
202,656 samples 估算，hidden tensor 本体约 3.1 TB（约 2.8 TiB），另有单文件
和索引开销。当前 `/mnt/project` 仍有约 29 TB 可用，容量足够。

旧的 40k/前视/8 点 cache 约 1.3 小时、Stage2 约 6 小时；新链路样本量约
5.2 倍且每条有六视角长 context，首次粗估 cache 12–18 小时、Stage2 30–45
小时。该区间只是容量/旧吞吐外推，正式任务启动后的前 100 个样本/step 必须用
实测吞吐更新，不能当作承诺时间。
