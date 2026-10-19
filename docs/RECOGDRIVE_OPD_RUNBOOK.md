# 执行与复现

工作目录：`/mnt/project/VLA-AD/research/recogdrive-opd-20261020`。
解释器：`/root/miniconda3/envs/navsim/bin/python`。
有效产物根目录：`/mnt/project/VLA-AD/outputs/recogdrive_opd_20261020/official_v2`。权重、完整日志、候选与输入缓存不进 git。

```bash
PY=/root/miniconda3/envs/navsim/bin/python
$PY scripts/opd/prepare_data.py
$PY scripts/opd/audit.py --config configs/opd/base.yaml
$PY -m pytest tests/opd -q
$PY scripts/opd/validate.py --mode batch
$PY scripts/opd/validate.py --mode grpo
$PY scripts/opd/inputs.py verify
```

真实缓存与评测按 shard 分发；不同 shard 的全部模型、噪声、数据和评分身份必须相同。

```bash
$PY scripts/opd/collect_teacher_outputs.py --rank 0 --shards 16
$PY scripts/opd/evaluate.py --exp E0 --split train --rank 0 --shards 16
$PY scripts/opd/build_advantage_gate.py
$PY scripts/opd/train.py --exp E3 --stage tiny --seed 42
$PY scripts/opd/evaluate.py --exp E3 --name tiny_E3_42 --split cal
$PY scripts/opd/analyze.py --phase tiny
$PY scripts/opd/calibrate.py
```

实际两槽位学习率对照和三种子正式矩阵由 `jobs/*/spec.json` 完整保存命令。正式训练须传入 `selection/tuning.json` 的学习率及 `calibration/gradient_scales.json` 的辅助权重，否则会拒绝运行。新复现使用不同的 `OPD_RUN_ROOT`，不要覆盖已完成结果。

```bash
$PY scripts/opd/suite.py formal
$PY scripts/opd/analyze.py --suite E0-E6 --split dev
$PY scripts/opd/inputs.py manifest
$PY scripts/opd/inputs.py extract --split navtest --rank 0 --shards 32
$PY scripts/opd/suite.py test
$PY scripts/opd/train.py --exp E6 --stage smoke --variant IL_JOINT
```

`suite formal` 等待既定的14个调参任务完成，锁定选择后派发21个相同预算训练任务，随后在 cal/dev 评测。`suite test` 只允许完整测试输入缓存和锁定的 cal 选择；不会用 navtest 生成伪标签、gate 或选模型。E6 正式运行需要 `selection/E6_admission.json` 与资源预算文件；不满足证据门禁时不运行。

训练保存 optimizer、Python/NumPy/Torch/CUDA RNG、数据游标、完整场景顺序、实际学生链及固定噪声 probe。SIGUSR1 请求在当前更新完成后保存暂停；用同名参数加 `--resume` 恢复。已有完成标记的任务禁止覆盖。真实连续与中断恢复已独立比较参数和优化器状态。

压力脚本仅可按 `/proc` 命令、PID 与 start ticks 确认为已知压力 worker 后停止；真实他人工作不可终止。调度前重新读取 `/mnt/project/server_dispatch_policy.json` 并检查 GPU 实际空闲，结束后恢复本任务添加的 pause 标记。资源操作不改变科学协议。

## 官方前向契约与统一修正批次

首轮父目录中的 v1 是INVALID，只保留审计历史。不要将其权重、gate、教师chain或hidden states放入official_v2。物理metric cache和训练GT可复用。`base.yaml`中的official_conditioning必须为true，缓存的observation_contract及protocol_sha256会强制校验。

官方参考源码位于父目录references/official_recogdrive，HEAD必须是5831ee83c7a6a40bf6919715b96b0e9ba2b48335；仅审计使用。缺少时：

```bash
REF=/mnt/project/VLA-AD/outputs/recogdrive_opd_20261020/references/official_recogdrive
git clone --filter=blob:none --no-checkout https://github.com/xiaomi-research/recogdrive "$REF"
git -C "$REF" sparse-checkout set navsim/agents/recogdrive scripts/evaluation docs
git -C "$REF" checkout 5831ee83c7a6a40bf6919715b96b0e9ba2b48335
$PY scripts/opd/official_parity.py
```

Python3.9审计加载器只对官方模块推迟类型注解，不修改参考文件或数值运算。正式编码使用本仓库OPD显式兼容层，不依赖导入官方包。

实际修正编排命令：

```bash
$PY scripts/opd/prepare_data.py --manifest-only
$PY scripts/opd/inputs.py manifest
# 每个rank必须有独立合法GPU分配；一份manifest的全部32个rank均执行
$PY scripts/opd/inputs.py extract --split all --rank 0 --shards 32
$PY scripts/opd/official_rerun.py
$PY scripts/opd/suite.py confirm
$PY scripts/opd/suite.py e6
```

`official_rerun.py`等待名为official_inputs_0..31的真实资源任务完成，依次运行真实G0/G1、DDP/resume/在线一致性、teacher bank和E0、tiny、梯度尺度、两槽位调参、21个正式训练。它通过resources.launch和每次读取的live策略分配任务；不能绕过现有完成标记覆盖已完成结果。输入任务的实际jobs/spec.json可直接复查命令。新复现根目录须先建立相同的baseline SHA身份、protocol.registration.json和calibration/registration.json，再按这些任务规格执行；它们都随报告小体积证据和锁定protocol交付，不能用旧实验结果当新证据。

统计与最终审计：

```bash
$PY scripts/opd/analyze.py --phase baseline --split dev
$PY scripts/opd/analyze.py --split dev
$PY scripts/opd/analyze.py --phase baseline --split navtest
$PY scripts/opd/analyze.py --split navtest
$PY scripts/opd/additional_statistics.py --split dev
$PY scripts/opd/diagnostics.py chains
$PY scripts/opd/diagnostics.py gradients
$PY scripts/opd/gate_diagnostics.py register
$PY scripts/opd/gate_diagnostics.py analyze
$PY scripts/opd/figures.py --split dev --cases
$PY scripts/opd/figures.py --split navtest
$PY scripts/opd/certify.py models
$PY scripts/opd/certify.py evaluation --split dev
$PY scripts/opd/certify.py evaluation --split navtest
$PY scripts/opd/final_audit.py provenance
$PY scripts/opd/final_audit.py cache_versions
$PY scripts/opd/final_audit.py tests
$PY scripts/opd/report.py
$PY scripts/opd/export.py
```

训练描述集gate_diagnostics的评测需要先对注册manifest运行E3/E5三种子evaluate；不得把训练描述集混称dev泛化。完整每场景文件路径均由manifest和artifact_index给出。

## 完整独立重跑入口

不要在已有完成结果的目录执行初始化。下面入口会实际严格加载本地三个权重、建立固定成员manifest、运行CPU测试并派发32个真实输入任务，不生成dummy缓存：

```bash
export OPD_RUN_ROOT=/mnt/project/VLA-AD/outputs/recogdrive_opd_reproduce_NEW
$PY scripts/opd/pressure.py pause
$PY scripts/opd/initialize.py --dispatch-inputs
$PY scripts/opd/official_rerun.py
$PY scripts/opd/suite.py confirm &
CONFIRM_PID=$!
$PY scripts/opd/suite.py e6 &
E6_PID=$!
wait "$CONFIRM_PID"
wait "$E6_PID"
```

需要先有本仓库固定split源manifest、原始NAVSIM文件/地图、真实训练metric cache以及三个正式权重。`initialize.py --verify`核验现有根目录权重身份，不能将缺失核心权重标成通过。不同根目录不会覆盖正式结果。

额外的tsinghua-zt5在本次运行中空出后，先完成16个真实校准场景、四模型条件、5次采样的跨主机一致性验证，轨迹和评分误差均为0，才接入。正式测试路由锁定为40张独立A800、每GPU4个原生batch1进程、E0为80个shard（20组），每个已训练策略20个shard（5组），主测试共80个GPU任务。它只改变分片，不增加候选、采样种子或改变数值路径。科学protocol SHA不变。

新根目录默认只调度初始4台主机；只有执行zt5实际资格评测并生成`tests/real_new_host.json`后才允许第五台进入自动队列。可以不启用第五台，以原32-GPU路由复现相同数值结果。不要直接复制资格JSON冒充新检查。实际新增主机命令见`jobs/qualification_zt5/spec.json`，比较命令为：

```bash
$PY scripts/opd/verify_integrations.py new_host
$PY scripts/opd/final_audit.py hardware
$PY scripts/opd/final_audit.py model_sources
$PY scripts/opd/verify_integrations.py grpo_resume
```

最后确认所有实际任务完成后才执行`pressure.py resume`；只移除本任务拥有的标记，原压力自动化重新按30分钟均值观察，不启动临时填卡程序。若其他操作者已更改标记，脚本拒绝覆盖。
