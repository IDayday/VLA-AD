# 执行与复现

工作目录：`/mnt/project/VLA-AD/research/recogdrive-opd-20261020`。
解释器：`/root/miniconda3/envs/navsim/bin/python`。
产物根目录：`/mnt/project/VLA-AD/outputs/recogdrive_opd_20261020`。权重、完整日志、候选与输入缓存不进 git。

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
$PY scripts/opd/inputs.py extract --split navtest --rank 0 --shards 14 --with-metrics
$PY scripts/opd/suite.py test
$PY scripts/opd/train.py --exp E6 --stage smoke --variant IL_JOINT
```

`suite formal` 等待既定的14个调参任务完成，锁定选择后派发21个相同预算训练任务，随后在 cal/dev 评测。`suite test` 只允许完整测试输入缓存和锁定的 cal 选择；不会用 navtest 生成伪标签、gate 或选模型。E6 正式运行需要 `selection/E6_admission.json` 与资源预算文件；不满足证据门禁时不运行。

训练保存 optimizer、Python/NumPy/Torch/CUDA RNG、数据游标、完整场景顺序、实际学生链及固定噪声 probe。SIGUSR1 请求在当前更新完成后保存暂停；用同名参数加 `--resume` 恢复。已有完成标记的任务禁止覆盖。真实连续与中断恢复已独立比较参数和优化器状态。

压力脚本仅可按 `/proc` 命令、PID 与 start ticks 确认为已知压力 worker 后停止；真实他人工作不可终止。调度前重新读取 `/mnt/project/server_dispatch_policy.json` 并检查 GPU 实际空闲，结束后恢复本任务添加的 pause 标记。资源操作不改变科学协议。
