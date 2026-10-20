# ReCogDrive 过程改进复现入口

固定起点为 `560bb257cd493613130f4b3643a0a2e4a88a7115`。本轮规范全文及 SHA256 保存在研究报告的 `specification/` 中；实验协议与小规模实测吞吐在 `configs/process_improvement/protocol.yaml`，运行时锁位于产物根目录。旧 OPD 目录只读。新代码不读取旧 E5 的科学门禁。

```bash
cd /mnt/project/VLA-AD/research/recogdrive-process-improvement-20261020
export PROCESS_RUN_ROOT=/mnt/project/VLA-AD/outputs/recogdrive_process_improvement_20261020
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/root/miniconda3/envs/navsim/bin/python
$PY scripts/process_improvement/build_splits.py
$PY scripts/process_improvement/correct_diagnostic_sampling.py
$PY scripts/process_improvement/run_suite.py --phase parity --resume
$PY -m pytest tests/process_improvement tests/opd -q
```

已有锁定总体时，`build_splits.py` 只验证身份并返回。新增缓存缺失不构成剔除场景的理由。原始缓存实际构建命令为下面的 24 分片形式，须由允许服务器上的独立 GPU 运行；分片不改变随机计划。

```bash
$PY scripts/process_improvement/prepare_cache.py --split all --rank 0 --shards 24
$PY scripts/process_improvement/verify_inputs.py
$PY scripts/process_improvement/audit_parent_ids.py
```

所有 GPU 作业应使用 `dispatch.launch` 或 `run_suite.py --host HOST --gpus ... --job-name UNIQUE`。调度入口每次读取 `/mnt/project/server_dispatch_policy.json`，逐卡检查 UUID、显存、利用率和本研究的持有记录。不允许覆盖已有作业或静默重试失败作业。`coordinator.py` 是实际使用的阶段编排器，会等待校准结果、锁定学习率、派发扩大规模 OPD/Anchor 和独立 GRPO，并在独立校准记录上验证局部过程目标。

```bash
$PY scripts/process_improvement/coordinator.py --once
$PY scripts/process_improvement/controller_control.py start
$PY scripts/process_improvement/controller_control.py status
$PY scripts/process_improvement/run_suite.py --phase scaling --method OPD --seed 42 --population train_full
$PY scripts/process_improvement/run_suite.py --phase local_diagnostics --rank 0 --shards 24
$PY scripts/process_improvement/calibrate.py --stage mechanism
$PY scripts/process_improvement/diagnose.py --name local_parent_confirm --split diagnostic_parent_confirm \
  --config configs/process_improvement/selected_process_parenthash.yaml --rank 0 --shards 8 --scoring-workers 4
$PY scripts/process_improvement/calibrate.py --stage process_gate
```

`--dry-run` 只打印命令，不生成完成或验收结果。训练入口支持 `--resume`，要求相同 world size、全局 batch、微批、初始化、数据、配置和最大计划更新数。中断用该进程的 `SIGUSR1`，在真实更新边界保存完整优化器、RNG、游标和预算后退出。新 run_id 不得复用未完成目录。完整场景的轨迹是一项联合动作。

```bash
$PY scripts/process_improvement/train.py --method OPD --split train_25k \
  --epochs 1 --lr 1e-5 --name tune_OPD_1e-5
$PY -m torch.distributed.run --standalone --nproc_per_node=8 \
  scripts/process_improvement/train_grpo.py --init IL --name GRPO_IL_42 \
  --score-budget 4000000 --stop-after 977 --seed 42 --lr 1e-5
```

GRPO 的 977、3907、15625 更新分别越过 25 万、100 万、400 万真实候选评分端点；应报告实际值和完整更新造成的超额。`--stop-after` 保存后暂停，不构成科学完成。只能用锁定的 cal-confirm 伤害边界决定提前终止；否则继续下一端点。普通 OPD 的物理评分为零，不能用评分预算决定其训练长度。

部署评价默认保持原生 B1、FP32、DDIM5、每种子一条输出。`--cuda-graph` 仅在完整 E0 数值门禁通过且代码哈希一致时启用，同样输出原生单条轨迹。独立进程并发最多三个/卡。跨场景 batch=32 和 batch=8 的加速资格测试未通过分数容差，未用于正式评价；失败目录保留。

```bash
$PY scripts/process_improvement/evaluate.py --name MY_STUDENT \
  --checkpoint "$PROCESS_RUN_ROOT/runs/MY_RUN/head_2341.pt" --split dev --cuda-graph
$PY scripts/process_improvement/analyze.py --methods MY_STUDENT --baseline B0_IL \
  --screening B0_IL_screen --split dev
$PY scripts/process_improvement/analyze.py --local local_parent_confirm --split diagnostic_parent_confirm
$PY scripts/process_improvement/export_student.py --name MY_STUDENT \
  --checkpoint "$PROCESS_RUN_ROOT/runs/MY_RUN/head_2341.pt" --verify
```

导出验证从真实历史图像重新编码，独立子进程只加载导出的学生和 VLM。审计钩子禁止读取教师、原 IL 权重、metric cache、原始观测缓存、监督库和过程改进模块。需与训练侧缓存输入的原生轨迹逐元素一致。

`navtest` 需要 `selection/final.lock.json`，只能对锁定最终模型和强对照运行。旧实验已经查看过该官方测试集，不能称为全新盲测。三种训练种子必须分别保存，五种推理种子不算五次训练重复。所有主张分别依据完整父驾驶记录 bootstrap 及安全分项；无数据的项目报告 `NOT_RUN`。

最终候选清单是显式 JSON 列表，每项包含 `name`、`checkpoint`，可选 `dev_name`（默认同名）和 `screening`。以下锁定入口要求对应的完整 dev 记录与权重、角色和随机种子一致；不会读取 navtest 分数或自动排序。生成的小体积锁必须先正常提交，随后评价还会核验当前 HEAD 中的锁字节、候选名、权重 SHA、种子及完整总体，单有外部文件不能放行任意模型。

```bash
$PY scripts/process_improvement/lock_final.py --candidates-file "$PROCESS_RUN_ROOT/selection/final_candidates.json" \
  --calibration-locks learning_rates.json warm_start.json grpo_rates.json process_gate.json
git add configs/process_improvement/final_eval_lock.json
git commit -m 'Lock final developed candidates before full Navtest confirmation'
$PY scripts/process_improvement/run_suite.py --phase final_eval --rank 0 --shards 24
```

早期 token-hash 诊断和选择保留为探索结果。纠正脚本在没有任何确认标签时锁定父记录哈希子集；正式编排自动读取纠正记录，不会沿用早期选择。`selected_process_exploratory_tokenhash.yaml` 仅用于重建旧探索结果。未知 GPU 占用不会被当作压力进程终止，资源变化只影响排程；同 world size 的迁移保存完整原生训练状态，不重置模型或优化器。
