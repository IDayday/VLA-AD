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

八台授权 canonical 主机的空闲设备均参与动态检查。新 8-GPU GRPO 初期使用非本地主机池，本地用于推进全量训练及短校准/诊断。过程门禁最终为 `NOT_SUPPORTED` 或已用尽登记增样的 `INCONCLUSIVE`、warm 已锁定、所有当前短任务成功完成后，空闲本地整机也可自动接入新 GRPO 训练线；失败/未完成目录不能触发交接，已有恢复任务不更换原主机。过程门禁支持时保留其后续训练容量，不让独立长对照抢占前置验证。

GPU 空闲快照采用两次观察，启动前仍原子复核，利用率回升不会放宽阈值。若原子检查明确在创建 worker 前拒绝，可在核对没有任何 `launch/owner/child` 记录后运行 `review_admission.py --jobs ORIGINAL_JOB_ID`，以新的作业 ID 重排；旧失败目录不删除、不覆盖。这个命令不处理网络超时或可能已启动的 worker，也不能用于重放已运行的优化器更新。复核映射记录在 `controller/admission_retry_map.json`。恢复的 GRPO 主机以真实训练配置中已保存的 owner 为准，不使用被拒绝的预计划主机。

每个 `jobs/JOB/child_N.json` 保留实际启动的代码哈希。后续作业还保存 `source_snapshot_N.tar.gz` 的小源码归档、归档 SHA 及当时 Git HEAD，覆盖新旧复用模块、基础规划器/评分器和小 YAML；不包含模型、缓存或环境凭据。它是共享工作区在启动时的源码快照，不冒充隔离 checkout 执行保证；旧作业只有其原始哈希，不能事后声称存在当时的源码归档。归档预检查失败时记录 FAILED 并拒绝创建子进程。

为防止整组训练被单卡队列持续分割，较高优先级的整组请求可以等待本研究已有短作业自然排空，并暂缓向该主机补入较低优先级任务。仅当全部八张卡均已确认空闲或由本研究正在运行的作业持有时才采用这项预留；存在未知/外部占用的主机不会因此锁住其余空闲卡。没有中断已有作业，实际启动仍重新检查物理空闲。预留状态保存在 `controller/state.json` 的 `reserved_for_gangs` 字段。

若局部门禁已关闭且短任务真实完成，本地只有 4–7 卡空闲时，可使用额外验证的四卡 GRPO 分区。准入要求 `tests/native_GRPO_partition32.json` 的真实全局 B32、三起点 4/8 卡和精确恢复检查通过，且训练器、规划器、随机计划、资格脚本哈希一致；CPU 路由测试不能替代。仅作用于全新本地正式训练线，八卡全空时仍优先八卡。恢复始终使用保存的原主机和 world size，不将运行中的八卡线转成四卡。执行规则在 `configs/process_improvement/grpo_partition_execution.json`，实际资格已通过且原检查器失败保留。

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
  --score-budget 4000000 --stop-after 977 --seed 42 --lr 3e-5
```

GRPO 的 977、3907、15625 更新分别越过 25 万、100 万、400 万真实候选评分端点；应报告实际值和完整更新造成的超额。`--stop-after` 保存后暂停，不构成科学完成。只能用锁定的 cal-confirm 伤害边界决定提前终止；否则继续下一端点。普通 OPD 的物理评分为零，不能用评分预算决定其训练长度。

部署评价默认保持原生 B1、FP32、DDIM5、每种子一条输出。`--cuda-graph` 仅在完整 E0 数值门禁通过且代码哈希一致时启用，同样输出原生单条轨迹。独立进程并发最多三个/卡。跨场景 batch=32 和 batch=8 的加速资格测试未通过分数容差，未用于正式评价；失败目录保留。

```bash
$PY scripts/process_improvement/evaluate.py --name MY_STUDENT \
  --checkpoint "$PROCESS_RUN_ROOT/runs/MY_RUN/head_2341.pt" --split dev --cuda-graph
$PY scripts/process_improvement/analyze.py --methods MY_STUDENT --baseline B0_IL \
  --screening B0_IL_screen --split dev
$PY scripts/process_improvement/analyze.py --methods MY_STUDENT --baseline B0_IL \
  --screening B0_IL_screen --split dev --cluster log
$PY scripts/process_improvement/analyze.py --methods MODEL42 MODEL43 MODEL44 \
  --baseline B0_RL --screening B0_IL_screen --split dev --result-name GROUP_vs_T0
$PY scripts/process_improvement/analyze.py --methods GRPO_WARM_42_step977 GRPO_WARM_43_step977 GRPO_WARM_44_step977 \
  --baseline-methods full_OPD_42_epoch1 full_OPD_43_epoch1 full_OPD_44_epoch1 \
  --screening B0_IL_screen --split dev --result-name R2_step977_vs_own_initial
$PY scripts/process_improvement/failure_audit.py --methods MODEL42 MODEL43 MODEL44 \
  --name GROUP --split dev
$PY scripts/process_improvement/analyze.py --local local_parent_confirm --split diagnostic_parent_confirm
$PY scripts/process_improvement/analyze_confirmation.py
$PY scripts/process_improvement/render_confirmation.py
$PY scripts/process_improvement/summarize_local.py --name local_parent_confirm --split diagnostic_parent_confirm
$PY scripts/process_improvement/diagnose_dynamic.py --name dynamic_parent_teaching --limit 16 \
  --checkpoint "$PROCESS_RUN_ROOT/runs/full_OPD_42/head_2341.pt"
$PY scripts/process_improvement/export_student.py --name MY_STUDENT \
  --checkpoint "$PROCESS_RUN_ROOT/runs/MY_RUN/head_2341.pt" --verify
```

导出验证从真实历史图像重新编码，独立子进程只加载导出的学生和 VLM。审计钩子禁止读取教师、原 IL 权重、metric cache、原始观测缓存、监督库和过程改进模块。需与训练侧缓存输入的原生轨迹逐元素一致。

R2 的三个种子从各自的 warm 权重开始，分析新增 RL 收益时用 `--baseline-methods` 逐一配对，不能把 seed42 的 warm 当作全部初始策略；该模式必须给独立 `--result-name`。方法间 M_minus 风险差仍使用同一个固定 IL/T0 参照，不把不同初始化混成共同基准。上述 R2 命令须等对应真实评价全部完成才能执行。

`infer_export.py` 可额外指定 `--benchmark-output PATH --benchmark-repeats 20`，在同一真实场景五组噪声上分别测量动作头和输入准备/图像 IO/VLM/动作头的热启动延迟。每种测量先暖机五次、逐次同步 GPU，并核验输出逐元素一致。动作头计时期间完整 VLM 仍驻留，所以该部分的总显存不是“独立动作头显存”；同时记录驻留与增量峰值。单场景重复计时不冒充总体时延置信区间，批量训练耗时不能替代本测量。

动态教学诊断只使用父记录哈希训练子集前 16 个固定场景、单个固定 roll-in 种子。在每个刚访问的新状态重新构造目标，并真实推进下一步，使用 M16/fit8/guard8、全部五个位置，独立报告调用和评分成本。它不是主训练每次仅一个位置的预算，也不是原生部署成绩；可同时评价一个冻结学生，但不通过该诊断更新或选择学生。历史 `dynamic_teaching` 目录保留，纠正后的结果使用独立名称。

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

```bash
$PY scripts/process_improvement/summarize_costs.py --snapshot-name actual_costs
$PY scripts/process_improvement/render.py --split dev
$PY scripts/process_improvement/render_full.py
$PY scripts/process_improvement/render_scaling.py
$PY scripts/process_improvement/plot_cases.py --method full_OPD_42_epoch1 --split dev
$PY scripts/process_improvement/plot_cases.py --method full_ANCHOR_42_epoch1 --split dev
```

成本表区分独占 GPU 作业租用时间和训练内 ledger，两者属于同一工作的不同观察口径，不能相加。失败、中断、资格检查、调参和评价作业保留；尚未同步的 rank 快照及缺失遥测用明确状态或 null 表示。早期一个 GRPO 学习率试验没有 DiT hook 计数，不能拿另一次训练的计数补成实测。原 IL/RL/VLM 的预训练成本未知，作为共同前置资产说明；warm 起点的本轮实际训练成本仍需计入后续方法。

固定原生GRPO评分端点图（只绘制已经完成完整dev的三个独立训练种子组，缺失组不补值）：

    $PY scripts/process_improvement/render_grpo.py

实际首轮执行生成1个已完成组；后续在生成对应R1/R2/R3三种子分析JSON后，用同一命令刷新。该图横轴为新增原生候选评分；完整成本另按实际ledger/lease和warm前置成本分析。

所选warm一轮的实际继承训练成本（核验锁定检查点、world1和准确端点日志，拒绝用其他epoch/rank/seed代替）：

    $PY scripts/process_improvement/summarize_warm_costs.py

该成本应按seed与R2/P2配对，完整研究成本还包括选择、缓存与评价；不重复加到已经包含warm训练的全作业lease总量。

调度器在原生GRPO固定端点恢复前保存 `runs/<name>/endpoint_states/<step>/full.pt` 与SHA收据，固定的是原始 `latest.pt` inode；训练器后续仍使用原生原子替换。该文件包含优化器和每rank RNG，可作同world恢复输入的独立保留副本，不是只含动作头的 `head_<step>.pt`。已覆盖的历史状态不重建、不用较新状态冒充；首次保护的缺失清单见报告表 `endpoint_state_initial_preservation.json`。完整状态文件均在gitignore输出目录。

新增物理主机的资格分两阶段执行，第一阶段在回收压力worker后的zt4跑三起点四更新复核，并保存实际IL第二更新的完整状态；第二阶段在另一已用A800主机恢复同一训练线，检查参数、优化器、每rank RNG和游标逐元素一致。必须通过实际第二阶段后，调度器才允许已暂停且相同world4的线在该合格主机对之间安排。两阶段分别运行（设备由严格调度器分配）：

    $PY scripts/process_improvement/qualify_grpo_host.py --stage source
    $PY scripts/process_improvement/qualify_grpo_host.py --stage destination

重复命令不会覆盖已完成资格线；不能把`PASS_SOURCE_ONLY_NOT_MIGRATION_ADMISSION`当作迁移通过。科学协议、训练核心源码和登记预算保持不变。

共同完成的原生GRPO首评分端点训练日志图（九条实际线的原始记录点，非dev学习曲线）：

    $PY scripts/process_improvement/render_grpo_dynamics.py

保存的局部过程trace个案（仅读缓存，零新增评分），以及已完成原生端点的地图/事件案例（每次真实展示重评分另记成本）：

    $PY scripts/process_improvement/render_local_trace.py
    $PY scripts/process_improvement/plot_cases.py --method GRPO_IL_42_step3907 --split dev

zt6重新实际空闲后的八卡资格另有独立执行前登记 `grpo_host8_execution_registration.json`。源阶段复核三起点world8并保存真实第二步状态；只有源阶段通过后才可进行已登记的短暂本地维护。以下操作只针对该登记中两条本任务原生线，先停止CPU协调器，以SIGUSR1请求原生完整更新边界保存；不使用SIGKILL，不重置任何训练状态。目的阶段不论通过与否，都恢复这两条线到原来的world4、主机和3907目标，确认实际更新后才启动协调器：

    $PY scripts/process_improvement/qualify_grpo_host8.py --stage source
    $PY scripts/process_improvement/controller_control.py stop
    $PY scripts/process_improvement/maintenance_native.py pause
    $PY scripts/process_improvement/qualify_grpo_host8.py --stage destination
    $PY scripts/process_improvement/maintenance_native.py resume
    $PY scripts/process_improvement/controller_control.py start

GPU命令通过严格dispatch分配对应8张实际空闲卡；不要在未通过源资格或训练仍占卡时直接运行目的阶段。维护中间head只作完整恢复证据，不加入候选、cal/dev评价或选模。world8迁移须有独立的实际cross-host PASS及未变源码，不能借world4资格放行。zt6暂停标记也在真实研究结束后按本任务tag单独释放。
