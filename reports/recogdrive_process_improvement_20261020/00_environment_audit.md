# 环境与数据总体审计

研究分支为 `research/recogdrive-process-improvement-20261020`，独立 worktree 从指定基准 `560bb257cd493613130f4b3643a0a2e4a88a7115` 创建。原工作区以及旧 OPD 产物保留。

完整读取的任务规范实际位于原工作区 `experiments/recogdrive_expert/recogdrive_process_policy_improvement_codex.md`；副本在 specification 目录，SHA256 为 `4e0e126af6fb9e7d37c8dd8ac3aaeb175453f6f43bcbf9518d92559428622268`。

实际重建官方 navtrain 的 1,192 个日志、103,288 个唯一 token，全部合法，无重复合法 token、无缺失必要图像/历史/未来/地图，metric cache 全部存在。没有用旧隐藏状态缓存的覆盖范围决定训练总体。

按完整父驾驶记录划分后，训练集 74,885 场景、903 个日志、127 个父记录。嵌套子集 25,958 / 50,039 场景。历史 4,115 场景全部保留为独立控制。保留旧 cal 并按固定 hash 补充至 20 个父记录后，校准集实际为 16,005 场景；按父记录拆为 tune 5,316、confirm 5,589、promotion 5,100。完整父记录约束使其大于 3,000 场景最低目标。

既有开发集保持 1,696 场景、16 个日志、15 个父记录；其父记录所有其他片段同样从本轮训练排除。navtest 12,146 场景仅作最终锁定确认，历史上已经查看过，不能称完全盲测。底模可能见过全部官方 navtrain，本轮留出不代表底模从未见过。

主环境 `/root/miniconda3/envs/navsim/bin/python`，依赖锁见 environment.lock.txt。现场核验的授权服务器为 vla-zt、rl-zt2 和 tsinghua 系列；仅使用现场确认空闲或已授权回收的压力 worker GPU。保留其他真实研究进程。压力暂停操作只针对 `/tmp/zt-idle-pressure-v1/agent.py --worker GPU-*` 的精确 argv、UID、PID/start ticks；没有新建或手动启动压力程序。

扩大训练期间补充核验了全部八台授权 canonical 主机：共 64 个不同物理 GPU UUID（`tables/eight_host_physical_uuid_inventory.json`）。这不等于 64 张卡均空闲：本研究最初并用五台的 40 张卡，随后多台出现无法归属到容器可见 PID 的负载，未终止未知进程或重置 GPU。OPD 的两条训练线在更新 664 保存完整原生状态后迁移至本地，同 world size 恢复继续；现场可直接使用的资源一度为本地和 rl-zt2 共 16 张卡。调度器持续检查全部八台的实时权限、物理显存/利用率与持有记录，有空闲时自动接入。完整 8-GPU GRPO 的新调度暂保留本地给短校准及机制诊断，避免长训练阻塞这些前置工作。

2026-10-21 06:44 UTC复核发现部分远端没有`rg`，此前`resources.py`的GPU物理清单仍有效，但该工具的进程过滤子命令失败，不能将它当作完整进程可见性证据。现改为优先rg、缺失时grep的明确回退，并兼容cgroup v1/v2。实际重查确认Tsinghua主节点有受保护的独立Qwen训练；Tsinghua-zt4新出现八个精确匹配的`/tmp/zt-idle-pressure-v1/agent.py --worker GPU-*`。按已有用户授权、UID/PID/start ticks/argv检查，仅暂停这八个压力worker，之后八卡物理显存与利用率均归零。其他节点仍无法归属的负载不处理。原检查、错误及修正后的进程证据保留在外部audit目录。

zt4加入本轮第六个压力暂停标记，结束时仅释放本任务自己的标记。正式迁移须先完成新增三起点真实host数值检查和跨主机同world完整状态恢复资格；不将新空闲资源直接视为数值资格，不移动正在训练的线，不改变world、优化器或科学预算。执行前登记见 `configs/process_improvement/grpo_host4_execution_registration.json`，对应真实资格评分预算4,096，不计为新训练种子或研究性能结果。

08:42 UTC前后，zt6原先不可归属的负载自行退出；未向它们发送信号。现场六卡物理全空、另外两卡仅为本任务端点评价，调度器已自动接入评价。设置本任务第七个压力暂停标记时释放worker数为0，仅防止等待资格阶段重新启动已知填充器。八卡训练另登记独立资格及一次本地两条自有线的原生安全保存/恢复，详见 `grpo_host8_execution_registration.json`；源阶段尚未通过前不暂停训练，目的阶段无论结果如何均恢复原有科学线，不让工程准入失败阻断独立实验。

本轮原始 VLM 缓存按 token 前缀分块，复用旧 official_v2 合法缓存；写入新缓存有 SHA256、源 VLM、输入契约与可恢复索引。共享磁盘现场约 2.3 TiB 空闲；本研究新缓存上限 900 GiB，并保留至少 512 GiB 空间，避免预先要求 TB 级全量缓存。

可追溯数据总体、排除原因和源日志哈希在 tables/population_summary.json 及 configs/process_improvement/splits 中。所有结果尚须按后续完整实验判定，工程门禁不构成研究假设支持。

2026-10-22 00:26:39 UTC，共享实时资源策略明确撤回 training-vlawm-tsinghua-zt6。00:38复核后停止该主机所有新安排，调度器现先按实时许可过滤队列，再查询容量；原子dispatch仍做第二次许可检查。未继续SSH或观察该主机，未运行world8目的资格或维护暂停，四条真实活动训练未中断。此前未执行的00:29维护修订也保留并被显式取消；记录见 grpo_host8_execution_withdrawal.json。历史源资格不是迁移PASS；既有七个暂停标记中zt6的标记不在撤回后释放，不以恢复压力为由重新连接撤回主机。其余六个本任务标记在实际结束时按实时权限和拥有证据处理。

00:38对其余七台的实际检查没有发现可归属的空闲卡，多个主机有其他真实工作；本轮不终止它们。原生GRPO仍有24张实际物理卡上的四条线，其他五条在各自1M端点完整暂停，排队等待容量。共享执行可能影响wall time；科学评分预算、global batch和优化器目标保持原锁定值。

2026-10-22用户明确要求停止本实验。所有相关训练/调度已保存并退出，资源释放；停止后的高预算及最终Navtest不再执行。完整收据、163项CPU检查及状态/成本归档见08_user_stop_and_resource_release.md。本段覆盖此前按时间记录的运行/等待状态，不改写历史实验事实。
