# 环境与数据总体审计

研究分支为 `research/recogdrive-process-improvement-20261020`，独立 worktree 从指定基准 `560bb257cd493613130f4b3643a0a2e4a88a7115` 创建。原工作区以及旧 OPD 产物保留。

完整读取的任务规范实际位于原工作区 `experiments/recogdrive_expert/recogdrive_process_policy_improvement_codex.md`；副本在 specification 目录，SHA256 为 `4e0e126af6fb9e7d37c8dd8ac3aaeb175453f6f43bcbf9518d92559428622268`。

实际重建官方 navtrain 的 1,192 个日志、103,288 个唯一 token，全部合法，无重复合法 token、无缺失必要图像/历史/未来/地图，metric cache 全部存在。没有用旧隐藏状态缓存的覆盖范围决定训练总体。

按完整父驾驶记录划分后，训练集 74,885 场景、903 个日志、127 个父记录。嵌套子集 25,958 / 50,039 场景。历史 4,115 场景全部保留为独立控制。保留旧 cal 并按固定 hash 补充至 20 个父记录后，校准集实际为 16,005 场景；按父记录拆为 tune 5,316、confirm 5,589、promotion 5,100。完整父记录约束使其大于 3,000 场景最低目标。

既有开发集保持 1,696 场景、16 个日志、15 个父记录；其父记录所有其他片段同样从本轮训练排除。navtest 12,146 场景仅作最终锁定确认，历史上已经查看过，不能称完全盲测。底模可能见过全部官方 navtrain，本轮留出不代表底模从未见过。

主环境 `/root/miniconda3/envs/navsim/bin/python`，依赖锁见 environment.lock.txt。现场核验的授权服务器为 vla-zt、rl-zt2 和 tsinghua 系列；仅使用现场确认空闲或已授权回收的压力 worker GPU。保留其他真实研究进程。压力暂停操作只针对 `/tmp/zt-idle-pressure-v1/agent.py --worker GPU-*` 的精确 argv、UID、PID/start ticks；没有新建或手动启动压力程序。

本轮原始 VLM 缓存按 token 前缀分块，复用旧 official_v2 合法缓存；写入新缓存有 SHA256、源 VLM、输入契约与可恢复索引。共享磁盘现场约 2.3 TiB 空闲；本研究新缓存上限 900 GiB，并保留至少 512 GiB 空间，避免预先要求 TB 级全量缓存。

可追溯数据总体、排除原因和源日志哈希在 tables/population_summary.json 及 configs/process_improvement/splits 中。所有结果尚须按后续完整实验判定，工程门禁不构成研究假设支持。
