# 本机教师重建执行代码

本目录保存本轮实际使用的执行代码；不包含模型、原始数据或生成张量。方法只选择 **D / C_to_P / K8 / double**。数学研究源码来自审计分支提交 `f85d4e557b12a7d05f5deb8d1640834cacd94d32`，原冻结教师交付证据基线为 `6bb7119583513094a169b4c2f2dc2a05ac293aa3`。

参见 [具体方法和数值要求](METHOD_PROTOCOL.md) 与 [本地协议补充](../docs/LOCAL_PROTOCOL_ADDENDUM.md)。旧 PPU 参数和选择不进入本机新教师拟合或学生监督。

当前实例在独立实验目录运行。代码副本与当前实例的对应关系由 `CODE_MANIFEST.json` 记录；运行前核实资产配置和代码 SHA。恢复现有实例时先读取 owner 锁、实际 PID、状态和已完成产物，不要额外启动第二个控制器。

执行顺序：

1. `native_base_worker.py` 从本地原始数据准备配对优化输入及 Validation post384，按 token/global index 核对旧本地 Base 缓存。Train/Test/Validation 的完整 FP32 mmap 由 `student_distillation.feature_cache` 保存。
2. `gt_opt_worker.py` 复用原冻结优化函数执行 GT 配对目标的完整有限搜索；`gt_opt_controller.py` 保留原始控制器，`gt_controller_fast.py` 只优化校验文件的读取，沿用原科学函数和严格选优规则。
3. `local_teacher_suite.py` 等待全部已登记 GT 工作者实际成功退出，再执行内层 D48/K8 拟合、pilot、校准、最终重拟合、冻结及完整 Train/Test Oracle 评分。随后使用额外评价 GPU 独立完成 Validation 教师诊断。
4. `round1_prepare_and_run.py` 等待本地 Train/Test 教师与实现验收完成，导出独立的 Train/Test 教师目标、完成 Train-only 定标和真实多卡/梯度/恢复验收，再真正执行八个完整学生 run、所有规定评价与报告。

脚本默认路径对应当前已核实实例。另建实验时，先把 `code/` 与所需 `engineering/` 文件复制到新的实验根目录，按 CLI 的 `--root`、`--round1-repo`、`--features`、`--paths`、`--source-root` 等参数明确指定新位置；勿直接在代码证据目录内生成数据。部分源码快照保留原作者路径，仅用于追溯；应按交接指南完成当前机器的路径映射。

运行环境使用已验收的本机 Native Python，设置 `ZT_AUTO_PROXY=0`、`OPENBLAS_CORETYPE=Sandybridge`，BLAS/OMP 线程数为 1。学生 CUDA 进程另使用已核验的 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。原 VLM、采样器、完整 padding 和头部 FP32 数值路径保留。

长任务通过稳定启动器保存 commands、PID、日志与实际退出状态；检查点和缓存原子提交。资源文件区分正式训练与独立诊断 GPU，并在每次启动时核对当前授权、设备 UUID 和空余显存。

`engineering/` 下的核验程序和成本/配对比较汇总器也属于交付代码。成本按 GPU UUID 合并重叠时间区间，分别报告物理占用时间与进程墙钟时间；S0–S4 是动作噪声重复，两次训练 seed 独立报告。

只有完整产物和报告通过最终验收才能标记 `REPORT_COMPLETE`。代码可执行、工程检查通过、后台作业已启动均不是实验完成。
