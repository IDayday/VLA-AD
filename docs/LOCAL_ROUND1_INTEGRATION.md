# 本机教师重建与 Round1 集成契约

用户最新纠正：本轮八个蒸馏 run 必须使用**本机完整重拟合并实测、随后冻结的教师**；不能回退为原 PPU 教师目标。原 `6bb7119` 的资产、历史分数和指南中的历史 gain1 计数仅作证据核查。学生均从原 ReCogDrive RL Base 初始化，所有八组共享同一冻结本地教师。

代码工作树：`/mnt/project/recogdrive-no-routing-round1-20261026`，分支 `research/no-routing-round1-20261026`。冻结证据树保持不变，新增模块置于 `student_distillation/`。正式模型/缓存/报告根目录：`/mnt/project/recogdrive-no-routing-round1-results-20261026`。

本地重拟合工作目录：`/mnt/project/VLA-AD/experiments/recogdrive_dc8_refit_20261026`。
源码补包：该目录的 `source_snapshot/ReCogDrive/`，提交来源 `f85d4e557b12a7d05f5deb8d1640834cacd94d32`。静态检查4547文件、2607Python文件及最终8021/2021会话隔离已通过。

## 统一接口

- `data.SceneDataset(manifest, population, target_version)`：manifest 为新目标/特征身份 JSON 路径。`__getitem__` 返回 FP32 `base_post[L,384]`、bool `image_mask[L]`、`text_mask[L]`、FP32 `state24[24]`、`teacher_actual_shift[384]`、`history12[12]`、`status8[8]`。`metadata` 保留 token/log/index/来源等审计字段；模型前向只读取前述普通输入。data 模块另提供 `batch_scene(item, device)` 增加 batch1 维并搬运张量。
- `adapter.ContinuousD48Adapter(basis, d0, token_mean, token_std, state_mean, state_std, cap_in_d0)`，forward 依指南返回字典，至少含 `post`、`delta`、`coefficients_raw`、`actual_rms`、`valid_mask`。
- `native_bridge.NativeDDIMBridge(frozen_head)`：`encode_context(history12[B,12],status8[B,8])`、`mean_from_post(actions,step_index:int,post,context)`、`states_from_post(post,context,noise_tape[6,B,8,3],stop_before:int|None)`。完整返回含 states/means/trajectory；stop_before 时返回动作 tensor。
- `losses.scene_loss(batch,adapter,bridge,cfg,update,calibration)` 返回 loss 和可序列化统计；cfg 可用 mapping，禁止根据来源标签作分支。`enable_transition_loss=False` 的 A/B 从不运行桥接。
- `noise` 由模型/桥接开发者负责，提供 `train_noise_seed`、`training_schedule(token,seed,epoch_index)` 返回 Python 整数 time_index/noise_bank，以及 `make_noise_tape(seed,device)` 按原生形状连续六次抽样。评价噪声须支持已校验的显式参考带，不只设 seed。
- calibration JSON 至少含 `h_mse_scale`、`mu_mse_scale[5]`、`token_mean[384]`、`token_std[384]`、`state_mean[24]`、`state_std[24]` 以及固定 Train 清单和身份 SHA。
- `asset_identity.load_library_bundle(path)` 读取本机新冻结库 manifest，返回 basis_FP64[384,48]、D0、cap_in_d0、参数/源码/本地教师选择身份；未就绪必须明确报未就绪，不回退旧库。
- `effective_config` 采用指南 common.yaml 结构，另有 `paths`：`frozen_repo`、`checkpoint`、`vlm`、`teacher_bundle`、`feature_manifest`、`targets_manifest`、`calibration`、`results_root`。正式配置在本机教师冻结、资产验收后由主 agent 展开并注册。

## 开发所有权

数据开发者：asset_identity.py、build_targets.py、data.py、feature_cache.py、calibration.py、data相关 tests。

模型开发者：adapter.py、native_bridge.py、noise.py、losses.py、model/bridge相关 tests。

训练评测开发者：distributed.py、checkpointing.py、train.py、evaluate.py、select_checkpoint.py、aggregate.py、run_suite.py、experiment_registry.py、四组yaml/启动脚本、training相关 tests。

主 agent：本地教师重拟合流水线、真实资产/数值验收、公共接口集成、统一GPU调度、实验注册和最终报告。其他开发者不能自行启动长期GPU任务或同时改所有权外核心文件；发现接口问题立即发送消息讨论。

八个新 run 为 A/B/C/D×20261027/20261028，每个85813×40 epochs、1341 updates/epoch、53640总更新，valid17475、test12146。旧九组学生实验仍暂停。完整任务到 REPORT_COMPLETE 才完成；不能把 smoke 或后台提交当交付。
