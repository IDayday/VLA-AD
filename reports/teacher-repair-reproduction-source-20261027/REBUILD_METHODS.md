# 复现顺序：输入、配对状态、离线库和蒸馏训练

本文件将需要重新生成的大缓存与保留的算法/配置分开。所有路径以下均相对于解压后的 `ReCogDrive/`。原代码和配置按字节保存；此交付不启动任何生成、优化、训练或 DLC 作业。

## 1. 固定身份，再重新生成缓存

先阅读 `PATH_MAPPING_AND_EXTERNAL_ASSETS.json`、`REPRODUCTION_KEY_CONFIGS.json` 和 `AUDIT_CONTEXT/`。核对 VLM/规划器权重 SHA、tokenizer/预处理源码、人口清单、全局 index/token/session 对应关系、精度、噪声规则、实际 Torch/设备库版本和 attention/norm 设置。

原路径为 `/mnt/workspace/ReCogDrive`、`/mnt/data`、`/mnt/yehr_workspace`。建议在独立容器中映射这些路径，给新实验分配独立输出目录；不要覆盖冻结的计划、源文件或验收记录。原配置的缓存 SHA 是历史身份，重新生成后必须给新清单和新配置记录实际 SHA 与原配置来源，不能只改旧文件中的 SHA 或跳过检查。

固定场景 `0002267a294e52a7` 的原始输入仍在仓库 `reports/ppu-a800-embedding-audit-20261027-v2.tar.gz`。先通过该场景的输入/参数/算子审计，再扩大缓存生成。相同 seed 和源码不能保证不同设备构建逐位相同；`AUDIT_CONTEXT/PPU_CONFIGURATION_AND_A800_REQUIREMENTS.md` 保留了本次对齐要求，不代表全链已对齐。

## 2. 原生 Base 与配对状态生成

| 步骤 | 算法/实现 | 输入与输出约束 |
| --- | --- | --- |
| 原生输入和特征 | `1016-rl_representation_analysis/code/worker.py`、`vlm_core.py`、`vlm_intervene_v4.py`、`code/upstream/` | 从原始场景按原预处理构造实际模型输入，记录身份；模型/评分配置见该阶段 `configs/`。最终 portable 的 `vendor/` 保存交付使用的模型与评分源码。 |
| 可微修正与后缀 | `1018-representation_repair_batch2/code/r2_counterfactual.py` | 冻结参数，通过完整原去噪后缀预测梯度；实际舍入后的有界编辑与真实评分由源代码检查。 |
| 配对状态优化 | `1022-gt_assisted_representation_optimization/code/g5_runtime.py` 的 `optimize`、`configurations` | 基础配置是 pre/post 两个接口，cap 为 0.005/0.02/0.1，24 步 Adam，shared 模式，lr_multiplier=1；实际任务可以提供自己的 `task['configs']`，必须沿用对应任务配置。 |
| 选择与导出 | 同文件 `export_teacher`；`g5_replay.py`、`g5_export_saved_selection.py`、`g5_accept_export.py` | 使用真实 S0 PDMS 选择；导出实际输入、Base/教师接口、选中 delta、轨迹/质量和 seed；pre 编辑位置和 post384 状态不可互换。 |
| 接续导出 | `resume_20261023_shared/code/` 和 `g5_export_retry_v1.py` | 保留接续版本、依赖身份和原控制器方法。旧 JobId、节点名和配额仅为历史配置，新作业应使用接收端自己的资源。 |

`configs/EFFECTIVE_GT_OPTIMIZATION.json` 固定 root_seed=20261008、`sha256(token|root_seed|j)` 的噪声规则、5 个去噪时刻 `[80,60,40,20,0]`、FP32 planner 和评分权重。该原 GT 优化协议使用 S0 选中修正、S1/S2 转移验证；最终 frozen portable 的五噪声场景选择/输出协议以它自己的源码和 manifest 为准，不将两种协议混用。

大型 `TEACHER.npz`、逐场景优化轨迹、hidden-state、评分缓存和原生 Base 缓存均重新生成。`REGENERATE_INSTEAD_OF_BUNDLING.json` 记录未附产物的原路径、长度和 SHA；这些 SHA 用于追溯历史身份，不要求新平台的生成物未经验证就与其相同。

## 3. 配对数据、几何和最终六库

1. `stage2/code/p2_u1_paired_assets_v1.py` 读取已有状态并构建 Base/教师均值、残差、角色 mask 与统计；`p2_u1_paired_mapping_math_v1.py` 提供配对映射。先恢复真实状态，不能用选中 delta 或伪造 post 状态替代。
2. `offline_library/code/ol_l0_prepare_v1.py`、`ol_l0_replay_v1.py` 构造/复核 L0 原生配对状态；`ol_l1_data_pack_v1.py` 生成实际 pairs/ordinary 数据包。`pack-pairs`、`pack-ordinary` 的 CLI 需要 `--full1850`、`--release`、`--indices`、`--output`，可配置 `--workers`。完成实际数据验收后才能生成后续 data proof。
3. `ol_session_cv_metadata_v1.py` 提供 session 划分算法；实际冻结的 `preparation/session_cv_metadata_v1/SPLITS.json` 已附，seed=2026102531。应优先复用同一划分，而不是在新缓存上重新随机抽样。
4. `ol_l1_geometry_v1.py`、`ol_l1_operators_v1.py`、`ol_l1_pilot_fitter_v1.py`、`ol_l2_cal_pack_v1.py`、`ol_l3_outer_refit_v1.py` 实现几何、13 项算子、拟合、校准与外折重拟合。各 JSON 计划、合约和代码内固定超参已附；不将 inner calibration 或 outer held 教师混入拟合。
5. `ol_stage_a_final_factory_v1.py` 重建需要复用的 A 库；`ol_stage_b_final_factory_v1.py` 提供 generalized 数学实现。最终入口是 `full_six_evaluation/code/full_six_factory_v2.py`，配套 `preparation/v2/` 与 `analysis/v2/`。其命令阶段为 `prepare-catalog` → `prepare-reused` → `prepare-inner` → `data/space/fit-inner` → `pack-pilot/pack-cal` → 原生校准/collect → `freeze` → `prepare-alltrain` → `refit-alltrain` → `finalize-catalog`。原生评分、数据/执行许可检查在中间必须实际完成，不能按箭头跳过。
6. 六个 family 固定为 D_P7_original、D_C8_original、R_P8_double、R_C8_double、D_P8_double、D_C8_double；位于 feature_encoder 后、DiT 前，宽度384，最终共享子空间48。精确 route/order/K/dose/槽位方法见 `USER_FIXED_CATALOG.json`、`FINAL_SLOT_FREEZE.json` 和最终六份 program。

上述 `stage2/` 路径均位于 `1024-representation_distillation_diagnosis_batch5/` 下，`offline_library/` 位于该 `stage2/` 内。`CLI_INDEX.json` 从源码静态提取所有 argparse 参数和位置，方便查真实参数名；没有运行任何实验模块的 `--help`。

原代码保留 ROOT/release、source hash 和数据验收要求。重新生成数据后，应为新 run 完成实际检查、形成自己的 release 与验收记录；包内旧计划/许可用于复核方法，不是新计算已获验收的凭据。

## 4. 训练和损失校准

| 路线 | 原入口和关键配置 | 原始主要设置 |
| --- | --- | --- |
| 第四批主蒸馏 | `1021-representation_distillation_batch4/code/d4_train_v4.py`；`configs/distillation_main_v4/{H,T,HT}.json`；`d4_calibrate_losses_v1.py`、`LOSS_CALIBRATION_V1.json` | 各组完整配置已附。以 H 为例：LoRA rank16/alpha32/dropout0，LoRA LR=5e-5、projection LR=1e-4，AdamW，global batch64、microbatch1、5 epochs、warmup200、clip1、weight_decay0；种子与 H/T/HT loss 权重以各自 JSON 为准。 |
| 局部蒸馏 stage2 | `stage2_local_distillation/code/s2_train_v2.py`；`configs/formal_v1/{H,H_interface,T,HT}.json` 和 FREEZE 记录 | 以 H 为例：LR=1e-5，10 epochs，global batch64、microbatch1、warmup133；local scales、lambda_T 和具体 train_indices 均保留原值。 |

第一条入口需要 `--config --run --rank --world-size`，可选 `--resume`。第二条入口需要 `--config --group --run-dir`，可选 `--engineering --max-steps --resume`。分布式启动方式和 worker 资源依赖原 launcher/配置；这些参数说明不意味着可绕过数据检查直接开训。

训练/校准索引、五折 session 清单以及 portable train/validation/test 人口清单已附。训练所需的大型 scene inventory 和 teacher_targets 可按 `d4_prepare_sources_v1.py`、`d4_training_sources_v2.py`、`s2_prepare.py` 及相应源码重新生成；需记录新产物 SHA，并保持相同 index/token/session 及固定样本顺序。历史实验有多条研究路线，不能把不同配置的 loss、数据资格或选择规则拼成一条未经验证的新协议。

先运行包根 `verify_sources.py`，确认源码/配置字节、关键模块和精确划分通过，再准备新 run。权重和完整 NAVSIM 数据仍由接收端提供，本包没有安装或改变环境。
