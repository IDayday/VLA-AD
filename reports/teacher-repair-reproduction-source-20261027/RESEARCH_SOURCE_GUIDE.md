# 教师特征修复：算法、训练配置与缓存重建交付

本包从 `/mnt/data_and_weight/WM_Group/sjc_workspace/ReCogDrive` 只读采集，保留原目录结构和文件字节。它补齐此前 portable 包中 8 份算法摘录的研究依赖，并包含配对状态优化、拟合、校准、六库生成和交付脚本。没有启动训练、重新生成教师特征，也没有修改原研究目录、模型或计算设置。

**版本边界：** `1026-interface_teacher_portable/repo` 冻结提交为 `6bb7119583513094a169b4c2f2dc2a05ac293aa3`。相邻历史研究目录并不属于该提交；它们由 `SOURCE_MANIFEST.json` 的逐文件 SHA256 冻结。两个 portable 仓库各有独立的 `*_GIT_PROVENANCE.json`。原始 8 份算法档案与其完整研究树中的对应源码会逐一核对字节。

## 收录范围与入口

压缩包解压后，源码统一位于 `ReCogDrive/`。所有列出的研究阶段，其算法源码、脚本、`config/` 或 `configs/` 完整收录；manifests 保留精确划分与关键身份清单。大型缓存/逐场景结果/冗长资产库存按用户要求省略并记录重建身份。

| 原目录 | 收录内容与阅读入口 |
| --- | --- |
| `1016-rl_representation_analysis/` | 第一批表征/修复研究、`paired_observation.py`、修复引擎、模型/NAVSIM upstream 源码、baseline 与 paired 配置。 |
| `1018-representation_repair_batch2/` | 第二批修复代码与协议；`r2_counterfactual.py` 提供可微冻结后缀与有界修正；包括 DLC 常驻执行器和提交/队列代码。 |
| `1021-representation_repair_batch3/` | 第三批修复、教师/候选库工程代码、配置和资产清单。 |
| `1021-representation_distillation_batch4/` | 蒸馏、损失校准、训练/验证索引；包含 `stage2_local_distillation/` 的完整代码、配置和清单。 |
| `1022-gt_assisted_representation_optimization/` | GT 辅助逐场景状态优化；`code/g5_runtime.py` 的 Adam 优化、真实 PDMS 选择、回放与状态导出；包含 `provenance/source_v*` 原版本及 `resume_20261023_shared/code/` 接续代码。关键配置为 `EFFECTIVE_GT_OPTIMIZATION.json`、`FROZEN_MODEL_AND_SCORING.json`。 |
| `1024-representation_distillation_diagnosis_batch5/` | 第五批诊断与 session 划分；完整 `code/` 和 `stage2/code/`。 |
| `…/stage2/offline_library/` | 完整离线库研究源码，包括此前缺失的 `ol_l1_data_pack_v1.py`、`ol_stage_b_final_factory_v1.py`、`ol_l3_outer_refit_v1.py` 及其本地依赖、元数据准备、拟合、校准、选择和原生评测脚本。 |
| `…/offline_library/full_six_evaluation/` | 最终六库的 `full_six_factory_v2.py`、工厂/校准/评测流水线；`preparation/v2/` 的实际目录、INNER/ALLTRAIN 计划、执行许可记录、最终 catalog 与六份 program；`analysis/v2/` 的实际槽位冻结配置。 |
| `1020-representation_teacher_portable/repo/`、`1026-interface_teacher_portable/repo/` | 两份 portable 教师仓库的全部算法/运行时/vendor 源码、配置、人口清单与合约；小型拟合库参数、golden 和逐场景选择产物本包不重新附带，可由保存的方法和冻结方案重建。 |
| portable 导出目录及 `1022-representation_research_synthesis/code/` | 导出、打包、回放补充、发布和研究综合脚本。 |

此外收录 preparation/provenance 中的源代码与配置元数据、冻结选择记录和可静态解析到的已有配置文件。边界记录在 `REGENERATE_INSTEAD_OF_BUNDLING.json`、`SNAPSHOT_NOTES.json`；正在被后台任务更新的监控/锁文件不属于冻结配置，已排除。

## 配对状态与算法调用链

1. `1018…/code/r2_counterfactual.py` 与 `1022…/code/g5_runtime.py`：冻结模型后缀、优化 delta、实际打分选择和按噪声回放；保留 GT 辅助/特权输出的研究属性。
2. `stage2/code/p2_u1_paired_assets_v1.py`：读取已有 Base/选中教师状态，提取真实配对均值、残差和角色 mask；不把 pre1536 编辑伪装成 post384 教师。
3. `offline_library/code/ol_l1_data_pack_v1.py`：普通输入与真实配对数据的身份、资格与读取约束。
4. `ol_l1_geometry_v1.py`、`ol_l1_operators_v1.py`、`ol_l1_pilot_fitter_v1.py`：空间、聚类、13 项算子、MLP/配对拟合；`stage2/code/p2_u1_paired_mapping_math_v1.py` 提供配对映射数学实现。
5. `ol_l2_cal_pack_v1.py`、`ol_l3_outer_refit_v1.py`、`ol_stage_a_final_factory_v1.py`、`ol_stage_b_final_factory_v1.py`：校准、外折重拟合、原有库工厂。
6. `full_six_evaluation/code/full_six_factory_v2.py`：实际最终六库入口，通过私有数值命名空间复用原算法，绑定最终 v2 元数据。早期 stage B 或 v1 计划不应替代最终 v2 计划。
7. `1026-interface_teacher_portable/repo/interface_teacher/`：冻结六库的最终推理/导出实现。

`LOCAL_IMPORT_GRAPH.json` 是全体源码的静态 import 清单；`DEPENDENCY_AND_SPLIT_VALIDATION.json` 记录从原始 8 份源码出发的本地依赖闭包及所需外部 Python 包。检查使用 AST，未导入/执行这些实验模块。

## 训练/校准划分

最关键的最终离线库划分文件：

```
ReCogDrive/1024-representation_distillation_diagnosis_batch5/stage2/
  offline_library/preparation/session_cv_metadata_v1/SPLITS.json
```

其 seed 为 `2026102531`，选中 10,042 条样本、130 个 session；由 1,850 条 failure、2,048 条 mid、6,144 条 high 组成。最终 family 固定后的 inner fit 为 **8,021 条 / 104 sessions**，inner calibration 为 **2,021 条 / 26 sessions**。这是离线库拟合人群，和完整蒸馏训练人口不是同一个概念。

| 外折 | outer train | outer held | inner fit | inner calibration |
| --- | ---: | ---: | ---: | ---: |
| 0 | 8055 | 1987 | 6374 | 1681 |
| 1 | 8013 | 2029 | 6356 | 1657 |
| 2 | 8028 | 2014 | 6419 | 1609 |
| 3 | 8115 | 1927 | 6445 | 1670 |
| 4 | 7957 | 2085 | 6303 | 1654 |

附带真实 index/session/token 清单，不仅是计数或 seed。检查程序验证五折 held 恰好覆盖全部选中样本、各折训练/校准/held 的样本和 session 不交叉、最终 8021/2021 完整分割。

相关的其他划分也保留在原位置：第五批 `manifests/SESSION_SPLIT_V1.json`、stage2 `preparation/u1_session_split_v1/SESSION_SPLIT.json`、第四批训练/验证/LOSS_CALIBRATION 索引，以及最终 portable 的 `train.csv.gz`、`validation.csv.gz`、`test.csv.gz` 和 `POPULATIONS.json`。最终 portable 人口为 train 85813、validation 17475、test 12146；index 是原全局索引，不能当 CSV 行号。

## 解包与只读核验

仓库外层 `ARCHIVE_PARTS.json` 给出整包与各卷的长度、SHA256；`SHA256SUMS` 校验交付文件。压缩包按 GitHub 单文件限制分卷，需取齐，不能单独解某一卷。外层 `assemble_archive.py` 只使用 Python 标准库：

```bash
python3 -B reports/teacher-repair-reproduction-source-20261027/assemble_archive.py \
  --archive /YOUR/OUTPUT/teacher-repair-reproduction-source-20261027.tar.gz \
  --extract-to /YOUR/OUTPUT/extracted \
  --verify-sources
```

请使用新的输出位置。程序拒绝覆盖已有压缩包和已有同名解压目录，先核验各卷和整包 SHA，再安全解包，最后执行 `verify_sources.py`。也可在解压后单独运行：

```bash
python3 -B /YOUR/OUTPUT/extracted/teacher-repair-reproduction-source-20261027/verify_sources.py
```

它验证逐文件字节、全部 Python 语法、原始 8 份算法与研究源文件一致、点名缺失模块和配对模块存在、关键本地依赖、实际 session 划分。它不加载模型、不启动队列、不训练、不生成教师输出。

## 路径、依赖与适用边界

- 原研究源码的历史路径、源码哈希保护和执行许可检查全部保留。`PATH_MAPPING_AND_EXTERNAL_ASSETS.json` 给出旧 DLC 根目录、当前采集根目录、模型位置与权重 SHA，以及解压后的对应关系。该文件是映射说明，不会自动改写原配置或原源码。接收端应在独立容器中映射路径，保留原始快照，给新实验分配新的输出目录。
- 本包交付源码、训练/校准配置、精确划分和缓存重建方法，按用户要求不重新附带生成缓存或拟合库数值。完整 VLM/规划器权重、全量 NAVSIM 图像/地图/评分缓存、全量 hidden-state/配对状态缓存、历史每场景优化张量和全部拟合 run payload 仍是外部资产；它们不是缺失的 Python 模块。其身份和引用保留在配置、清单和 provenance 中，不能以占位数据代替。
- `LITERAL_CONFIG_REFERENCES.json` 是保守的静态路径检查，保留不存在的引用供审查。它包含未执行/废弃路线的生成物、错误状态文件、动态模板和存放在档案位置后失去原父目录语义的源码副本；不能将所有 `included=false` 都解释为最终六库缺配置。最终使用 `full_six_evaluation/preparation/v2/` 与冻结 portable repo。
- 静态依赖检查不证明全历史流水线可直接从零训练重现。若重跑拟合，需要上述外部状态数据和原有 ROOT/release/身份检查；本次没有绕过这些检查或制造旧验收记录。
- 原研究库版本见 portable 的 `requirements-inference.txt`、reference/provenance。`AUDIT_CONTEXT/` 另附本机 PPU 实测配置、历史完整链路记录和 A800 对齐要求，区分历史原环境与本机视觉核查环境。不要把它们混写成一次运行的配置，也不要据此宣称 PPU/A800 全链已对齐。
- 前次仓库的 `reports/ppu-a800-embedding-audit-20261027-v2.tar.gz` 已包含固定场景 `0002267a294e52a7` 的原始输入；本包补研究源码与配置，不重新生成那份输入或原 golden。

复现步骤请先读 [REBUILD_METHODS.md](REBUILD_METHODS.md)，关键配置见 `REPRODUCTION_KEY_CONFIGS.json`，CLI 参数索引见 `CLI_INDEX.json`。本包验证结果、文件数和压缩包校验值以同目录的机器可读清单为准。所有原研究源码均按原字节归档；新增工具位于包根目录和 `packaging/`，原始资产与本次工具清楚分开。
