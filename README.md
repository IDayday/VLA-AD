# ReCogDrive 第五批：接口修复教师的跨平台交付

本仓库把已完成的**六套冻结接口修复库、全量逐场景选择、重放代码和两份实验报告**交给接收方。接收方已有 RL Base 和 NAVSIM 数据，即可在当地生成修复后的内部教师表征和轨迹，再自行尝试全参更新、局部解冻、LoRA 或外挂修复器，将能力固定到参数中。

这里交付的是离线教师，不是已经达到教师成绩的单模型。此前蒸馏/修复器的正负结果均保留在报告中；本仓库不预设某种学生训练方案一定有效，也不启动训练。

## 先从这里开始

如果由Codex接手，请让它在仓库根目录启动：根目录`AGENTS.md`会自动指向完整交接；也可直接让它读取[给训练智能体的交接](docs/给训练智能体的交接.md)。不需要原作者补充口头背景。

1. [接收方逐步操作指南](docs/接收方操作指南.md)：克隆、配置、验收、导出、训练接入和分片接续。
2. [教师定义与数据字典](docs/教师定义与数据字典.md)：专家、空间、力度、种子、选择和文件字段。
3. 两份完整报告：[接口表征分析与修复](reports/interface_representation_repair.html)、[诊断与参数吸收实验](reports/representation_distillation_diagnosis.html)。HTML含折叠细节与内嵌图表，下载/克隆后用本地浏览器打开。

## 最短使用路径

```bash
git clone https://github.com/sjc786526-coder/-recogdrive-representation-teacher-2.git
cd -- -recogdrive-representation-teacher-2
python -B verify_package.py
cp configs/paths.example.json paths.local.json
# 编辑 paths.local.json，填写当地 RL checkpoint、VLM、地图、logs/sensor/metric 路径。
python -B check_assets.py --paths paths.local.json
CUDA_VISIBLE_DEVICES=0 python -B export_teacher.py \
  --paths paths.local.json --out outputs/smoke \
  --population train --library D_C8_double --source raw \
  --limit 2 --save-full-interface --score
```

使用现有兼容的研究环境；包不自动安装依赖或下载模型。核心接口API只需要NumPy/PyTorch，原始图像推理和评分使用vendor快照及原ReCogDrive/NAVSIM依赖。

## 交付的六套教师

以下是**同一场景固定选一个Base/专家，跨S0–S4不变**的事后选择。训练列为实际85,813例/130会话；测试为实际12,146例。每场景等权，先平均五噪声，平局优先Base。没有逐噪声重选，也没有六库合并择优。

| 教师库 | 专家数 | 完整训练PDMS | 相对Base | 完整测试PDMS | 相对Base |
|---|---:|---:|---:|---:|---:|
| Base | 0 | 94.9163 | — | 90.4793 | — |
| D_P7_original | 7 | 96.8531 | +1.9368 | 95.1925 | +4.7132 |
| D_C8_original | 8 | 96.8983 | +1.9820 | 95.2789 | +4.7996 |
| R_P8_double | 8 | 96.9201 | +2.0037 | 95.6781 | +5.1988 |
| R_C8_double | 8 | 96.9157 | +1.9994 | 95.5814 | +5.1020 |
| D_P8_double | 8 | 97.1685 | +2.2522 | 96.1528 | +5.6735 |
| D_C8_double | 8 | 97.3053 | +2.3890 | 96.2389 | +5.7596 |

全部修复都在VLM→动作头投影输出 `[L,384]`，通过一个48维子空间解码出384维修正，广播到有效token，padding保持。D系列共用一个实际D48空间；R系列共用另一个实际R48空间。最终库实际入选的是定标中心方向、近邻配对修正和小MLP，完整参数已打包，不需要重新搜索/聚类/拟合。

## 已保存的“每个场景怎么修”

```python
from interface_teacher import Library, DecisionTable

teacher = Library("D_C8_double")
choices = DecisionTable("D_C8_double", "train")
record = choices.row(token)  # choice、method、五噪声分数及增量

# frozen_Base_post: 原生FP32 [1,L,384]，来自一份固定的RL Base。
# mask: 同一输入的image/有效text位置，不能把padding当text。
target = teacher.selected_target(
    frozen_Base_post, image_mask, valid_text_mask, token, choices
)
# target即修复后完整接口；可以直接作为detach的教师目标。
```

`choice=0`不修复，其他编号对应该库JSON中的固定专家顺序。每套库的训练/测试选择各自独立保存，没有遗漏正常场景可获益的修复。测试选择用于离线上限复现，不作为学生训练标签；另提供原17,475例/32会话的验证清单。

## 包含什么，哪些需要当地生成

| 已带入仓库 | 由接收方提供或生成 |
|---|---|
| 两套实际48维基底、小专家参数、力度/限幅 | RL Base和配套VLM大权重 |
| 训练/验证/测试场景身份及数据相对路径 | NAVSIM原始图像、logs、地图、评分缓存 |
| 六库训练/测试逐场景选择与全部候选五噪声均分 | 全量逐token教师表征和完整轨迹缓存 |
| 独立接口API、端到端/缓存重放、教师导出 | 接收方自行决定的学生训练方案和训练输出 |
| 种子、实际golden轨迹/评分/请求、SHA清单 | 不同硬件和软件版本上的完整重放验收 |
| 两份报告、来源与本地验收记录 | 原研究所有搜索中间产物，复现教师不需要 |

文件主要为小型文本/压缩数组；没有图像数据集或Base权重。全量接口缓存不小，默认支持保存紧凑实际修复量，或按需在当地生成完整接口。

## 复现身份与边界

- Base为本实验实际使用的 `ReCogDrive-2B-RL`，checkpoint SHA256 `23fb949b10387efeff87e093703d5faf41b61b9e5ead16d4076661e3c7c08a20`。名称相近不等于同一模型。
- `verify_package.py` 检查所有打包文件SHA、真实golden请求和全量保存选择；`reference/LOCAL_VALIDATION.json`列出实际工程验收，不把未执行的跨平台全量复现写成已通过。
- 固定种子公式在`seeds.py`；原生FP32动作头、batch=1、TF32关闭和精确token布局均会影响边界轨迹。换硬件先做小规模验收。
- 表征目标必须从冻结Base生成。学生训练中投影/VLM更新后，不要把它自己输出的表征重新当作原教师输入。
- 本仓库使用相对路径，不依赖阿里云/DLC、CPFS、原`/mnt/...`或账号凭证。原路径只在报告/来源档案中用于追溯。
- 报告的原资产链接改为[来源索引](reports/evidence_index.html)，正文、核心表格、折叠细节及图片保留。完整大缓存未冒充已上传。

上游模型与NAVSIM代码来源保留在vendor及来源清单中，适用其原有许可。本仓库没有替第三方代码或数据重新授予许可。
