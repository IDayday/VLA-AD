**真实 C 已接收并完成单例因果隔离：首层 embedding 可逐位对齐，但后续视觉计算仍有独立差异。全链尚未修复，全量继续暂停。**

本次读取 PPU 审计分支提交 `36e9f40123213d23ad4951cc062f24b161a2ae78`。教师源代码仍为 `6bb7119583513094a169b4c2f2dc2a05ac293aa3`，固定场景 `0002267a294e52a7`，固定 `D_C8_double` 选择 6，无重新拟合或选择。

**下载、输入与噪声已核对**

v2 压缩包 SHA 为 `4e080828292355b2e42e4fd81fee213df276a175a8f2c75175ca00e05f81a65c`，包内 62 项文件的大小和 SHA 全部通过。

实际 C 文件 SHA 为 `9c1abc99ea9ba94db1160cdd89c566c5c2df4d2efacaed8e83d801deaa3a302d`；解码后的 BF16 原始字节 SHA 为 `c04efdfc37e1d5d2fc0ce8bf618df389103f20a7c183e53e708eaf3dbaf2389b`，均命中要求。

A800 当前实际使用的 5 项模型输入、3 项动作辅助状态、3 项角色 mask 和 FP32 pixels，逐项与补包完全一致。五组完整显式噪声 `[5,6,1,8,3]` 的 SHA 为 `c39e5183f1127b0faf42e649b148743cb57a34dc70c30657f0d881a9cf210f10`，也与原 PPU 完全一致。差异不能再归因于这些输入或只设 seed。

**两端原生卷积的精确误差**

以下均为真实张量的逐元素比较，方向为 PPU 减 A800，相对 L2 分母为 A800 对应张量范数。

| 张量 | 总元素 | 不同元素 | 最大绝对误差 | 相对 L2 |
|---|---:|---:|---:|---:|
| 原生 BF16 无 bias C | 9,437,184 | 5,989 | 0.0078125 | 0.00005233997318 |
| 加各自相同 BF16 bias 后的 patch | 9,437,184 | 2,179 | 0.0078125 | 0.00005279721763 |

A800 的 `cudnn.allow_tf32=False/True` 两种配置得到相同原生 BF16 C，和 PPU 的差异也相同。此前 631 是 PPU 内部 Native 与诊断 B 的差异；现在的 5,989/2,179 才是两端对应原生张量的直接比较。

**将同一个真实 PPU C 注入 A800 后发生什么**

在 Torch 2.4、语言 eager、视觉非 Flash、VLM BF16、投影/动作 FP32、两个 TF32=False 的原始推理配置下，运行原生和 C 注入两组单例诊断。只在该次 VLM 前向临时替换 patch convolution 输出为真实 `PPU C + 相同 BF16 bias`；其余计算照常。前向后立即移除 hook，参数版本保持不变。

| 阶段 | A800 原生是否命中原 PPU | 注入 PPU C 后是否命中原 PPU |
|---|---|---|
| 完整 vision embedding | 否 | **是** |
| 视觉末层 | 否 | 否 |
| bridge | 否 | 否 |
| 语言首层、末层 | 否 | 否 |
| 完整 FP32 Base 接口，含 padding | 否 | 否 |

注入后完整 embedding SHA 为 `ba0df59744dadae4460575d1cf6786ffb3d5636d286f7267d57a113b2d789d74`，逐位命中原 PPU。

但是视觉末层变为 `32ee715174c5a3c02006b92c819b3a4397325ef8c9b116d0d53ddb27ea646dea`，仍不等于原 PPU 的 `ba7dbbd0efbb5109fa9c593a28cf3528b8b6d3ed1507fa7a832e36cc7abcfc9d`。

因此，首层卷积差异是一个真实原因，**并不是全部原因**。相同 embedding 进入后续视觉计算后，两平台仍产生不同结果。此时尚不能把后续差异具体归到 LayerNorm、矩阵乘或 softmax 中某一项，需要对照实际算子输出。更不能把剩余语言层分歧直接认作语言算子独立问题，因为语言输入本身还不同。

**五次轨迹与独立真实评分**

下表只用于单例定位；不是正式 A800 基线、方法改进或可用缓存的验收。PDMS 为五次真实评分平均值，轨迹偏差对照原 PPU 五条配对轨迹。

| 条件 | Base PDMS | 教师 PDMS | Base XY 最大偏差 | 教师 XY 最大偏差 |
|---|---:|---:|---:|---:|
| 原 PPU 历史参考 | 90.919160 | 91.663342 | 0 | 0 |
| A800 双 eager 原生 | 91.054756 | 91.816774 | 0.116749 m | 0.134693 m |
| A800 注入真实 PPU C | 91.015583 | 91.784556 | 0.074093 m | 0.093832 m |

三者均使用固定选择 6 和相同五组噪声。新轨迹独立重新评分，没有复用旧评分。NC/DAC/TTC/comfort 在这两组 A800 单例仍均为 1，但七项完整评分和轨迹均未与原记录完全一致。分数较高不能用来宣布复现成功。

**下一步 PPU 只需执行一次算子级核查，不再重导出输入或 C**

已提供 `trace_first_vision_block.py` 和 A800 的 `A800_FIRST_BLOCK_REFERENCE.json`。在 A800 上，独立 block 0 的输出与真实完整 VLM 前向的 block 0 输出逐位一致；可选的独立视觉后缀输出也与实际 C 注入前向的视觉末层逐位一致，均已验收。

请在 PPU 已加载的冻结 eval 模型与原 BF16 pixels 上调用：

```python
import json
import runpy
from pathlib import Path

folder = Path('reports/ppu-a800-alignment-followup-20261027')
trace = runpy.run_path(str(folder / 'trace_first_vision_block.py'))['trace_first_vision_block']
record = trace(
    engine.backbone.model,
    inputs['pixel_values'],
    folder / 'ppu_native_no_bias.npz',
    reference_file=folder / 'evidence/A800_FIRST_BLOCK_REFERENCE.json',
    verify_full_vision=True,
)
Path('/一个独立已有目录/PPU_FIRST_BLOCK_TRACE.json').write_text(
    json.dumps(record, ensure_ascii=False, indent=2) + '\n'
)
```

变量名按原审计实际对象替换，`folder` 相对当前 VLA-AD 仓库。此函数不自动加载模型、不改变精度标志、不改 attention 或参数；它从已知 PPU C 重建并验证完整 embedding，然后记录 block 0 的 norm、QKV、softmax、projection、MLP 等实际输出指纹。JSON 含参数逐项匹配结果、首个不同事件、两种 LayerNorm 算术控制及精度状态。除显式指定的 JSON 写入外，函数自身只打印，不输出大张量。

`verify_full_vision=True` 另执行纯视觉 encoder，不运行语言、动作或教师。首先看 `vision_last_matches_historical_PPU`：PPU 此次补交环境的 SDK 记录为 2.1.0，原完整重放报告为 2.0.0，必须确认当前视觉后缀仍命中旧 PPU SHA，不能自动假定两份环境等价。若不命中，请报告实际值和环境差异；此时新 trace 只能作为当前 PPU 的算子对照，不能替代历史参考。

请优先使用原 Runtime 实际前向配置（原记录为语言 eager、视觉非 Flash、无 autocast、两个 TF32=False），保留当前进程的模型便于必要的后续比较。若只能使用补交 C 时 TF32=True 的环境，照实记录，不悄悄切换或把它称为原路径。此函数的首个视觉 block 不运行 Qwen，因此不会把当前 transformers 4.57 的首层审计扩展成原 4.37 全语言链证明。

请把 `PPU_FIRST_BLOCK_TRACE.json` 回传同一审计分支。无需重新提供原图、prompt、输入、C 或再次搜索教师。若 block 0 全部相同而视觉末层不同，下一步才沿后续 block 二分定位；若首个不一致在 norm1/QKV/softmax，则直接围绕该算子比较，避免继续盲试整套环境。

**当前边界**

此次注入只是因果诊断，不是通用的 A800 修复，也不能据单例 C 生成其他场景。当前已有证据支持计算后端数值不等价，没有硬件故障证据；具体内核累加/舍入语义尚未完全确定。正式教师资产、原缓存隔离与全量暂停状态保持不变。


**可复核记录**

- [真实 C 和 patch 精确比较](evidence/EXACT_C_AND_PATCH_COMPARISON.json)
- [C 注入完整分阶段、逐层指纹与真实评分](evidence/REAL_C_INJECTION_RESULT.json)
- [配对噪声 SHA 与五次评分比较](evidence/PAIRED_SINGLE_SCENE_SCORES.json)
- [首个 block 对照参考](evidence/A800_FIRST_BLOCK_REFERENCE.json)及[当地完整前向一致性检查](evidence/FIRST_BLOCK_VERIFICATION.json)
- [PPU 最小核查函数](trace_first_vision_block.py)、[A800 注入代码](diagnostic_source/inject_ppu_native_C.py)、[当地 block 核查驱动](diagnostic_source/run_first_vision_block_trace.py)
