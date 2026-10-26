**已确认卷积之后的首个分歧是视觉 block 0 的 `norm1`。本次 PPU 纯视觉末层仍逐位命中历史参考；全链尚未对齐。**

在 A800 提交 `5cf8440` 基础上执行。固定教师提交 `6bb7119583513094a169b4c2f2dc2a05ac293aa3`，唯一场景 `0002267a294e52a7`。未重新导出输入或 C、未运行语言/动作/教师，也未搜索或生成批量教师。

完整回包：[PPU_FIRST_BLOCK_TRACE.json](PPU_FIRST_BLOCK_TRACE.json)；包括原脚本记录、构造与精度来源，以及本机 norm1 数值补充的 first16。文件 SHA 见同名 `.sha256`。原版 [trace_first_vision_block.py](trace_first_vision_block.py) 未修改。

**输入、参数与历史参考核验**

| 项目 | 结果 |
|---|---|
| 实际 BF16 pixels、真实 PPU C | 命中原指纹，复用此前交付数据。 |
| 从 C 重建的完整 embedding | `ba0df59744dadae4460575d1cf6786ffb3d5636d286f7267d57a113b2d789d74`，与历史 PPU、A800 注入后的输入相同。 |
| embedding 与 block 0 的 18 个实际 BF16 参数 | shape、dtype、原始字节 SHA 全部与 A800 参考相同。 |
| 当前 PPU 纯视觉末层 | `ba7dbbd0efbb5109fa9c593a28cf3528b8b6d3ed1507fa7a832e36cc7abcfc9d`，**逐位命中历史 PPU**。 |
| 视觉源代码 | `1978e330ea5533dda4d1379096aded271fc741347fc48cb4924846b56a2a6a4b`，与 A800 相同。 |
| 参数和临时 hook | 全部视觉参数的版本及 SHA 前后不变，所有 trace hook 已移除。 |

**block 0 的第一个事件已经不同**

`norm1` 是 `torch.nn.LayerNorm`，PPU 实际 `normalized_shape=[1024]`、`eps=1e-6`；输出为 BF16 `[9,1025,1024]`。其输入是上表相同的完整 embedding，此时尚未执行 block 0 的 QKV 或 attention。

| 对照 | PPU SHA256 | A800 SHA256 | 一致 |
|---|---|---|---|
| 原生 BF16 norm1 | `d8007d98b0846101fd9c86e3c983cc9b053574239b6e76101364173cc7643ffe` | `5030ad21c11c8533d7db00b8fa6f84cca5f9fc754dbe632e399a516f04bcc839` | 否 |
| FP32 layer_norm 后转 BF16 | `2bc72b9cd2aa4b6b244df882743257f48795d628c9990852222b0985cb50c045` | `5030ad21c11c8533d7db00b8fa6f84cca5f9fc754dbe632e399a516f04bcc839` | 否 |
| 显式 FP64 公式后转 BF16 | `c2d3724196d0317280585c3e838ea6bc0835507ae04b2790b07ef465c999757e` | `c2d3724196d0317280585c3e838ea6bc0835507ae04b2790b07ef465c999757e` | 是 |

13 个事件按实际执行顺序记录。`first_divergent_event="norm1"`，之后 QKV、softmax 后的 attn_drop、投影、MLP 与 block0_output 均已不同。**这些后续不一致不能单独证明这些算子还有独立差异，因为它们已收到不同的 norm1 输出。** 同理，此次没有新增语言层自身存在独立问题的证据。

FP64 控制只证明该诊断公式最终转 BF16 的输出一致；不证明未舍入的 FP64 中间值全部一致，也不能替代原生教师。PPU 的原生 BF16 norm1、FP32 控制和 FP64 控制三者均不同，所以“只把 norm 转 FP32”尚不能作为跨平台逐位修复。

**PPU 内部 norm1 数值对照**

原版 trace 完成后，在保留的同一模型进程中，仅重建相同 embedding 并重复 norm1 与两种临时算术控制；未重复全视觉核查。三个输出指纹分别与原 trace 一致。全部 9,446,400 个元素在 CPU 做原始字节及数值比较；相对 L2 分母是表中第二个张量。

| 比较，方向为第一项减第二项 | 不同元素 | 最大绝对误差 | 相对 L2 |
|---|---:|---:|---:|
| PPU 原生 norm1 vs PPU FP32→BF16 | 25 | 0.00390625 | 0.000003911146885 |
| PPU 原生 norm1 vs PPU FP64→BF16 | 247 | 0.0078125 | 0.000012842914730 |
| PPU FP32→BF16 vs PPU FP64→BF16 | 240 | 0.0078125 | 0.000012233179750 |

**25/247/240 均不是两端原生 norm1 的精确差异数。** A800 实际 norm1 全量载荷不在本机，不能从 SHA 推算直接跨平台误差。每组前 16 个不同展平索引、两项数值和差值已写入 `PPU_FIRST_BLOCK_TRACE.json → norm1_local_numeric_supplement`，并单列在 [PPU_NORM1_CONTROLS.json](PPU_NORM1_CONTROLS.json)。

**实际执行环境与保持项**

本机 PPU SDK 为 `2.1.0-a5f865`，Torch `2.4.0`、git `36615d2dc445b0232500ca5fadc986949833ae46`，transformers `4.57.0`。本次只验证视觉路径，不能将其扩展成旧 transformers 4.37 的完整语言链复现。旧 PPU SDK 2.0 与本机 SDK 2.1 的版本差异未被隐瞒；本次通过了文档要求的历史视觉末层 SHA 核验。

原进程已退出，本次通过原 `RecogDriveBackbone` 入口重载同一冻结 VLM。在模型构造时，按接续文档和 A800 驱动设置 `use_flash_attn=False`，实际 24 个视觉层均为非 Flash；没有对已构造模型切换 attention 或 norm。没有手动设置全局精度标志。

| 标志 | PPU 本次初始/加载后/执行后 | A800 参考 |
|---|---|---|
| matmul TF32 | False / False / False | False |
| cuDNN TF32 | True / True / True | False |
| BF16 / FP16 reduced reduction | True / True，始终保持 | True / True |
| autocast / benchmark / deterministic | False / False / False，始终保持 | False / False / False |

因此，PPU 本次精度状态与 A800 参考并非每个字段都相同；已按接续文档允许的补交 C 环境如实记录，没有悄悄切换。当前视觉末层命中原历史结果，仍不能代替具体算子累积/舍入实现的解释。

模型按要求保留在原 PPU Python 进程，便于接续比较。机器本地会话索引在 `/mnt/workspace/project/ppu-first-block-session-20261027.json`；该本地会话索引不上传仓库。捕获驱动和数值补充脚本在 `diagnostic_source/`，含本机执行路径，不是接收方需要安装的环境。

**接续最小工作**

下一步应围绕相同 embedding 和 norm1 做对照，补记 A800 的实际 `eps`、`normalized_shape` 与输入 stride。若需要精确的两端 norm1 误差或注入，需双方真实原生 norm1 输出；当前 JSON 中只有 PPU 内部数值对照，不能冒充已完成直接跨平台比较。随后用相同 norm1 输出测试 QKV，才能判断矩阵乘是否另有独立差异。暂不需要二分后续 block、重跑动作评分或尝试整套环境切换。
