**结论：已定位首个数值分歧，尚未修复到原 PPU 教师数据逐位一致。全量生成继续暂停。**

本文针对名称含 `20261027` 的审计补包；没有把补包名称作为本地执行日期。输入仓库冻结为 `6bb7119583513094a169b4c2f2dc2a05ac293aa3`。

**已完成的核查**

已下载审计分支 `audit/ppu-a800-embedding-20261027`，固定到提交 `0a1b1eb0813643b9d7c1ba7711009943133cc7d8`。压缩包 SHA256 为 `5b3505b59439d839ca4add76323ff0e51d3e4e8d806be092634912dfc08d9e75`，包内清单校验通过。

| 环节 | 结果及含义 |
|---|---|
| 图像预处理与输入排布 | 原三例 pixels FP32/BF16、input IDs、attention mask、position IDs 全部逐位一致。 |
| 实际参数与源码 | 首层四个实际 BF16 参数一致；三个关键模型源码 SHA 一致。 |
| attention 配置 | 已在构造时匹配语言 eager、视觉非 Flash；实际 4D 因果 mask 已检查。原 A800 Native 使用 Flash 是第一处协议差异，已经修正诊断路径。 |
| 第一处剩余分歧 | 视觉 patch embedding 的原生 BF16 **无 bias 卷积**，发生在所有 attention 层之前。 |
| bias、位置编码 | PPU 原生卷积与“其无 bias 结果加 BF16 bias”一致；位置插值、CLS 与对应加法控制一致。差异已经出现在加 bias 之前。 |
| 独立 A800 节点 | zt4 与 zt6 的单例首层输入、参数及中间量一致，未见单节点特有异常。 |

本次采用收到的 `audit_embedding.py` 在 A800 重跑一个固定场景。源脚本的报告字段引用了未定义的 `v`；仅在独立副本改为 `tensors[k]`，没有修改计算。原补包保持原样，修改与源码 SHA 记录在 `PATCH_RECEIPT.json`。

**新确认的跨平台算术控制**

场景为 `0002267a294e52a7`，卷积输出 `[9,1024,32,32]`，共 9,437,184 个 BF16 元素。

定义诊断路径：

- A：输入和权重转 FP32，带 bias 卷积，然后转 BF16。
- B：输入和权重转 FP32，无 bias 卷积，先转 BF16，再加 BF16 bias。

收到的补充审计实际使用 `cudnn.allow_tf32=True`。在 A800 匹配这个设置后，A、B 的完整 SHA 都与 PPU 补充审计逐位相同：

| 路径 | 两端共同 SHA256 |
|---|---|
| A | `6d4534341b80eaf4786525ec546db64933ae28e7f742b68ab162c166218ae2b3` |
| B | `4b68fe75dd277df7216e32ea2d3c2ac476d88e36c17615b6e96ddee0bfd61386` |

但是两端**原生 BF16 无 bias 卷积**仍不一致：

| 平台 | 原生 BF16 无 bias SHA256 |
|---|---|
| PPU | `c04efdfc37e1d5d2fc0ce8bf618df389103f20a7c183e53e708eaf3dbaf2389b` |
| A800 | `bf31e9e59faa5cadc54c23000cd17dc14254b956aca0e9ae821e9842340228b0` |

因此，“原生 BF16 卷积路径数值不等价”已有直接证据；不能归因于模型权重、图像输入或 attention。厂商 kernel 的具体内部累加/舍入实现尚未确认，不能宣称发现了硬件故障，也不能只安装同名 Torch 2.4 就视为解决。

已检查 cuDNN v7/v8 API、NCHW/channels_last、deterministic 和 benchmark 的 16 组常见设置，均未命中 PPU 原生结果。这不是穷举所有 cuDNN 算法。

PPU 原始完整重放记录的 `cudnn.allow_tf32=False`，而新补充首层审计为 True；两份记录必须区分。A800 两种设置都已保存，**没有为生成教师而改用 True，也没有用 FP32 诊断输出替代原生教师**。两种 BF16 诊断输出相同也不能证明未舍入 FP32 结果全部相同。

**目前能量化到什么程度**

由于 B 两端逐位一致，可以把它作为共同参考：PPU 原生带 bias 输出与 B 有 631 个不同元素；A800 原生带 bias 输出与 B 有 2614 个不同元素。由此得到两端原生带 bias 输出之间的不同元素数在 **1983～3245** 之间。**631 不是两端原生输出的直接比较结果。**精确的原生对原生误差尚缺 PPU 实际张量，不能从 SHA 或前 16 个差异值恢复。

相关严格界限见 [DERIVED_BOUNDS.json](evidence/DERIVED_BOUNDS.json)。首层差异小不能保证最终接口和轨迹可用；此前全链诊断仍未命中原 Base 接口 SHA，不能按“误差可忽略”放行。

**现在只需 PPU 补充一个实际张量**

请在原冻结模型与固定场景 `0002267a294e52a7` 上导出原生 BF16 无 bias 卷积结果 C，形状 `[9,1024,32,32]`，未压缩约 18 MiB。现有 32 KB 审计包只有统计、SHA 和少量坐标，没有真实张量文件。

使用 [export_ppu_native_C.py](export_ppu_native_C.py)，在原审计已经取得实际模型和 BF16 pixels 后调用：

```python
import runpy
export = runpy.run_path('/path/export_ppu_native_C.py')['export_ppu_native_C']
export(
    engine.backbone.model,
    inputs['pixel_values'],
    '/一个已存在的独立诊断目录/ppu_native_no_bias.npz',
)
```

变量名按原审计实际对象替换。函数不加载模型，不改变 attention、参数或精度标志；先验证输入与权重 SHA，再验证 C SHA 必须为上表 `c04efdfc...`，只以独占创建方式写新的 NPZ，已有文件不会覆盖。NPZ 用原 dtype 连续内存的原始 uint8 字节保存 C，并附 shape、dtype、SHA 与精度标志；不用 pickle。

请将这一个 NPZ 与打印的文件 SHA 推到同一审计分支或提供可读路径。无需再次核查整套环境，无需导出全量教师。收到后可直接算精确跨平台误差，并将 PPU C 注入 A800 的首层后续路径，验证后续视觉层/语言层是否还存在独立差异。**这项单例隔离预计收到数据后 10～20 分钟；严格全链对齐的修复时间仍不能可靠估计。**

**本地可复核证据与数据状态**

- [A800 TF32=False 完整记录](evidence/A800_tf32_false.json)
- [A800 TF32=True 完整记录](evidence/A800_tf32_true.json)
- [v7 控制](evidence/CUDNN_v7.json)、[v8 控制](evidence/CUDNN_v8.json)
- [本次发布文件校验清单](MANIFEST.json)；A800 实际对照张量保留在本地，此次不上传大张量。
- [原补包下载校验](evidence/DOWNLOAD_RECEIPT.json)、[报告脚本修正记录](evidence/PATCH_RECEIPT.json)

原冻结仓库工作区干净。旧 Native 缓存继续隔离，未标记 research-ready；没有恢复全量教师生成或原九组训练实验。


**供 PPU Codex 执行的交接要求**

在原冻结提交和原实际模型/输入上执行上面的单例导出。请勿重拟合专家、重选教师或开启全量生成。目的仅为拿到已知 SHA 的真实 C 张量，精确比较两端原生结果，并在 A800 注入相同 C 检查后续层。返回 NPZ 文件、文件 SHA256 和实际精度标志；如果 C SHA 与预期不同，先报告实际值和环境差别，不把新值当作旧参考。

成功文件建议放到当前审计分支的 `reports/ppu-a800-alignment-followup-20261027/ppu_native_no_bias.npz`，并附同名 `.sha256` 文件。请保留原 dtype 的原始字节、shape 和 dtype 元数据；只输出 hash、截图或前若干元素不足以继续隔离。该要求不需要全量教师缓存或模型权重。
