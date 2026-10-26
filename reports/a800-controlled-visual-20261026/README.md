# A800 按 PPU 精确可移植配置重测结果

**严格可移植门禁已通过，原生输出仍未逐位一致。** 本轮没有修改 norm、激活、卷积公式，也没有用 PPU C 替代本机原生输出来宣布对齐。

参考为审计提交 `3d9b0979f0e052bb9c2b6afdf73f595af463a3aa`，冻结教师源码 `6bb7119583513094a169b4c2f2dc2a05ac293aa3`，场景 `0002267a294e52a7`。实际 GPU 为 `training-vlawm-tsinghua-zt6` 的 GPU1，NVIDIA A800-SXM4-80GB；任务已正常退出，退出码 0。

## 已经匹配的条件

- 精确 Python 3.10.13；Torch 通用版本 2.4.0；torchvision0.19.0、transformers4.57.0、tokenizers0.22.1、timm1.0.20、NumPy1.26.4、Pillow10.1.0、safetensors0.6.2、accelerate1.5.2、einops0.8.1、decord0.6.0。
- 独立新进程按原配置构建 BF16、非 Flash、eval 模型，使用 no_grad，不使用 autocast/inference_mode。matmul TF32=False、cuDNN TF32=True、线程8/8，其他数值开关及全部声明环境项完全匹配。未调用会覆盖这些设置的 Runtime/worker。
- 同一 v2 数据包的全部12项实际输入与辅助量、文件字节、shape、dtype、stride、放置约定及前向参数匹配。
- 同一原始 VLM safetensors 的完整文件 SHA、全部340个设备端视觉参数、341个视觉模块配置/训练状态、buffers、模型配置及相关 Python 算子源码身份匹配。
- `A800_CONTROLLED_GATE.json` 中 portable_differences=[]，不是放宽容差或修改门禁后的通过。

## 原生卷积与共同输入后的独立差异

原生无 bias BF16 卷积 C：9,437,184 个元素中 5,989 个不同（约0.06346%），最大绝对误差0.0078125，相对L2为5.2339973181182126e-05。含 bias patch 有2,179个元素不同，相对L2为5.279721763236374e-05。这里的相对L2分母按随包比较器定义为第二个张量，即本轮 A800 输出的L2。

原生 embedding SHA 仍不同。随后仅为隔离后缀算子，使用真实 PPU C 在当地加相同 bias、CLS 和位置编码，重建的 embedding 字节、布局及设备约定完全命中 PPU norm1 输入门禁。在这个相同输入下，首个分歧仍是原生 LayerNorm `norm1`；后续首 block 与从共同 embedding 起算的24层视觉后缀也未命中。

| 检查 | 逐位一致 |
|---|---|
| 原生 BF16 无 bias 卷积 C | 否 |
| 原生含 bias patch | 否 |
| 原生 embedding | 否 |
| PPU C 重建的共同 embedding | 是 |
| 共同 embedding 的原生 norm1 | 否 |
| 共同 embedding 的视觉末层 | 否 |
| FP32 LayerNorm 后转 BF16 | 否 |
| FP64 公式后转 BF16 | 是 |

FP64 公式后转 BF16 的诊断对照可以命中，而原生 BF16 与 FP32 LayerNorm 后转 BF16 均不命中。这支持差异处于实际数值执行后端，而不是输入数据、参数或已声明的前端设置；尚未测得具体内核的累加、归约或中间舍入步骤，不能把某一种实现细节当成已经证明的原因。

## 尚未匹配的平台后端

| 项目 | PPU | A800 |
|---|---|---|
| Torch 构建 | 2.4.0，git36615d2d… | 2.4.0+cu124，gite4ee3be4… |
| cuDNN兼容API报告 | 8905 | 90700 |
| Python编译器 | GCC11.4.0 | GCC12.3.0 |
| 设备/运行库 | PPU-ZW810E、SDK2.1厂商库 | NVIDIA A800、NVIDIA CUDA/cuDNN库 |

同名版本不等于同一二进制。通用版本和开关匹配没有消除原生卷积与 LayerNorm 的差异；当前不能唯一归因于物理硬件，也不能声称整个软件栈相同。把 PPU SDK 库直接替换进 A800 并不构成兼容解决方案。

## 范围和接续

本轮参考是当前 PPU 的纯视觉路径（transformers4.57/cuDNN TF32=True），不是历史完整链（4.37/False）。未执行语言、动作、PDMS或新教师生成。上表“共同 embedding 的视觉末层”使用PPU C隔离后缀；没有将它冒称为完全本机原生卷积起算的视觉末层。原生 embedding 已经足以否定本轮整条视觉输出逐位一致。

所有改动仅在独立诊断目录。九组旧学生训练保持暂停，已验收的本地 Native 教师数据未覆盖。完整本地 D／先聚类／K8／增强重拟合仍按用户要求等待拟合源码补包；已整理的85,813场景 Base 输入保留。

独立Python前缀与复制的通用包用于本轮诊断，Torch2.4私有包与其他依赖通过明确记录的只读路径加载；没有安装或升级共享环境。`CPU_ENVIRONMENT_PREFLIGHT.json` 与执行清单记录实际路径。最初的torchvision版本元数据预检查不通过及备用Python下载超时均保留失败收据，未作为GPU比较结果。随后安装未修改的官方torchvision0.19.0包、完成Python3.10.13前缀后才执行上述通过门禁的GPU任务。

## 张量指纹

- 原生 BF16 无 bias 卷积 C
  - PPU：`c04efdfc37e1d5d2fc0ce8bf618df389103f20a7c183e53e708eaf3dbaf2389b`
  - A800：`bf31e9e59faa5cadc54c23000cd17dc14254b956aca0e9ae821e9842340228b0`
- 原生含 bias patch
  - PPU：`0d0a5f0e48d4e03f6635140f9043a2fc79b9f8b800a4e95b541bd28e3e55cd6b`
  - A800：`5c2bc73d5f9638bd46a2f05b40b555e00b0ca10795a61d2b62f38d9d8f2b74a6`
- 原生 embedding
  - PPU：`ba0df59744dadae4460575d1cf6786ffb3d5636d286f7267d57a113b2d789d74`
  - A800：`f6baad2b747217f687d472eefae293619ba84c6b7c56ea8681381961c3a72ae8`
- PPU C 重建的共同 embedding
  - PPU：`ba0df59744dadae4460575d1cf6786ffb3d5636d286f7267d57a113b2d789d74`
  - A800：`ba0df59744dadae4460575d1cf6786ffb3d5636d286f7267d57a113b2d789d74`
- 共同 embedding 的原生 norm1
  - PPU：`d8007d98b0846101fd9c86e3c983cc9b053574239b6e76101364173cc7643ffe`
  - A800：`5030ad21c11c8533d7db00b8fa6f84cca5f9fc754dbe632e399a516f04bcc839`
- 共同 embedding 的视觉末层
  - PPU：`ba7dbbd0efbb5109fa9c593a28cf3528b8b6d3ed1507fa7a832e36cc7abcfc9d`
  - A800：`32ee715174c5a3c02006b92c819b3a4397325ef8c9b116d0d53ddb27ea646dea`
- FP32 LayerNorm 后转 BF16
  - PPU：`2bc72b9cd2aa4b6b244df882743257f48795d628c9990852222b0985cb50c045`
  - A800：`5030ad21c11c8533d7db00b8fa6f84cca5f9fc754dbe632e399a516f04bcc839`
- FP64 公式后转 BF16
  - PPU：`c2d3724196d0317280585c3e838ea6bc0835507ae04b2790b07ef465c999757e`
  - A800：`c2d3724196d0317280585c3e838ea6bc0835507ae04b2790b07ef465c999757e`

## 机器可读证据

`RESULTS.json`、`A800_CONTROLLED_CONFIGURATION.json`、`A800_CONTROLLED_GATE.json`、`A800_CONTROLLED_NORM1_INPUT_GATE.json`、`A800_CONTROLLED_TRACE.json` 和 `FRESH_PROCESS_SETTINGS.json` 均来自同一次受控任务。执行源码及原始数据身份见 `EXECUTION_MANIFEST.json`、`SOURCE_RECEIPT.json`；原始脚本没有被修改。所有文件 SHA 见 `SHA256SUMS`。
