**A800 本轮执行要求：先完整匹配本文件及机器清单中的可移植条件，使用审计仓库同一份原始数据和输入，再运行单场景比较。任何可移植配置或输入门禁失败，停止卷积/norm/encoder 核查，先回传差异清单。**

本说明在提交 `c3e1441` 的算子核查基础上补充。当前 PPU 模型保留在同一进程；此次只读采集配置、重新哈希原 VLM 文件，没有再次运行模型或改变 PPU 设置。

- [完整当前实测配置：PPU_CONTROLLED_CONFIGURATION.json](PPU_CONTROLLED_CONFIGURATION.json)，含 340 个实际视觉参数、341 个模块的标量/训练状态、完整 runtime config、实际输入布局、运行标志、代码/权重身份、已加载库及全部 Python distribution 版本。
- [历史完整链参考：PPU_HISTORICAL_FULL_CHAIN_REFERENCE.json](PPU_HISTORICAL_FULL_CHAIN_REFERENCE.json)。历史未采集项仍标为未知，不以当前进程补写。
- [已有 A800 的已知差异/缺项](A800_KNOWN_CONFIGURATION_GAPS.json)。**这不是 A800 已经完成新门禁的结果；目前尚未收到新的受控 A800 回包。**

**1. 先区分两份 PPU 参考，不混用设置**

| 参考 | SDK / transformers | matmul TF32 / cuDNN TF32 | 实际验证范围 |
|---|---|---|---|
| 本轮可执行视觉对照 | 2.1.0-a5f865 / 4.57.0 | False / **True** | 单场景 C、完整视觉 embedding、纯视觉末层均命中历史 SHA；首个 block 已逐算子采集。未运行语言或动作。 |
| 原 PPU 历史完整重放 | 2.0.0-715aa1 / 4.37.0 | False / **False** | 历史图像→视觉→语言末层→FP32 Base 接口→五组轨迹及评分均逐位重现。不是当前进程新跑的全链。 |

**A800 本轮先匹配第一行。** 不要调用会改回 False/1 线程的完整 Runtime/worker 后，又将结果称为匹配当前 PPU 视觉参考。也不要把第一行成功扩展为当前 transformers 4.57 完整语言链已复现。

冻结教师源代码：`6bb7119583513094a169b4c2f2dc2a05ac293aa3`。PPU 原入口是冻结 `vendor/upstream/model/recogdrive_backbone.py::RecogDriveBackbone`，构造时 `model_type=internvl`、VLM BF16、device=cuda、eval、`use_flash_attn=False`。原已构造模型的参数、attention、norm 未切换。A800 使用同一原入口及构造参数。

VLM 原文件 `model.safetensors` 为 **4,178,048,344 字节**，本次重新读取全文件的 SHA 为 `79fb39297e322cd2d3dc68d4f23b86ff85806b336a4d5d0ed5db9b66e4034a3c`。A800 必须核对实际文件及全部 340 个视觉设备参数，不能只检查四个首层参数或目录名称。历史配套 RL checkpoint SHA 是 `23fb949b10387efeff87e093703d5faf41b61b9e5ead16d4076661e3c7c08a20`；本轮不用加载它，因为不运行动作头。

**2. 全部数值与执行设置**

| 项目 | 本次 PPU 实测，A800 必须逐项匹配 |
|---|---|
| tf32_matmul | `False` |
| tf32_cudnn | `True` |
| cudnn_enabled | `True` |
| bf16_reduced | `True` |
| fp16_reduced | `True` |
| matmul_precision | `highest` |
| autocast_cuda | `False` |
| autocast_cpu | `False` |
| autocast_gpu_dtype | `torch.float16` |
| autocast_cpu_dtype | `torch.bfloat16` |
| cudnn_benchmark | `False` |
| cudnn_deterministic | `False` |
| deterministic_algorithms | `False` |
| deterministic_warn_only | `False` |
| inference_mode_enabled | `False` |
| distributed_initialized | `False` |
| default_dtype | `torch.float32` |
| default_device | `cpu` |
| num_threads | `8` |
| num_interop_threads | `8` |
| operator_execution_context | `torch.no_grad(), eval; no autocast; ordinary tensors, not inference_mode` |

`num_threads=8` 和 `num_interop_threads=8` 来自保留的实际 PPU 模型进程，不是猜测。所有视觉模块的 `training=False`；`drop_path_rate=0.1` 保留原配置，在 eval 下不随机丢弃。诊断算子在 `torch.no_grad()` 中执行，autocast 和 inference_mode 均未开启；不通过改 norm、激活或算子公式来制造命中。

| 视觉配置 | 实际值 |
|---|---|
| image_size | 448 |
| patch_size | 14 |
| num_channels | 3 |
| hidden_size | 1024 |
| intermediate_size | 4096 |
| num_hidden_layers | 24 |
| num_attention_heads | 16 |
| hidden_act | gelu |
| norm_type | layer_norm |
| layer_norm_eps | 1e-06 |
| qkv_bias | True |
| qk_normalization | False |
| dropout | 0.0 |
| attention_dropout | 0.0 |
| drop_path_rate | 0.1 |
| use_flash_attn | False |
| use_moe | False |

`norm1` 实际类为 `LayerNorm`，`normalized_shape=[1024]`、`eps=1e-6`、`elementwise_affine=True`。权重和 bias 都是 BF16；具体 SHA 及 stride 见机器清单 `all_vision_parameters.encoder.layers.0.norm1.*`。全部 24 层视觉 attention 为非 Flash，qk_normalization=False。GELU 类、Python LayerNorm/Linear/function 源码、模块参数和 buffer 均逐项记录并纳入门禁。

**3. 通用库与环境条件**

| 通用软件 | 本次 PPU 精确版本 |
|---|---|
| Python | 3.10.13 |
| Torch 通用版本 | 2.4.0 |
| torchvision | 0.19.0 |
| transformers | 4.57.0 |
| tokenizers | 0.22.1 |
| timm | 1.0.20 |
| numpy | 1.26.4 |
| Pillow | 10.1.0 |
| safetensors | 0.6.2 |
| accelerate | 1.5.2 |
| einops | 0.8.1 |
| decord | 0.6.0 |

精确约束另见 [requirements-visual-common.txt](requirements-visual-common.txt)。A800 应使用独立诊断环境准备这些通用版本，不要直接升级正在运行实验的公共环境。脚本不会安装包；版本不同或缺失即停止比较。Torch 通用版本固定 2.4.0，平台专用构建、SDK/driver/cuDNN 的差异必须单独披露。

完整安装清单见配置 JSON 的 `backend_identity.all_distribution_versions`；未执行的动作/评分依赖不纳入本轮纯视觉可移植门禁。执行相关环境逐项如下，未设置与设置为某个值不能混淆：

| 环境项 | 当前 PPU 值，A800 新子进程匹配 |
|---|---|
| OMP_NUM_THREADS | 未设置 |
| MKL_NUM_THREADS | 未设置 |
| OPENBLAS_NUM_THREADS | 未设置 |
| NUMEXPR_NUM_THREADS | 未设置 |
| CUDA_MODULE_LOADING | `LAZY` |
| CUBLAS_WORKSPACE_CONFIG | 未设置 |
| NVIDIA_TF32_OVERRIDE | 未设置 |
| TORCH_ALLOW_TF32_CUBLAS_OVERRIDE | 未设置 |
| TORCH_CUDNN_V8_API_DISABLED | 未设置 |
| TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT | 未设置 |
| KMP_DUPLICATE_LIB_OK | `True` |
| KMP_INIT_AT_FORK | `FALSE` |
| ATEN_CPU_CAPABILITY | 未设置 |
| DISABLE_ADDMM_CUDA_LT | 未设置 |
| CUBLASLT_WORKSPACE_SIZE | 未设置 |
| CUDNN_CONV_WSCAP_DBG | 未设置 |
| PYTORCH_CUDA_ALLOC_CONF | 未设置 |
| PYTORCH_ALLOC_CONF | 未设置 |

`configure_fresh_receiver()` 只允许在尚未初始化 CUDA 的**新 A800 诊断子进程**显式调用；它按已采集值设置这些环境项、线程和数值标志，不触碰已有训练/推理进程。PPU 当前进程从未调用这个配置函数。

**4. 强制使用仓库里的同一条数据，排除原始数据差异**

唯一场景 `0002267a294e52a7`。不得改用 A800 原有缓存、同名 JPG、自己的重分词结果或其他场景。使用同一分支的 [v2 数据包](../ppu-a800-embedding-audit-20261027-v2.tar.gz)：

- 压缩包 SHA256：`4e080828292355b2e42e4fd81fee213df276a175a8f2c75175ca00e05f81a65c`。
- 当前 CAM_F0 原 JPG，1920×1080；原日志 frame_index=103，四帧历史为 100–103，元数据保留 ndarray dtype/shape/原字节。不使用其他相机或未来帧。
- 原始 JPG、四帧 JSON、完整 question/query、FP32/BF16 pixels、五项 VLM 输入、三项状态和三项角色 mask 全部在包里。包内原始数据各文件 SHA 见下表。
- 模型直接读取包中的已验证原始张量字节，BF16 只按 dtype 重解释，不经过 FP32 重算。若另行重建图像预处理或分词，也必须先命中相应全量 SHA，才能进入算子比较。

| 仓库数据文件（inputs/） | 字节数 | SHA256 |
|---|---|---|
| formatted_query.txt | 31825 | ee68adb75c5310dc60373a5000988b0d01b672dc7d9b23184771cb70866c899c |
| metadata.json | 8419 | 3bbc39c363a1741ed5a6eb576d2e4fe18986685b3fe480f5a7b80086f12d4e74 |
| question.txt | 551 | d55f24057ece88729b2ec8a8a47b0f4dcf67d5988f557249b10bd9b1bdba42ab |
| raw_frames.json | 123429 | 9baf01a464216502bf3c89d3c2fe165b52f9b1a8fe677d09b847fcf8492f7ab1 |
| raw_front.jpg | 199740 | a5afee9b743aa413b3f4ccc735424f54be26325fdddde4a6eeb3fcd157eafe0b |
| tensors.npz | 10333845 | 61677e5a650a741ebd9f6aeaf0de4bc45fba2f232db03964a88d5edab545cedc |

实际模型/辅助量必须同时满足原字节、shape、dtype、stride、storage_offset、layout 和 CPU/cuda 放置约定。`image_flags` 保持 CPU，其余四项实际 VLM 输入在 cuda；辅助状态和角色 mask 在 CPU。5 个模型 tensor 及 7 项补充量共 12 项如下：

| 实际输入/辅助量 | shape / dtype / stride | 原始字节 SHA256 |
|---|---|---|
| pixel_values | `[9, 3, 448, 448]` / `torch.bfloat16` / `[602112, 200704, 448, 1]` | `f0d5c4a3814c428ac6cca1a0ec5293c5d624fe2d5669c926cc9d3c594e7ab4ac` |
| input_ids | `[1, 2800]` / `torch.int64` / `[2800, 1]` | `f47363f01c681976fa54b1911f3db310a3f474a942b8bcae86df74974105de37` |
| attention_mask | `[1, 2800]` / `torch.int64` / `[2800, 1]` | `327128d0b57d5d5888ab4f01d5f9c759b37539a94f0029106ab5e30306446d7e` |
| position_ids | `[1, 2800]` / `torch.int64` / `[2800, 1]` | `6a66ecc3aee4ddf1751c21598346235104f5d19a961df6abb94feb272e0ef307` |
| image_flags | `[9]` / `torch.int64` / `[1]` | `df070c0849900928e1e85833cce32020c89d2303a9e960510c3c3b79746448b7` |
| history_trajectory | `[4, 3]` / `torch.float32` / `[3, 1]` | `b1819fc792ee0380343de7a91d61c7442728bf4373e82e8c9355c9df530a723f` |
| high_command_one_hot | `[4]` / `torch.float32` / `[1]` | `6855960ee428276148fc8745625c44bde7ae835a9e2fa181cc917af9bcad0d6a` |
| status_feature | `[8]` / `torch.float32` / `[1]` | `5dc1484333407fc1f3d843d003c150e8188a98ee9b636cbaa06d065c0a8e8cd9` |
| pixels_FP32 | `[9, 3, 448, 448]` / `torch.float32` / `[602112, 200704, 448, 1]` | `dd60c99b68580be8e5240951069e390b683a5c28bb8e05125a0bd649c73cd55d` |
| role_image | `[2800]` / `torch.bool` / `[1]` | `621c053c24b6db399240153747c0cb18cfe1c40daf1f6a7c344214f1ef1ed5e6` |
| role_validtext | `[2800]` / `torch.bool` / `[1]` | `9301bb7936486d9b956679b1902cbdd095f94abce86e44dee01f605d9b886db9` |
| role_padding | `[2800]` / `torch.bool` / `[1]` | `bbdfe78369f0d4464d26c1936f2e1bf8958a8856a444d0177afd20e043af3e28` |

VLM forward 选项为 `use_cache=False / output_hidden_states=True / return_dict=True`。左 padding，序列长度 2800，有效 2793、padding 7、image token 2304、有效文本 489；9 个图块、每块 256 个图像上下文 token。全部 token 值在 NPZ 中，不从摘要推导。

**5. norm1 之前还有一个独立输入门禁**

从仓库真实 PPU C 加相同 BF16 bias、CLS/position 重建相同 embedding。C 原字节 SHA 为 `c04efdfc37e1d5d2fc0ce8bf618df389103f20a7c183e53e708eaf3dbaf2389b`。真实 NPZ 文件 SHA 为 `9c1abc99ea9ba94db1160cdd89c566c5c2df4d2efacaed8e83d801deaa3a302d`。

norm1 输入必须为 BF16 `[9,1025,1024]`，stride `[1049600,1024,1]`，torch.strided、contiguous=True、storage_offset=0、device_type=cuda，原字节 SHA 必须为 `ba0df59744dadae4460575d1cf6786ffb3d5636d286f7267d57a113b2d789d74`。不符合时停止 norm1/QKV/encoder 比较；不能用不同输入判断算子独立差异。

**6. 后端软件差异必须保留，不能冒充“全部外部条件相同”**

| 后端身份 | PPU 当前实际值 | 已有 A800 记录 |
|---|---|---|
| 设备 | PPU-ZW810E | NVIDIA A800-SXM4-80GB |
| Torch 构建 | 2.4.0 | 2.4.0+cu124 |
| Torch git | 36615d2dc445b0232500ca5fadc986949833ae46 | e4ee3be4063b7c430974252fdf7db42273388d86 |
| PPU SDK | version: 2.1.0-a5f865 | 非 PPU 平台 |
| cuDNN API 报告 | 8905 | 90700（9.7.0，旧 A800 审计；新进程须重采） |
| CUDA 构建标记 | 12.4 | 须在新进程采集 |
| 实际已加载库 | 33 项，完整路径与大小见 JSON | 须返回实际映射，不能仅比较 pip 版本 |

PPU 的 8905 是兼容 API 报告；即使 A800 改成同一标签，也不能证明其二进制内核与 PPU 相同。现有 A800 cuDNN 90700 与 PPU 8905 **明确没有匹配**。若有兼容构建，可进一步做同版本后端对照；不能强行替换不兼容 ABI，或静默略过差异。

当前 PPU 主机为 Ubuntu 22.04.5 LTS，内核与 glibc 标记见配置 JSON；兼容查询报告驱动 `1.3.2-d7f5a2`、显存 97920 MiB，ppu-smi 为 v1.28（2026-04-17，9c3eb8d）。当前容器镜像 tag 未在本进程取得，已明确记为未知；历史 DLC job 的镜像与命令另保存在历史参考 JSON，不冒充当前容器身份。

门禁把可移植控制字段与平台后端身份分开报告。可移植字段全部匹配后，允许进行“相同输入/参数/配置下的后端比较”；`backend_differences` 和 `all_external_conditions_identical` 仍如实保留。**只有前者通过不等于整个软件栈相同，更不能直接归因于物理硬件设计。** 不同 Torch 构建、算子库、编译和驱动实现仍是可能变量。

**7. A800 单命令流程与必须回传的文件**

先更新审计分支，记录审计仓库 `git rev-parse HEAD`，保持教师仓库为固定 6bb7119。用压缩包及 C 的同名 `.sha256` 验证文件；将 v2 解包到独立目录，再运行其中 `load_inputs.py`。不要覆盖旧 A800 证据。

```bash
# 以下变量按 A800 现有路径填写；先使用准备好精确通用版本的新环境。
AUDIT_FOLLOWUP=/path/to/VLA-AD/reports/ppu-a800-alignment-followup-20261027
AUDIT_BUNDLE=/path/to/ppu-a800-embedding-audit-20261027-v2
TEACHER_REPO=/path/to/frozen/teacher/repo
ORIGINAL_VLM=/path/to/original/ReCogDrive-VLM-2B

python3 -B "$AUDIT_BUNDLE/load_inputs.py"
python3 -B "$AUDIT_FOLLOWUP/run_a800_controlled_visual.py" \
  --bundle "$AUDIT_BUNDLE" \
  --frozen-repo "$TEACHER_REPO" \
  --vlm "$ORIGINAL_VLM" \
  --output-dir /path/to/new/a800_controlled_visual_001
```

该驱动先显式配置新子进程，再按原入口构造非 Flash BF16 eval 模型；直接读取仓库张量。只读采集全部配置、源代码和权重文件/设备参数，然后调用严格门禁。

- 不一致：输出 `A800_CONTROLLED_CONFIGURATION.json` 和 `A800_CONTROLLED_GATE.json`，退出码 2；不执行卷积、norm 或 encoder。先解决列出的缺项/差异，不用新的算子 SHA 代替匹配条件。
- 可移植门禁通过：重建并核对 norm1 输入，保存 `A800_CONTROLLED_NORM1_INPUT_GATE.json`；再计算原生 C/patch/embedding，与真实 PPU C/patch 做精确误差比较，并执行原版首个 block 与纯视觉后缀核查。
- 成功执行的结果为 `A800_CONTROLLED_TRACE.json`。回传以上 JSON、实际环境配置和新子进程设置 `FRESH_PROCESS_SETTINGS.json`；异常时回传 `A800_CONTROLLED_FAILURE.json`，不要删掉失败记录后声称已通过。

脚本不执行语言、动作或教师。原 trace helper 名称中的 `*_A800` 在此引用的是 PPU JSON，结果额外提供 `*_PPU` 别名和明确参考身份，避免比较方向混淆。不得把 FP32/FP64 控制替换成原生教师输出。历史五组显式动作噪声 SHA 为 `c39e5183f1127b0faf42e649b148743cb57a34dc70c30657f0d881a9cf210f10`；本轮不运行动作，后续全链重放仍必须用该完整噪声，而不是只统一 seed。

当前只完成 PPU 配置归档和 A800 门禁/执行要求，**尚未在 A800 执行新的受控对照，不宣称其已匹配全部条件或完成全链对齐**。
