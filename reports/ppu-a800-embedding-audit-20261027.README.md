**最新 PPU 算子核查：**[结果说明](ppu-a800-alignment-followup-20261027/PPU_FIRST_BLOCK_RESULTS.md)、[PPU_FIRST_BLOCK_TRACE.json](ppu-a800-alignment-followup-20261027/PPU_FIRST_BLOCK_TRACE.json)。当前 PPU 视觉末层命中历史 SHA；相同 embedding 后首个分歧为 block 0 的 `norm1`。

# PPU 与 A800 单场景特征核查：真实数据、张量和诊断

已交付固定场景 `0002267a294e52a7` 的原始数据和完整实际模型输入，以及真正的 PPU 原生 BF16 无 bias 卷积张量 C。原教师冻结提交为 `6bb7119583513094a169b4c2f2dc2a05ac293aa3`。

- [v2 补充压缩包](ppu-a800-embedding-audit-20261027-v2.tar.gz)：原始 JPG、四帧历史元数据、完整 prompt、FP32/BF16 pixels、IDs/mask/position/image_flags、车辆状态与角色 mask；**12 项全量 SHA 全部命中原 PPU**。另含原 PPU/本机库版本、390 项本机安装版本、原 PPU 全链关键指纹、冻结预处理源码/完整 tokenizer、既有单场景 golden 数值及 20 节离线 HTML 文档。
- [真正的 PPU C 张量](ppu-a800-alignment-followup-20261027/ppu_native_no_bias.npz)：`[9,1024,32,32]`、BF16，原始数据 18 MiB、NPZ 14.9 MB；不在 v2 包内重复存储。原始字节 SHA 精确命中 `c04efdfc37e1d5d2fc0ce8bf618df389103f20a7c183e53e708eaf3dbaf2389b`。
- [A800 后续核验与 PPU 交付说明](ppu-a800-alignment-followup-20261027/README.md)、[C 实算记录](ppu-a800-alignment-followup-20261027/PPU_EXPORT_RECEIPT.json)及同目录文件 SHA。

```bash
cd reports
sha256sum -c ppu-a800-embedding-audit-20261027-v2.tar.gz.sha256
tar -xzf ppu-a800-embedding-audit-20261027-v2.tar.gz
cd ppu-a800-embedding-audit-20261027-v2
python3 -B load_inputs.py
```

打开 `REPORT.html`，新增内容在第 15–20 节；实际输入身份、shape/dtype/stride 和加载/比较方式见第 18 节。`load_inputs.py` 默认 CPU 读取，BF16 按原始字节重解释，无模型加载或前向。`compare_inputs(actual_inputs)` 可直接打印 A800 已有输入相对所附完整输入的逐项误差及前 16 个不同索引。元数据/NPZ 都不使用 pickle。

当前证据：A800 匹配 `cudnn.allow_tf32=True` 后，FP32 诊断 A/B 与 PPU 一致；剩余首个分歧为原生 BF16 无 bias 卷积。真实 C 已提供，接续工作是精确比较双方 C，并在 A800 将相同 C 接入后续路径检查是否还有独立差异。**尚未完成全链对齐，不恢复批量教师生成。**

原 PPU 的完整链结果来自历史实测，和本机首层/C 实测分开标注。原核查进程已退出，本次 C 沿用原入口重载同一冻结模型并使用已逐位验证的输入，没有修改参数或精度标志。原始输入为从原文件按冻结源码重新导出，并经全量旧 PPU 指纹核验；没有把重导出称为旧进程内存快照。

最初的 [32 KB 审计包](ppu-a800-embedding-audit-20261027.tar.gz) 和其原 SHA 保留原样，供已下载方追溯；旧采集脚本报告字段的单行修正在 v2 内记录，数值操作未变。本包不含完整大模型权重或全量数据集。`run_export_native_C.py` 是本机实际捕获过程的驱动记录，含本机路径；接收方使用后续目录原版导出函数或第 18 节的输入读取器。
