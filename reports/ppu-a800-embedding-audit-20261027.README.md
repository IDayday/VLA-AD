# PPU 与 A800 首层视觉 embedding 核查交接

**A800 后续核验与 PPU 最小补充要求已更新：**见 [后续说明及导出脚本](ppu-a800-alignment-followup-20261027/README.md)。当前尚未完成原生 BF16 全链对齐，批量生成继续暂停。

压缩包：`ppu-a800-embedding-audit-20261027.tar.gz`。内容与此前交付的 ZIP 逐文件一致，包含详细 HTML 报告、实际测量 JSON、A800 采集脚本、操作说明和校验清单。

```bash
cd reports
sha256sum -c ppu-a800-embedding-audit-20261027.tar.gz.sha256
tar -xzf ppu-a800-embedding-audit-20261027.tar.gz
```

解包后打开 `ppu-a800-embedding-audit-20261027/REPORT.html`。报告为独立离线文件，已内嵌完整测量与脚本；A800 具体操作见报告第 9 节。

固定场景：`0002267a294e52a7`。源教师冻结提交：`6bb7119583513094a169b4c2f2dc2a05ac293aa3`。

本机已逐位复现首层视觉 embedding，首个 PPU–A800 分歧位于 patch 卷积。完整视觉后缀、语言末层和动作接口特征尚未由本轮验证。631 个不同元素是本机 native patch 与本机参考 B 的差异，不能当作跨平台误差。

包中不包含模型权重、原始图像或实际完整张量载荷。采集脚本默认只打印，必须在已加载模型和实际输入的会话中调用，不会自动加载模型或运行教师批处理。
