# 教师特征修复：算法源码、训练配置与缓存重建

源目录：`/mnt/data_and_weight/WM_Group/sjc_workspace/ReCogDrive`。冻结 teacher repo：`6bb7119583513094a169b4c2f2dc2a05ac293aa3`；历史研究树另由逐文件 SHA256 冻结。

- 原始文件 **4,547** 份，Python 源码 **2,607** 份；从原始 8 份档案出发的本地依赖闭包 **197** 份源码，关键本地模块无缺失。
- 已补 `ol_l1_data_pack_v1`、`ol_stage_b_final_factory_v1`、`ol_l3_outer_refit_v1`，以及配对状态优化、配对映射、L0–L3、stage A/B、最终六库工厂、DLC 执行器和交付脚本。
- 收录完整算法源码和训练/校准配置、精确 index/session 划分、冻结选择方案、两份 portable 的运行时/vendor 源码与人口清单；大型缓存、状态数组和运行结果由接收端重建。
- 最终库拟合划分：10,042 条 / 130 sessions；inner fit 8,021 条 / 104 sessions；inner calibration 2,021 条 / 26 sessions。原 8 份算法字节、全部 Python 语法、关键依赖和实际 session 隔离均已检查通过。

[缓存重建顺序与训练入口](REBUILD_METHODS.md) · [关键配置及原始 SHA](REPRODUCTION_KEY_CONFIGS.json) · [完整源码/配置目录](RESEARCH_SOURCE_GUIDE.md) · [依赖与划分核验结果](DEPENDENCY_AND_SPLIT_VALIDATION.json) · [路径映射和外部大资产](PATH_MAPPING_AND_EXTERNAL_ASSETS.json) · [逐文件清单（gzip）](SOURCE_MANIFEST.json.gz)

压缩包 `teacher-repair-reproduction-source-20261027.tar.gz`：**95,694,352 bytes**，SHA256：

```
98e4bc7c315fcf9326dac2fa3972d1d89aeb066094d2dc0169cde3537ecf2005
```

压缩包分为 **3 卷**，各卷不超过 40 MiB。取齐本目录文件后，在仓库根目录执行（输出位置自行指定）：

```bash
python3 -B reports/teacher-repair-reproduction-source-20261027/assemble_archive.py \
  --archive /YOUR/OUTPUT/teacher-repair-reproduction-source-20261027.tar.gz \
  --extract-to /YOUR/OUTPUT/extracted \
  --verify-sources
```

该命令检查各卷与整包 SHA、安全解包，再做只读源码与划分检查；不加载模型或启动训练。请使用新的输出路径。

本次保留原始源码和哈希保护，不改写历史路径、不绕过 ROOT/release 检查。按用户要求不附大型缓存、已拟合库数值或历史运行结果；包内 `REGENERATE_INSTEAD_OF_BUNDLING.json` 保存未附产物的原始路径和 SHA，`REBUILD_METHODS.md` 给出生成/优化/拟合/校准/训练顺序。完整权重和 NAVSIM 数据仍需接入；未宣称已从零复现实验或完成 PPU/A800 全链对齐。原始固定场景数据沿用仓库已有的 `reports/ppu-a800-embedding-audit-20261027-v2.tar.gz`。

[各卷清单](ARCHIVE_PARTS.json) · [压缩包逐文件回读核验](ARCHIVE_ROUNDTRIP_VALIDATION.json) · [接收端合并/解包/复验演练](RECEIVER_REPLAY_VALIDATION.json) · [SHA256SUMS](SHA256SUMS)
