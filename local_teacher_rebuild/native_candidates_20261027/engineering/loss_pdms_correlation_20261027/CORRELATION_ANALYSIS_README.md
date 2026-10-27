# A组固定checkpoint loss与完整Test PDMS分析

`correlation_analysis.py`只执行CPU分析，不启动GPU、训练、评测或日志轮询。
输入证据必须已经由真正的原生评测产生并完成封存。该脚本不把已提交作业、
部分场景、loss下降或历史PPU分数当作完整Test结果。

固定点为A / H_full / seed20261027的epoch **1、4、8、12、16**，按训练进度和
训练loss跨度预先冻结，不按测试成绩挑点。第8/16轮复用正式Test评测，其他三点
使用独立诊断路径；不改变正式权重选择的五个epoch端点。

每个点要求：

- 完整checkpoint、`.COMPLETE.json`、原训练资产身份及真实场景曝光；
- 完整Test **12,146场景 × 同一S0–S4**，真实配对轨迹和全部七项PDM分量；
- 同一固定Train **8,192面板**、full教师目标的冻结权重表征误差；
- 冻结逐epoch统计，完整85,813场景/1,341次更新、相同checkpoint SHA；
- 所有权重的Test token/log顺序、显式噪声、配对Base、教师、特征和定标一致。

脚本复用原始`verify_evaluation`和`read_checkpoint`，并从逐场景文件重新计算
Test PDMS及面板H，核对SUMMARY。对缺失噪声、SHA不符、换人群、换Base或不完整
真实曝光均拒绝。缺任一点时输出`INCOMPLETE`、退出码2；不输出相关系数。

两种loss分开报告：

1. **整轮在线loss/H**：该轮内权重不断更新时的85,813场景加权均值。
2. **固定Train8192面板H**：同一个已保存权重在相同面板上的冻结表征误差。
   此项是面板指标，不冒充完整Train H或完整Train行为分数。

Pearson和Spearman均按五个checkpoint的loss与完整Test PDMS配对计算。
负相关表示这些权重中loss较低时Test分数较高。该任务只有一条训练轨迹上的五个
相关权重，提供描述性关联，不给把它们视作独立训练样本的显著性p值。
附加95%区间仅按Test log成对重采样，条件于固定权重和固定Train损失，不能
解读成训练种子不确定性。S0–S4也不能算作五个训练种子。

Test按用户要求参与模型选择及方法比较，不再声称是独立held-out最终测试。

## 执行

```bash
ZT_AUTO_PROXY=0 OPENBLAS_CORETYPE=Sandybridge OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /mnt/project/recogdrive-distillation-first42-a800-20261022/.venv/bin/python -B \
  correlation_analysis.py --manifest INPUT_MANIFEST.json --out analysis
```

产物：`CORRELATION_ANALYSIS.json`、`CHECKPOINT_LOSS_TEST.csv`、
`LOSS_PDMS_CORRELATION_REPORT.md`、`LOSS_VS_TEST_PDMS.png/svg`。
完整结果才生成散点图和系数；缺失项逐条列出。完整扫描与bootstrap预计1–3分钟，
繁忙IO时预留5分钟缓冲，此范围是运行前估计，不是已测成本。

输入manifest格式见当前固化的`INPUT_MANIFEST.json`；`epoch_statistics`与`request`
都要求`{path,sha256}`。脚本不会修改这些输入、checkpoint或冻结训练目录。

CPU检查：

```bash
ZT_AUTO_PROXY=0 OPENBLAS_CORETYPE=Sandybridge OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /mnt/project/recogdrive-distillation-first42-a800-20261022/.venv/bin/python -B \
  -m unittest -v test_correlation_analysis
```

检查覆盖正/负关联、Spearman并列rank、相同loss/分数、配对次序、真实场景归一化、
checkpoint关联证据、Test log配对bootstrap，以及原生完整格式验证器对缺噪声的拒绝。
CPU格式fixture不是正式Test评测，不使用GPU或生产输出路径。
