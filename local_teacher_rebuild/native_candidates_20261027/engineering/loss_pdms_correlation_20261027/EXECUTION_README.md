# A组不同loss权重的完整测试与相关性

本任务预先固定epoch1/4/8/12/16，以完整Test12146×S0-S4评价A的高到低loss权重，报告线上整轮H与固定Train8192面板H的Pearson/Spearman。Test8/16仅复用原正式owner，额外Test1/4/12分别与B/C/D使用相同固定4卡资源；没有额外物理GPU或临时服务器。四组正式训练保持运行。

`REQUEST.json`、`INPUT_MANIFEST.json`、`A_EPOCH_LOSS_SNAPSHOT.json`在读取Test结果前冻结。`METHODS_FOUR_ARMS.md`给出实际四组方法、共同参数和源码依据。

`correlation_owner.py`真正执行补评与全部5个面板H，等待正式Test8/16的完整证据，然后调用冻结`correlation_analysis.py`生成实际结果；不完整会明确失败而非编造相关系数。`SCOPE_FREEZE.json`与`ANALYSIS_CODE_FREEZE.json`分别绑定执行和分析代码。

最终输出`analysis/CORRELATION_ANALYSIS.json`、`CHECKPOINT_LOSS_TEST.csv`、`LOSS_VS_TEST_PDMS.png/svg`及`LOSS_PDMS_CORRELATION_REPORT.md`。额外补充`LOSS_COMPONENTS.csv`/JSON/MD拆开保持与修复损失，供解释整体loss与行为的差异。

以下为读取完成证据后的复现命令，不应在当前owner活跃时重复派发：

```bash
ZT_AUTO_PROXY=0 OPENBLAS_CORETYPE=Sandybridge OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 /mnt/project/recogdrive-distillation-first42-a800-20261022/.venv/bin/python -B correlation_analysis.py \
 --manifest INPUT_MANIFEST.json --out analysis
ZT_AUTO_PROXY=0 OPENBLAS_CORETYPE=Sandybridge OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 /mnt/project/recogdrive-distillation-first42-a800-20261022/.venv/bin/python -B supplemental_loss_metrics.py \
 --manifest INPUT_MANIFEST.json --analysis analysis/CORRELATION_ANALYSIS.json --out analysis
```

五个checkpoint来自同一训练轨迹，相关系数是本次描述性分析，S0-S4不是五个训练seed。Test已按用户要求用于模型比较，不称独立held-out测试。固定面板H不称完整Train误差；配对Base单列，不作为额外有loss的数据点。
