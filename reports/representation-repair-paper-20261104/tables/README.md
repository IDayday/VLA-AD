# 汇总表与绘图输入

所有文件均为人口汇总或已发表报告中的学习曲线，不包含逐场景轨迹/状态数组。

- `main_results.csv`：三力度及正式迁移、续训后修复；指标、人口、Base、均值、SD与结果角色分列。
- `ablation_contrasts.csv`：四个普通TEST核心对照及会话描述性区间。
- `course_comparison.csv`：两种课程、三个固定人口、三训练seed。
- `main_failure_and_preservation.csv`：低/中/高分组的总体贡献与损伤。
- `deployment_cost.csv`：单代表原图P50/P95与峰值内存。
- `synthetic_rl_curves.csv`：六条课程共78个完整真实TUNE点。
- `figure_data.json`：全部图表的轻量、可移植输入。
- `main_results.tex`、`core_ablations.tex`：可放进论文的booktabs表片段。

`training_seed_SD`是样本标准差，不是置信区间或集成误差。navhard为一份TUNE代表模型的五动作噪声评价，另行标明。CSV分数以0—100计。
