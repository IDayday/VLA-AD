# E0 基线结果（已完成）

固定开发集1696场景/16完整log，5个配对推理种子。IL PDMS=0.911777，RL=0.948395，RL−IL=0.0366 [0.0230, 0.0535]。RL/COPY逐步输出和全量评分误差为0。

RL教师 M_minus=0.0095 [0.0052, 0.0159]，共有93场景的五次平均下降超过0.02。独立五次采样的 IL 自比较：{'PDMS': 0.9119058790363713, 'G': 0.00012886560388567818, 'M_plus': 0.005773177936076329, 'M_minus': 0.00564431233219065, 'material_drop_fraction': 0.02358490566037736, 'zero_scene_fraction': 0.01474056603773585, 'zero_output_fraction': 0.026650943396226416, 'positive_to_zero_fraction': 0.00294811320754717, 'stable_high_to_zero_fraction': 0.0, 'stable_high_retention': 1.0, 'zero_to_positive_fraction': 0.003537735849056604}。

IL/RL 活动头各347张量，326张量不同，其余调度/固定buffer完全一致。RL旧策略的347张量与官方IL逐项一致，实际教师加载活动RL而非该参考。所有核心参数missing/unexpected/shape mismatch均为空。

完整baseline分项、区间和逐场景结果在外部产物analysis/baseline_dev.json和evaluation/dev/E0。正式E1–E5比较尚在开发集评测，不在本基线报告预填成绩。
