# Naming Audit

目标命名：完整框架 **AMPT**（Aligned Multi-Trajectory Policy Training）；Stage 1 **PC-MTS**；Stage 2 **FF-PGRPO**；Stage 3 **APR**。`Score`、`Pareto`、`PC-MTS` 作为数据构建变体保留。

## 1. 论文中的命名冲突

| 位置 | 发现 | 结论/处理 |
|---|---|---|
| `docs/AuthorKit27 (7).pdf`，正文贡献 bullet（PDF 第 2 页） | `Iterative Pareto-guided policy refinement (IPR)` | 与全文其他位置的 `Adaptive Pareto-Guided Policy Refinement (APR)` 冲突；建议正文仅改为 `Adaptive Pareto-Guided Policy Refinement (APR)` |
| PDF 其余方法叙述/图表 | AMPT、PC-MTS、FF-PGRPO、APR | 作为附录规范名 |
| PDF 数据构建对照 | GT、Score、Pareto、PC-MTS | 按指南保留，不把 `Score/Pareto` 改成方法阶段名 |

本次未找到主稿 `.tex`，因此无法给出 IPR 的 LaTeX source line，也未直接修改正文。

## 2. 源码/历史分支中的旧名

| 旧名 | commit:path | 实际含义 | 附录处理 |
|---|---|---|---|
| FIRST-Drive V7 curriculum | `41a8613:navsim/agents/recogdrive/curriculum/`; `.../first_drive_v7_agent.py` | PC-MTS/Stage 1 的最接近实现谱系，同时含后续 bridge 机制 | 论文叙述统一写 PC-MTS；首次代码注释可写“implemented in historical FIRST-Drive V7 curriculum” |
| Phase A/B/C | `41a8613:navsim/planning/script/config/experiment/first_drive_v7_phase_{a,b,c}.yaml` | V7 内部 retention/repair/progress/mode curriculum 阶段 | **不得**直接映射为 AMPT Stage 1/2/3；附录配置表用原配置名并解释 |
| FF-Pareto / Feasibility-First Anchor Pareto v3 | `41a8613:navsim/agents/recogdrive/stage3_ff_pareto.py:FF_PARETO_VERSION` | FF-PGRPO 的 credit core 历史名 | 论文统一 FF-PGRPO；代码定位保留原 symbol |
| Core-Pareto GRPO v2 | `88579c8` 谱系；checkpoint manifest `checkpoints/recogdrive/stage3_sota_backups/.../manifest.md` | 早期 scalar/Pareto Stage3 checkpoint，PDMS 0.910274 | 作为 historical baseline/implementation predecessor，不直接重命名为最终 FF-PGRPO |
| LFP-GRPO / Safe DiffGRPO / SDR / GSPO / DR-GRPO / AWAC / DPO / CEM | `codex/a5-epdms-stage3-20260726:scripts/stage3/` 与 `scripts/training/` | 中间实验算法/分支 | 只在 provenance 或消融需要时保留，不放入 AMPT 三阶段主叙述 |
| RAP | `ab9124c:scripts/stage3/launch_pdms91_rap_dual_branches_vla.sh` 及相关 generator | 外部 RAP 模型/候选源名 | **绝不能**机械改为 APR |
| teacher refinement / strict teacher / safety repair teacher | `fb96a03` 与后续 `scripts/stage3/build_pdms_*teacher*.py` | APR 的机制实现名 | 附录统一描述为 APR teacher construction/audit，代码注释保留 symbol |
| TIES / anchored task-vector graft / reverse conflict residual | `d24b7c7`, `64f5531`; 最终 checkpoint output dir | 可选 checkpoint consolidation | 不改名为 APR；写作中说明是 APR round consolidation（若正文保留） |

## 3. A/B/C、A0–A5、A1/A2/A3 的歧义

仓库存在至少三组互不相同的字母命名，均不是论文 AMPT 三阶段：

1. `first_drive_v7_phase_a/b/c.yaml`：V7 curriculum 内部阶段；
2. `scripts/run_recogdrive_expert_ablation_plan.py` 与 `configs/ablations/recogdrive2b_A*.yaml`：A1=JEPA-only、A2=VGGT-only、A3=context-only 等专家消融；
3. `docs/PTA_FS_DiT_Stage2_Implementation.md`、`stage2_dpsi_fs_x0_anchorbank_plan.md`：A1/A2/A3 是 FS/PTA/DPSI 组件消融；另有旧 CoT 的 Stage A1/A3。

附录不可出现孤立的“A/B/C”或“A1/A2/A”；若必须引用旧实验，需带完整实验族前缀，例如 `V7 Phase B`、`expert ablation A2 (VGGT-only)`。

## 4. 当前工作树中的论文名覆盖率

`git grep` 检查 `ba5908d`：

- tracked source 中未发现 `AMPT`、`PC-MTS`、`FF-PGRPO`、`APR` 或 `IPR` 方法实现 symbol；
- `AMPT` 主要出现在未跟踪、论文面向的 `outputs/ampt_*` provenance/定性资产中；
- 因而附录不能写“见当前源码 `AMPT` 类”，必须引用历史 commit:path:function。

## 5. 建议的术语映射规则

| 论文术语 | 允许的代码定位别名 | 禁止的替换 |
|---|---|---|
| AMPT | 整体训练谱系/最终 submission | 不把任一 `stage3_*` 单独称为完整 AMPT |
| PC-MTS | FIRST-Drive V7 curriculum mechanisms | 不把 V7 Phase A/B/C 当 AMPT 三阶段 |
| FF-PGRPO | `stage3_ff_pareto` credit assignment + GRPO loss integration | 不把所有 Core-Pareto/LFP runs 自动归为最终 FF-PGRPO |
| APR | teacher audit + interpolation/re-evaluation + retention + dynamic rebuild | 不把 RAP、TIES 或单次 checkpoint interpolation 单独改名 APR |
| Score/Pareto/PC-MTS | supervision construction variants | 保持原样 |

## 6. 文件/图表自动检查建议

最终 supplementary 构建前执行大小写敏感检查：

```bash
rg -n '\b(IPR|FIRST-Drive|FF-Pareto|Phase [ABC]|A[0-5])\b' supplementary \
  --glob '*.tex' --glob '*.md' --glob '*.csv'
rg -n '\b(APR|AMPT|PC-MTS|FF-PGRPO)\b' supplementary/sections
```

允许命中仅限：审计文件、显式 historical implementation 注释、或带完整解释的 provenance。最终 PDF 不应出现 IPR；`APR/IPR` 不得混用。
