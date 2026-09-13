"""Report every method/seed/endpoint, without selecting a best checkpoint."""
from common_r import *
import pandas as pd
from figures_r import LABELS
def main():
    d=pd.read_csv(OUT/'metrics/seed_means.csv');g=pd.read_csv(OUT/'metrics/gain_bootstrap.csv');paired=pd.read_csv(OUT/'metrics/paired_gain_contrasts.csv');old=pd.read_csv(V3/'metrics/F_seed_means.csv');oldgain=pd.read_csv(OUT/'metrics/formal_minus_old_recipe_gain.csv')
    lines=[];a=lines.append
    a('# V3 GRPO rerun with the formal stage3 recipe — 2026-09-13\n')
    a('本报告重跑 V3 的 downstream GRPO 实验，复用原六组 micro-SFT 初始化，并增加未经额外 SFT 的 official IL。V1/V2/V3 与已暂停的 Progressive quick test 均保留；这里的结果属于新配置实验，不能回填或替换旧 V3 CSV。\n')
    a('## 1. 实验范围与正式配置\n')
    a('700 个训练场景、300 个诊断 holdout、两个 seed（1701/2903）、七种初始化。每组正式训练 10 epochs，8 GPU × 每卡 8 场景，accumulation=1。DistributedSampler 每 epoch 将 700 补齐到 704（原生 sampler 的 4 次场景重复），每卡 88 场景，即 11 updates/epoch、共 110 个固定训练 step。保存 0/10/50/100/110，主终点为 110；100 专门与旧 V3 的相同 step 预算比较。不挑最好 checkpoint。\n')
    a('所有快照在全部 300 holdout 场景上生成 64 条 CRN 轨迹，使用 V3 同一冻结推理类、FP32、相同 token-keyed noise；step0 复用已验证的 V3 评测缓存。PDMS 为随机 rollout 的均值，不是挑最高轨迹的 oracle 分数。共新增 1,075,200 条评测轨迹。场景属于原 NAVTRAIN 中的内部机制划分，holdout 没有进入本轮权重更新；不能称为未经原官方 IL 预训练见过的外部测试集。\n')
    a('| 项目 | 旧 V3 wrapper | 本轮正式 stage3 |\n|---|---|---|')
    for row in [('GRPO forward','原生 forward_grpo','当前正式 ReCogDriveAgent + AgentLightningDiT'),('AdamW betas','(0.9, 0.999)','(0.9, 0.95)'),('weight decay','0.01','0.0001'),('LR','1e-4 常数','1e-4 起，按 epoch 的 10-epoch cosine，0 warmup'),('scene batch','8，逐 scene 累积','8 GPU × 每卡 8 = 64，正式 padding/collate'),('training precision','FP32','16-mixed，原生 AMP'),('BC reference','所有方法统一 official IL','每组自身的冻结初始化，遵循正式 shell 的 CHECKPOINT 规则'),('BC coefficient','0.1','0.1'),('group / denoise','8 / 5','8 / 5'),('std / noise clip','0.04 / 5','0.04 / 5'),('EP/TTC/comfort reward','10 / 5 / 2','10 / 5 / 2'),('snapshots','0/10/50/100','0/10/50/100/110')]:a('| '+' | '.join(row)+' |')
    a('\n训练直接使用 `ReCogDriveAgent.get_optimizers()`、`AgentLightningDiT`、原生 Lightning Trainer 和正式 collate；没有再次手写 optimizer/scheduler 或修改 advantage。唯一计算加速是标量等价的批量 NAVSIM reward。四场景 FP32/16-mixed 审计 reward/loss 差为 0；FP32 梯度最大差低于 1.4e-9，16-mixed 梯度差为 0。每组当前 action head 和其 reference 的有效 tensor 均与指定初始化完全相同。当前代码新增但已默认冻结的 action-aware auxiliary head 不参与原 GRPO。\n')
    a('正式入口：`scripts/training/run_recogdrive_train_multi_node_rl_2b.sh`、`navsim/planning/script/run_training_recogdrive_rl.py`。根配置虽含 `use_deepspeed` 字段，正式入口实际创建普通 `pl.Trainer`；本轮遵循实际入口，不激活额外 DeepSpeed。\n')
    a('历史成功 run 为 `stage3_rl_2b_official_chunkcache_retry_20260608T225423Z`：85,109 train scenes、1,330 updates/epoch、13,300 updates；watcher 中 epoch0/8/9 为 88.2648/90.4199/90.5500（完整 Navtest）。epoch0 已训练 1,330 updates，不是 untouched IL step0。这里保留正式单步行为与 10-epoch 调度，但仅训练 V3 的 700 scenes，不是完整历史训练重放。这些历史分数不能与本报告的 300-scene 随机轨迹均分直接相减。\n')
    a('## 2. 主结果：全部 300 holdout，两个 seed\n')
    a('| 初始化 | Step0 PDMS | Step110 PDMS | ΔPDMS [scene-paired 95% CI] | Δ feasible (pp) | Scene win fraction |\n|---|---:|---:|---:|---:|---:|')
    means=d.groupby(['method','step']).mean(numeric_only=True)
    for m in CFG['methods']:
        z=means.loc[(m,0)];f=means.loc[(m,110)];gg=g[(g.method==m)&(g.step==110)&(g.metric=='PDMS_gain')].iloc[0]
        a(f'| {LABELS[m]} | {z.PDMS:.4f} | {f.PDMS:.4f} | {gg["mean"]:+.4f} [{gg.ci_low:+.4f}, {gg.ci_high:+.4f}] | {(f.feasible_rate-z.feasible_rate)*100:+.3f} | {gg.win_fraction:.3f} |')
    a('\n主统计先在每个 scene 内平均两 seed，再对同 scene 的差值做 3,000 次 bootstrap；原始每 scene/seed 和 mean/median/95% CI/win fraction 均保存在 CSV。场景 bootstrap 不等于跨训练 seed 的泛化置信区间，两 seed 仍是有限重复。\n')
    a('### 全部固定快照与 seed\n')
    a('| 初始化 | Seed | Step0 | Step10 | Step50 | Step100 | Step110 |\n|---|---:|---:|---:|---:|---:|---:|')
    for m in CFG['methods']:
        for sd in CFG['seeds']:
            r=d[(d.method==m)&(d.seed==sd)].set_index('step');a('| '+LABELS[m]+' | '+str(sd)+' | '+' | '.join(f'{r.loc[s,"PDMS"]:.4f}' for s in CFG['snapshots'])+' |')
    a('\n## 3. 与旧 V3 的并列比较\n')
    a('| 初始化 | 旧 V3 step100 ΔPDMS | 正式 recipe step100 ΔPDMS | 配置重跑带来的 gain 差 [95% CI] |\n|---|---:|---:|---:|')
    for m in CFG['methods'][1:]:
        oo=old[(old.method==m)&(old.step==100)].PDMS.mean()-old[(old.method==m)&(old.step==0)].PDMS.mean()
        nn=means.loc[(m,100),'PDMS']-means.loc[(m,0),'PDMS'];rr=oldgain[(oldgain.method==m)&(oldgain.step==100)&(oldgain.metric=='PDMS_gain')].iloc[0]
        a(f'| {LABELS[m]} | {oo:+.4f} | {nn:+.4f} | {rr["mean"]:+.4f} [{rr.ci_low:+.4f}, {rr.ci_high:+.4f}] |')
    a('\n上述差值是整个 recipe 的对照，不是 beta、weight decay、batch、precision、schedule 或 BC-reference 中任何单项的独立因果效应。尤其相同 optimizer step 下，本轮 rollout scene budget 是旧 V3 的 8 倍；不能把两者称为 compute-matched。新七组之间则使用完全相同预算、采样顺序与参考设置规则。\n')
    a('## 4. 安全、轨迹分布与优化干扰\n')
    a('| 初始化 | Final feasible % | Hard failure % | EP | TTC | Policy-change ADE m | Spread-AUC | Hit@8 | IL retention loss |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for m in CFG['methods']:
        x=means.loc[(m,110)];a(f'| {LABELS[m]} | {x.feasible_rate*100:.3f} | {x.hard_failure*100:.3f} | {x.EP:.4f} | {x.TTC:.4f} | {x.CRN_policy_change_ADE:.4f} | {x.Spread_AUC:.4f} | {x.Hit8:.4f} | {x.IL_retention_loss:.6g} |')
    a('\n`training_safety_dynamics.csv` 保存每一步 EP/NC/DAC/TTC/DDC、feasible、zero-score、PIA、LR、梯度 norm 与 AMP skip。PIA 严格为 `count(A>0 AND infeasible)/count(infeasible)`，不是占所有 rollout 的比例。训练 reward 权重 10/5/2 与标准评测 5/5/2 按原正式实现保留；不能把 training reward 上升直接解释为 PDMS 上升。\n')
    a('## 5. 本轮可以与不可以支持的判断\n')
    a('- **SUPPORTED（本固定实验范围）**：正式 recipe 的 14 个运行在第 110 步均高于各自 step0；与旧 V3 相比，相同第 100 步的平均 gain 改善约 4.90–6.76 points，所有六组 paired CI 均高于 0。但新旧训练样本预算不同，不能把它解释成某个单一超参数的因果效应。对各组绝对增益，除 Old-PC 外的 scene-bootstrap CI 仍跨 0，两个 seed 的正向一致性不等于普遍显著性。')
    il=g[(g.method=='official_il')&(g.step==110)&(g.metric=='PDMS_gain')].iloc[0]
    state='SUPPORTED' if il.ci_high<0 else 'NOT SUPPORTED'
    a(f'- “未额外 SFT 的 official IL 在本短程设置下也下降”：**{state}**。固定终点增益 {il["mean"]:+.4f}，95% CI [{il.ci_low:+.4f}, {il.ci_high:+.4f}]。这只适用于本冻结子集预算。')
    for m in ['score','pareto','conditional_pc']:
        c=paired[(paired.endpoint==110)&(paired.method==m)&(paired.reference=='gt_only')&(paired.metric=='PDMS_gain')].iloc[0]
        a(f'- {LABELS[m]} 相对 GT-SFT 的 GRPO gain 差：{c["mean"]:+.4f} [{c.ci_low:+.4f}, {c.ci_high:+.4f}]；scene win fraction={c.win_fraction:.3f}。'+('有正向差异证据。' if c.ci_low>0 else '有负向差异证据。' if c.ci_high<0 else '区间跨 0，不能声称显著优于或劣于 GT-SFT。'))
    final_pairs=pd.read_csv(OUT/'metrics/paired_final_contrasts.csv')
    for metric,factor,label in [('PDMS',1,'最终 PDMS'),('feasible_rate',100,'最终 feasible rate (pp)')]:
        c=final_pairs[(final_pairs.endpoint==110)&(final_pairs.method=='conditional_pc')&(final_pairs.reference=='gt_only')&(final_pairs.metric==metric)].iloc[0]
        a(f'- Conditional-PC 减 GT-SFT 的{label}：{c["mean"]*factor:+.4f} [{c.ci_low*factor:+.4f}, {c.ci_high*factor:+.4f}]。')
    a('- **PARTIALLY SUPPORTED**：Conditional-PC 相对 GT-SFT 有较小的 GRPO policy-change ADE（少约 0.0093 m）、较低 IL retention loss（差约 -0.000333）和较高 Hit@8（约 +1.44 pp）；这些辅助指标的 paired CI 不跨 0。然而它们没有转化为 PDMS gain 或整体安全性的显著优势，不能据此声称已经解决 downstream optimization。')
    a('- Global-Pareto 的最终 PDMS 比 GT-SFT 高约 +0.4586 [0.1324, 0.8083]，但 feasible rate 低约 -0.5573 pp [-1.1225, -0.0417]；Score-MTS 的最终 feasible rate 也低约 -1.1745 pp [-1.8464, -0.6094]。因此仍存在 quality/safety tradeoff，和“GRPO 持续退化”是不同结论。')
    a('- **NOT SUPPORTED**：把旧 V3 的普遍退化归因于“高分轨迹 SFT 必然破坏 GRPO”。旧实验的 GT-SFT 同样明显退化，而且当时没有 untouched official IL 直接对照，训练 recipe 也未完整匹配正式 stage3。')
    a('- **UNTESTED**：具体哪一个配置差异解释改善/退化；完整 85,109-scene 训练是否重现历史最终性能；本轮 PC-MTS 是否在多个完整预算和更多 seed 上具有稳定优势。')
    a('- V3 静态 learnability 的 source/scene/quality 控制结果继续保留，但不能用静态 diffusion loss 相关性替代本轮 downstream 因果结果。PC-MTS 的核心贡献地位应以它相对 GT-SFT 的实际 quality/safety tradeoff 为准，不以“比旧错误 wrapper 好”代替必要性证据。\n')
    a('### 为什么旧的下降不能归因于高分 SFT\n')
    a('**已测得的事实**是整套正式 recipe 与旧 wrapper 给出了不同的学习轨迹。**机制解释仍属待拆分验证**：更大的 batch 会改变梯度估计及其方差；按 epoch 衰减 LR 会减小后期更新；以自身初始化作为 BC reference 会保留不同 SFT 初始化已有的行为，而不是把所有方法拉向同一个官方 IL reference。AdamW 的 beta2 与 weight decay 也不同，但本实验没有单因素 ablation，不能断言其中哪一个是主因，尤其不能仅凭参数差异夸大 weight decay 的作用。正式 batch collate 的 padding 行为也按原入口保留。\n')
    a('早期不稳定并未被保证消除。原始 GRPO 使用组内相对 reward advantage；组内较好的轨迹仍可能不满足 TTC/DDC 的 conservative feasibility。实现使用 advantage-weighted diffusion log-probability 与 BC，并没有 PPO-style probability-ratio clipping 的 trust region。这些代码特征提供了应继续检查的机制，但不能仅凭本次共同出现的早期下降就视作已经证明的因果解释。\n')
    a('### PC-MTS 是否已足以作为论文核心贡献\n')
    pc=paired[(paired.endpoint==110)&(paired.method=='conditional_pc')&(paired.reference=='gt_only')&(paired.metric=='PDMS_gain')].iloc[0]
    if pc.ci_low<=0:
        a('**目前不足。** 本轮尚不能证明 Conditional-PC 相对关键 GT-SFT baseline 具有稳定的 downstream GRPO gain 优势。即使其绝对 PDMS 上升，也不能把共同 recipe 修正带来的恢复当成 PC-MTS 独有贡献。可保留“quality alone 不能完整描述 supervision 的可学习性”作为静态机制观察；“PC-MTS 因果地解决 GRPO 退化”这一更强表述尚未获得本轮必要证据。\n')
    else:
        a('本轮给出 Conditional-PC 相对 GT-SFT 的短程 gain 优势证据，但两个 seed、700-scene 训练子集不能单独支持完整训练性能和普遍必要性。是否作为核心贡献仍需结合本表的 safety tradeoff、更多 seed 和完整预算的 matched causal chain。\n')
    a('## 6. 审计、资源与文件\n')
    a(f'Primary YAML SHA256：`{sha(CONFIG)}`。完整输入、源代码、checkpoint、split、CRN 与计数审计见 `outputs/pc_mts_v3_stage3_rerun/manifests/`。')
    a('独立训练分配至四台完整 8-GPU 主机，并在第五台的 5 张空闲卡和训练主机空余显存上并行评测。一台主机 GPU 5–7 存在本容器不可见的占用，该节点两次在第一步优化前发生 OOM；失败缓存和日志保留于 `invalid/`，该 run 按原配置/seed 从相同初始化转移到完整空闲节点。没有停止不可见的占用进程。')
    a('环境审计识别出一台默认 Python 为 3.10，切换到现有 Python 3.9 环境；所有使用的运行时均为 Torch 2.5.1+cu124、Lightning 2.6.0、Transformers 4.57.6。跨主机冻结 V3 public reference 轨迹与 PDMS 都一致。维护中的 root 推理类曾显示数个 FP32 ULP 的 batch 差异，因此遵循本实验原定“复用 V3 评测定义”，采用其冻结推理类；未改变评测数值、阈值或 scene。正式训练仍使用 root stage3。相关失败审计日志保留。')
    a('并行 CPU 评分曾把正在原子写入的 `.tmp.npz` 当作已完成输入，触发 token lookup 错误；已改为只发现精确匹配 manifest token 的完成文件并增加回归测试。现有已完成缓存原样保留、按输入 hash 校验后续算，没有重评分后挑选有利版本。')
    a('\n- 代码：`tools/analysis/pc_mts_v3_stage3_rerun/`\n- 配置：`configs/pc_mts_v3_stage3_rerun/primary.yaml`\n- 原始绘图数据：`outputs/pc_mts_v3_stage3_rerun/metrics/*.csv`\n- 图（PNG + PDF）：`outputs/pc_mts_v3_stage3_rerun/figures/Fig-R1_Formal_GRPO_gain_and_safety`、`Fig-R2_Old_vs_formal_recipe`、`Fig-R3_EP_safety_and_policy_shift`、`Fig-R4_Training_safety_and_advantage`\n- 大型 rollout、原始日志与 checkpoints 保留服务器，不提交到 Git。')
    target=ROOT/'reports/PC_MTS_V3_FORMAL_STAGE3_GRPO_RERUN_20260913.md';target.write_text('\n'.join(lines)+'\n');print(target)
if __name__=='__main__':main()
