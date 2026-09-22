# ReCogDrive原版GRPO：EP=5、10与历史版本核验

结论：90.41这条历史original-GRPO训练链使用的官方版本，训练reward的EP系数为10；标准PDMS评测系数为5。本轮分析没有将原本为5的训练配置改成10。此前“从5提高到10”的表述应理解为两个打分器的配置差异，不是IL阶段先使用PDMS reward=5再被我们切换为10。

## 官方版本历史

| 官方commit | 日期 | GRPOConfig中EP/TTC/Comfort |
|---|---|---|
| `30212249d90bc6cd6248390365dd232d1c85c7d4`（first version） | 2025-08-21 | 30 / 5 / 2 |
| `1bc43060d3d6af5ed03bf46cbd1ac227e7cb38b3` | 2025-08-24 | 30 / 5 / 2 |
| `e3d5810a3529271c5a546c2532d29ae390bc650c` | 2025-10-04 | 10 / 5 / 2 |
| `8a200077601ef33414469cfbe9fd095e322cccb6`（本地90.41训练采用的版本） | 2026-03-13 | 10 / 5 / 2 |

从官方GitHub API取得该文件历史，再逐个下载固定commit的源文件验证，不能用本地shallow clone的 `git log -S` 推断首次修改日期。这里没有声称作者发布的所有历史RL权重都使用同一组参数。

- [官方首版GRPOConfig：EP=30](https://github.com/xiaomi-research/recogdrive/blob/30212249d90bc6cd6248390365dd232d1c85c7d4/navsim/agents/recogdrive/recogdrive_diffusion_planner.py#L90-L92)
- [2025-10-04版本GRPOConfig：EP=10](https://github.com/xiaomi-research/recogdrive/blob/e3d5810a3529271c5a546c2532d29ae390bc650c/navsim/agents/recogdrive/recogdrive_diffusion_planner.py#L90-L92)
- [本地所用官方版本GRPOConfig：EP=10](https://github.com/xiaomi-research/recogdrive/blob/8a200077601ef33414469cfbe9fd095e322cccb6/navsim/agents/recogdrive/recogdrive_diffusion_planner.py#L90-L92)
- [同一版本的标准评测配置：EP=5](https://github.com/xiaomi-research/recogdrive/blob/8a200077601ef33414469cfbe9fd095e322cccb6/navsim/planning/script/config/pdm_scoring/default_scoring_parameters.yaml#L20-L22)

## 90.41权重的具体证据

历史run：`stage3_rl_2b_official_chunkcache_retry_20260608T225423Z`。

1. `commands.log`指定从归档 `official_recogdrive` 目录加载代码，开启 `agent.grpo=True`，IL初始化及reference均为官方IL。
2. run保存的 `code/hydra/config.yaml` 没有root scorer，也没有agent奖励权重覆盖；`overrides.yaml`亦无对应覆盖。
3. 归档planner、agent与PDM scorer三个文件均与官方commit `8a200077…` 的Git blob逐字节一致；planner SHA256为 `c349bf196611034ebcdfb3dd30fe02e1ac8ee4fc6a76d65934b70dbf583d994a`。
4. agent中的 `make_recogdrive_config` 使用默认 `GRPOConfig`；仅覆盖metric cache路径和reference checkpoint路径。`_init_grpo`将 `cfg.scorer_config` 交给 `train_scorer`；`reward_fn`调用的正是这个scorer。
5. CPU上调用真实配置factory并实例化真实PDMScorer，得到权重数组 `[10,5,2,0]`（EP/TTC/Comfort/DDC），没有构建policy、没有GPU推理或权重更新。
6. 原run `epoch=8-step=11970.ckpt` 与评测watcher的 `epoch8_step11970.ckpt` SHA256完全相同，均为 `b51951abbba86e9661fc82f85406d09ae26c6eff6aa3651a40462363866fb326`。

checkpoint没有保存 `hyper_parameters`，`hparams.yaml`为空。因此不是声称从权重张量读取出了奖励系数；上述判断依据是保存的启动记录、Hydra配置、实际配置构造链、源码一致性和checkpoint身份。

## 两种打分的区别

```text
标准评测PDMS = NC × DAC × (5 EP + 5 TTC + 2 Comfort) / 12
本版本训练reward = NC × DAC × (10 EP + 5 TTC + 2 Comfort) / 17
```

通用 `PDMScorerConfig()` 默认EP=5，但GRPO显式创建了EP=10的专用配置，覆盖这个默认值。修改评测YAML中的5并不会自动修改这条原版GRPO训练分支。

此前5000场景报告中的PDMS均值、分项分布使用标准5/5/2评分；native advantage相关统计使用该归档训练分支的10/5/2 reward。两类变量已分别保存，无需重算或修改已有结果。不能由此推广到其他本地LFP、PSI、EPDMS等分支；那些分支仍须按各自实际执行路径核验。

机器可读证据：`outputs/grpo_component_diversity/audits/reward_weight_provenance_20260922.json`。
复现脚本：`tools/analysis/grpo_component_diversity/audit_reward_weight_provenance.py`。
