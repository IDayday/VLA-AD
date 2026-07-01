# PSI-Drive Random-Init Full103k GT-Supp Preflight Supplement

生成时间：2026-06-25T18:59:20Z
最近更新：2026-06-26T01:14:15Z

本文档是当前主实验的 preflight supplement，用于区分早先 `preflight_20260624T213741Z.md` 中的 IL-init Clean 默认设置。当前主实验按用户要求采用 Stage2 随机初始化、完整 navtrain cache `103288`、GT-only 缺候选场景补充后的 APSD Clean support index。

## 当前主实验结论

- Stage2 随机初始化训练已经启动并持续运行。
- 当前训练使用官方 ReCogDrive Stage1 hidden cache，不启用 JEPA/VGGT/LastRD/LastVLA/two-expert/external feature injection。
- 当前训练集为完整 cache `103288` 条；验证/log-val 仍为 GT 目标，仅作训练期日志，不作为最终模型选择依据。
- 固定 val6000 token file 和 navtest overlap audit 已存在；最终 Stage2/Stage3 选择和报告必须使用 fixed val6000 exact PDM 与 full navtest exact PDM。
- 当前 checkpoint store 正式对象审计通过；存在 4 个隐藏 `.tmp` 硬链接残留，已由新版 audit 作为 warning 暴露，未删除任何 checkpoint/source/object 文件。

## Git / 分支

- branch: `feature/psi-drive-stage2-stage3`
- HEAD: `d2a09e03e42e069d617f5b8dd74a1e39fb4a0d04`
- worktree: 含 PSI-Drive 代码、脚本、报告和测试改动；未执行 destructive git 操作。

## Stage2 训练设置

- run root: `/mnt/project/VLA-AD/outputs/psi_drive_stage2_apsd_random_init_full103k_gt_supp_20260625T162149Z`
- launch script: `scripts/psi_drive/run_stage2_apsd_8gpu.sh`
- resolved command: `/mnt/project/VLA-AD/outputs/psi_drive_stage2_apsd_random_init_full103k_gt_supp_20260625T162149Z/resolved_command.sh`
- init mode: `agent.checkpoint_path=` + `agent.allow_random_init=true`
- epochs: `200`
- devices: `8`
- batch size per device: `16`
- LR: `1e-4`
- optimizer weight decay: `1e-4`
- requested precision: `16-mixed`
- model parameter dtype: fp32
- checkpoint layout: `psi`
- raw checkpoint cadence: every epoch

关键禁用项：

```text
agent.use_jepa=false
agent.use_vggt=false
agent.use_last_rd=false
agent.use_last_vla=false
agent.use_two_expert_slots=false
agent.use_expert_features=false
agent.expert_feature_source=none
```

## 数据和 Cache

- official Stage1 hidden cache: `/mnt/project/VLA-AD/cache/recogdrive_official_stage1_hidden_navtrain_2b`
- loader mode: `official-cache-loader-all-cache-train-log-val`
- `cache_train_all_records=true`
- train records: `103288`
- train unique sample tokens: `103288`
- validation/log-val records: `18179`
- validation/log-val target source: `gt`
- train/val overlap in this full-cache run: `18179`

解释：当前主实验按用户要求使用完整 `103k` cache 训练，因此 train 与 log-val 有重叠。固定 val6000 文件仍保留独立 audit：train-complement 与 val6000 overlap 为 `0`，navtest overlap 为 `0`。最终模型选择必须以 fixed val6000 exact PDM 为准，不以训练期 log-val loss 为准。

## Support Index

- support index: `/mnt/project/VLA-AD/outputs/psi_drive/support_index/stage2_pareto_support_clean_full_gt_supplemented_20260625T162351Z.pt`
- SHA256: `817ada27e18a92062e2fb4bd5fddacaa3f75cbd30a0f8d903119bc04ab4bbf27`
- source mode: `clean`
- unique tokens: `103288`
- support count histogram: `1=58287`, `2=34054`, `3=10947`
- fallback: `gt_fallback=69`
- weights sum min/max: `1 / 1`
- finite trajectories/scores/weights: `true / true / true`
- quality band: `0.02`
- min descriptor distance: `0.75`

source histogram:

```text
gt=40407
progress_endpoint=32694
policy=32015
il=26639
progress_speed=10268
endpoint_lateral=7404
progress_gamma=5109
timing_delay=1909
timing_slow_first=1643
lateral_offset=1148
```

显式 overlap audit：

- passed: `true`
- train-complement overlap: `97288`
- fixed val6000 overlap: `6000`
- navtest overlap: `0`

解释：support index 覆盖完整 `103288` 训练 cache，因此包含 fixed val6000 tokens；这是当前 full-cache 训练设置的预期结果。navtest overlap 为 `0`，仍满足训练目标不得使用 navtest 的要求。

## Fixed Val6000

- token file: `artifacts/splits/navtrain_val6000_seed260306049.txt`
- token count: `6000`
- SHA256: `1b6355bfd1f1fbf9438897d34c64320d3f46d07c62437fe8da7df5e42c5cbc54`
- train complement file: `artifacts/splits/navtrain_val6000_seed260306049_train_complement.txt`
- train complement count: `97288`
- train complement SHA256: `68904df7e36871a6ba5bd809113a3d4fe3ac0e42a7794dd866e219e364eac8c2`
- split audit: `artifacts/splits/navtrain_val6000_seed260306049_audit.json`
- split audit train/val overlap: `0`
- split audit navtest overlap: `0`

## Checkpoint Store Snapshot

快照来自 run summary `2026-06-26T01:14:15Z`。

- latest raw checkpoint: `epoch_082.ckpt`
- latest raw mtime: `2026-06-26T01:10:02Z`
- latest archived checkpoint id: `epoch_082`
- latest archived SHA256: `a2883f5bdae7d43faa8d81ee575e31942737fafd12b1c31e1cf50c3b4f571624`
- raw checkpoint count: `84`
- inventory rows: `84`
- unique SHA256: `84`
- object files: `84`
- checked object rows: `84`
- audit passed: `true`
- audit failures: `0`
- audit warnings: `4`
- warning type: hidden object-store `.tmp` files present
- top5 entries: `0`

临时文件 warning 不作为对象校验失败；它们不是 Top-5 或正式 immutable object。`2026-06-26T01:14:06Z` 审计后，84 个 inventory 行和 84 个 object 文件均校验通过；4 个隐藏 `.tmp` 条目仍作为 warning 显式记录，未删除任何 checkpoint/source/object 文件。归档工具已补充同 inode 并发 race 后只清理自身临时路径的回归修复，`epoch_082` 归档后 warning 数保持为 `4`，未继续增长。

## Evaluation / Stage3 Gate

- val6000 eval status rows: `1 started`
- navtest eval status rows: `0`
- val6000 Top-5 manifest: not yet created
- navtest Top-3 backup manifest: not yet created
- remote val6000 watcher: `training-rl-zt3` PID `3015882`
- remote navtest Top-5 watcher: `training-rl-zt3` PID `3016634`
- Stage3 orchestrator PID: `1548272`
- Stage3 launch rule: wait for Stage2 val6000 eval `done=200/200`, select val6000 Top-1 object checkpoint, then launch SR-PGRPO.
- navtest is evaluation-only and must not select Stage3 initialization.

当前 val6000 exact PDM 已改由 `training-rl-zt3` 远端资源执行，`epoch_001` 于 `2026-06-26T01:49:19Z` 标记为 `started`。navtest watcher 不再等待全部 200 个 val6000 完成，而是等待 `rankings/val6000/current_top5.tsv`，只评估 val6000 Top-5 checkpoint；完成后会备份 navtest Top-3。

## Verification Added In This Supplement

本次补强的 checkpoint audit 变更：

```text
python -m py_compile scripts/checkpoints/audit_checkpoint_store.py
pytest -q tests/test_checkpoint_store_audit.py
```

结果：`3 passed`

新增行为：

- audit 继续对正式 `.ckpt` object、inventory rows 和 Top-5 rows 做强校验；
- object hash/size/symlink/missing object 仍作为 failure；
- object store 中隐藏 `.tmp` 文件作为 warning 输出到 JSON/MD；
- warning 不会把已经可校验的正式 checkpoint store 判为失败。

## 尚未完成

- Stage2 random-init full103k 训练尚未跑满 `200` epochs。
- Stage2 fixed val6000 exact PDM eval 已在远端启动，但尚未产生 `done` 结果。
- Stage2 full navtest exact PDM eval 尚未产生结果，等待 val6000 Top-5。
- Stage2 val6000 Top-5 backup 和 navtest Top-3 backup 尚未产生，等待 ranking 结果。
- Stage3 SR-PGRPO 尚未启动，仍等待 val6000-selected Stage2 checkpoint。
- Stage3 val6000/navtest eval、Top-5、趋势和诊断报告尚未产生。

因此当前不能声明 Stage2 或 Stage3 benchmark 结果，也不能声明 SOTA。
