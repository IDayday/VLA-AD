# ReCogDrive Stage1 NAVSIM Data Discovery

- status: official NAVSIM JSONL data found
- source dataset: `owl10/ReCogDrive_Pretraining`
- NAVSIM trajectory replay file: `Navsim_Traj/dataset_navsim_traj.jsonl`
- NAVSIM QA file: `Navsim_ReCogDrive/dataset_navsim_recogdrive.jsonl`

For this Stage1-v2 run, trajectory replay CE uses only official `NAVSIM-Traj`.
`NAVSIM-ReCogDrive` is NAVSIM driving QA, not guaranteed to parse as `[8,3]`
trajectory labels, so it is not mixed into the trajectory replay cache.

The run cache was built from:

- JSONL: `/mnt/project/VLA-AD/experiments/two_expert_stage1_v2_20260615_174154/data/recogdrive_pretraining_jsonl/Navsim_Traj/dataset_navsim_traj.jsonl`
- replay cache: `/mnt/project/VLA-AD/experiments/two_expert_stage1_v2_20260615_174154/data/stage1_replay_cache`
- matched records: `84918`
- unmatched official JSONL rows: `191`
- parse failures: `0`

This is not NAVSIM GT fallback. The replay labels come from ReCogDrive official
NAVSIM-Traj JSONL; JEPA teacher diagnostics were used only to map image paths to
the local `sample_token` namespace after the old base chunk root had been removed.
