# Trajectory Risk Critic Training Report

Labels: `experiments/risk_vla/v2/R2_formal_riskvla_strategy_full7095_20260606T032838Z/candidate_assets_train2000/strategy_utility_labels.jsonl`
Candidates: `experiments/risk_vla/v2/R2_formal_riskvla_strategy_full7095_20260606T032838Z/candidate_assets_train2000/candidate_trajectories.npz`
Train/val tokens: `1600` / `400`
Best epoch: `19`
Best val loss: `0.728272`
Pairwise accuracy: `0.625000`
Selection accuracy vs constrained anchor: `0.582500`

| Selection | PDMS | Zero | DAC0 | NC0 | TTC0 | Progress | Comfort |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 0.918833 | 11 | 9 | 2 | 6 | 0.853700 | 1.000000 |
| selected | 0.918460 | 11 | 9 | 2 | 7 | 0.855305 | 1.000000 |
| oracle | 0.924798 | 9 | 8 | 1 | 5 | 0.861014 | 1.000000 |

Risk AUROC/AUPRC:
- `comfort_auprc`: `None`
- `comfort_auroc`: `None`
- `interaction_nc_auprc`: `0.005130964693487017`
- `interaction_nc_auroc`: `0.658893346706787`
- `low_score_auprc`: `0.05559524816998759`
- `low_score_auroc`: `0.7230995656924091`
- `path_dac_auprc`: `0.05945912327443204`
- `path_dac_auroc`: `0.7648081841432225`
- `progress_auprc`: `0.0756944137442075`
- `progress_auroc`: `0.7541703409758965`
- `ttc_auprc`: `0.020103419215056043`
- `ttc_auroc`: `0.6234787018255578`

Leakage guard: training mode rejects split names containing `test` or `navtest`.
