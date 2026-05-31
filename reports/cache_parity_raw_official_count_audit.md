# Cache Parity Raw Official Count Audit

## Summary

Generated with `scripts/verify_chunk_cache_against_official_raw_cache.py` using
`CUDA_VISIBLE_DEVICES=''` and `--num-workers 32`.

Important correction: the first raw-count pass ignored the official
`navtrain.yaml` `tokens:` whitelist and over-counted raw sliding-window samples.
The fixed report applies the same token whitelist that official `SceneLoader`
uses.

Artifacts:

- Corrected full count report: `reports/cache_parity_raw_official_count_navtrain_fixed.json`
- Superseded over-count report: `reports/cache_parity_raw_official_count_navtrain.json`
- Same-token smoke report: `reports/cache_parity_raw_official_smoke.json`

## Same-Token Field Parity

Status: aligned for the sampled token(s).

Compared fields:

- `history_trajectory`
- `high_command_one_hot`
- `status_feature`
- `trajectory`
- `image_path_tensor` by normalized sensor-path suffix

`last_hidden_state` was not regenerated in this CPU-only pass. That requires
explicit `--compare-hidden-state` and a VLM checkpoint/device.

## Count Parity

The corrected count pass uses the official `navtrain` scene filter intersection:

- scene_filter token whitelist: 103288
- train log files after official train/val split intersection: 978
- val log files after official train/val split intersection: 214

| split | raw official tokens | chunk tokens | overlap | raw minus chunk | chunk minus raw |
|---|---:|---:|---:|---:|---:|
| train | 85109 | 84918 | 84918 | 191 | 0 |
| val | 18179 | 18118 | 18118 | 61 | 0 |

Interpretation: the local chunk cache is a near-complete subset of the official
`navtrain` token whitelist. It is count-compatible for final training, with 252
missing official tokens total.

## Operational Notes

- Multiprocess raw split counting is available via `--num-workers`.
- The script defaults `--device cpu`, so count/smoke checks do not allocate GPU
  memory unless hidden-state regeneration is explicitly requested.
- Use `CUDA_VISIBLE_DEVICES=''` for CPU-only audit runs.
