# NAVTRAIN Missing Token Backfill Report

Date: 2026-05-30

## Summary

The 252 tokens missing from `cache/recogdrive_expert_chunks/full_v1` were
backfilled for A0 official-aligned training.

The original `full_v1` cache was not modified. Instead, an A0-specific overlay
root was created:

```text
cache/recogdrive_expert_chunks/full_v1_a0_complete
```

This overlay contains symlinks to the existing train/val chunks and one new
hidden-only backfill chunk:

```text
cache/recogdrive_expert_chunks/full_v1_a0_complete/train_a0_official_missing_chunk_000000
```

## Why Hidden-only

All 252 missing tokens have the current frame and 4 history CAM_F0 images
available, so A0 `last_hidden_state` can be regenerated.

The 4 future CAM_F0 images needed for JEPA target tokens are missing for all
252 tokens. Therefore these tokens cannot be faithfully backfilled as full
A4/alignment expert samples without obtaining the missing future sensor blobs.

## Outputs

- Missing-token manifest:
  `reports/navtrain_missing_252_manifest.jsonl`
- Image availability report:
  `reports/navtrain_missing_252_image_availability.json`
- Backfill log:
  `reports/navtrain_missing_252_backfill.log`
- Corrected count report:
  `reports/cache_parity_raw_official_count_navtrain_a0_complete.json`

## Backfill Result

The generated chunk contains:

- `num_records`: 252
- `contains_vlm_hidden`: true
- `contains_jepa`: false
- `contains_vggt`: false
- sample keys:
  - `history_trajectory`
  - `high_command_one_hot`
  - `last_hidden_state`
  - `status_feature`
  - `trajectory`
  - `image_path_tensor`
  - `sample_token`
  - `scene_token`

The checked `last_hidden_state` shape is `[2800, 1536]`, dtype `float32`, all
finite.

## Final Token Count

Verification command wrote:

```text
reports/cache_parity_raw_official_count_navtrain_a0_complete.json
```

Corrected final counts:

| split | official tokens | overlay chunk tokens | raw minus chunk | chunk minus raw |
|---|---:|---:|---:|---:|
| train | 85109 | 85109 | 0 | 0 |
| val | 18179 | 18179 | 0 | 0 |
| total | 103288 | 103288 | 0 | 0 |

## Usage

For A0 official-aligned local-loader training, use:

```text
CACHE_PATH=/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1_a0_complete
```

Do not use this overlay as an A4 alignment cache, because the new backfill chunk
does not contain JEPA/VGGT expert target fields.
