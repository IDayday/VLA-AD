# VLA-AD Old Cache Size Inventory

- Generated: 2026-06-13T17:17:56.512347Z
- Disk: `100.97.192.3@tcp:100.97.192.4@tcp:/hbTcp001/78ebbad2386e922193d44ac249bbb12e_kl4gv5vivlcf06o2  310T  309T  1.3T 100% /mnt/project`
- Scope: `/mnt/project/VLA-AD/cache`, cache-like directories under `/mnt/project/VLA-AD/outputs`, and cache-like directories under `/mnt/project/VLA-AD/experiments`.
- I/O policy: low-priority `ionice/nice`, no current training process was killed. Current two-expert active cache is marked as keep, not a deletion candidate.
- Post-inventory cleanup: the old LoRA/cache smoke paths
  `/mnt/project/VLA-AD/outputs/last_vla_v2/tmp_b_lora_cache_batch2_smoke` and
  `/mnt/project/VLA-AD/outputs/last_vla_v2/decoupled_highcap_no_risk_navtest_cot_lora_eval_20260607T142428Z/logs/B_epoch003_lora_navtest_full_cache`
  were deleted after this inventory.

## Main Findings

| Cache | Size | Status / note |
|---|---:|---|
| `/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/train_full_highcap_chunks` | 1.79 TiB | exact by summing 27 child chunks |
| `/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/navtest_full_highcap_chunks` | 215.45 GiB | exact |
| `/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1` | 196.36 GiB | exact |
| `/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/navtest_geometry192_overlay` | 14.88 GiB | exact |
| `/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/navtest_geometry192_overlay_raw` | 14.88 GiB | exact |
| `/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/navtest_jepa128_overlay` | 5.98 GiB | exact |
| `/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/navtest_jepa128_overlay_raw` | 5.98 GiB | exact |
| `/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1_fast_pickle` | 8.43 GiB | exact by summing 257 children |
| `/mnt/project/VLA-AD/cache/metric_cache_navtest_full_v1` | 3.07 GiB | exact by summing 137 scene dirs |
| `/mnt/project/VLA-AD/cache/metric_cache_train_full` | 29.06 GiB | estimated from 547/1193 dirs; scanned 13.32 GiB |
| `/mnt/project/VLA-AD/outputs/hidden_cache_equiv_navtest2` | 4.51 GiB | exact |
| `/mnt/project/VLA-AD/experiments/recogdrive_expert/smoke_tests/full_v1_cache_smoke_20260527_124745` | 2.47 GiB | exact |
| `/mnt/project/VLA-AD/outputs/last_vla_v2/tmp_b_lora_cache_batch2_smoke` | 36.45 MiB | exact; deleted after inventory |

## Largest Old Cache Candidates

| Priority | Path | Size | Recommendation |
|---|---|---:|---|
| A | `/mnt/project/VLA-AD/outputs/last_vla_v2/tmp_b_lora_cache_batch2_smoke` | 36.45 MiB | Deleted after inventory. |
| A | `/mnt/project/VLA-AD/outputs/hidden_cache_equiv_navtest2` | 4.51 GiB | Can delete if no one needs old smoke outputs; small savings. |
| A | `/mnt/project/VLA-AD/experiments/recogdrive_expert/smoke_tests/full_v1_cache_smoke_20260527_124745` | 2.47 GiB | Can delete if no one needs old smoke outputs; small savings. |
| B | `/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/train_full_highcap_chunks` | 1.79 TiB | largest old cache, but current hidden-cache builder is reading it now; do not delete until current cache + audit complete. |
| B | `/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/navtest_full_highcap_chunks` | 215.45 GiB | needed if later rebuilding navtest hidden cache from old highcap base. |
| B | `/mnt/project/VLA-AD/cache/recogdrive_expert_chunks/full_v1` | 196.36 GiB | old ReCogDrive expert feature chunks, not current two-expert route; verify no other route uses it. |
| B | `/mnt/project/VLA-AD/cache/last_vla_v2/decoupled_highcap_no_risk/navtest_geometry192_overlay* + navtest_jepa128_overlay*` | 41.71 GiB | old LastVLA overlay teacher caches. |

## Keep / Do Not Delete Now

- `/mnt/project/VLA-AD/cache/two_expert_slot/real_20260612_172446/hidden_navtrain_stage1_lora_bf16`: current two-expert hidden cache generation target, active.
- `/mnt/project/VLA-AD/cache/two_expert_slot/real_20260612_172446/jepa_dynamic`: current strict JEPA teacher cache, 103,036 records; keep.
- `/mnt/project/VLA-AD/cache/two_expert_slot/real_20260612_172446/vggt_feature23`: current strict VGGT Feature(23) teacher cache, 103,036 records; keep.
- Stage3/AWAC/IQL caches were not evaluated for deletion because that is another active project.

## Notes on Timeouts / Estimates

- Parent `du` on `train_full_highcap_chunks` timed out, but child chunk summation completed: 27 chunks, total `1,967,628,066,816` bytes = 1.79 TiB.
- `metric_cache_train_full` exact root scan timed out. Previous low-priority scan covered 547/1193 scene dirs: 13.32 GiB; estimated full size is 29.06 GiB. This is not a major space source compared with highcap chunks.
- `vlm_text_anchor_2bbase_train_*` timed out because of many small files. Based on navtest anchor size and sample count it is expected to be sub-GB to low-GB, not a major deletion target.
- Current two-expert teacher cache `du` also timed out due many small sample files; it is intentionally excluded from old-cache cleanup candidates.

## Raw Inventory Files

- `/mnt/project/VLA-AD_last_vla_dev/reports/two_expert_slot/cache_size_inventory_train_full_highcap_chunks_20260613T170445Z.tsv`
- `/mnt/project/VLA-AD_last_vla_dev/reports/two_expert_slot/cache_size_inventory_targeted_20260613T164346Z.tsv`
- `/mnt/project/VLA-AD_last_vla_dev/reports/two_expert_slot/cache_size_inventory_remaining_20260613T170047Z.tsv`
- `/mnt/project/VLA-AD_last_vla_dev/reports/two_expert_slot/cache_size_inventory_fast_20260613T163836Z.tsv`
- `/mnt/project/VLA-AD_last_vla_dev/reports/two_expert_slot/cache_subdir_inventory_20260613T161241Z.txt`

## Practical Recommendation

1. Do not delete current two-expert caches or teacher caches.
2. Do not delete `train_full_highcap_chunks` until the active hidden-cache builder and audit finish, because the running process is reading it.
3. If disk pressure becomes urgent after hidden-cache generation completes, the biggest reclaim target is `train_full_highcap_chunks` at 1.79 TiB, followed by old navtest highcap chunks at 215.45 GiB and old expert chunks at 196.36 GiB.
4. Small LoRA/cache smoke directories can be deleted, but they save only a few GiB or less.
