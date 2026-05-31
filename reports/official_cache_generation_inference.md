# Official ReCogDrive Cache Generation Inference

Date: 2026-05-30

## Executive Summary

Correction: the earlier raw-count report was wrong because it scanned the raw
logs with `log_names + frame_interval=1` but failed to apply the official
`navtrain.yaml` `tokens:` whitelist. The official `SceneLoader` does apply that
token whitelist.

After fixing the audit script, the current local chunk cache is count-compatible
with the official `navtrain` cache definition at the final-training level:

| Split | Official navtrain tokens | Current chunk tokens | Coverage | Missing from chunk |
| --- | ---: | ---: | ---: | ---: |
| train | 85109 | 84918 | 99.78% | 191 |
| val | 18179 | 18118 | 99.66% | 61 |
| total | 103288 | 103036 | 99.76% | 252 |

With the standard effective batch size 128, this is effectively the same epoch
scale:

| Split | Official nominal batches/epoch | Current chunk nominal batches/epoch |
| --- | ---: | ---: |
| train | 665 | 664 |
| val | 143 | 142 |

Therefore, based on the repository's official cache generation script and
`navtrain` scene filter, the current chunk cache is not missing "most" official
training samples. It is a near-complete subset of the official token whitelist,
missing 252 tokens total.

## Why Disk Size Still Matches

The current chunk cache size is 1.9T:

```text
du -sh cache/recogdrive_expert_chunks/full_v1
1.9T
```

That is consistent with official ReCogDrive hidden-cache scale. A sampled local
chunk file is about 16.55 MiB. Almost all of that is:

```text
last_hidden_state: [2800, 1536] float32 ~= 16.4 MiB
```

The official feature builder also stores `last_hidden_state.squeeze(0).float().cpu()`.
Thus about 103k samples naturally lands around 1.7T plus pickle/filesystem
overhead. The JEPA/VGGT expert fields add only about 0.14 MiB per sample in the
sampled files, so they do not explain the terabyte scale.

## Official Cache Generation Evidence

The official cache launcher uses `TRAIN_TEST_SPLIT=navtrain` and calls
`navsim/planning/script/run_dataset_caching_multi_node.py` with
`agent=recogdrive_agent`, `agent.cache_hidden_state=True`,
`agent.cache_mode=True`, and `cache_path=$CACHE_PATH`.

The cache generation script builds a `SceneLoader` from the full
`cfg.train_test_split.scene_filter`, then rank 0 collects all unique
`scene_loader.tokens`. The important detail is that the scene filter includes
both `log_names` and `tokens`.

`navsim/common/dataloader.py` applies the token filter:

```python
if scene_filter.tokens is not None:
    tokens = set(scene_filter.tokens)
...
if filter_tokens and token not in tokens:
    continue
```

The official `navtrain.yaml` in this repository contains:

- `log_names`: 1192
- `tokens`: 103288

The training cache is then consumed by `CacheOnlyDataset`, split by
`cfg.train_logs` and `cfg.val_logs`.

## Corrected Count Evidence

Corrected report:

```text
reports/cache_parity_raw_official_count_navtrain_fixed.json
```

Observed counts:

- scene_filter token whitelist: 103288
- current chunk train tokens: 84918
- current chunk val tokens: 18118
- official-style raw train tokens after token whitelist: 85109
- official-style raw val tokens after token whitelist: 18179
- `chunk_minus_raw_count = 0` for both train and val
- `raw_minus_chunk_count = 191` for train
- `raw_minus_chunk_count = 61` for val

This means every current train/val chunk token is in the official raw
`navtrain` token set, and only 252 official whitelist tokens are absent from the
current chunk cache.

## Same-token Field Parity

For sampled overlapping tokens, the current chunk cache matches official raw
feature/target generation on the core non-expert training fields:

- `history_trajectory`
- `high_command_one_hot`
- `status_feature`
- `trajectory`
- `image_path_tensor` after normalizing the dataset root prefix

`last_hidden_state` was not regenerated in this CPU-only pass, because that
requires the VLM. The cache does contain `last_hidden_state`; numeric hidden
state parity remains the next small check if we need byte-level confidence.

## Final-training Interpretation

The current chunk cache is very likely equivalent to the official Stage2 cache
in sample identity and epoch scale, except for 252 missing tokens. The previous
651k-token raw estimate should be discarded.

Remaining differences to audit are no longer "cache total amount"; they are:

1. Hidden-state numeric parity for a small fixed token list.
2. Loader/collate/Lightning batch semantics.
3. Optimizer, scheduler, precision, validation, and checkpoint selection.
4. Whether the 252 missing official whitelist tokens are due to failed cache
   writes, missing sensor/log data, or intentional filtering.

## Recommended Minimal Next Checks

1. Run hidden-state parity on 8-16 fixed overlapping tokens with the official VLM.
2. List the 252 missing official tokens and inspect failure causes.
3. Continue aligning the training loader and Lightning trainer behavior; cache
   size and token coverage are no longer the primary blocker.
