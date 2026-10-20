"""Incremental, sharded raw-observation cache; shared VLM never enters optimization."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import shutil
import time
from scripts.process_improvement.common import *


def main():
    import torch
    from scripts.opd.inputs import agent, features, encode, metric
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train_full')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--with-metrics', action='store_true')
    args = parser.parse_args()
    configure()
    cfg = config()
    data = membership('train_full')+membership('cal') if args.split == 'all' else membership(args.split)
    if args.limit:
        data = data[:args.limit]
    data = data[args.rank::args.shards]
    priority = {r['token'] for name in ['diagnostic_train','diagnostic_cal'] for r in membership(name)}
    data.sort(key=lambda r: (r['token'] not in priority, r['log_name'], r['start_idx']))
    # A conservative upper bound is checked before allocation, never an up-front
    # requirement to reserve TBs. Cache absence remains a repairable state.
    new_rows = [r for r in membership('train_full')+membership('cal') if not str(r['observation_path']).startswith(str(HISTORY))]
    projected_bytes = len(new_rows)*(2800*1536*2+65536)
    if projected_bytes > cfg['cache_max_bytes']:
        raise OSError('Population exceeds locked cache cap; use a measured block eviction plan before extraction')
    model = None
    started = time.time()
    built, reused = 0, 0
    receipts = ROOT/'cache_receipts'/f'{args.split}_{args.rank}_of_{args.shards}.jsonl'
    receipts.parent.mkdir(parents=True, exist_ok=True)
    with receipts.open('a') as receipt:
        for index, row in enumerate(data):
            path = Path(row['observation_path'])
            if path.exists():
                observation(row)
                reused += 1
            else:
                if shutil.disk_usage(ROOT).free < cfg['shared_free_floor_bytes']:
                    raise OSError('Shared disk free-space floor reached; preserve all prior artifacts')
                if model is None:
                    model = agent(official=True)
                f = features(model, row)
                tokens = encode(model, f)
                assert tokens.shape == (1, 2800, 1536)
                save(path, dict(token=row['token'], tokens=tokens, features={k:v.cpu() for k,v in f.items()},
                               cache_format='opd_native_vlm_v1', vlm_sha256=cfg['vlm_sha256'],
                               protocol_sha256=CONTRACT, observation_contract='recogdrive_official_fixed2800_v2',
                               input_contract='Current front image + navigation + four historical ego states; padded raw VLM output',
                               source_code=identity()['code_sha']))
                receipt.write(json.dumps(dict(token=row['token'], path=str(path), bytes=path.stat().st_size,
                                              sha256=sha(path), key=digest([row['token'],cfg['vlm_sha256'],CONTRACT,'bfloat16'])))+'\n')
                receipt.flush()
                built += 1
            if args.with_metrics:
                metric(row)
            if (index+1)%64 == 0:
                print('RAW_CACHE', args.split, args.rank, index+1, len(data), built, reused, time.time()-started, flush=True)
    write(ROOT/'cache_jobs'/f'{args.split}_{args.rank}_of_{args.shards}.json', dict(status='COMPLETE', built=built,
          reused=reused, scenes=len(data), seconds=time.time()-started, owner=identity(), contract=CONTRACT))


if __name__ == '__main__':
    main()
