"""Freeze explicit evaluation tapes; retain old S0-S4, record local new noise."""
from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import os
import time
import uuid


def main(args):
    import numpy as np
    import torch
    from .noise import make_noise_tape, scene_seed
    from .checkpointing import atomic_json, sha256
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    marker = out/'EVALUATION_NOISE_MANIFEST.json'
    if marker.exists():
        value=json.loads(marker.read_text())
        for entry in value['populations'].values():
            assert sha256(entry['path']) == entry['sha256']
            assert sha256(entry['tokens_path']) == entry['tokens_sha256']
        print('EXISTING_EXPLICIT_NOISE_VERIFIED',flush=True)
        return
    protocol_path=Path(args.reference_protocol).resolve()
    protocol=json.loads(protocol_path.read_text())
    entries={x['cohort']:x for x in protocol['files']}
    reference=protocol_path.parent.parent
    old_train=reference/entries['navtrain']['file']
    old_test=reference/entries['navtest']['file']
    assert sha256(old_train)==entries['navtrain']['sha256']
    assert sha256(old_test)==entries['navtest']['sha256']
    train_tokens=Path(args.train_index).resolve()
    test_tokens=Path(args.test_index).resolve()
    rows=json.loads(train_tokens.read_text())
    lookup={x['token']:i for i,x in enumerate(rows)}
    with gzip.open(Path(args.frozen_repo)/'manifests/validation.csv.gz','rt') as stream:
        val=list(csv.DictReader(stream))
    assert len(val)==17475 and len(set(x['token'] for x in val))==17475
    bank=np.load(old_train,mmap_mode='r',allow_pickle=False)
    assert bank.shape==(103288,5,6,1,8,3) and bank.dtype==np.float32
    path=out/'validation_S0_S9.npy'
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    final=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.float32,shape=(17475,10,6,1,8,3))
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    # Demonstrate that the isolated generator preserves the native six-call
    # sampling rule; this is not a claim of PPU equivalence for the new indices.
    t=val[0]['token'];seed=scene_seed(t,5)
    with torch.random.fork_rng(devices=[0]):
        torch.manual_seed(seed)
        native=torch.stack([torch.randn((1,8,3),device='cuda',dtype=torch.float32) for _ in range(6)])
    assert torch.equal(native,make_noise_tape(seed,'cuda'))
    began=time.monotonic()
    for i,row in enumerate(val):
        token=row['token'];source=rows[lookup[token]]
        assert source['subset']=='validation' and source['session']==row['session']
        final[i,:5]=bank[lookup[token]]
        for j in range(5,10):
            final[i,j]=make_noise_tape(scene_seed(token,j),'cuda').cpu().numpy()
        if (i+1)%128==0:
            atomic_json(out/'LIVE.json',{'status':'GENERATING_ONLY_NEW_VALIDATION_S5_S9','completed':i+1,'expected':17475,'seconds':time.monotonic()-began})
    final.flush();del final
    with temporary.open('rb') as stream:os.fsync(stream.fileno())
    temporary.replace(path)
    index=out/'validation_token_index.json'
    atomic_json(index,[{'token':x['token'],'global_index':rows[lookup[x['token']]]['global_index'],'session':x['session']} for x in val])
    check=np.load(path,mmap_mode='r',allow_pickle=False)
    for i,row in enumerate(val):
        assert check[i,:5].tobytes()==bank[lookup[row['token']]].tobytes()
    assert np.isfinite(check).all()
    def entry(p,t,e,source):
        return {'path':str(p),'sha256':sha256(p),'dtype':'float32','shape':e['shape'],
                'tokens_path':str(t),'tokens_sha256':sha256(t),'noise_indices':list(range(e['shape'][1])),'source':source}
    value={'schema':'native_explicit_evaluation_noise_v1','populations':{
        'train':entry(old_train,train_tokens,entries['navtrain'],'Original verified PPU explicit reference S0-S4; full103288 token index covers the85813Train subset'),
        'test':entry(old_test,test_tokens,entries['navtest'],'Original verified PPU explicit reference S0-S9'),
        'validation':entry(path,index,{'shape':[17475,10,6,1,8,3]},'S0-S4 copied bytewise from original reference; S5-S9 generated once using local native CUDA six-call scene_seed protocol')},
        'reference_protocol_sha256':sha256(protocol_path),'local_extra_noise_numerics':{'Torch':str(torch.__version__),'torch_git':torch.version.git_version,'CUDA':torch.version.cuda,'device':torch.cuda.get_device_name(0)},
        'per_noise_origin':{'validation':{str(j):('original_PPU_reference' if j<5 else 'new_local_A800_reference') for j in range(10)}},
        'no_seed_fallback_when_consuming':True,'same_bank_for_all_models':True,'PPU_bitwise_claim_for_new_noise':False}
    atomic_json(out/'VALIDATION_NOISE_PROTOCOL.json',{'files':[{'cohort':'validation','file':str(path),'sha256':sha256(path),'shape':[17475,10,6,1,8,3],'dtype':'float32','bytes':path.stat().st_size}]})
    atomic_json(marker,value)
    atomic_json(out/'COMPLETE.json',{'status':'COMPLETE_EXPLICIT_EVALUATION_NOISE','manifest_sha256':sha256(marker),'validation_scenes':17475,'all_first5_original_bytes_verified':True,'new_noise_six_call_native_parity':True,'seconds':time.monotonic()-began})
    print(json.dumps({'status':'COMPLETE','manifest':str(marker),'sha256':sha256(marker)},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--frozen-repo',required=True)
    p.add_argument('--reference-protocol',required=True)
    p.add_argument('--train-index',required=True)
    p.add_argument('--test-index',required=True)
    p.add_argument('--out',required=True)
    main(p.parse_args())
