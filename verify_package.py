"""Verify transferred hashes, actual native request fixtures and all decisions."""
from pathlib import Path
import argparse,csv,gzip,hashlib,json
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_k]='1'
import numpy as np
ROOT=Path(__file__).resolve().parent


def verify(integrity=True,decisions=True,golden=True):
    from interface_teacher import Library,DecisionTable
    import torch
    torch.set_num_threads(1)
    checks={}
    def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if integrity:
        manifest=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text())
        for r in manifest['files']:
            p=ROOT/r['path']
            if sha(p)!=r['sha256']:raise AssertionError(('File SHA mismatch',r['path']))
        checks['files_SHA256']=len(manifest['files'])
    if golden:
        references=json.loads((ROOT/'reference/GOLDEN.json').read_text())['real_native_records'];n=0;experts=0
        for r in references:
            lib=Library(r['library'])
            with np.load(ROOT/r['arrays']['path'],allow_pickle=False) as z:
                cpu=torch.get_rng_state().clone()
                pack=lib.requests(z['live_Base_pooled_float64'],z['live_token_counts_int64'])
                assert torch.equal(cpu,torch.get_rng_state()),'Expert evaluation changed global RNG'
                for key in ('requested_shifts_float64','requested_shifts_float32'):
                    assert pack[key].dtype==z[key].dtype and pack[key].shape==z[key].shape
                    assert pack[key].tobytes()==z[key].tobytes(),(r['library'],r['token'],key)
                assert pack['cap_RMS_float64'].tobytes()==z['cap_RMS'].tobytes()
                n+=1;experts+=lib.config['K']
        # Runtime cap: zero must be literal identity and padding must not change.
        from interface_teacher.native_cap import apply
        gen=torch.Generator().manual_seed(44)
        base=torch.randn((1,13,384),generator=gen);image=torch.tensor([[1]*7+[0]*6],dtype=torch.bool)
        text=torch.tensor([[0]*7+[1]*4+[0]*2],dtype=torch.bool)
        req=torch.randn((3,2,384),generator=gen);req[0]=0;cap=torch.tensor([0.,.1,.25],dtype=torch.float64)
        cpu=torch.get_rng_state().clone();out=apply(base,image,text,req,cap)
        assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(out['interfaces'][0],base[0])
        assert torch.equal(out['interfaces'][:,11:],base[:,11:].expand(3,-1,-1))
        assert bool((out['actual_RMS']<=cap+out['tolerance']).all())
        checks.update(real_native_records=n,expert_requests_bitwise_exact=experts,zero_padding_cap=True)
    if decisions:
        exp=json.loads((ROOT/'reference/EXPECTED.json').read_text());seen=0;libs=0
        for population in ('train','test'):
            for name,expected in exp[population].items():
                table=DecisionTable(name,population,allow_test=True);total=0;bs=ts=0.;counts=np.zeros(expected['K']+1,np.int64)
                with gzip.open(ROOT/f'manifests/{population}.csv.gz','rt') as f:scenes=list(csv.DictReader(f))
                for chunk in table.manifest['chunks']:
                    assert sha(ROOT/chunk['path'])==chunk['sha256']
                    with np.load(ROOT/chunk['path'],allow_pickle=False) as z:
                        choices=z['choice'];means=z['mean_PDMS_score'];indices=z['indices']
                        assert np.array_equal(choices,means.argmax(1)),'Not a Base-first fixed-scene argmax'
                        for q,(idx,ch,scores) in enumerate(zip(indices,choices,means)):
                            row=scenes[total+q];assert int(row['index'])==idx
                            r=table.row(row['token']);assert int(r['choice'])==ch
                            base=[float(r[f'Base_S{j}']) for j in range(5)];teacher=[float(r[f'teacher_S{j}']) for j in range(5)]
                            assert abs(np.mean(base)-scores[0]*100)<1e-11
                            assert abs(np.mean(teacher)-scores[ch]*100)<1e-11
                        bs+=means[:,0].sum();ts+=means[np.arange(len(choices)),choices].sum()
                        counts+=np.bincount(choices,minlength=len(counts));total+=len(choices)
                assert total==expected['scenes']==len(table.rows)
                assert abs(bs/total*100-expected['Base_PDMS'])<1e-9
                assert abs(ts/total*100-expected['PDMS'])<1e-9
                assert counts.tolist()==expected['choices_count']
                seen+=total;libs+=1
        checks.update(complete_population_library_pairs=libs,all_scene_library_decisions=seen,full_population_scores_reproduced=True)
    result=dict(status='PASS',checks=checks)
    print(json.dumps(result,ensure_ascii=False,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--no-integrity',action='store_true');p.add_argument('--no-decisions',action='store_true');a=p.parse_args()
    verify(integrity=not a.no_integrity,decisions=not a.no_decisions)
