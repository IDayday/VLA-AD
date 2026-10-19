"""Complete-population paired statistics; absent experiments never get scores."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse,csv
from scripts.opd.common import *

EXPERIMENTS=['E1','E2','E3','E4-A','E4-B','E4-C','E5']
FIELDS=['NC','DAC','EP','TTC','Comfort','DDC','PDMS']


def bank(split,name,model=None):
    selected=rows(split);path=ROOT/'evaluation'/split/name
    complete=list(path.glob('complete_*.json'))
    if not complete:raise FileNotFoundError(f'No completion marker: {path}')
    markers=[read(p) for p in complete];shards=markers[0]['shards']
    if len(markers)!=shards or {r['rank'] for r in markers}!=set(range(shards)):raise FileNotFoundError(f'Incomplete shards: {path}')
    if sum(r['rows'] for r in markers)!=len(selected):raise ValueError(f'Wrong evaluated population: {path}')
    values=[]
    for row in selected:
        p=path/'scores'/f"{row['token']}.json";v=read(p)
        key=model or (next(iter(v['scores'])) if len(v['scores'])==1 else name)
        y=np.array(v['scores'][key]['scores'],dtype=float)
        if y.shape!=(5,7) or not np.isfinite(y).all() or (y< -1e-6).any() or (y>1+1e-6).any():raise ValueError(f'Invalid PDMS {p}')
        values.append(y)
    return np.stack(values)


def paired_quantities(new,base,stable,w=None):
    # new[training seed,scene,inference seed,field], base[scene,inference seed,field]
    n=new.shape[1];w=np.ones(n) if w is None else w;den=w.sum()
    j=new[...,-1].mean(-1);b=base[...,-1].mean(-1);d=j-b[None]
    mean=lambda v:float(np.sum(v*w[None])/den/len(new))
    st=w*stable;stden=st.sum()
    return dict(PDMS=mean(j),G=mean(d),M_plus=mean(np.maximum(d,0)),M_minus=mean(np.maximum(-d,0)),
        material_drop_fraction=mean(d<-.02),zero_scene_fraction=mean(j<=1e-12),
        zero_output_fraction=mean((new[...,-1]<=1e-12).mean(-1)),
        positive_to_zero_fraction=mean((b[None]>1e-12)&(j<=1e-12)),
        stable_high_to_zero_fraction=mean(stable[None]&(j<=1e-12)),
        stable_high_retention=float(np.sum((j>=.8)*st[None])/stden/len(new)) if stden else None,
        zero_to_positive_fraction=mean((b[None]<=1e-12)&(j>1e-12)))


def bootstrap(new,base,logs,stable,reps=2000):
    rng=np.random.default_rng(20261020);point=paired_quantities(new,base,stable);samples={k:[] for k,v in point.items() if v is not None}
    unique,inverse=np.unique(logs,return_inverse=True);ns,n,nz,_=new.shape
    for _ in range(reps):
        si=rng.integers(0,ns,ns);zi=rng.integers(0,nz,nz)
        w=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))[inverse]
        v=paired_quantities(new[si][:,:,zi],base[:,zi],stable,w)
        for k in samples:
            if v[k] is not None:samples[k].append(v[k])
    return {k:dict(estimate=v,lower95=float(np.quantile(samples[k],.025)),upper95=float(np.quantile(samples[k],.975)),replicates=len(samples[k])) if v is not None else None for k,v in point.items()}


def compare(a,b,base,logs,stable,reps=2000):
    rng=np.random.default_rng(20261020);p=paired_quantities(a,base,stable);q=paired_quantities(b,base,stable)
    keys=['PDMS','G','M_minus','stable_high_to_zero_fraction'];draws={k:[] for k in keys};unique,inverse=np.unique(logs,return_inverse=True)
    for _ in range(reps):
        si=rng.integers(0,len(a),len(a));zi=rng.integers(0,a.shape[2],a.shape[2]);w=np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))[inverse]
        aa=paired_quantities(a[si][:,:,zi],base[:,zi],stable,w);bb=paired_quantities(b[si][:,:,zi],base[:,zi],stable,w)
        for k in keys:draws[k].append(aa[k]-bb[k])
    return {k:dict(estimate=p[k]-q[k],lower95=float(np.quantile(draws[k],.025)),upper95=float(np.quantile(draws[k],.975))) for k in keys}


def baseline(split):
    il=bank(split,'E0','IL');rl=bank(split,'E0','RL');copy=bank(split,'E0','COPY');screen=bank(split,'E0','IL_screen')
    assert np.array_equal(rl,copy),'E0-COPY score mismatch'
    stable=(screen[...,-1]>=.8).all(-1);logs=[r['log_name'] for r in rows(split)]
    result=dict(split=split,scenes=len(il),logs=len(set(logs)),copy_max_error=0.,stable_high_count=int(stable.sum()),
        components={n:dict(zip(FIELDS,y.mean((0,1)).tolist())) for n,y in [('E0-IL',il),('E0-RL',rl),('E0-COPY',copy)]},
        paired_RL_vs_IL=bootstrap(rl[None],il,logs,stable),
        score_median={n:float(np.median(y[...,-1].mean(-1))) for n,y in [('IL',il),('RL',rl)]},
        teacher_worse_scene_count=int((rl[...,-1].mean(-1)<il[...,-1].mean(-1)-.02).sum()),
        independent_five_draw_self_comparison=paired_quantities(screen[None],il,stable),
        single_draw_IL_PDMS_std=float(il[...,-1].std(-1,ddof=1).mean()))
    write(ROOT/'analysis'/f'baseline_{split}.json',result);return result


def tiny():
    b=bank('cal','E0','IL');result={}
    for exp in EXPERIMENTS:
        v=bank('cal',f'tiny_{exp}_42');mean=float(v[...,-1].mean());difference=mean-float(b[...,-1].mean())
        result[exp]=dict(PDMS=mean,Delta_PDMS=difference,zero_output_fraction=float((v[...,-1]<=1e-12).mean()),
            catastrophic=difference<-.10,finite_valid=True)
    passed=not any(v['catastrophic'] for v in result.values())
    write(ROOT/'analysis/G2.json',dict(status='PASS' if passed else 'STOP_CATASTROPHIC',results=result,
        rule='All seven fixed256-scene qualification runs and full D_cal eval must be valid; stop if mean PDMS drops by0.10'))
    print(result)


def select_tuning():
    if read(ROOT/'analysis/G2.json')['status']!='PASS':raise ValueError('G2 not passed')
    cfg=config();results={};selected={}
    for exp in EXPERIMENTS:
        slots=[]
        for slot,lr in enumerate(cfg['lr_candidates']):
            name=f'tune_{exp}_{slot}';value=bank('cal',name)
            slots.append(dict(name=name,lr=lr,PDMS=float(value[...,-1].mean())))
        results[exp]=slots;selected[exp]=max(slots,key=lambda r:(r['PDMS'],-r['lr']))
    path=ROOT/'selection/tuning.json'
    if path.exists():raise ValueError('Tuning selection already locked')
    write(path,dict(status='LOCKED',selected=selected,candidates=results,source='D_cal only; fixed100-step runs,seed41, two equal slots each',
        gradient_weights=read(ROOT/'calibration/gradient_scales.json')['weights']))
    print(selected)


def formal(split='dev'):
    cfg=config();il=bank(split,'E0','IL');rl=bank(split,'E0','RL');screen=bank(split,'E0','IL_screen')
    logs=[r['log_name'] for r in rows(split)];stable=(screen[...,-1]>=cfg['high_threshold']).all(-1)
    result={};arrays={}
    experiments=read(ROOT/'selection/navtest.json')['methods'] if split=='navtest' else EXPERIMENTS
    for exp in experiments:
        value=np.stack([bank(split,f'train_{exp}_{seed}') for seed in cfg['train_seeds']]);arrays[exp]=value
        metrics=bootstrap(value,il,logs,stable,cfg['bootstrap_replicates'])
        j=value[...,-1].mean(-1);rj=rl[...,-1].mean(-1);ij=il[...,-1].mean(-1)
        groups={'A':(ij<.8)&(rj>=.8),'B':(ij>=.8)&(rj<.8),'C':(ij>=.8)&(rj>=.8),'D':(ij<.8)&(rj<.8)}
        result[exp]=dict(status='COMPLETE',training_seeds=cfg['train_seeds'],metrics=metrics,
            components=dict(zip(FIELDS,value.mean((0,1,2)).tolist())),median_PDMS=float(np.median(j)),
            median_paired_change=float(np.median(j-ij[None])),
            new_component_violations={field:dict(single_output_fraction=float(((value[...,k]<1-1e-12)&(il[None,...,k]>=1-1e-12)).mean()),
                scene_fraction=float(((value[...,k].mean(-1)<1-1e-12)&(il[None,...,k].mean(-1)>=1-1e-12)).mean())) for k,field in enumerate(FIELDS[:-1]) if field!='EP'},
            categories={k:dict(scenes=int(m.sum()),PDMS=float(j[:,m].mean()) if m.any() else None,IL_PDMS=float(ij[m].mean()) if m.any() else None,RL_PDMS=float(rj[m].mean()) if m.any() else None) for k,m in groups.items()},
            teacher_positive_student_zero=int(((rj[None]>1e-12)&(j<=1e-12)).sum()),
            teacher_zero_student_positive=int(((rj[None]<=1e-12)&(j>1e-12)).sum()),
            both_zero_student_improved=int(((ij[None]<=1e-12)&(rj[None]<=1e-12)&(j>.02)).sum()))
    contrasts={}
    pairs=[('E3','E1'),('E3','E2'),('E4-A','E3'),('E4-B','E3'),('E4-C','E3'),('E5','E3'),('E5','E4-C')]
    choice=ROOT/'selection/navtest.json'
    if choice.exists():
        selected=('E5',read(choice)['E4_variant'])
        if selected not in pairs:pairs.append(selected)
    for a,b in pairs:
        if a not in arrays or b not in arrays:continue
        contrasts[a+'_vs_'+b]=compare(arrays[a],arrays[b],il,logs,stable,cfg['bootstrap_replicates'])
    write(ROOT/'analysis'/f'formal_{split}.json',dict(status='COMPLETE',split=split,scenes=len(il),logs=len(set(logs)),methods=result,contrasts=contrasts,
        interval_unit='Paired complete logs, independent training seeds, paired inference seeds. Pooled scene weighting; no scene/pair pseudoreplication.',
        high_subset='Independent5 IL screening seeds, threshold0.8 in every screen draw; evaluate using a different5 seeds.'))
    print({k:v['metrics']['PDMS'] for k,v in result.items()})


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',default='E0-E6');p.add_argument('--phase',choices=['baseline','tiny','select','formal'],default='formal');p.add_argument('--split',default='dev');a=p.parse_args()
    if a.phase=='baseline':print(baseline(a.split))
    elif a.phase=='tiny':tiny()
    elif a.phase=='select':select_tuning()
    else:formal(a.split)

if __name__=='__main__':main()
