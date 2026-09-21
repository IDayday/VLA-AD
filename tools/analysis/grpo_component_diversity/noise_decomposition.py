from shared import *
import sample_extra as sampler
import argparse,time
def run(m):
    path=OUT/'manifests/noise_decomposition.json';spec=ROOT/'configs/grpo_component_diversity/noise_decomposition.yaml'
    if not path.exists():
        tokens=[r['token'] for r in sorted(scenes(),key=lambda r:previous.digest(['noise_decomposition_20260921',r['token']]))[:64]]
        save(path,dict(config_sha256=sha(spec),tokens=tokens,timestamp=time.time(),outcome_conditioned=False))
    frozen=read(path);assert frozen['config_sha256']==sha(spec)
    p,meta=sampler.sample.load(m);before=previous.state_hash(p);rows=[]
    for t in frozen['tokens']:
        vl,action=previous.inputs(t);original=p.p_mean_variance;capture=[]
        def hook(*a,**kw):
            result=original(*a,**kw);capture.append(result[0].detach().clone());return result
        p.p_mean_variance=hook
        try:full=sampler.native(p,vl,action,t,0,128)
        finally:p.p_mean_variance=original
        assert len(capture)==5
        # Clipping affects Gaussian variance; compute observed residual too.
        mean=p.denorm_odo(capture[-1].clamp(-p.final_action_clip_value,p.final_action_clip_value)).cpu().numpy()
        cached=arrays(m,t);err=float(abs(full-cached).max());assert err<1e-4
        def width(x):
            d=previous.distance(x,x);return float(d[np.triu_indices(128,1)].mean())
        rows.append(dict(model=m,token=t,pairwise_ADE=width(full),pre_final_noise_mean_pairwise_ADE=width(mean),residual_pairwise_ADE=width(full-mean),mean_to_output_ADE=float(np.linalg.norm(full[...,:2]-mean[...,:2],axis=-1).mean()),cache_parity=err,x_residual_std=float((full-mean)[...,0].std()),y_residual_std=float((full-mean)[...,1].std())))
    assert previous.state_hash(p)==before
    csv(f'noise_decomposition_{m}.csv',rows)
    save(OUT/'audits'/f'noise_decomposition_{m}.json',dict(status='PASS',scenes=64,state_unchanged=True,supplementary_hash=sha(spec),max_cache_error=max(x['cache_parity'] for x in rows)))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--model',required=True);run(a.parse_args().model)
