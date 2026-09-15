"""Native LFP credit assignment on new read-only groups and historical references.

This is a fixed-batch rule diagnostic, not replay of historical optimizer batches.
The universal vanilla-zscore proxy in analyze.py is explicitly a separate control.
"""
from common_support import *
from analyze import train_reward,safe
import torch,argparse

def main(args):
    torch.set_num_threads(1);m=models()[args.model];assert m['family'] in ['a5','v6'];sys.path.insert(0,m['code_root'])
    from navsim.agents.recogdrive.stage3_lfp_grpo import coerce_lfp_grpo_config,compute_lfp_advantages
    from navsim.agents.recogdrive.stage3_metric_adapter import Stage3MetricAdapter
    from navsim.agents.recogdrive.stage3_reference_cache import Stage3ReferenceCache
    cfg=coerce_lfp_grpo_config(m['grpo_config']['lfp_grpo_cfg']);reference=Stage3ReferenceCache(cfg.reference_cache_path)
    assert reference.metadata['stage2_checkpoint_sha256']==models()[m['family']+'_sft']['sha256']
    adapter=Stage3MetricAdapter('navsim_v1');ordered=sorted(scenes(),key=lambda s:digest(['native_credit_batch',s['token']]))
    names=['no_at_fault_collisions','drivable_area_compliance','ego_progress','time_to_collision_within_bound','history_comfort','driving_direction_compliance','pdms'];rows=[]
    for protocol in CFG['protocols']:
        for replicate in range(4):
            # Formal effective batch is 64 scenes; the last batch is reported.
            for begin in range(0,len(ordered),64):
                chunk=ordered[begin:begin+64];tokens=[s['token'] for s in chunk]
                scores=np.stack([np.load(OUT/'cache/scores/rollouts'/args.model/f'{t}.npz')[protocol][replicate] for t in tokens])
                components={name:torch.tensor(scores[...,i],dtype=torch.float32) for i,name in enumerate(names)}
                components['pdms']=torch.tensor(train_reward(scores),dtype=torch.float32)
                metrics=adapter.canonicalize(components,batch_size=len(tokens),group_size=16);ref=reference.get(tokens,'cpu',torch.float32)
                for s in chunk:
                    hist=np.asarray(reference.records[s['token']]['gt_trajectory']);assert distance(hist[None],np.asarray(s['gt'])[None])[0,0]<1e-3
                kw={}
                if m['family']=='v6':
                    from navsim.agents.recogdrive.stage3_policy_geometry import compute_group_trajectory_spread
                    tr=np.stack([np.load(OUT/'cache/rollouts'/args.model/f'{t}.npz')[protocol][replicate] for t in tokens])
                    kw['group_pairwise_ade_m']=compute_group_trajectory_spread(torch.tensor(tr)).pairwise_ade_m
                output=compute_lfp_advantages(metrics,ref,cfg,**kw);adv=output.advantages.numpy();conservative=safe(scores)
                for j,t in enumerate(tokens):
                    pos=adv[j]>1e-8;neg=adv[j]<-1e-8;invalid=~conservative[j]
                    rows.append(dict(model=args.model,protocol=protocol,token=t,group=replicate,batch_size=len(tokens),positive_count=pos.sum(),negative_count=neg.sum(),zero_count=(~pos&~neg).sum(),no_positive_group=not pos.any(),rule_feasible_count=int(output.feasible[j].sum()),progress_pass_count=int(output.progress_ok[j].sum()),ttc_pass_count=int(output.ttc_ok[j].sum()),pareto_count=int(output.pareto_front[j].sum()),infeasible_count=invalid.sum(),positive_infeasible_count=(pos&invalid).sum(),mean_positive_PDMS=float(scores[j,pos,6].mean()*100) if pos.any() else np.nan,positive_above_reference_count=(pos&(components['pdms'][j].numpy()>float(ref.scalar[j]))).sum(),ref_scalar=float(ref.scalar[j])*100,ref_EP=float(ref.ep[j]),ref_TTC=float(ref.ttc[j]),mean_advantage=float(adv[j].mean()),std_advantage=float(adv[j].std()),all_zero=bool((abs(adv[j])<=1e-8).all())))
                    delta=components['pdms'][j].numpy()-float(ref.scalar[j])
                    # Tolerance only distinguishes float32 equality, not eligibility.
                    rows[-1].update(positive_below_reference_count=(pos&(delta< -1e-6)).sum(),positive_equal_reference_count=(pos&(abs(delta)<=1e-6)).sum(),positive_gain1point_count=(pos&(delta>=.01)).sum(),positive_advantage_mass=float(adv[j,pos].sum()),positive_below_reference_advantage_mass=float(adv[j,pos&(delta< -1e-6)].sum()))
    csv(f'native_credit_{args.model}.csv',rows)
    save(OUT/'audits'/f'native_credit_{args.model}.json',dict(status='PASS',protocol_hash=identity(),runtime_hash=sha(Path(m['code_root'])/'navsim/agents/recogdrive/stage3_lfp_grpo.py'),reference_path=cfg.reference_cache_path,reference_hash=sha(cfg.reference_cache_path),groups=len(rows),batch_size=64,last_batch_size=40,scope='Native archived LFP advantage function on fixed token-hash diagnostic batches. Actual historical minibatches/curriculum and optimizer updates are not replayed.',PSI='UNTESTED native SR credit assignment: universal vanilla proxy is not substituted.'))
    print(args.model,'NATIVE CREDIT COMPLETE',len(rows),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--model',required=True);main(a.parse_args())
