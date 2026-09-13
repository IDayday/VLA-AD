"""Audit the original scalar reward/forward loss against the batched logging hook."""
import copy,types,lzma,pickle,inspect,concurrent.futures,multiprocessing
from native import *
from train import attach_grpo
def main():
    assert (OUT/'manifests/audit_E.json').exists();device=setup();p=model();ref=model();lookup={r['token']:r for r in scenes()};recipe=attach_grpo(p,ref,lookup)
    from navsim.agents.recogdrive.recogdrive_diffusion_planner import GRPOConfig
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
    sampling=TrajectorySampling(time_horizon=4,interval_length=.1);p.simulator=PDMSimulator(sampling);p.train_scorer=PDMScorer(sampling,GRPOConfig().scorer_config);p.eval();p.requires_grad_(False)
    from scoring import init,online
    original=p.reward_fn;checks=[]
    with concurrent.futures.ProcessPoolExecutor(1,mp_context=multiprocessing.get_context('spawn'),initializer=init) as pool:
        for token in tokens('train')[:4]:
            vl,action=inputs(feature(token),device);capture={}
            def scalar(self,trajs,toks,caches):
                val=original(trajs,toks,caches);capture['native_reward']=val.cpu().numpy().copy();capture['trajs']=trajs.cpu().numpy().copy();return val
            p.reward_fn=types.MethodType(scalar,p);torch.manual_seed(seed(token,'v3_GRPO_parity'))
            with torch.no_grad():a=p.forward_grpo(vl,action(1),[token],sample_time=8,bc_coeff=.1,use_bc_loss=True)
            def batched(self,trajs,toks,caches):
                standard,reward=pool.submit(online,lookup[token],trajs.cpu().numpy()).result();capture['batch_reward']=reward[:,6];return torch.as_tensor(reward[:,6],device=device,dtype=trajs.dtype)
            p.reward_fn=types.MethodType(batched,p);torch.manual_seed(seed(token,'v3_GRPO_parity'))
            with torch.no_grad():b=p.forward_grpo(vl,action(1),[token],sample_time=8,bc_coeff=.1,use_bc_loss=True)
            diffs={k:float(abs(a[k]-b[k])) for k in ['loss','policy_loss','bc_loss','reward']};assert max(diffs.values())<1e-6,diffs
            # Float64 evaluator to float32 native return is the original conversion.
            err=float(abs(capture['native_reward']-capture['batch_reward'].astype(np.float32)).max());assert err<1e-8,err
            checks.append(dict(token=token,loss_differences=diffs,reward_max_abs=err))
    stage_audit('F_recipe',checks=checks,recipe=recipe,implementation='Direct original p.forward_grpo; only reward function receives equivalent batched scorer plus logging',source_sha256=sha(Path(inspect.getfile(p.__class__))),original_reward_weights=dict(EP=10,TTC=5,comfort=2),evaluation_reward_weights=dict(EP=5,TTC=5,comfort=2),reference_checkpoint_sha256=v1.models()[0]['sha256'],optimizer_updates_in_audit=0)
    print('Original GRPO parity PASS',checks,flush=True)
if __name__=='__main__':main()
