"""Execute archived advantage statements on CUDA; verify actual scorer algebra."""
from shared import *
from metrics import reward
import ast,types,torch,lzma,pickle,dataclasses,inspect
def extract_advantage(path):
    source=Path(path).read_text();tree=ast.parse(source)
    fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='forward_grpo')
    start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='rewards_matrix')
    end=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='num_denoising_steps')
    body=fn.body[start:end]+[ast.Return(value=ast.Name(id='advantages',ctx=ast.Load()))]
    wrapper=ast.FunctionDef(name='native_adv',args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=x) for x in ['rewards','B','G','self']],kwonlyargs=[],kw_defaults=[],defaults=[]),body=body,decorator_list=[])
    module=ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[]));scope={'torch':torch};exec(compile(module,str(path),'exec'),scope)
    return scope['native_adv'],'\n'.join(source.splitlines()[fn.body[start].lineno-1:fn.body[end-1].end_lineno])
def main():
    identity();torch.set_num_threads(1);ss=scenes();mm=models();rows=[]
    fun,source=extract_advantage(mm[CFG['models'][0]]['runtime_planner_path'])
    (OUT/'audits/native_advantage_excerpt.py.txt').write_text(source+'\n')
    settings=types.SimpleNamespace(clip_advantage_lower_quantile=0.,clip_advantage_upper_quantile=1.)
    for m in CFG['models']:
        scores=np.stack([arrays(m,r['token'],True) for r in ss]);out={}
        for g in CFG['group_sizes']:
            r=reward(scores.reshape(-1,7)).reshape(-1,g);tensor=torch.tensor(r,dtype=torch.float32,device='cuda')
            a=fun(tensor.reshape(-1),len(tensor),g,settings).reshape(5000,128).cpu().numpy()
            out[f'G{g}']=a
            cpu=fun(tensor.cpu().reshape(-1),len(tensor),g,settings).reshape(5000,128).numpy()
            rows.append(dict(model=m,G=g,cpu_vs_cuda_max_abs=float(abs(cpu-a).max()),native_code_sha256=mm[m]['runtime_planner_sha256']))
        npz(OUT/'cache/advantages'/f'{m}.npz',dict(protocol_hash=identity(),tokens=[s['token'] for s in ss],runtime_code_sha256=mm[m]['runtime_planner_sha256'],device='cuda',dtype='fp32'),**out)
    # Actual archived scorer (not a new scoring definition).
    code=Path(mm[CFG['models'][0]]['runtime_planner_path']).parents[3];sys.path.insert(0,str(code))
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.common.dataclasses import Trajectory
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer,PDMScorerConfig
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    sampling=TrajectorySampling(num_poses=40,interval_length=.1);sim=PDMSimulator(sampling)
    checks=[]
    for m in CFG['models']:
        for row in ss[:4]:
            with lzma.open(row['metric_cache_path'],'rb') as f:cache=pickle.load(f)
            t=arrays(m,row['token']);s=arrays(m,row['token'],True)
            for i in [0,15,64,127]:
                for ep_weight in [5,10]:
                    scorer=PDMScorer(sampling,PDMScorerConfig(progress_weight=float(ep_weight),ttc_weight=5.,comfortable_weight=2.))
                    result=pdm_score(cache,Trajectory(t[i].astype(np.float64)),sampling,sim,scorer)
                    expected=reward(s[i:i+1],ep_weight==10)[0];err=abs(float(dataclasses.asdict(result)['score'])-expected)
                    assert err<=1e-8,(m,row['token'],i,ep_weight,err)
                    checks.append(dict(model=m,token=row['token'],member=i,EP_weight=ep_weight,max_abs_error=err))
    save(OUT/'audits/native_reward_advantage.json',dict(status='PASS',protocol_hash=identity(),runtime_code_path=mm[CFG['models'][0]]['runtime_planner_path'],native_excerpt_sha256=sha(OUT/'audits/native_advantage_excerpt.py.txt'),scorer_path=inspect.getfile(PDMScorer),scorer_sha256=sha(inspect.getfile(PDMScorer)),scoring_checks=checks,advantage_checks=rows,advantage_source='Unmodified AST statements from archived forward_grpo executed in FP32 on CUDA; before denoising discounts, logprob clipping and BC. No backward/update.',group_size_note='Hypothetical diagnostic groups; original successful training used G8.'))
if __name__=='__main__':main()
