"""Resumable phase driver. Every subprocess must pass before the next stage."""
import argparse, os, subprocess, time, threading
from common import *

def main(a):
    cfg=config(a.config);out=ROOT/cfg['output_dir'];logs=out/'logs';logs.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONUNBUFFERED='1')
    base=['--config',a.config,'--manifest',a.manifest]
    scripts=ROOT/'tools/analysis/pc_mts_diagnostics'
    def run(label,cmd):
        start=time.time();print(json.dumps(dict(event='start',stage=label,command=cmd,time=start)),flush=True)
        with (logs/(label+'.log')).open('a') as f:
            p=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            save(out/'manifests/active_process.json',dict(pid=p.pid,command=cmd,stage=label,started=start))
            stopped=threading.Event()
            def monitor():
                import psutil
                with (logs/(label+'_resources.jsonl')).open('a') as log:
                    while not stopped.is_set():
                        try:gpus=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=4).strip()
                        except (subprocess.SubprocessError,OSError):gpus='unavailable'
                        log.write(json.dumps(dict(time=time.time(),cpu_percent=psutil.cpu_percent(),ram_used_bytes=psutil.virtual_memory().used,ram_available_bytes=psutil.virtual_memory().available,gpus=gpus))+'\n');log.flush();stopped.wait(2)
            monitor_thread=threading.Thread(target=monitor,daemon=True);monitor_thread.start()
            rc=p.wait()
            stopped.set();monitor_thread.join(timeout=5)
        save(out/'manifests'/(label+'_execution.json'),dict(command=cmd,returncode=rc,wall_seconds=time.time()-start))
        if rc:raise RuntimeError(f'{label} failed; see {logs/(label+".log")}')
        print(json.dumps(dict(event='complete',stage=label,seconds=time.time()-start)),flush=True)
    if a.phase in ['gpu','all']:
        torchrun=[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=8',str(scripts/'distributed_rollout.py')]+base
        run('features',torchrun+['--features'])
        for m in a.checkpoints:run('rollout_'+m,torchrun+['--checkpoint',m])
    if a.phase in ['score','all']:
        run('score_rollouts',[PY,str(scripts/'evaluate_cached_rollouts.py')]+base+['--stage','rollouts','--workers',str(cfg['cpu_workers']),'--validate'])
    if a.phase in ['candidates','all']:
        method_args=['--methods']+a.methods
        run('build_raw',[PY,str(scripts/'build_candidate_reservoir.py')]+base+['--selection-local'])
        for stage in ['raw_candidates','selection_local']:
            run('score_'+stage,[PY,str(scripts/'evaluate_cached_rollouts.py')]+base+['--stage',stage,'--workers',str(cfg['cpu_workers'])])
        run('select_pools',[PY,str(scripts/'select_candidate_pools.py')]+base+['--empty-policy',a.empty_policy]+method_args)
        run('analyze_positions',[PY,str(scripts/'analyze_candidates.py')]+base+method_args)
        run('validate_pools',[PY,str(scripts/'validate_candidate_pools.py')]+base+method_args)
        run('heldout_generation',[PY,str(scripts/'evaluate_local_robustness.py')]+base+method_args)
        run('score_heldout',[PY,str(scripts/'evaluate_cached_rollouts.py')]+base+['--stage','heldout','--workers',str(cfg['cpu_workers'])]+method_args)
        run('candidate_analysis',[PY,str(scripts/'analyze_candidates.py')]+base+method_args)
        run('gt_sensitivity',[PY,str(scripts/'gt_threshold_sensitivity.py')]+base)
    if a.phase in ['analysis','all']:
        run('distribution_analysis',[PY,str(scripts/'analyze_policy_distribution.py')]+base)
        run('readiness_analysis',[PY,str(scripts/'analyze_grpo_readiness.py')]+base)
        run('figures',[PY,str(scripts/'make_figures.py')]+base)
    if a.phase=='pc-partial':
        run('pc_partial_select',[PY,str(scripts/'select_candidate_pools.py')]+base+['--methods','pc_mts','--defer-empty'])
        run('pc_partial_positions',[PY,str(scripts/'analyze_candidates.py')]+base+['--matched-pc-only'])
        run('pc_partial_validate',[PY,str(scripts/'validate_candidate_pools.py')]+base+['--allow-pending'])
        run('pc_partial_heldout_generation',[PY,str(scripts/'evaluate_local_robustness.py')]+base+['--methods','pc_mts','--allow-pending'])
        run('pc_partial_score_heldout',[PY,str(scripts/'evaluate_cached_rollouts.py')]+base+['--stage','heldout','--workers',str(cfg['cpu_workers']),'--methods','pc_mts','--allow-pending'])
        run('pc_partial_analysis',[PY,str(scripts/'analyze_candidates.py')]+base+['--matched-pc-only'])
        run('pc_partial_report',[PY,str(scripts/'report_partial_pc.py')]+base)
    if a.phase=='pc-coverage':
        run('pc_coverage_select',[PY,str(scripts/'select_candidate_pools.py')]+base+['--methods','pc_mts','--empty-policy','coverage'])
        run('pc_coverage_positions',[PY,str(scripts/'analyze_candidates.py')]+base)
        run('pc_coverage_validate',[PY,str(scripts/'validate_candidate_pools.py')]+base)
        run('pc_coverage_heldout_generation',[PY,str(scripts/'evaluate_local_robustness.py')]+base+['--methods','pc_mts'])
        run('pc_coverage_score_heldout',[PY,str(scripts/'evaluate_cached_rollouts.py')]+base+['--stage','heldout','--workers',str(cfg['cpu_workers']),'--methods','pc_mts'])
        run('pc_coverage_analysis',[PY,str(scripts/'analyze_candidates.py')]+base)
        run('pc_coverage_figures',[PY,str(scripts/'make_figures.py')]+base)
        run('pc_coverage_report',[PY,str(scripts/'report_coverage.py')]+base)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--manifest',required=True);p.add_argument('--phase',choices=['gpu','score','candidates','analysis','all','pc-partial','pc-coverage'],default='all');p.add_argument('--empty-policy',choices=['error','missing','emergency','coverage'],default='error');p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS);p.add_argument('--checkpoints',nargs='+',choices=MODELS,default=MODELS);main(p.parse_args())
