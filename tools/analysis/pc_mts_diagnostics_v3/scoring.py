"""Isolated CPU workers use the validated NAVSIM implementation, never surrogate scores."""
import argparse,concurrent.futures,lzma,pickle
from common_v3 import *
from evaluate_cached_rollouts import init_worker,score_arrays,STATE
CACHES={}
SCORERS={}
def init():
    init_worker()
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer,PDMScorerConfig
    SCORERS['evaluation']=STATE['scorer'];SCORERS['training']=PDMScorer(STATE['sampling'],PDMScorerConfig(progress_weight=10.,ttc_weight=5.,comfortable_weight=2.))
def cache(scene):
    token=scene['token']
    if token not in CACHES:
        with lzma.open(scene['metric_cache_path'],'rb') as f:CACHES[token]=pickle.load(f)
    return CACHES[token]
def online(scene,trajectories):
    c=cache(scene);STATE['scorer']=SCORERS['evaluation'];standard=score_arrays(c,trajectories);STATE['scorer']=SCORERS['training'];reward=score_arrays(c,trajectories);STATE['scorer']=SCORERS['evaluation']
    return standard,reward
def task(args):
    path,scene=args;relative=path.relative_to(OUT/'cache/rollouts');dest=OUT/'cache/scores'/relative;meta=dict(identity=identity(),input_sha256=sha(path))
    if valid(dest,meta):return str(dest)
    a=np.load(path);STATE['scorer']=SCORERS['evaluation'];scores=score_arrays(cache(scene),a['trajectories']);npz(dest,meta,scores=scores);return str(dest)
def main(stage):
    rows={s['token']:s for s in scenes()};paths=list((OUT/'cache/rollouts'/stage).glob('*/*/*.npz'));assert paths
    start=time.time()
    with concurrent.futures.ProcessPoolExecutor(96,initializer=init) as ex:
        for j,_ in enumerate(ex.map(task,[(p,rows[p.stem]) for p in paths],chunksize=1)):
            if j%500==0:print(stage,'CPU scores',j+1,len(paths),time.time()-start,flush=True)
    stage_audit(stage+'_scoring',files=len(paths),workers=96,seconds=time.time()-start)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage');main(a.parse_args().stage)
