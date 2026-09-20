"""Predeclared step128 learning-curve statistics while final inference runs."""
from common_matched import *
import importlib.util,concurrent.futures
spec=importlib.util.spec_from_file_location('matched_analysis',WORK/'tools/analysis/psi_matched_sft/analyze.py');mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
def one(s):return mod.analyze_scene(s,require_final=False,only_step=128)[0]
def main():
    rows=sorted(scenes('holdout'),key=lambda s:digest([CFG['split_seed'],'early',s['token']]))[:1000]
    for s in rows:
        for method in CFG['methods']:
            for r in CFG['train_seeds']:assert mod.loc(f'{method}_seed{r}_step0128',s['token'],True).exists()
    result=[]
    with concurrent.futures.ProcessPoolExecutor(12) as pool:
        for i,a in enumerate(pool.map(one,rows,chunksize=4)):
            result+=a
            if i%100==0:print('EARLY',i,flush=True)
    f=table('early1000_scene_metrics.parquet',result)
    assert f.groupby(['method','seed','protocol']).size().eq(1000).all()
    avg=f.groupby(['method','protocol','token'],as_index=False)[mod.METRICS].mean()
    table('early1000_summary.csv',avg.groupby(['method','protocol'])[mod.METRICS].mean().reset_index())
    save(OUT/'audits/early1000_analysis.json',dict(status='PASS',step=128,scenes=1000,not_checkpoint_selection=True,primary_final_step=512))
if __name__=='__main__':main()
