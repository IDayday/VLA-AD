"""Index server-local caches and validate the small, reviewable publication bundle."""
import ast,concurrent.futures,importlib.metadata,re
from v2_common import *
def record(p):return dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p))
def main():
    for p in (ROOT/'tools/analysis/pc_mts_diagnostics_v2').glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    report=ROOT/'reports/PC_MTS_POLICY_DIAGNOSTICS_V2_20260913.md'
    for target in re.findall(r'\]\(([^)]+)\)',report.read_text()):
        if not target.startswith('http'):assert (report.parent/target).exists(),target
    arrays=[]
    for folder in ['il_banks','raw_candidates','positions','selection_local','candidate_pools','heldout','heldout_seeded','evaluator','denoising','chains']:
        arrays.extend(p for p in (OUT/folder).rglob('*') if p.is_file())
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as ex:cache=list(ex.map(record,arrays))
    save(OUT/'manifests/CACHE_INDEX.json',dict(identity=ident(),files=cache,file_count=len(cache),total_bytes=sum(r['bytes'] for r in cache),storage='Server-local immutable-input/resumable-result caches; repository publishes metrics, figures, manifests and report.'))
    publication=[report]+list((ROOT/'tools/analysis/pc_mts_diagnostics_v2').glob('*'))+list((ROOT/'configs/pc_mts_diagnostics_v2').glob('*'))+list((OUT/'metrics').glob('*'))+list((OUT/'figures').glob('*'))+list((OUT/'report').glob('*'))
    publication=[p for p in publication if p.is_file()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:items=list(ex.map(record,publication))
    save(OUT/'manifests/RESULT_ARTIFACTS.json',dict(identity=ident(),files=items,file_count=len(items),core_figures=11,supplementary_figures=6,primary_scene_count=1000,contrastive_scene_count=276,policy_rollouts_per_checkpoint_scene=64,denoising_scene_count=500,robustness_primary='heldout_seeded; initial deterministic results preserved in audit_archives'))
    packages=['torch','numpy','scipy','pandas','matplotlib','scikit-learn','joblib','threadpoolctl'];versions={}
    for package in packages:
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]='isolated OUT/deps installation'
    save(OUT/'manifests/ENVIRONMENT.json',dict(python=sys.version,packages=versions,threads=dict(OMP=1,MKL=1,OPENBLAS=1),gpu_processes=8,cpu_evaluator_workers=96))
    print('Indexed caches',len(cache),'published artifacts',len(items),flush=True)
if __name__=='__main__':main()
