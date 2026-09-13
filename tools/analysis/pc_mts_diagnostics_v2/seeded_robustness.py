"""Fix unused held-out shape seeds; preserve original deterministic probes for audit."""
import concurrent.futures,shutil
from v2_common import *
from perturbations import bank,perturb
PROTOCOL=dict(reason='The initial three held-out families passed seeds but did not consume them. Fix seed use for independent local validation only.',shape='Before max-XY amplitude normalization, multiply displacement at u by 1+0.1*z*sin(pi*u), z~Uniform(-1,1) from the assigned independent probe seed.',amplitudes=[.05,.15,.30,.50],families=['lateral','curvature','progress'],directions=[-1,1],selection='No selection, bridge alpha, q threshold, contrastive rule or checkpoint rollout changed.',archive='Original deterministic heldout and evaluator/heldout retained; initial derived tables/report copied to audit_archives/deterministic_shape_initial.',rule='Single implementation correction fixed before seeded robustness evaluation; no coefficient search or outcome tuning.')

def work(scene):
    token=scene['token']
    for method in METHODS:
        src=OUT/'candidate_pools'/method/f'{token}.npz';dest=OUT/'heldout_seeded'/method/f'{token}.npz';meta=dict(identity=ident(),protocol=PROTOCOL,pool_sha256=sha(src),stream='v2_heldout_robustness')
        if not valid(dest,meta):npz(dest,meta,trajectories=np.stack([bank(t,token,'v2_heldout_robustness',True,True) for t in np.load(src)['trajectories']]))
    return token

def main():
    marker=OUT/'manifests/seeded_robustness_correction.json'
    if not marker.exists():
        dest=OUT/'audit_archives/deterministic_shape_initial';dest.mkdir(parents=True,exist_ok=True)
        shutil.copytree(OUT/'metrics',dest/'metrics');shutil.copytree(OUT/'figures',dest/'figures')
        if (ROOT/'reports/PC_MTS_POLICY_DIAGNOSTICS_V2_20260913.md').exists():shutil.copyfile(ROOT/'reports/PC_MTS_POLICY_DIAGNOSTICS_V2_20260913.md',dest/'report.md')
        save(marker,dict(identity=ident(),protocol=PROTOCOL))
    t=np.asarray(scenes()['scenes'][0]['gt']);a=perturb(t,.3,'lateral',1,1,True);b=perturb(t,.3,'lateral',1,2,True);assert not np.array_equal(a,b);np.testing.assert_array_equal(a,perturb(t,.3,'lateral',1,1,True))
    with concurrent.futures.ProcessPoolExecutor(max_workers=96) as ex:rows=list(ex.map(work,scenes()['scenes'],chunksize=1))
    save(OUT/'manifests/heldout_seeded_generation.json',dict(identity=ident(),protocol=PROTOCOL,scenes=len(rows),same_seed_replay=True,different_seeds_change_shape=True))

if __name__=='__main__':main()
