"""One common 192-candidate space; source-balanced bridges, no outcome retuning."""
import argparse,concurrent.futures
from v2_common import *
from compatibility import fit,position
from perturbations import perturb,bank

def work(scene):
    token=scene['token'];src=OUT/'il_banks'/f'{token}.npz';il=np.load(src);meta=dict(ident(),token=token,il_banks_sha256=sha(src),external_sources='first16_V1_external_per_model')
    dest=OUT/'raw_candidates'/f'{token}.npz'
    if not valid(dest,meta):
        gt=np.asarray(scene['gt']);trajs=[gt];sources=['GT'];ids=['gt_exact'];alphas=[np.nan];anchor_ids=[''];reference_ids=[-1]
        for i in range(31):
            family=['lateral','progress','curvature','endpoint','spline'][i%5];amp=CFG['gt_amplitudes_m'][(i//5)%5]
            trajs.append(perturb(gt,amp,family,1 if i%2 else -1,seed(token,'v2_gt',i)));sources.append('GT');ids.append(f'gt_{i:02d}');alphas.append(np.nan);anchor_ids.append('');reference_ids.append(-1)
        for i,t in enumerate(il['native']):trajs.append(t);sources.append('IL-native');ids.append(f'il_native_{i:02d}');alphas.append(np.nan);anchor_ids.append('');reference_ids.append(-1)
        anchors=[];external_hashes={}
        for model in MODELS[1:]:
            parent=V1/'rollouts'/model/f'{token}.npz';external_hashes[model]=sha(parent);external=np.load(parent)['external'][:16];scores=np.load(V1/'evaluator/rollouts'/model/f'{token}.npz')['external'][:16]
            safe=feasible(scores);chosen=sorted(range(16),key=lambda i:(-int(safe[i]),-scores[i,6],i))[:4]
            for i,t in enumerate(external):trajs.append(t);sources.append('External-MTS' if model.startswith('mts') else 'External-RL');ids.append(f'{model}_{i:02d}');alphas.append(np.nan);anchor_ids.append('');reference_ids.append(-1)
            anchors.extend((external[i],f'{model}_{i:02d}') for i in chosen)
        assert len(trajs)==128 and len(anchors)==16
        for t,anchor in anchors:
            j=int(distance(t[None],il['R'])[0].argmin());rho=il['R'][j]
            for alpha in CFG['bridge_alphas']:
                bridge=(1-alpha)*rho+alpha*t
                angle=np.arctan2(np.sin(t[:,2]-rho[:,2]),np.cos(t[:,2]-rho[:,2]));bridge[:,2]=np.arctan2(np.sin(rho[:,2]+alpha*angle),np.cos(rho[:,2]+alpha*angle))
                trajs.append(bridge);sources.append(f'Bridge-{alpha:g}');ids.append(f'bridge_{anchor}_{alpha:g}');alphas.append(alpha);anchor_ids.append(anchor);reference_ids.append(j)
        assert len(trajs)==192
        npz(dest,dict(meta,external_hashes=external_hashes),trajectories=np.stack(trajs),source=np.asarray(sources),candidate_id=np.asarray(ids),bridge_alpha=np.asarray(alphas),anchor_id=np.asarray(anchor_ids),reference_index=np.asarray(reference_ids))
    raw=np.load(dest);cal=fit(il['R'],il['Q']);pmeta=dict(meta,raw_sha256=sha(dest));pdest=OUT/'positions'/f'{token}.npz'
    if not valid(pdest,pmeta):npz(pdest,pmeta,**position(raw['trajectories'],cal),query_reference_distances=cal['qr'],query_mahalanobis=cal['qm'],covariance_shrinkage=np.array(cal['shrinkage']),covariance_jitter=np.array(cal['jitter']))
    local=OUT/'selection_local'/f'{token}.npz'
    if not valid(local,pmeta):npz(local,pmeta,trajectories=np.stack([bank(t,token,'v2_selection',False) for t in raw['trajectories']]))
    return token

def main(a):
    rows=scenes()['scenes']
    if a.smoke:rows=[r for r in rows if r['token'] in set(subset('smoke'))]
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(96,len(rows))) as ex:
        for i,t in enumerate(ex.map(work,rows,chunksize=1)):
            if i%100==0:print('Built',i+1,len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');main(p.parse_args())
