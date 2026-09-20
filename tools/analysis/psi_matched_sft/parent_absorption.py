"""Descriptive post-training audit of actual supervised parents on fixed train probes.

No selector, checkpoint, or primary endpoint is changed by this supplementary audit.
Zero hits describe the finite 64-sample bank, not impossibility or unlearnability.
"""
from common_matched import *


def main():
    identity()
    assert (OUT/'audits/data_ready.json').exists()
    def loc(name, token):
        path = baseline(token) if name=='official_il' else Path('/nonexistent')
        return path if path.exists() else OUT/'cache/rollouts'/name/f'{token}.npz'
    probes = scenes('train')[:512]
    rows = []
    for method in CFG['methods']:
        for runseed in CFG['train_seeds']:
            ledger = pd.read_parquet(OUT/'metrics'/f'ledger_{method}_seed{runseed}.parquet')
            counts = ledger.groupby(['token', 'raw_index']).size().to_dict()
            name = f'{method}_seed{runseed}_step0512'
            for scene in probes:
                token = scene['token']
                pool = read(OUT/'cache/selection'/f'{token}.json')['methods'][method]
                with np.load(OUT/'cache/raw'/f'{token}.npz') as raw:
                    trajectories = raw['trajectories'][pool['indices']]
                with np.load(loc(name, token)) as sampled, np.load(loc('official_il', token)) as base:
                    for protocol in CFG['evaluation']['protocols']:
                        distance = legacy.distance(trajectories, sampled[protocol].reshape(64,8,3))
                        initial = legacy.distance(trajectories, base[protocol].reshape(64,8,3))
                        for j, idx in enumerate(pool['indices']):
                            row = dict(token=token, log=scene['log'], method=method,
                                       seed=runseed, protocol=protocol, raw_index=idx,
                                       target_weight=pool['weights'][j],
                                       actual_presentations=int(counts.get((token,idx),0)),
                                       nearest_ADE=float(distance[j].min()),
                                       initial_nearest_ADE=float(initial[j].min()))
                            for radius in [.5, 1.]:
                                suffix=str(radius).replace('.', 'p')
                                row['hits64_'+suffix]=int((distance[j]<=radius).sum())
                                row['initial_hits64_'+suffix]=int((initial[j]<=radius).sum())
                            rows.append(row)
    frame=table('parent_absorption.parquet', rows)
    summaries=[]
    for key,g in frame.groupby(['method','seed','protocol']):
        trained=g[g.actual_presentations>0]
        record=dict(zip(['method','seed','protocol'],key))
        record.update(probe_scenes=g.token.nunique(), selected_unique_parents=len(g),
                      actually_presented_unique_parents=len(trained),
                      never_presented_parents=int((g.actual_presentations==0).sum()))
        for radius in ['0p5','1p0']:
            # Pooled parent counts and equally weighted scene estimates are both explicit.
            hit='hits64_'+radius
            absent=trained[hit]==0
            record['trained_parent_zero_hit_fraction_'+radius]=float(absent.mean())
            record['scene_equal_trained_parent_zero_hit_fraction_'+radius]=float(
                trained.assign(absent=absent).groupby('token').absent.mean().mean())
            record['initial_zero_hit_fraction_same_trained_parents_'+radius]=float(
                (trained['initial_'+hit]==0).mean())
        summaries.append(record)
    table('parent_absorption_summary.csv', summaries)
    save(OUT/'audits/parent_absorption.json',dict(
        status='PASS', protocol_hash=identity(), fixed_train_probe_scenes=512,
        rows=len(frame), posthoc_descriptive=True, changes_to_selection_or_training=False,
        statement='Finite-bank coverage of actually presented targets; not an impossibility claim.'))


if __name__=='__main__': main()
