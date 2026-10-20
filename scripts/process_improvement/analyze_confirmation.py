"""Separate real finite-bank, raw-mean and guarded gains on held-out parents."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.process_improvement.common import *


def main():
    from navsim.agents.recogdrive.process_improvement.statistics import local_noise_bootstrap
    plan=diagnostic_plan();gate=read(ROOT/'selection/process_gate.json')
    assert not gate['expansion_allowed']
    root=ROOT/'diagnostics'/gate['source'];rows=membership(plan['confirm_split'])
    datasets={};accept={};payloads={}
    for row in rows:
        path=root/'scenes'/f"{row['token']}.json";record=read(path);payloads[str(path)]=sha(path)
        for source in ['IL','OLD_OPD','RL']:
            values=sorted([r for r in record['records'] if r['source']==source],key=lambda r:(r['index'],r['bank']))
            assert len(values)==10
            own=[]
            for v in values:
                base_scores=np.asarray(v['P0']);base=base_scores[:,-1]
                choice=np.asarray(v['P1'])[:,-1]-base
                raw=np.asarray(v['proposed'])[:,-1]-base
                guarded=np.asarray(v['P2'])[:,-1]-base
                fields=[choice,raw,guarded,choice-raw,raw-guarded,choice-guarded]
                for role in ['P1','proposed','P2']:
                    metrics=[0,1,3] if role!='P2' else [0,1,3,2,4,5]
                    fields.extend(np.asarray(v[role])[:,j]-base_scores[:,j] for j in metrics)
                own.append(np.stack(fields,axis=-1))
            own=np.asarray(own);L=own.shape[1]
            datasets.setdefault(source,[]).append(own.reshape(5,2,L,own.shape[-1]).mean(0))
            accept.setdefault(source,[]).append(sum(int(v['accepted']) for v in values))
    results={}
    for source,data in datasets.items():
        value=local_noise_bootstrap(data,[r['driver_record'] for r in rows])
        value.update(accepted_targets=sum(accept[source]),scenes_with_any_accepted_target=sum(v>0 for v in accept[source]),
                     total_targets=len(rows)*10)
        results[source]=value
    result=dict(status='COMPLETE',fields=['P1_minus_P0','raw_projection_minus_P0','guarded_projection_minus_P0',
        'P1_minus_raw_projection','raw_minus_guarded_projection','P1_minus_guarded_projection',
        'P1_delta_NC','P1_delta_DAC','P1_delta_TTC','raw_delta_NC','raw_delta_DAC','raw_delta_TTC',
        'guarded_delta_NC','guarded_delta_DAC','guarded_delta_TTC','guarded_delta_EP','guarded_delta_Comfort','guarded_delta_DDC'],results=results,
        gate=gate['status'],gate_sha256=sha(ROOT/'selection/process_gate.json'),
        script_sha256=sha(__file__),statistics_sha256=sha(SOURCE/'navsim/agents/recogdrive/process_improvement/statistics.py'),
        source_payloads_sha256=digest(payloads),
        scope='512 independent cal-confirm scenes; fixed rollin, paired banks/test-noise; local programs, not deployed students. Differences are procedure comparisons, not KL projection error.')
    write(ROOT/'analysis/confirmation_decomposition.json',result)
    write(REPORTS/'tables/confirmation_decomposition.json',result)
    for name in [gate['source']+'.json','process_gate.json']:
        path=(ROOT/'analysis'/name) if name!='process_gate.json' else ROOT/'selection'/name
        write(REPORTS/'tables'/name,read(path))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
