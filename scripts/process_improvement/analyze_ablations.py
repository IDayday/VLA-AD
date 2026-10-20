"""Paired local negative-control analysis from real retained MC score deltas."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.process_improvement.common import *


def main():
    from navsim.agents.recogdrive.process_improvement.statistics import local_noise_bootstrap
    root=ROOT/'diagnostics/local_parent_ablations'
    result=read(root/'results.json');assert result['status']=='COMPLETE'
    configuration=read(root/'configuration.json')
    tokens=configuration['scene_tokens'];parents=[];data={};payloads={}
    for token in tokens:
        path=root/'scenes'/f'{token}.json';record=read(path)
        assert record['token']==token and len(record['records'])==80
        parents.append(record['parent']);payloads[str(path)]=sha(path)
        for mode in ['antithetic','iid']:
            for variant in ['PROCESS','UNIFORM_TARGET','SHUFFLED_VALUE','NO_GUARD']:
                values=sorted([r for r in record['records'] if r['mode']==mode and r['variant']==variant],
                              key=lambda r:(r['index'],r['bank']))
                assert len(values)==10
                data.setdefault((mode,variant),[]).append(np.asarray([v['test_delta'] for v in values]).reshape(5,2,16).mean(0)[...,None])
    contrasts=[]
    for new,control in [
        (('antithetic','PROCESS'),('antithetic','SHUFFLED_VALUE')),
        (('antithetic','PROCESS'),('antithetic','NO_GUARD')),
        (('antithetic','PROCESS'),('antithetic','UNIFORM_TARGET')),
        (('iid','PROCESS'),('antithetic','PROCESS'))]:
        delta=np.asarray(data[new])-np.asarray(data[control])
        value=local_noise_bootstrap(delta,parents)
        contrasts.append(dict(new=list(new),control=list(control),**value))
    output=dict(status='COMPLETE',scenes=len(tokens),parents=len(set(parents)),contrasts=contrasts,
        original_result_sha256=sha(root/'results.json'),source_scene_payloads_sha256=digest(payloads),
        analysis_script_sha256=sha(__file__),statistics_sha256=sha(SOURCE/'navsim/agents/recogdrive/process_improvement/statistics.py'),
        scope='Fixed64 D_fit negative controls, historical OPD rollin; no calibration change; conditional fixed rollin seed; PDMS only, not a safety noninferiority or student result')
    write(ROOT/'analysis/local_parent_ablation_contrasts.json',output)
    write(REPORTS/'tables/local_parent_ablation_contrasts.json',output)
    write(REPORTS/'tables/local_parent_ablations.json',result)
    print(json.dumps(output,indent=2),flush=True)


if __name__=='__main__':main()
