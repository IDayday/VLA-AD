"""Paired continuation-policy diagnostic with retained real test-noise axis."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from scripts.process_improvement.common import *


def main():
    from navsim.agents.recogdrive.process_improvement.statistics import local_noise_bootstrap
    root=ROOT/'diagnostics/suffix_mismatch/selected_full_seed42'
    original=read(root/'results.json');assert original['status']=='COMPLETE' and original['scenes']==128
    rows=membership(diagnostic_plan()['confirm_split'])[:128]
    data=[];accepted=0;payloads={}
    for row in rows:
        path=root/'scenes'/f"{row['token']}.json";record=read(path)
        assert record['token']==row['token'] and record['driver_record']==row['driver_record']
        positions=sorted(record['records'],key=lambda r:r['index']);assert len(positions)==5
        own=[]
        for position in positions:
            ref=np.asarray(position['reference_suffix']);student=np.asarray(position['student_suffix'])
            assert ref.shape==student.shape==(2,16,7)
            dg=ref[1,:,-1]-ref[0,:,-1];ds=student[1,:,-1]-student[0,:,-1]
            own.append(np.stack([dg,ds,ds-dg],axis=-1));accepted+=int(position['accepted'])
        data.append(np.asarray(own).mean(0)[None]);payloads[str(path)]=sha(path)
    result=local_noise_bootstrap(data,[r['driver_record'] for r in rows])
    result.update(status='COMPLETE',fields=['reference_suffix_gain','student_suffix_gain','student_minus_reference_gain'],
        target_constructions=640,accepted_targets=accepted,
        conditional_on_fixed_target_construction=True,source_checkpoint_sha256=original['checkpoint_sha256'],
        input_result_sha256=sha(root/'results.json'),scene_payloads_sha256=digest(payloads),
        analysis_script_sha256=sha(__file__),statistics_sha256=sha(SOURCE/'navsim/agents/recogdrive/process_improvement/statistics.py'),
        scope='128 fixed cal-confirm scenes, one fixed target bank per state; only suffix policy changes under same current-step reference kernel. No training, recalibration or deployed-student gain claim.')
    write(ROOT/'analysis/selected_full_suffix_mismatch_paired.json',result)
    write(REPORTS/'tables/selected_full_suffix_mismatch_paired.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
