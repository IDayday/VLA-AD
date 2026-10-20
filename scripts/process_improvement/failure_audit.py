"""Descriptive paired failure/retention counts from complete real evaluations.

Positive means PDMS > locked zero tolerance, not a road-safety success claim.
Rows remain separate training policies; no trajectory or policy ensemble is
constructed. Inferential intervals belong to the hierarchical main analysis.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import csv
import re

import numpy as np

from scripts.process_improvement.common import *
from scripts.process_improvement.analyze import predictions,analysis_source


def main():
    p=argparse.ArgumentParser();p.add_argument('--methods',nargs='+',required=True)
    p.add_argument('--name',required=True);p.add_argument('--split',default='dev');a=p.parse_args()
    if not re.fullmatch('[A-Za-z0-9_.-]+',a.name):raise ValueError('Invalid audit identity')
    source=analysis_source(a.split);source['audit_script_sha256']=sha(__file__)
    source.pop('bootstrap_seed',None) # Deterministic counts do not resample.
    base,rows=predictions('B0_IL',a.split)
    teacher,_=predictions('B0_RL',a.split);screen,_=predictions('B0_IL_screen',a.split)
    assert base.shape==teacher.shape==screen.shape and base.shape[1:]==(5,7)
    zero=1e-8;high=(screen[:,:,-1]>=.9).all(1)
    il=base[:,:,-1].mean(1);rl=teacher[:,:,-1].mean(1);both=(il<=zero)&(rl<=zero)
    result=[]
    for name in a.methods:
        values,_=predictions(name,a.split);assert values.shape==base.shape
        pdms=values[:,:,-1];mean=pdms.mean(1)
        record=dict(method=name,scenes=len(rows),inference_seeds=5,
            PDMS=float(pdms.mean()),median_scene_PDMS=float(np.median(mean)),
            zero_output_count=int((pdms<=zero).sum()),zero_scene_mean_count=int((mean<=zero).sum()),
            IL_positive_to_student_zero_scenes=int(((il>zero)&(mean<=zero)).sum()),
            IL_zero_to_student_positive_scenes=int(((il<=zero)&(mean>zero)).sum()),
            teacher_positive_to_student_zero_scenes=int(((rl>zero)&(mean<=zero)).sum()),
            teacher_zero_to_student_positive_scenes=int(((rl<=zero)&(mean>zero)).sum()),
            both_baselines_zero_scenes=int(both.sum()),both_zero_repaired_scenes=int((both&(mean>zero)).sum()),
            independent_IL_high_scenes=int(high.sum()),
            IL_high_to_student_zero_scenes=int((high&(mean<=zero)).sum()),
            IL_high_to_student_near_zero_scenes=int((high&(mean<=.01)).sum()),
            IL_high_material_drop_scenes=int((high&(mean-il<-.02)).sum()),
            IL_high_retained_scenes=int((high&(mean>=.9)).sum()))
        for index,label in [(0,'NC'),(1,'DAC'),(3,'TTC')]:
            record[label+'_new_zero_outputs_vs_IL']=int(((base[:,:,index]>zero)&(values[:,:,index]<=zero)).sum())
            record[label+'_component_decreased_outputs_vs_IL']=int((values[:,:,index]<base[:,:,index]-zero).sum())
        result.append(record)
    value=dict(status='COMPLETE_DESCRIPTIVE_COUNTS',split=a.split,policies=result,
        analysis_source=source,zero_tolerance=zero,high_score_threshold=.9,material_drop=.02,
        both_failure_definition='Both IL and T0 five-inference-seed scene means <=1e-8',
        interpretation='One row per independent training policy; counts are finite paired observations, not independent scene/noise training replications or verified per-scene expectation changes')
    target=ROOT/'analysis'/f'failures_{a.split}_{a.name}.json';write(target,value)
    write(REPORTS/'tables'/target.name,value)
    with (REPORTS/'tables'/f'failures_{a.split}_{a.name}.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(result[0]),lineterminator='\n');writer.writeheader();writer.writerows(result)
    print('REAL_PAIRED_FAILURE_AUDIT',a.name,result,flush=True)


if __name__=='__main__':main()
