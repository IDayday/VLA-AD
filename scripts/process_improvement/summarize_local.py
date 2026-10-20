"""Posthoc position/variance/clamp diagnostics from saved real local outputs.

This performs no optimization, selection, candidate resampling, or new scoring.
Trace statistics use only the predefined hash subset and are labeled as such.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
from collections import defaultdict
import numpy as np
from scripts.process_improvement.common import *


def main():
    import torch
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--split',required=True)
    a=p.parse_args();plan=diagnostic_plan()
    choice=read(ROOT/'selection'/plan['choice_file'])['chosen']
    temperature,rho=choice['temperature'],choice['rho']
    rows=membership(a.split);root=ROOT/'diagnostics'/a.name
    grouped=defaultdict(list);payloads={}
    for row in rows:
        path=root/'scenes'/f"{row['token']}.json"
        if not path.is_file():raise ValueError('Complete population required: '+row['token'])
        payloads[str(path)]=sha(path)
        records=[r for r in read(path)['records'] if r['temperature_base']==temperature and r['rho']==rho]
        assert len(records)==30  # Three sources, five positions, two banks.
        for record in records:
            values=[record[k] for k in ['selection_gain','projection_gain','guarded_gain','accepted','flat','ess','radius','fit_value_range']]
            scores=np.asarray(record['P0'])[:,-1]
            values.append(float(scores.var(ddof=1)))
            grouped[(record['source'],record['index'])].append(values)
    fields=['selection_gain','projection_gain','guarded_gain','accepted_fraction','flat_fraction','ESS',
            'standardized_shift_RMS','fit_candidate_mean_value_range','reference_suffix_return_variance']
    results=[]
    for (source,index),values in sorted(grouped.items()):
        assert len(values)==2*len(rows)
        results.append(dict(source=source,index=index,targets=len(values),mean=dict(zip(fields,np.asarray(values).mean(0).tolist()))))
    traces=defaultdict(list);trace_sources={};trace_tokens=set()
    for path in sorted((root/'traces').glob(f'*_{temperature}_{rho}.pt')):
        data=torch.load(path,map_location='cpu',weights_only=False)
        record=data['record'];raw=data['raw_noise'].numpy();candidates=data['candidates'].numpy()
        fit=np.asarray(data['fit_scores'])[:,:,-1];independent=np.asarray(data['independent_scores'])[:,:,-1]
        std=data['std'].numpy();reference=data['reference_mean'].numpy();proposed=data['proposed_mean'].numpy()
        shift=proposed-reference
        traces[(record['source'],record['index'])].append([
            float((np.abs(raw)>1).mean()),float((np.abs(candidates)>=1).mean()),
            float(std.min()),float(std.max()),float(np.sqrt((shift**2).mean())),
            float((shift==0).mean()),float(independent.var(axis=1,ddof=1).mean()),
            float(np.abs(fit.mean(1)-independent.mean(1)).mean())])
        trace_tokens.add(data['row']['token']);trace_sources[str(path)]=sha(path)
    trace_fields=['preclip_candidate_noise_fraction','candidate_coordinates_at_or_beyond_unit_bound',
                  'native_std_parameter_min','native_std_parameter_max','rounded_proposed_mean_shift_RMS',
                  'unchanged_mean_coordinates_fraction','within_candidate_suffix_return_variance',
                  'fit_vs_independent_candidate_mean_absolute_gap']
    trace_results=[dict(source=source,index=index,saved_states=len(values),
                        mean=dict(zip(trace_fields,np.asarray(values).mean(0).tolist())))
                   for (source,index),values in sorted(traces.items())]
    result=dict(status='COMPLETE',name=a.name,split=a.split,scenes=len(rows),parents=len({r['driver_record'] for r in rows}),
                selected_temperature=temperature,selected_rho=rho,by_position=results,
                trace_hash_subset=dict(unique_scenes=len(trace_tokens),by_position=trace_results,
                    rule='Original fixed hash digest([scene_token,case]) modulo64 equals0; bank0 only',
                    interpretation='Subset-only diagnostic; unit-bound saturation is final clamp only at index4; earlier coordinates may exceed bounds'),
                provenance=dict(script_sha256=sha(__file__),population_sha256=sha(ROOT/'manifests'/f'{a.split}.json'),
                    choice_sha256=sha(ROOT/'selection'/plan['choice_file']),scene_payloads_sha256=digest(payloads),trace_sources=trace_sources),
                interpretation='Descriptive complete-position coverage and predefined trace subset; no new significance tests, no revised calibration or targets')
    write(ROOT/'analysis'/f'{a.name}_details.json',result)
    write(REPORTS/'tables'/f'{a.name}_details.json',result)
    print('REAL_LOCAL_DETAIL_AUDIT',a.name,len(rows),len(trace_tokens),flush=True)


if __name__=='__main__':main()
