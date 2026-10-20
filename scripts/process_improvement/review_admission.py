"""Explicitly review a pre-launch GPU rejection; never retry running/failed work.

All old directories are preserved. This CLI only handles the exact atomic
GPU-occupied rejection that occurs before dispatch creates any worker. Network
errors or an ambiguous launch require separate process-identity investigation.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import time

from scripts.process_improvement.common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--jobs',nargs='+',required=True);a=p.parse_args()
    identity();path=ROOT/'controller/admission_retry_map.json'
    value=read(path) if path.exists() else dict(current={},reviews=[])
    for original in a.jobs:
        previous=value['current'].get(original,original);directory=ROOT/'jobs'/previous
        failure=read(directory/'admission_failure.json')
        if 'RuntimeError: GPU occupied:' not in failure['error']:
            raise ValueError('Ambiguous admission failure needs a separate process audit: '+previous)
        if any((directory/name).exists() for name in ['launch.json','owner.json','complete.json']) or list(directory.glob('child_*.json')):
            raise ValueError('A potentially launched/finished worker cannot use this review: '+previous)
        number=1+sum(r['original']==original for r in value['reviews'])
        replacement=f'reviewed_retry{number}_{original}'
        if (ROOT/'jobs'/replacement).exists():raise FileExistsError('Preserve prior retry identity: '+replacement)
        value['reviews'].append(dict(original=original,rejected_attempt=previous,replacement=replacement,
            failure_sha256=sha(directory/'admission_failure.json'),spec_sha256=sha(directory/'spec.json'),
            reason='Atomic GPU-occupied check rejected before Popen; no model, scoring or optimizer worker launched',
            original_artifacts_untouched=True,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
        value['current'][original]=replacement
    write(path,value)
    print('EXPLICIT_PRELAUNCH_ADMISSION_REVIEW',value['current'],flush=True)


if __name__=='__main__':main()
