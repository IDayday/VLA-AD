"""Allow a new physical host only after real same-world migration parity."""
from scripts.process_improvement.common import ROOT,read
from scripts.process_improvement.qualify_grpo_host import sources
from scripts.process_improvement.qualify_grpo_host8 import sources as host8_sources


def resume_hosts(run,world):
    original=read(run/'config.json')['owner']['host']
    if world not in [4,8]:return [original]
    gate=ROOT/('tests/native_GRPO_cross_host_resume.json' if world==4 else 'tests/native_GRPO_cross_host_resume8.json')
    if not gate.exists() or not (run/'paused.json').exists():return [original]
    value=read(gate)
    if (value.get('status')!='PASS' or value.get('world')!=world or
            value.get('global_batch')!=32 or not value.get('optimizer_exact') or
            not value.get('all_rank_RNG_exact') or not value.get('cursor_reference_budget_exact') or
            value.get('validated_initializations')!=['IL','RL','WARM'] or
            value.get('qualification_source_hashes')!=(sources() if world==4 else host8_sources())):return [original]
    if original!=value['destination_host']:return [original]
    return [original,value['source_host']]
