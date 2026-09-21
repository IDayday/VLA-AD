from shared import *
import datetime,subprocess
def main():
    assert not (OUT/'manifests/protocol.json').exists(), 'Never overwrite frozen protocol'
    ss=scenes();assert len(ss)==len({r['token'] for r in ss})==5000
    for r in ss:assert Path(r['metric_cache_path']).exists()
    manifests=PREV/'outputs/il_rl_psi_distribution_5000/manifests'
    mm={k:models()[k] for k in CFG['models']}
    for m in mm.values():assert sha(m['checkpoint_path'])==m['sha256'];assert sha(m['runtime_planner_path'])==m['runtime_planner_sha256']
    save(OUT/'manifests/scenes.json',dict(scenes=ss))
    save(OUT/'manifests/models.json',mm)
    save(OUT/'manifests/protocol.json',dict(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),config_sha256=sha(CFG_PATH),scenes_sha256=sha(manifests/'scenes.json'),parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),inherited_cache_inventory=str(manifests/'cache_hashes.json'),inherited_cache_inventory_sha256=sha(manifests/'cache_hashes.json'),reused_sampling_code={str(p):sha(p) for p in [Path(previous.base.__file__),Path('/mnt/project/VLA-AD/tools/analysis/historical_sft_grpo_support/sample.py')]},new_random_groups=[4,5,6,7],old_random_groups=[0,1,2,3],note='Nested group-size diagnostics; native G parity required. Not historical optimizer minibatch replay. No new fitting.'))
    save(OUT/'audits/resources.json',dict(gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total,utilization.gpu','--format=csv'],text=True),cpu_count=os.cpu_count(),stopped_authorized_pressure_process=dict(pid=2601186,command='python gpu_stress.py',children=[2601286,2601287,2601288,2601289,2601290,2601291,2601292,2601293,2601294])))
    print('FROZEN',identity(),flush=True)
if __name__=='__main__':main()
