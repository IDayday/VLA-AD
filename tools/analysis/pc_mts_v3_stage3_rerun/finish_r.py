"""Run the fixed final analyses after all committed artifacts exist."""
from common_r import *
import subprocess
def main():
    while True:
        counts=[len(list((OUT/'manifests').glob('train_*.json'))),len(list((OUT/'manifests').glob('eval_*.json'))),len([p for p in (OUT/'cache/scores').glob('*/*/*.npz') if len(p.stem)==16])]
        print('Completion counters: train/eval shards/scored scenes',counts,flush=True)
        if counts==[14,224,16800]:break
        time.sleep(20)
    for name in ['analyze_r.py','figures_r.py','audit_r.py','report_r.py','tests.py']:
        print('RUN',name,flush=True);subprocess.run([sys.executable,str(Path(__file__).parent/name)],check=True,cwd=ROOT)
    save(OUT/'manifests/COMPLETED.json',dict(identity(),completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),report_sha256=sha(ROOT/'reports/PC_MTS_V3_FORMAL_STAGE3_GRPO_RERUN_20260913.md')))
if __name__=='__main__':main()
