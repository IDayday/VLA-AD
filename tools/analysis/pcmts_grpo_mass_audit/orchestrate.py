"""Background pipeline: A-only preparation overlaps inference; B waits for seal."""
from common_mass import *
import time,subprocess,concurrent.futures
from select_pools import task,variants,seal_selection

def main():
    rows=scenes();pending={};done={};start=time.time()
    with concurrent.futures.ProcessPoolExecutor(24) as ex:
        while len(done)<1000:
            busy={s['token'] for s in pending.values()}
            for s in rows:
                t=s['token']
                if t in done or t in busy:continue
                paths=[OUT/'cache/raw'/f'{t}.npz',OUT/'cache/scores/raw'/f'{t}.npz',OUT/'cache/rollouts/A'/f'{t}.npz',OUT/'cache/scores/rollouts/C'/f'{t}.npz']
                if not all(p.exists() for p in paths):continue
                if len(pending)>=48:break
                pending[ex.submit(task,s)]=s
            for f in [f for f in pending if f.done()]:
                s=pending.pop(f);done[s['token']]=f.result()
            save(OUT/'manifests/selection_progress.json',dict(done=len(done),pending=len(pending),seconds=time.time()-start,B_read=False))
            print('A selection',len(done),'/1000','pending',len(pending),flush=True)
            if len(done)<1000:time.sleep(10)
    hashes=[done[s['token']] for s in rows]
    seal_selection(done,time.time()-start)
    print('ALL SELECTIONS SEALED',digest(hashes),flush=True)
    while not all((OUT/'cache/scores/rollouts/B'/f'{s["token"]}.npz').exists() for s in rows):time.sleep(10)
    for stage in ['analyze_B.py','summarize.py','figures.py','examples.py','coordinate_audit.py','ddv2_lineage.py','audit.py','final_review.py','write_report.py']:
        command=[sys.executable,'-u',str(Path(__file__).parent/stage)];print('BEGIN',stage,utc(),flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
    save(OUT/'manifests/pipeline_finished.json',dict(completed_at=utc(),status='AWAITING_FINAL_REVIEW_TESTS_COMMIT_PUSH'))
if __name__=='__main__':main()
