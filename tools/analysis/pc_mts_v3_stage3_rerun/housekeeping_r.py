"""Apply AGENT.md 1.1 only to idle devices after this campaign's GPU work."""
from common_r import *
import subprocess
def main():
    assert len(list((OUT/'manifests').glob('train_*.json')))==14
    assert len(list((OUT/'manifests').glob('eval_*.json')))==224
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    occupants=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    records=[]
    for row in rows:
        index,uid,mem=[v.strip() for v in row.split(',')]
        if int(mem)>0 or uid in occupants:
            records.append(dict(index=int(index),uuid=uid,action='leave_existing_occupants_untouched'));continue
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=uid);path=OUT/'logs'/f'idle_pressure_{socket.gethostname()}_gpu{index}.log'
        command=[sys.executable,'/mnt/project/gpu_stress.py','--gpus','0','--memory-gb','75','--status-interval','60','--yield-check-interval','2']
        with open(path,'a') as f:p=subprocess.Popen(command,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        records.append(dict(index=int(index),uuid=uid,action='restore_disposable_idle_pressure',pid=p.pid))
    save(OUT/'manifests'/f'housekeeping_{socket.gethostname()}.json',dict(nonexperimental=True,policy='AGENT.md section 1.1',devices=records))
if __name__=='__main__':main()
