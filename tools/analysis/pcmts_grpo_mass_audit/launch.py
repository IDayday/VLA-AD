"""Launch genuine scene-sharded inference only; never occupy idle GPUs."""
from common_mass import *
import subprocess, argparse
def main():
    p=argparse.ArgumentParser();p.add_argument('kind',choices=['native','external']);p.add_argument('--ranks',nargs='+',type=int,required=True);p.add_argument('--source',choices=['ddv2','drivor']);a=p.parse_args()
    env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',NUPLAN_MAPS_ROOT='/mnt/navsim/maps',NUPLAN_MAP_VERSION='nuplan-maps-v1.0')
    rows=[]
    for rank in a.ranks:
        assert 0<=rank<8
        name=f'{a.kind}_{a.source or ""}_rank{rank}';path=OUT/'logs'/f'{name}.log';path.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u',str(Path(__file__).parent/('sample_native.py' if a.kind=='native' else 'external_candidates.py'))]
        command+=['--rank',str(rank),'--world','8'] if a.kind=='native' else [a.source,'--rank',str(rank)]
        with open(path,'a') as f:process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        rows.append(dict(pid=process.pid,rank=rank,command=command,log=str(path),host=os.uname().nodename,started_at=utc()))
    save(OUT/'manifests'/f'launch_{a.kind}_{a.source or "native"}_{a.ranks[0]}.json',rows)
    print(rows)
if __name__=='__main__':main()
