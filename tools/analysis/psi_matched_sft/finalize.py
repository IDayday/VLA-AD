from common_matched import *
def main():
    while not (OUT/'audits/data_ready.json').exists():time.sleep(10)
    py='/root/miniconda3/envs/navsim/bin/python';root=WORK/'tools/analysis/psi_matched_sft'
    for name,args in [('analyze.py',[]),('supervision_audit.py',[]),('figures.py',[]),('test_matched.py',[]),('audit.py',['--final']),('report.py',[])]:
        subprocess.run([py,str(root/name)]+args,cwd=WORK,check=True)
    save(OUT/'audits/finalize_complete.json',dict(status='PASS',protocol_hash=identity()))
if __name__=='__main__':main()
