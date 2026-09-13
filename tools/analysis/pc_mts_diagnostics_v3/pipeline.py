"""Explicit sequential execution; negative effects never change frozen parameters."""
import subprocess,argparse
from common_v3 import *
HERE=ROOT/'tools/analysis/pc_mts_diagnostics_v3'
def run(name,argv):
    command=[PY,str(HERE/'run_phase.py'),name,'--']+argv
    print('START',name,flush=True);rc=subprocess.call(command,cwd=ROOT);assert rc==0,(name,rc)
def python(name,script,*args):run(name,[PY,str(HERE/script),*args])
def gpu(name,script,*args):run(name,[str(Path(PY).parent/'torchrun'),'--standalone','--nproc_per_node=8',str(HERE/script),*args])
def main():
    while not (OUT/'manifests/C_gpu_execution.json').exists():time.sleep(5)
    assert read(OUT/'manifests/C_gpu_execution.json')['returncode']==0
    python('C_statistics','analyze_c.py')
    gpu('D_gpu','gradient.py')
    python('D_statistics','analyze_d.py')
    python('tests_before_E','tests.py')
    python('E_training','launch_training.py','sft')
    gpu('E_evaluation','evaluate.py','sft')
    python('E_scoring','scoring.py','sft')
    python('E_statistics','analyze_training.py','sft')
    # Original algorithm parity is audited before any GRPO update.
    python('F_recipe_audit','grpo_audit.py')
    python('F_training','launch_training.py','grpo')
    gpu('F_evaluation','evaluate.py','grpo')
    python('F_scoring','scoring.py','grpo')
    python('F_statistics','analyze_training.py','grpo')
    gpu('G_progressive','progressive.py')
    python('G_statistics','analyze_progressive.py')
    gpu('G_null','null_support.py')
    python('G_null_statistics','analyze_null.py')
    save(OUT/'manifests/pipeline_complete.json',dict(identity=identity(),complete=True))
if __name__=='__main__':main()
