"""Read-only, allowlisted resource inventory; never kills or creates workloads."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import concurrent.futures
import subprocess
from scripts.process_improvement.common import *


def inventory():
    policy_path = PROJECT.parent / 'server_dispatch_policy.json'
    policy = read(policy_path)
    hosts = sorted(set(policy['allowed_hosts']) - set(policy.get('global_dispatch_prohibitions', {})))
    command = """hostname; date -u +%FT%TZ; nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv,noheader; nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory,process_name --format=csv,noheader; ps -eo pid,ppid,uid,lstart,args | rg 'zt-idle-pressure|scripts/process_improvement|python.*(train|campaign|serve)' | head -45; cat /sys/fs/cgroup/cpu.max; free -m; df -h /mnt/project /; test -x /root/miniconda3/envs/navsim/bin/python && /root/miniconda3/envs/navsim/bin/python -c 'import torch;print(torch.__version__,torch.cuda.is_available())'"""
    def one(host):
        args = ['bash', '-c', command] if host == 'training-vla-zt' else ['ssh', '-oBatchMode=yes', '-oConnectTimeout=12', host, command]
        try:
            out = subprocess.run(args, capture_output=True, text=True, timeout=75)
            return host, dict(returncode=out.returncode, stdout=out.stdout, stderr=out.stderr)
        except Exception as exc:
            return host, dict(error=repr(exc))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        result = dict(pool.map(one, hosts))
    write(ROOT / 'audit/resources.json', dict(policy_sha256=sha(policy_path), hosts=result))
    for host, result in result.items():
        print(host, result.get('returncode', result.get('error')), '\n', result.get('stdout', '')[:1800])


if __name__ == '__main__':
    inventory()
