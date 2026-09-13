"""Honor AGENT.md after GPU experiments; no signals are sent to any process."""
import importlib.util
import subprocess
from common_v3 import *

def main():
    assert (OUT / 'manifests/audit_G_null.json').exists(), 'V3 GPU work is not complete'
    stress = ROOT.parent / 'gpu_stress.py'
    spec = importlib.util.spec_from_file_location('pressure_readonly', stress)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    gpu_rows = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
        text=True).splitlines()
    observations = []
    for line in gpu_rows:
        index, uuid = [s.strip() for s in line.split(',')]
        # The helper only reads the selected physical device's occupancy.
        os.environ['CUDA_VISIBLE_DEVICES'] = uuid
        real, pressure = mod.gpu_cuda_occupants(0)
        row = dict(index=int(index), uuid=uuid, real_pids=real, pressure_pids=pressure)
        if real or pressure:
            row['action'] = 'leave_existing_processes_untouched'
        else:
            command = [PY, str(stress), '--gpus', '0', '--memory-gb', '75',
                       '--status-interval', '60', '--yield-check-interval', '2']
            with (OUT / 'logs' / f'pressure_restore_gpu{index}.log').open('a') as log:
                p = subprocess.Popen(command, env=dict(os.environ), cwd=ROOT,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            row.update(action='start_cooperatively_yielding_pressure', pid=p.pid,
                       command=command)
        observations.append(row)
    save(OUT / 'manifests/GPU_PRESSURE_RESTORED.json',
         dict(completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              instructions='AGENT.md section 1.1; never stop unrelated processes',
              observations=observations))
    print(json.dumps(observations, indent=2))

if __name__ == '__main__':
    main()
