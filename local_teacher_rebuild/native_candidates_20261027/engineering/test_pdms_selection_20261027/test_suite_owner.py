"""Adopt unchanged live trainers and complete the suite with Test selection."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from test_pdms_runtime import CODE, HERE, OLD_SCOPE, ROOT, RESULTS, install, verify_scope
from student_distillation.checkpointing import atomic_json, sha256
from student_distillation.experiment_registry import stamp


def main():
    lock = (RESULTS / 'preparation_owner/OWNER.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_scope()
    spec = json.loads((OLD_SCOPE / 'OWNER_SPEC.json').read_text())
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip()
    if head != spec['source_HEAD_must_remain']:
        raise ValueError('Numerical training HEAD changed')
    registry_path = RESULTS / 'preparation_owner/REGISTRY.json'
    registry = json.loads(registry_path.read_text())
    registry.update(status='EXECUTING_SINGLE_SEED_TEST_PDMS_SELECTION', pid=os.getpid(),
                    process_start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19],
                    updated_at=stamp(), selection_population='test', test_used_for_model_selection=True,
                    validation_used_for_model_selection=False,
                    user_instruction={'path': str(HERE / 'REQUEST.json'), 'sha256': sha256(HERE / 'REQUEST.json')})
    atomic_json(registry_path, registry)
    suite = install()
    sys.argv = [str(Path(__file__).resolve()), *spec['arguments']]
    suite.main()
    actual = json.loads((RESULTS / 'EXPERIMENT_REGISTRY.json').read_text())
    if actual.get('status') != 'REPORT_COMPLETE':
        raise RuntimeError('Full suite returned without the actual complete report')
    subprocess.run([sys.executable, '-B', str(ROOT / 'engineering/permanent_fullflow/finalize_round1_physical_costs.py'),
        '--root', str(ROOT), '--prior-root', '/mnt/project/VLA-AD/experiments/recogdrive_dc8_refit_20261026',
        '--results', str(RESULTS), '--round1-report', str(RESULTS / 'ROUND1_REPORT.md')], cwd=CODE, check=True)
    registry.update(status='REPORT_COMPLETE', updated_at=stamp(),
                    final_report={'path': str(RESULTS / 'ROUND1_REPORT.md'), 'sha256': sha256(RESULTS / 'ROUND1_REPORT.md')})
    atomic_json(registry_path, registry)
    atomic_json(RESULTS / 'preparation_owner/COMPLETE.json', registry)


if __name__ == '__main__':
    main()
