"""Public reproducible entry points for the trajectory-effect validation."""
from __future__ import annotations
import importlib
import sys

COMMANDS={
    'audit':('audit',[]),
    'collect_updates':('updates',[]),
    'build_candidates':('candidates',[]),
    'train_probes':('probe_runner',['train']),
    'evaluate_probes':('probe_runner',['evaluate']),
    'train_rl':('rl_runner',['train']),
    'evaluate_rl':('rl_runner',['evaluate']),
    'report':('report',[]),
}


def main():
    if len(sys.argv)<2 or sys.argv[1] not in COMMANDS:
        print('Usage: python -m tools.trajectory_effect {'+','.join(COMMANDS)+'} [arguments]')
        raise SystemExit(2)
    module,prefix=COMMANDS[sys.argv[1]];sys.argv=[sys.argv[0],*prefix,*sys.argv[2:]]
    importlib.import_module('.'+module,__package__).main()


if __name__=='__main__':main()
