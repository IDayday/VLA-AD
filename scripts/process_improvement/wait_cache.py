"""Wait for incremental cache producers without manufacturing missing inputs."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
import time
from scripts.process_improvement.common import *


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',default='train_25k');a=p.parse_args()
    remaining=[r['observation_path'] for r in membership(a.split)]
    while remaining:
        remaining=[path for path in remaining if not Path(path).is_file()]
        print('WAIT_CACHE',a.split,len(remaining),flush=True)
        if remaining:time.sleep(10)
