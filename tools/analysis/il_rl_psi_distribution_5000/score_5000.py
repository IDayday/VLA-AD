"""Reuse identical NAVSIM-v1 persistent scoring implementation."""
from common_5000 import *
import score, argparse
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=80);a.add_argument('--watch',action='store_true')
    score.main(a.parse_args())
