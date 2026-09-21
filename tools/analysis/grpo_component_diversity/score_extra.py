"""Reuse the exact persistent, scalar-equivalent NAVSIM evaluator."""
from shared import *
import score,argparse
def main():
    a=argparse.ArgumentParser();a.add_argument('--workers',type=int,default=80);a.add_argument('--watch',action='store_true');args=a.parse_args()
    identity();score.OUT=OUT;score.identity=identity;score.scenes=scenes
    score.main(args)
if __name__=='__main__':main()
