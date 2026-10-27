"""Bootstrap the retained-run Test gate in future evaluation subprocesses."""
import argparse,runpy,sys
from single_seed_scope import install
p=argparse.ArgumentParser();p.add_argument('--module',choices=['student_distillation.evaluate'],required=True)
args,rest=p.parse_known_args();rest=rest[1:] if rest and rest[0]=='--' else rest
install();sys.argv=[args.module,*rest];runpy.run_module(args.module,run_name='__main__')
