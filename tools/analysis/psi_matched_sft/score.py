"""Shared persistent scoring implementation, in new namespace only."""
from common_matched import *
import importlib.util
def main():
    setup_legacy();legacy.scenes=lambda:scenes()
    spec=importlib.util.spec_from_file_location('historical_scoring',MAIN/'tools/analysis/historical_sft_grpo_support/score.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    import argparse
    m.main(argparse.Namespace(workers=64,watch=True))
if __name__=='__main__':main()
