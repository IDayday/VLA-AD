"""Print one scene's saved expert and exact frozen configuration."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from interface_teacher import Library,DecisionTable


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--token',required=True);p.add_argument('--library',default='D_C8_double')
    p.add_argument('--population',choices=['train','test'],default='train');p.add_argument('--allow-test-diagnostic',action='store_true');a=p.parse_args()
    lib=Library(a.library);table=DecisionTable(a.library,a.population,allow_test=a.allow_test_diagnostic)
    row=table.row(a.token);ch=int(row['choice'])
    print(json.dumps(dict(library=a.library,population=a.population,decision=row,
        frozen_expert=lib.config['experts'][ch-1] if ch else dict(choice=0,method='Base'),
        raw_shift_factor=lib.config['raw_shift_factor'],cap_factor=lib.config['cap_factor'],space=lib.config['space']),ensure_ascii=False,indent=2))
