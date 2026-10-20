"""Online image-to-raw-VLM parity against both reused and newly produced cache."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import inspect
import time
from scripts.process_improvement.common import *


def main():
    import torch
    from scripts.opd.inputs import agent,features,encode
    configure();started=time.time();model=agent(official=True)
    lookup={r['token']:r for r in membership('train_full')+membership('cal')}
    selected=membership('legacy_4k')[:2]
    for rank in [0,8,16]:
        path=ROOT/'cache_receipts'/f'all_{rank}_of_24.jsonl'
        first=json.loads(path.open().readline())
        selected.append(lookup[first['token']])
    records=[]
    for row in selected:
        actual=encode(model,features(model,row));expected=observation(row)['tokens']
        assert torch.equal(actual,expected),(row['token'],float((actual.float()-expected.float()).abs().max()))
        records.append(dict(token=row['token'],cache_path=row['observation_path'],raw_token_max_error=0.))
    owner=identity()
    write(ROOT/'tests'/('online_input_'+owner['host']+'.json'),dict(status='PASS',records=records,owner=owner,
          seconds=time.time()-started,backbone_source_sha256=sha(inspect.getfile(model.backbone.__class__)),
          transformers_runtime=__import__('transformers').__version__,torch_runtime=torch.__version__))
    print('ONLINE_INPUT_HOST_PASS',owner['host'],records,flush=True)


if __name__=='__main__':main()
