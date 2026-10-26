"""Capture A800 first-block operator fingerprints from the verified real PPU embedding."""
from pathlib import Path
import sys
ROOT=Path('/mnt/project/recogdrive-teacher2-a800-20261026')


def main(verify_full_vision=False):
    sys.path.insert(0,str(ROOT/'diagnostics/torch240_packages'))
    import torch
    import importlib.util
    from cache_common import REPO,read,write,stamp
    sys.path.insert(0,str(REPO))
    out=ROOT/'diagnostics/ppu_audit_20261027_v2'/('FIRST_VISION_BLOCK_WITH_SUFFIX' if verify_full_vision else 'FIRST_VISION_BLOCK')
    if (out/'RESULT.json').exists():raise FileExistsError(out/'RESULT.json')
    from interface_teacher.runtime import bootstrap,Runtime
    bootstrap(ROOT/'paths.local.json',out)
    from transformers import AutoModel
    original=AutoModel.from_pretrained
    def load(*a,**kw):kw['use_flash_attn']=False;return original(*a,**kw)
    AutoModel.from_pretrained=load
    engine=Runtime('raw')
    from vlm_core import prepare
    features=torch.load(ROOT/'diagnostics/backend/inputs/0002267a294e52a7.pt',map_location='cpu',weights_only=False)
    inputs,_=prepare(engine.backbone,features)
    script=ROOT/'ppu_alignment_check/trace_first_vision_block.py'
    spec=importlib.util.spec_from_file_location('first_block_trace',script);trace=importlib.util.module_from_spec(spec);spec.loader.exec_module(trace)
    report=trace.trace_first_vision_block(engine.backbone.model,inputs['pixel_values'],ROOT/'diagnostics/ppu_audit_20261027_v2/ppu_native_no_bias.npz',verify_full_vision=verify_full_vision)
    injection=read(ROOT/'diagnostics/ppu_audit_20261027_v2/C_INJECTION/RESULT.json')['cases'][1]
    expected=injection['layer_fingerprints']['vision_0']
    assert report['events'][-1]['tensor']==expected,'One-block isolated execution differs from actual full VLM first block'
    if verify_full_vision:assert report['vision_last']==injection['fingerprints']['vision_last']
    write(out/'A800_REFERENCE.json',report)
    write(out/'RESULT.json',dict(status='COMPLETE_ONE_BLOCK_TRACE_FROM_PPU_EMBEDDING',time=stamp(),full_forward_first_block_match=True,reference=str(out/'A800_REFERENCE.json'),bulk_generation_resumed=False))


if __name__=='__main__':main('--verify-full-vision' in sys.argv)
