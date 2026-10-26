"""Fresh-process A800 reproduction with strict conditions and repository inputs.

Explicitly configures this new diagnostic child to the captured PPU settings.
Never run inside an existing training/inference process. No language/action
forward, teacher selection, package installation or model-file modification.
"""
from pathlib import Path
import argparse
import json
import runpy
import sys
import traceback


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True,help='Extracted verified v2 bundle')
    parser.add_argument('--frozen-repo',required=True,help='Teacher source at frozen 6bb7119')
    parser.add_argument('--vlm',required=True,help='Original VLM directory, not a converted/merged model')
    parser.add_argument('--output-dir',required=True,help='New independent diagnostic output directory')
    parser.add_argument('--skip-visual-suffix',action='store_true')
    args=parser.parse_args()
    sys.dont_write_bytecode=True
    folder=Path(__file__).resolve().parent
    bundle=Path(args.bundle).resolve();repo=Path(args.frozen_repo).resolve();vlm=Path(args.vlm).resolve()
    out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    ns=runpy.run_path(str(folder/'controlled_conditions.py'))
    try:
        configured=ns['configure_fresh_receiver'](folder/'PPU_CONTROLLED_CONFIGURATION.json')
        (out/'FRESH_PROCESS_SETTINGS.json').write_text(json.dumps(configured,ensure_ascii=False,indent=2)+'\n')
        sys.path.insert(0,str(bundle));sys.path.insert(0,str(repo/'vendor/upstream'))
        sys.modules['wandb']=None
        from transformers import AutoModel
        from model.recogdrive_backbone import RecogDriveBackbone
        original=AutoModel.from_pretrained
        def non_flash(*args,**kwargs):
            kwargs['use_flash_attn']=False
            return original(*args,**kwargs)
        AutoModel.from_pretrained=non_flash
        try:backbone=RecogDriveBackbone('internvl',str(vlm),'cuda')
        finally:AutoModel.from_pretrained=original
        load=runpy.run_path(str(bundle/'load_inputs.py'))['load_inputs']
        model_inputs,auxiliary,tensors,metadata=load(bundle/'inputs',device='cuda')
        result=ns['run_controlled_check'](backbone.model,model_inputs,auxiliary=auxiliary,
            role_tensors=tensors,bundle_dir=bundle,vlm_dir=vlm,frozen_repo=repo,folder=folder,
            output_dir=out,verify_full_vision=not args.skip_visual_suffix)
        return 0 if result.get('portable_gate_passed') else 2
    except Exception as exc:
        failure={'status':'FAILED_WITHOUT_ALIGNMENT_CLAIM','exception':type(exc).__name__,
                 'message':str(exc),'traceback':traceback.format_exc(),
                 'full_chain_alignment_complete':False,'bulk_generation_resumed':False}
        (out/'A800_CONTROLLED_FAILURE.json').write_text(json.dumps(failure,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(failure,ensure_ascii=False,indent=2))
        return 1


if __name__=='__main__':
    raise SystemExit(main())
