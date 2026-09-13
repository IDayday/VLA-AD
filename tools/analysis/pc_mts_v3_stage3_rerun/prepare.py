"""Freeze formal settings and derive loader-format initializations without changing tensors."""
from common_r import *
import torch, subprocess
def main():
    frozen=OUT/'manifests/frozen.json'
    assert not frozen.exists(), 'Already frozen; do not regenerate the primary protocol'
    assert len(tokens('train'))==700 and len(tokens('holdout'))==300
    assert not set(tokens('train')) & set(tokens('holdout'))
    sources=[CFG['formal_launcher'],CFG['formal_entrypoint'],
      'navsim/planning/script/config/training/default_training.yaml',
      'navsim/planning/script/config/common/agent/recogdrive_agent.yaml',
      'navsim/agents/recogdrive/recogdrive_agent.py',
      'navsim/agents/recogdrive/recogdrive_diffusion_planner.py',
      'navsim/agents/recogdrive/utils/lr_scheduler.py',
      'navsim/planning/training/agent_lightning_module.py']
    records=[]
    for m in CFG['methods']:
        for sd in CFG['seeds']:
            source=Path(v3.v1.models()[0]['checkpoint_path']) if m=='official_il' else V3/'checkpoints/sft'/run_name(m,sd)/'step0200.pt'
            state=state_only(source);dest=initial_path(m,sd);dest.parent.mkdir(parents=True,exist_ok=True)
            torch.save({'state_dict':{'agent.action_head.'+k:v for k,v in state.items()},
                        'provenance':{'source':str(source),'source_sha256':sha(source),'tensor_changes':False}},dest)
            loaded=state_only(dest);assert all(torch.equal(v,loaded[k]) for k,v in state.items())
            records.append(dict(method=m,seed=sd,source=str(source),source_sha256=sha(source),path=str(dest),sha256=sha(dest),tensors=len(state)))
    save(OUT/'manifests/initializations.json',records)
    save(frozen,dict(config_sha256=sha(CONFIG),split_sha256=sha(V3/'manifests/splits.json'),base_commit=CFG['base_commit'],
      frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_sha256={p:sha(ROOT/p) for p in sources},
      train_scenes=700,holdout_scenes=300,ddp_samples_per_rank=88,steps_per_epoch=11,total_updates=110,
      padding_samples_per_epoch=4,holdout_used_in_updates=False,hardware_execution_only_may_change=True))
    print('Frozen',identity(),flush=True)
if __name__=='__main__':main()
