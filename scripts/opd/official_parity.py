"""Independent pinned upstream inference audit; never silently patch baseline."""
import sys,importlib.util,importlib.abc,importlib.machinery,types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.opd.common import *

REFERENCE=Path('/mnt/project/VLA-AD/outputs/recogdrive_opd_20261020/references/official_recogdrive')
REFERENCE_SHA='5831ee83c7a6a40bf6919715b96b0e9ba2b48335'

class _ReferenceLoader(importlib.machinery.SourceFileLoader):
    def get_code(self,fullname):
        # Upstream uses Python3.10 union annotations. Postpone annotations under
        # the project Python3.9 environment without editing pinned source/math.
        return compile('from __future__ import annotations\n'+self.get_data(self.path).decode(),self.path,'exec')

class _ReferenceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        prefix='_opd_upstream_reference.'
        if not fullname.startswith(prefix):return None
        base=REFERENCE/'navsim/agents/recogdrive'/fullname[len(prefix):].replace('.','/')
        file=base/'__init__.py' if base.is_dir() else base.with_suffix('.py')
        if not file.is_file():return None
        return importlib.util.spec_from_file_location(fullname,file,loader=_ReferenceLoader(fullname,str(file)),
            submodule_search_locations=[str(base)] if base.is_dir() else None)
sys.meta_path.insert(0,_ReferenceFinder())

def module(name):
    package='_opd_upstream_reference'
    if package not in sys.modules:
        p=types.ModuleType(package);p.__path__=[str(REFERENCE/'navsim/agents/recogdrive')];sys.modules[package]=p
    return __import__(package+'.'+name,fromlist=[name])

def original_agent():
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.agents.recogdrive.opd.checkpoint_io import restore
    actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REFERENCE,text=True).strip()
    assert actual==REFERENCE_SHA
    c=config();a=module('recogdrive_agent').ReCogDriveAgent(TrajectorySampling(time_horizon=4,interval_length=.5),
        vlm_path=c['vlm'],checkpoint_path=c['student_init'],cam_type='single',vlm_type='internvl',dit_type='small',
        vlm_size='small',sampling_method='ddim',cache_hidden_state=False,cache_mode=False,grpo=False,train_backbone=False)
    a.action_head.config.use_action_aware_aux=False
    restore(a.action_head,c['student_init']);a.eval();a.requires_grad_(False)
    return a

def main():
    import torch
    from scripts.opd.inputs import features,encode,agent
    from navsim.agents.recogdrive.opd.checkpoint_io import build_head,restore
    configure();a=original_agent();corrected=agent(official=True);records=[]
    c=config();models={k:build_head(c[v],frozen=True) for k,v in [('IL','student_init'),('RL','teacher_init')]}
    for row in rows('train')[:4]+rows('cal')[:4]:
        f=features(a,row);h=encode(a,f)
        old=torch.load('/mnt/project/VLA-AD/outputs/trajectory_effect_validation_20261019/observations/'+row['token']+'.pt',map_location='cpu',weights_only=False)
        assert all(torch.equal(f[k].cpu(),old['features'][k].cpu()) for k in f)
        x,i=conditioning(dict(tokens=h,features=f));ox,oi=conditioning(old)
        corrected_features=features(corrected,row);ch=encode(corrected,corrected_features)
        assert all(torch.equal(f[k].cpu(),corrected_features[k].cpu()) for k in f)
        correction_error=float((h.float()-ch.float()).abs().max())
        assert correction_error==0.,correction_error
        item=dict(token=row['token'],official_shape=list(h.shape),modified_shape=list(old['tokens'].shape),corrected_token_error=correction_error)
        for label,path in [('IL','student_init'),('RL','teacher_init')]:
            restore(a.action_head,c[path]);seed=seed_for(row['token'],1101)
            seeds(seed)
            with torch.no_grad():reference=a.action_head.get_action(x,i)['pred_traj']
            seeds(seed)
            with torch.no_grad():local=models[label].get_action(x,i)['pred_traj']
            seeds(seed)
            with torch.no_grad():previous=models[label].get_action(ox,oi)['pred_traj']
            item[label]=dict(head_parity_max_error=float((reference-local).abs().max()),
                            observation_contract_trajectory_max_error=float((reference-previous).abs().max()),
                            official_trajectory=reference.cpu().tolist(),modified_trajectory=previous.cpu().tolist())
        records.append(item);print(item['token'],item['official_shape'],item['modified_shape'],item['IL']['head_parity_max_error'],item['IL']['observation_contract_trajectory_max_error'],flush=True)
    head=max(r[k]['head_parity_max_error'] for r in records for k in ('IL','RL'))
    observation_error=max(r[k]['observation_contract_trajectory_max_error'] for r in records for k in ('IL','RL'))
    write(ROOT/'audit/official_forward_parity.json',dict(reference_sha=REFERENCE_SHA,records=records,
        head_parity=head<=1e-6,observation_contract_parity=observation_error<=1e-6,corrected_token_parity=True,
        max_head_error=head,max_observation_trajectory_error=observation_error,owner=identity()))

if __name__=='__main__':main()
