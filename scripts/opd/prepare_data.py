import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import argparse
from scripts.opd.common import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--manifest-only',action='store_true');a=p.parse_args();cfg=config(a.config)
    from scripts.opd.splits import BUNDLE,population,verify
    if (ROOT/'protocol.lock.json').exists():
        verify();print('LOCKED_DATA_UNCHANGED: use a fresh OPD_RUN_ROOT to rebuild');return
    source=PROJECT/'outputs/trajectory_effect_validation_20261019/manifests';mapping={'train':'D_fit','cal':'D_cal','dev':'D_audit'};logs={};parents={}
    for split,old in mapping.items():
        data=population(split) if (BUNDLE/f'{split}.json').exists() else read(source/f'{old}.json')
        assert all(r['split']=='navtrain' for r in data)
        for r in data:
            if not Path(r['metric_cache_path']).is_file():r['metric_cache_path']=str(ROOT/'metric_cache'/split/f"{r['token']}.pkl")
        if not a.manifest_only:assert all(Path(r['metric_cache_path']).is_file() for r in data)
        if cfg.get('official_conditioning'):
            for r in data:r['observation_path']=str(ROOT/'observations'/split/f"{r['token']}.pt")
        logs[split]={r['log_name'] for r in data};parents[split]={r['driver_record'] for r in data}
        write(ROOT/'manifests'/f'{split}.json',data)
    for a in logs:
        for b in logs:
            if a!=b:assert not logs[a]&logs[b] and not parents[a]&parents[b]
    if '--manifest-only' in sys.argv:return
    # Read tensor shapes from PyTorch ZIP metadata without rereading all cached bytes.
    import zipfile,io
    class ShapeReader(pickle.Unpickler):
        def persistent_load(self,pid):return None
        def find_class(self,module,name):
            if module=='torch._utils' and name.startswith('_rebuild_tensor'):
                return lambda storage,offset,size,stride,*args:dict(shape=list(size))
            return super().find_class(module,name)
    lengths={}
    for split in mapping:
        for row in rows(split):
            with zipfile.ZipFile(row.get('observation_path',str(Path(cfg['observation_cache'])/f"{row['token']}.pt"))) as archive:
                meta=ShapeReader(io.BytesIO(archive.read(next(n for n in archive.namelist() if n.endswith('/data.pkl'))))).load()
            lengths[row['token']]=meta['tokens']['shape'][1]
    write(ROOT/'manifests/token_lengths.json',lengths)
    import torch
    from pyquaternion import Quaternion
    from nuplan.common.actor_state.state_representation import StateSE2
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import convert_absolute_to_relative_se2_array
    def targets_from_frames(frames):
        poses=np.array([[*f['ego2global_translation'][:2],Quaternion(*f['ego2global_rotation']).yaw_pitch_roll[0]] for f in frames[3:12]],dtype=np.float64)
        return torch.from_numpy(convert_absolute_to_relative_se2_array(StateSE2(*poses[0]),poses[1:]).astype(np.float32))
    groups={};targets={}
    for row in rows('train'):groups.setdefault(row['metadata_path'],[]).append(row)
    for path,group in groups.items():
        with open(path,'rb') as f:frames=pickle.load(f)
        for row in group:targets[row['token']]=targets_from_frames(frames[row['start_idx']:row['start_idx']+row['frame_count']])
    save(ROOT/'supervision/train_gt.pt',targets)
    write(ROOT/'manifests/identity.json',dict(splits={s:dict(scenes=len(rows(s)),logs=len(logs[s]),sha256=sha(ROOT/'manifests'/f'{s}.json')) for s in mapping},
        source=str(source),source_checkpoint='frozen VLM encoding; the action projection is not cached',
        caveat='All splits are official navtrain; held out from this new training, potentially seen by original pretraining'))
    print(read(ROOT/'manifests/identity.json'))

if __name__=='__main__':main()
