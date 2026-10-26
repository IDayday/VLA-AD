"""Optional single-device raw/cached ReCogDrive replay, independent of DLC."""
from pathlib import Path
import csv, gzip, hashlib, json, os, pickle, sys
from types import SimpleNamespace
import numpy as np
from .bank import ROOT


def bootstrap(paths,out):
    os.environ['RECOGDRIVE_PORTABLE_PATHS']=str(Path(paths).resolve())
    os.environ['RECOGDRIVE_PORTABLE_WORKDIR']=str(Path(out).resolve()/'runtime')
    import portable_common
    sys.modules['common']=portable_common
    portable_common.setup('teacher_export');portable_common.compatibility()
    sys.path.insert(0,str(ROOT/'vendor'))


def read_rows(population,paths):
    if population not in ('train','test'):raise ValueError(population)
    cohort='navtrain' if population=='train' else 'navtest'
    with gzip.open(ROOT/f'manifests/{population}.csv.gz','rt') as f:rows=list(csv.DictReader(f))
    for r in rows:
        r.update(raw_path=str(Path(paths[cohort+'_logs'])/r['raw_rel']),
            sensor_root=str(Path(paths[cohort+'_sensor'])/r['sensor_rel']),
            metric_cache=str(Path(paths[cohort+'_metric'])/r['metric_rel']))
        if paths.get(cohort+'_features'):r['feature_path']=str(Path(paths[cohort+'_features'])/r['cache_rel']/'internvl_feature.gz')
    return rows


class Runtime:
    def __init__(self,source='raw'):
        import torch
        from worker import load_model
        from portable_common import read
        self.cfg=read(ROOT/'configs/repair_final_baseline.json');self.agent,self.identity=load_model(self.cfg)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        self.source=source
        from model.recogdrive_backbone import RecogDriveBackbone
        if source=='raw':
            self.backbone=RecogDriveBackbone('internvl',self.cfg['vlm'],'cuda').eval().requires_grad_(False)
            self.backbone.model.language_model.lm_head.forward=lambda x:x
        elif source=='cached':
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(self.cfg['vlm'],trust_remote_code=True,use_fast=False)
            self.backbone=SimpleNamespace(tokenizer=tokenizer)
        else:raise ValueError(source)

    def scene(self,row):
        import torch
        from vlm_core import raw_features,prepare
        from model.recogdrive_backbone import IMG_CONTEXT_TOKEN
        observation=raw_features(row)
        inputs,_=prepare(self.backbone,observation)
        ids=inputs['input_ids'][0];valid=inputs['attention_mask'][0].bool()
        image=valid&(ids==self.backbone.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN));text=valid&~image
        if self.source=='cached':
            if 'feature_path' not in row:raise ValueError('Set navtrain_features / navtest_features for cached replay')
            data=Path(row['feature_path']).read_bytes()
            if hashlib.sha256(data).hexdigest()!=row['feature_sha256']:raise ValueError(f'Cached feature SHA mismatch: {row["token"]}')
            features=pickle.loads(gzip.decompress(data))
            hidden=features['last_hidden_state'].unsqueeze(0).cuda().float()
            for key in ('history_trajectory','high_command_one_hot','status_feature'):
                if not torch.equal(features[key].float().cpu(),observation[key].float().cpu()):raise ValueError(f'Observation/cache mismatch: {key}')
        else:
            features=observation
            with torch.no_grad():hidden=self.backbone.model(**inputs).hidden_states[-1].float()
        if hidden.shape[1]!=len(valid):raise ValueError('Native sequence length and mask layout differ')
        with torch.no_grad(),torch.autocast('cuda',enabled=False):base=self.agent.action_head.feature_encoder(hidden)
        return features,hidden,base,image,text,ids

    def trajectory(self,features,hidden,interface,token,j):
        import torch
        from seeds import scene_seed,reseed
        hook=self.agent.action_head.feature_encoder.register_forward_hook(lambda m,a,o:interface)
        try:
            batch={k:v.unsqueeze(0).cuda().float() for k,v in features.items() if k not in ('pixel_values','last_hidden_state')}
            batch['last_hidden_state']=hidden;reseed(scene_seed(token,j))
            with torch.no_grad(),torch.autocast('cuda',enabled=False):return self.agent.forward(batch)['pred_traj'][0].float().cpu().numpy()
        finally:hook.remove()
