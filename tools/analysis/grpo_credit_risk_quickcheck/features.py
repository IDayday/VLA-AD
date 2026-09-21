from common import *
import argparse,pickle,torch
from torch.utils.data import DataLoader
install_runtime();forbid_updates()
from navsim.common.dataclasses import AgentInput,SensorConfig
from navsim.agents.recogdrive.recogdrive_features import ReCogDriveFeatureBuilder
class Raw:
 def __init__(self,rows):self.rows=rows;self.logs={}
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i];p=Path('/mnt/navsim/trainval_navsim_logs/trainval')/(r['log']+'.pkl')
  if r['log'] not in self.logs:
   with p.open('rb') as f:self.logs[r['log']]=pickle.load(f)
  frames=self.logs[r['log']][r['frame_start']:r['frame_start']+4]
  assert len(frames)==4
  sensors=SensorConfig(cam_f0=True,cam_l0=False,cam_l1=False,cam_l2=False,cam_r0=False,cam_r1=False,cam_r2=False,cam_b0=False,lidar_pc=False)
  return r['token'],AgentInput.from_scene_dict_list(frames,Path('/mnt/navsim/trainval_sensor_blobs/trainval'),4,sensors,load_image_path=True)
def main(a):
 torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 rows=scenes(smoke=a.smoke)[a.rank::a.world];dest=OUT/'cache/features';dest.mkdir(parents=True,exist_ok=True)
 rows=[r for r in rows if not (dest/(r['token']+'.pt')).exists()]
 b=ReCogDriveFeatureBuilder(cache_hidden_state=True,cache_mode=True,model_type='internvl',checkpoint_path=str(ROOT/'checkpoints/recogdrive/ReCogDrive-VLM-2B'),device='cuda:0',use_expert_features=False)
 assert next(b.backbone.model.parameters()).dtype==torch.bfloat16
 b.backbone.model.requires_grad_(False);before=state_hash(b.backbone.model);start=time.time();records=[]
 for i,(token,input) in enumerate(DataLoader(Raw(rows),batch_size=None,num_workers=2,multiprocessing_context='fork')):
  with torch.inference_mode():f=b.compute_features(input)
  assert set(f)=={'last_hidden_state','history_trajectory','status_feature','high_command_one_hot'}
  payload={k:v.cpu() for k,v in f.items()};meta=dict(token=token,observation_only=True,observed_frames=4,vlm_path=str(ROOT/'checkpoints/recogdrive/ReCogDrive-VLM-2B'),vlm_state_sha256=before,feature_builder_sha256=sha(RUNTIME/'navsim/agents/recogdrive/recogdrive_features.py'),precision=str(f['last_hidden_state'].dtype),config_sha256=sha(OUT/'resolved_config.yaml'))
  payload['_metadata']=meta;p=dest/(token+'.pt');tmp=p.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(p)
  # The old FP32 diagnostic has the same four observations and command but is
  # explicitly NOT reused as the BF16 hidden state for native mixed precision.
  old=torch.load(ROOT/'outputs/pc_mts_diagnostics/features'/(token+'.pt'),map_location='cpu',weights_only=False)
  for key in ['history_trajectory','status_feature','high_command_one_hot']:assert torch.equal(payload[key],old[key]),(token,key)
  records.append(dict(token=token,path=str(p),sha256=sha(p),metadata=meta))
  if i%8==0:print('FEATURE',i+1,len(rows),time.time()-start,flush=True)
 after=state_hash(b.backbone.model);assert before==after
 save(OUT/'audits'/f"features_{'smoke' if a.smoke else 'formal'}_{a.rank}.json",dict(status='PASS',state_before=before,state_after=after,optimizer_steps=0,records=records))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--rank',type=int,default=0);p.add_argument('--world',type=int,default=1);main(p.parse_args())
