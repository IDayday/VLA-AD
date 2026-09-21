from common import *
import torch
from pairwise_credit import norm_match_delta

def main():
 for s in cfg()['tiny_update']['seeds']:
  base=OUT/'tiny'/str(s);a=torch.load(base/'A/step1.pt',map_location='cpu',weights_only=False)['state_dict'];d=torch.load(base/'D/step1.pt',map_location='cpu',weights_only=False)['state_dict'];start=torch.load(base/'A/step0.pt',map_location='cpu',weights_only=False)['state_dict'];names=list(read(base/'A/trainable_parameters.json'))
  matched,na,nd=norm_match_delta({k:start[k] for k in names},{k:a[k] for k in names},{k:d[k] for k in names});out=dict(start);out.update(matched);path=base/'A_norm_to_D';path.mkdir(exist_ok=True);torch.save(dict(state_dict=out,step=1,seed=s,arm='A_norm_to_D'),path/'step1.pt');actual=float(torch.sqrt(sum((out[k].double()-start[k].double()).square().sum() for k in names)))
  save(path/'control.json',dict(status='PASS',original_A_displacement=na,D_displacement=nd,norm_matched_actual_displacement=actual,rounding_error=actual-nd,scope='one-step diagnostic rescaling actual A parameter displacement; not an optimizer or advantage-RMS match'))
if __name__=='__main__':main()
