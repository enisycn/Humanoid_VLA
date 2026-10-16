"""Portable version of the recorded batch24 / new6 / replay18 training objective."""
from pathlib import Path
import argparse,hashlib,json,sys,random
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from training_objective import FrozenExpertReference,forward_objective
from model_io import load_policy,sha
KEYS=('observation.images.image','observation.images.image2')
def sample(r,t):
 import torch
 n=len(r['actions']);idx=np.arange(t,t+50)
 return {**{k:torch.from_numpy(r['images'][t,i].copy()).permute(2,0,1).float()[None]/255 for i,k in enumerate(KEYS)},'observation.state':torch.from_numpy(r['own'][t].copy())[None],'action':torch.from_numpy(r['actions'][np.minimum(idx,n-1)].copy()),'action_is_pad':torch.from_numpy(idx>=n),'task':r['language']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--model-dir',required=True,type=Path);p.add_argument('--initial-expert',required=True,type=Path);p.add_argument('--data',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--updates',type=int,default=1000);p.add_argument('--seed',type=int,default=2026101621);p.add_argument('--check-only',action='store_true');a=p.parse_args()
 import torch
 from safetensors.torch import save_file
 from torch.utils.data import default_collate
 from lerobot.scripts.lerobot_train import _preprocess_dataset_batch
 from lerobot.policies.factory import make_pre_post_processors
 manifest=json.loads((a.data/'manifest.json').read_text());records=[]
 for item in manifest['records']:
  f=a.data/item['file'];assert sha(f)==item['sha256']
  with np.load(f,allow_pickle=False)as z:r={k:z[k].copy()for k in ['images','own','actions']}
  r['language']=item['language'];assert len(r['actions'])==item['labels'];records.append(r)
 with np.load(a.data/'schedule.npz')as z:schedule=z['contexts']
 assert schedule.shape==(1000,24,2)and len(records)==64
 if a.check_only:print('Data and schedule verified. No model loaded.');return
 random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
 student,cfg=load_policy(a.model_dir,a.initial_expert);names={n for n,v in student.named_parameters()if v.requires_grad};assert len(names)==299
 teacher=FrozenExpertReference(student,names);frozen={k:v.detach().cpu().clone()for k,v in student.state_dict().items()if k not in names}
 pre,_=make_pre_post_processors(cfg,pretrained_path=str(a.model_dir),preprocessor_overrides={'device_processor':{'device':'cuda'}})
 parameters=[v for k,v in student.named_parameters()if k in names];opt=torch.optim.AdamW(parameters,lr=1e-6,betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
 a.output.mkdir(parents=True,exist_ok=False);history=[]
 for step,ids in enumerate(schedule[:a.updates],1):
  batch=_preprocess_dataset_batch(default_collate([sample(records[int(ep)],int(t))for ep,t in ids]),list(cfg.image_features),{},pre)
  opt.zero_grad(set_to_none=True);total,bc,kd,losses=forward_objective(student,teacher,batch,1.);total.backward();torch.nn.utils.clip_grad_norm_(parameters,10.)
  for group in opt.param_groups:group['lr']=1e-6*min(1.,step/10)
  opt.step()
  if step==1 or step%25==0:
   row={'step':step,'BC24':float(bc.detach()),'KD_old18':float(kd.detach()),'new6_BC':float(losses[:6].mean().detach())};history.append(row);print(json.dumps(row),flush=True)
  if step in [200,600,a.updates]:
   assert all(torch.equal(student.state_dict()[k].cpu(),v)for k,v in frozen.items())
   save_file({k:v.detach().cpu().contiguous()for k,v in student.state_dict().items()if k in names},a.output/f'expert-{step}.safetensors')
 (a.output/'losses.json').write_text(json.dumps(history,indent=2)+'\n')
 torch.save({'step':a.updates,'optimizer':opt.state_dict(),'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()},a.output/'resume.pt')
if __name__=='__main__':main()
