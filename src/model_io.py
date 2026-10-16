"""Load saved precision exactly; replace the complete expert once, before inference."""
from pathlib import Path
import hashlib,json

def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def load_policy(directory,expert=None,device='cuda'):
 import torch
 from safetensors import safe_open
 from lerobot.configs.policies import PreTrainedConfig
 from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
 cfg=PreTrainedConfig.from_pretrained(str(directory),local_files_only=True);cfg.device=device;cfg.push_to_hub=False;cfg.n_action_steps=10
 model=SmolVLAPolicy.from_pretrained(str(directory),config=cfg,local_files_only=True,strict=True)
 state=model.state_dict(keep_vars=True)
 with safe_open(Path(directory)/'model.safetensors',framework='pt')as f:
  assert set(f.keys())==set(state)and len(state)==788
  with torch.no_grad():
   for name,v in state.items():v.data=f.get_tensor(name).to(device)
 if expert:
  with safe_open(expert,framework='pt')as f:
   names={n for n,p in model.named_parameters()if p.requires_grad}
   assert set(f.keys())==names and len(names)==299
   with torch.no_grad():
    for name in names:
     v=f.get_tensor(name);assert v.dtype==torch.float32
     state[name].data=v.to(device)
 return model,cfg
