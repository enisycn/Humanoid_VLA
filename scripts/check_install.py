from pathlib import Path
import argparse,hashlib,json,importlib
p=argparse.ArgumentParser();p.add_argument('--part',choices=['native','policy','world-model'],required=True);a=p.parse_args()
modules={'native':['numpy','mujoco','robosuite','cv2','av','yaml'],'policy':['torch','torchvision','transformers','lerobot.policies.smolvla.modeling_smolvla'],'world-model':['torch','torchvision','einops']}
for name in modules[a.part]:importlib.import_module(name);print(name+' OK')
ROOT=Path(__file__).resolve().parents[1]
if a.part=='policy':
 config=json.loads((ROOT/'configs/model.json').read_text());f=ROOT/'weights/final-expert.safetensors'
 with f.open('rb')as s:assert hashlib.file_digest(s,'sha256').hexdigest()==config['final_expert_sha256']
 print('Final expert hash OK')
