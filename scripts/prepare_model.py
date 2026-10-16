"""Download the pinned public backbone and reuse the exact final processors."""
from pathlib import Path
import argparse,hashlib,json,os,shutil
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--public-out',type=Path);a=p.parse_args()
 from huggingface_hub import snapshot_download
 m=json.loads((ROOT/'configs/model.json').read_text());source=Path(snapshot_download(m['public_repo'],revision=m['revision']))
 with (source/'model.safetensors').open('rb')as f:assert hashlib.file_digest(f,'sha256').hexdigest()==m['base_sha256']
 a.out.mkdir(parents=True,exist_ok=False)
 for f in (ROOT/'configs/policy').rglob('*'):
  if f.is_file():
   q=a.out/f.relative_to(ROOT/'configs/policy');q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,q)
 try:os.link(source/'model.safetensors',a.out/'model.safetensors')
 except OSError:shutil.copyfile(source/'model.safetensors',a.out/'model.safetensors')
 if a.public_out:
  shutil.copytree(source,a.public_out,ignore=shutil.ignore_patterns('model.safetensors','.cache'))
  try:os.link(source/'model.safetensors',a.public_out/'model.safetensors')
  except OSError:shutil.copyfile(source/'model.safetensors',a.public_out/'model.safetensors')
 print(a.out.resolve())
if __name__=='__main__':main()
