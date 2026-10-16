from pathlib import Path
import hashlib,json,zipfile
from huggingface_hub import snapshot_download
import yaml
ROOT=Path(__file__).resolve().parents[1];m=json.loads((ROOT/'configs/environment.json').read_text());archive=ROOT/'third_party/libero-runtime.zip'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==m['libero_runtime_zip_sha256']
(ROOT/'vendor').mkdir(exist_ok=True)
with zipfile.ZipFile(archive)as z:z.extractall(ROOT/'vendor')
assets=snapshot_download(m['assets_repo'],repo_type=m['assets_repo_type'],revision=m['assets_revision'])
base=ROOT/'vendor/libero/libero';cfg=ROOT/'.cache/libero-config';cfg.mkdir(parents=True,exist_ok=True)
(cfg/'config.yaml').write_text(yaml.safe_dump({'benchmark_root':str(base),'assets':assets,'bddl_files':str(base/'bddl_files'),'init_states':str(base/'init_files'),'datasets':str(ROOT/'data/libero')}))
print('LIBERO_CONFIG_PATH='+str(cfg))
