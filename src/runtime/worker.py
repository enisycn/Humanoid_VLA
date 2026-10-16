"""Two isolated, learned components: stock VLA proposals or a tiny forward model.

Only builtins/bytes cross the local pipe. No simulator, target pose, case ID,
phase or clock can enter either prediction request.
"""
from pathlib import Path
import argparse, hashlib, json, os, sys, time, traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/runtime')]
from protocol import send,receive
IMAGE_KEYS=('observation.images.image','observation.images.image2')

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def setup_vla(args):
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.utils.random_utils import set_seed
    from safetensors import safe_open
    set_seed(args.seed)
    from model_io import load_policy
    overlay=os.environ.get('CONTACT_EXPERT_OVERLAY')or None
    policy,cfg=load_policy(args.model,overlay)
    overlay_identity=None if overlay is None else dict(path=overlay,sha256=sha(overlay))
    trainable=[k for k,v in policy.named_parameters()if v.requires_grad]
    policy.requires_grad_(False);policy.reset()
    pre,post=make_pre_post_processors(cfg,pretrained_path=args.model,preprocessor_overrides={'device_processor':{'device':'cuda'}})
    def predict(request):
        assert set(request)=={'op','request_id','rgb2','own8','task'} and request['op']=='predict'
        assert request['task'] in ['push the plate to the front of the stove','push the black bowl to the front of the stove','push the cream cheese to the front of the stove']
        assert type(request['rgb2']) is bytes and len(request['rgb2'])==2*256*256*3
        assert type(request['own8']) is bytes and len(request['own8'])==8*4
        images=torch.frombuffer(bytearray(request['rgb2']),dtype=torch.uint8).reshape(1,2,256,256,3)
        own=torch.frombuffer(bytearray(request['own8']),dtype=torch.float32).reshape(1,8)
        assert torch.isfinite(own).all()
        batch={**{k:images[:,i].permute(0,3,1,2).contiguous().float()/255 for i,k in enumerate(IMAGE_KEYS)},
            'observation.state':own,'task':[request['task']]}
        prepared=pre(batch);proposals=[]
        with torch.inference_mode():
            for _ in range(4):
                normalized=policy.predict_action_chunk(dict(prepared))
                assert normalized.shape==(1,50,7)
                # Use the same official per-action postprocessor as stock select_action.
                native=torch.stack([post(normalized[:,t])[0] for t in range(5)])
                assert native.shape==(5,7) and torch.isfinite(native).all()
                proposals.append(native.cpu().tolist())
        return dict(proposals=proposals,learned_proposals_only=True,no_simulator_input=True)
    meta=dict(kind='VLA',model_sha256=sha(Path(args.model)/'model.safetensors'),
        exact_saved_tensors=788,FP32_expert_tensors=sum(v.dtype==torch.float32 for k,v in policy.named_parameters() if k in trainable),frozen_VLM_unchanged=True,
        input_keys=['RGB2','own8','language'],proposal_count=4,predicted_chunk=50,external_execute=5,
        stock_predict_action_chunk_and_postprocessor=True,no_custom_noise_or_motor_correction=True)
    meta.update(expert_overlay=overlay_identity,parent_model_sha256=meta['model_sha256'])
    if overlay_identity is not None:meta['model_sha256']=overlay_identity['sha256']
    return predict,meta

def setup_wm(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    source=Path(os.environ['DINO_SOURCE']);checkpoint=Path(os.environ['DINO_CHECKPOINT'])
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert sha(checkpoint)=='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
    base=torch.hub.load(str(source),'dinov2_vits14',source='local',pretrained=False)
    base.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);base.cuda().eval();base.requires_grad_(False)
    encoder=lambda x:base.forward_features(x)['x_norm_patchtokens']
    transform=transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.Normalize([.5]*3,[.5]*3)])
    resize=transforms.Resize(196)
    data=torch.load(args.model,map_location='cuda',weights_only=True)
    mean,basis=data['zmean'],data['basis'];model=data['models']['visual']
    def predict(request):
        assert set(request)=={'op','request_id','rgb2','own8','past_rgb2','past_own8','candidates'} and request['op']=='predict'
        for key in ['rgb2','past_rgb2']:assert type(request[key]) is bytes and len(request[key])==2*256*256*3
        for key in ['own8','past_own8']:assert type(request[key]) is bytes and len(request[key])==8*4
        assert type(request['candidates']) is bytes and len(request['candidates'])==4*5*7*4
        images=torch.stack([torch.frombuffer(bytearray(request[k]),dtype=torch.uint8).reshape(2,256,256,3) for k in ['rgb2','past_rgb2']])
        im=images.flatten(0,1).permute(0,3,1,2).contiguous().cuda().float()/255
        with torch.inference_mode():
            patches=encoder(resize(transform(im)))
            pooled=F.adaptive_avg_pool2d(patches.transpose(1,2).reshape(-1,384,14,14),(7,7)).reshape(2,2*384*7*7)
            latent=(pooled-mean)@basis
            z=latent[0:1].expand(4,-1);zp=latent[1:2].expand(4,-1)
            own=torch.frombuffer(bytearray(request['own8']),dtype=torch.float32).cuda().reshape(1,8).expand(4,-1)
            op=torch.frombuffer(bytearray(request['past_own8']),dtype=torch.float32).cuda().reshape(1,8).expand(4,-1)
            a=torch.frombuffer(bytearray(request['candidates']),dtype=torch.float32).cuda().reshape(4,5,7)
            assert torch.isfinite(a).all() and (a.abs()<=1).all()
            am=a.mean(1)[:,:3]
            x=torch.cat([z,z-zp,own,own-op,a.flatten(1),(own[:,:,None]*am[:,None,:]).flatten(1),(z[:,:,None]*am[:,None,:]).flatten(1)],1)
            xb=torch.cat([(x-model['mean'])/model['std'],torch.ones(4,1,device='cuda')],1)
            y=xb@model['weight']
            assert y.shape==(4,73) and torch.isfinite(y).all()
        return dict(predicted_goal_error_change_m=y[:,-1].cpu().tolist(),predicted_own_delta8=y[:,64:72].cpu().tolist(),
            selected=int(y[:,-1].argmin()),no_true_goal_or_object_measurement=True)
    meta=dict(kind='WM',model_sha256=sha(args.model),forecast_commands=5,
        input_keys=['current RGB2','past RGB2','current own8','past own8','proposed5x7'],
        no_simulator_or_goal_pose_input=True,scoring_reward_labels_training_only=True)
    return predict,meta

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['VLA','WM'],required=True);ap.add_argument('--model',required=True);ap.add_argument('--seed',type=int,default=0);a=ap.parse_args()
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
    wire_out=os.fdopen(os.dup(sys.stdout.fileno()),'wb',buffering=0)
    os.dup2(sys.stderr.fileno(),sys.stdout.fileno());sys.stdout=sys.stderr
    try:
        predict,meta=(setup_vla(a) if a.kind=='VLA' else setup_wm(a))
        send(wire_out,dict(op='ready',**meta))
        while True:
            req=receive(sys.stdin.buffer)
            if req['op']=='close':send(wire_out,dict(op='closed'));return
            start=time.perf_counter();result=predict(req)
            send(wire_out,dict(op='prediction',request_id=req['request_id'],seconds=time.perf_counter()-start,**result))
    except BaseException as e:
        traceback.print_exc(file=sys.stderr)
        try:send(wire_out,dict(op='fatal',type=type(e).__name__,message=str(e)))
        except BaseException:pass
        raise

if __name__=='__main__':main()
