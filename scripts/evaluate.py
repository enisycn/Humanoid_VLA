"""Evaluate a public or post-trained VLA, optionally ranking four proposals."""
from pathlib import Path
import argparse,copy,json,os,select,subprocess,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/runtime')]
from protocol import send,receive
KEYS=('observation.images.image','observation.images.image2')
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
class Worker:
 def __init__(self,python,kind,model,seed,log,expert=None):
  self.log=log.open('wb');env=os.environ.copy();env['CONTACT_EXPERT_OVERLAY']=str(expert)if expert else'';self.index=0
  self.process=subprocess.Popen([str(python),str(ROOT/'src/runtime/worker.py'),'--kind',kind,'--model',str(model),'--seed',str(seed)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,env=env)
  self.metadata=self.get(300)
 def get(self,timeout=180):
  if not select.select([self.process.stdout],[],[],timeout)[0]:raise TimeoutError('Prediction timed out')
  r=receive(self.process.stdout)
  if r['op']=='fatal':raise RuntimeError(r)
  return r
 def predict(self,**kw):
  self.index+=1;send(self.process.stdin,dict(op='predict',request_id=self.index,**kw));return self.get()
 def close(self):
  if self.process.poll()is None:
   try:send(self.process.stdin,dict(op='close',request_id=0));self.get(5);self.process.wait(5)
   except Exception:self.process.terminate();self.process.wait(10)
  self.log.close()
def measured(obs):
 return np.stack([np.rint(obs[k][0].numpy().transpose(1,2,0)*255).astype(np.uint8)for k in KEYS]),obs['observation.state'][0].numpy().astype(np.float32)
def gate_check(case,points):
 walls=case.get('walls',[])
 if len(walls)!=2:return True
 a,b=[np.array(w['center_xyz_m'][:2])for w in walls];t=(b-a)/np.linalg.norm(b-a);n=np.array([-t[1],t[0]]);mid=(a+b)/2;v=(points-mid)@n
 if v[-1]<v[0]:n=-n;v=-v
 crossing=False
 for i in np.flatnonzero((v[:-1]<0)&(v[1:]>=0)):
  u=-v[i]/(v[i+1]-v[i]);xy=points[i]*(1-u)+points[i+1]*u
  crossing|=abs((xy-mid)@t)<np.linalg.norm(b-a)/2-.073
 return bool(v[0]<=-.073 and v[-1]>=.073 and crossing)
def wall_xml(walls):
 pieces=[]
 for i,w in enumerate(walls):
  shape=w.get('shape','box');size=w['half_size_xyz_m'];size=[size[0],size[2]]if shape=='cylinder'else size;yaw=w.get('yaw_rad',0.)
  sizes=' '.join(map(str,size));rgba=' '.join(map(str,w['rgba']));quat=' '.join(map(str,[np.cos(yaw/2),0,0,np.sin(yaw/2)]))
  pieces.append(f'<body name="scenario_wall_{w["name"]}" pos="{8+i} 8 .925"><geom name="scenario_wall_{w["name"]}_geom" type="{shape}" size="{sizes}" quat="{quat}" rgba="{rgba}" group="1" contype="1" conaffinity="1" condim="3" friction=".6 .005 .0001"/></body>')
 return ''.join(pieces)
def main():
 p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,default=ROOT/'configs/evaluation-cases.json');p.add_argument('--indices',type=int,nargs='+',default=[0]);p.add_argument('--list',action='store_true');p.add_argument('--model-dir',type=Path);p.add_argument('--expert',type=Path);p.add_argument('--policy-python',type=Path);p.add_argument('--wm-python',type=Path);p.add_argument('--world-model',type=Path);p.add_argument('--libero-config',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
 entries=json.loads(a.cases.read_text())
 if a.list:print(json.dumps(entries,indent=2));return
 for name in ['model_dir','policy_python','libero_config','output']:
  if getattr(a,name)is None:p.error('--'+name.replace('_','-')+' is required')
 if a.world_model and not a.wm_python:p.error('--wm-python is required with --world-model')
 os.environ['LIBERO_CONFIG_PATH']=str(a.libero_config.resolve());os.environ.setdefault('MUJOCO_GL','egl')
 import torch
 from work.challenge2026.three_scenario_scene_v1.environment import ThreeScenarioEnvironment,SceneContactMonitor,EpisodeScorer
 import work.challenge2026.scenario_scene as scene
 scene.wall_xml=wall_xml
 class Environment(ThreeScenarioEnvironment):
  def __init__(self,case,out):
   self.case=copy.deepcopy(case);self.out=out;out.mkdir(parents=True,exist_ok=False);self.native=scene.make_env(case,450,10);self.env,self.sim=self.native.env,self.native.env.sim;self._prepare()
 a.output.mkdir(parents=True,exist_ok=False);summary=[];torch.set_num_threads(1)
 for index in a.indices:
  case=entries[index]['case'];folder=a.output/f'episode-{index:06d}';env=monitor=v=w=None;images=[];states=[];actions=[];clocks=[];plans=[];result=None
  try:
   env=Environment(case,folder);monitor=SceneContactMonitor(env);scorer=EpisodeScorer(env);obs=env.observation();im,own=measured(obs);images.append(im);states.append(own);history=[(im,own)]
   v=Worker(a.policy_python,'VLA',a.model_dir,case['model_seed'],folder/'policy.log',a.expert);write(folder/'policy-identity.json',v.metadata)
   if a.world_model:w=Worker(a.wm_python,'WM',a.world_model,case['model_seed'],folder/'world-model.log');write(folder/'world-model-identity.json',w.metadata)
   def step(command,phase):
    before=float(env.sim.data.time);k=env.native.steps;monitor.begin_command();info=env.step(command);monitor.finish_command(k,phase);sample=env.score_sample();scorer.update_route(sample,k,phase);im,own=measured(env.observation());images.append(im);states.append(own);actions.append(command.copy());clocks.append(dict(command_index=k,phase=phase,simulation_time_before_s=before,simulation_time_after_s=float(env.sim.data.time),scoring_only=sample));return sample,im,own
   while True:
    pi,po=history[max(0,len(history)-11)];prediction=v.predict(rgb2=im.tobytes(),own8=own.tobytes(),task=case['language']);raw=np.asarray(prediction['proposals'],np.float32);candidates=np.clip(raw,-1,1);chosen=0;rank=None
    if w:rank=w.predict(rgb2=im.tobytes(),own8=own.tobytes(),past_rgb2=pi.tobytes(),past_own8=po.tobytes(),candidates=candidates.tobytes());chosen=int(rank['selected'])
    plans.append(dict(start_command=len(actions),raw_proposals=raw.tolist(),chosen=chosen,world_model=rank))
    for command in candidates[chosen]:
     issued=command.astype(np.float64);sample,im,own=step(issued,'policy');history.append((im,own))
     if sample['native_success']or env.native.steps>=450:break
    if sample['native_success']or env.native.steps>=450:break
   n=len(actions);scorer.set_stop(sample)
   for _ in range(10):sample,im,own=step(np.r_[np.zeros(6),issued[-1]],'hold');scorer.add_hold(sample)
   result=scorer.result(monitor);samples=[clocks[n-1]['scoring_only']]+[c['scoring_only']for c in clocks[n:]];points=np.array([case['plate_start_xy_m']]+[c['scoring_only']['plate_xyz'][:2]for c in clocks]);route=gate_check(case,points)
   near=all(c['centre_error_mm']<=50 for c in samples)and result['maximum_lift_mm']<=20
   result.update(functional_success=bool((result['native_goal_success']or near)and route and result['added_obstacle_collision_free']),route_complete=route,policy_commands=n,simulated_policy_time_s=n*.05,gripper_3d_travel_cm=float(np.linalg.norm(np.diff(np.asarray(states[:n+1])[:,:3],axis=0),axis=1).sum()*100),plate_travel_mm=float(np.linalg.norm(np.diff(points[:n+1],axis=0),axis=1).sum()*1000))
   summary.append(dict(title=entries[index]['title'],index=index,case=case,**result));write(a.output/'summary.json',summary);print(json.dumps({'index':index,'success':result['functional_success']}),flush=True)
  finally:
   if actions:
    np.savez_compressed(folder/'actual-native-trace.npz',**{k:np.stack(images)[:,i].transpose(0,3,1,2)for i,k in enumerate(KEYS)},**{'observation.state':np.stack(states),'issued_native_actions':np.stack(actions)})
    write(folder/'actual-native-trace.json',dict(case=case,commands=clocks,result=result,object_truth_used_for_actor=False,trace_status='completed_terminal'if result is not None else'nonterminal_partial'));write(folder/'plans.json',plans)
   for worker in [v,w]:
    if worker:worker.close()
   if monitor:monitor.close()
   if env:env.close()
if __name__=='__main__':main()
