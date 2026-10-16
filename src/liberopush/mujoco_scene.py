"""One simple physical scene and shared all-substep contact scoring."""
import numpy as np
def simplify_scene(env):
 sim=env.env.sim
 for name in ['cream_cheese_1','akita_black_bowl_1','wine_bottle_1']:
  body=sim.model.body_name2id(name+'_main');joint=sim.model.body_jntadr[body]
  adr=sim.model.jnt_qposadr[joint];vel=sim.model.jnt_dofadr[joint]
  sim.data.qpos[adr:adr+3]=[5.,5.,.95];sim.data.qvel[vel:vel+6]=0
 sim.forward();env.raw=env.env.env._get_observations(force_update=True)

def prepare_recording_scene(env):
 """Recording helper only: remove measured approach-blocking fixed fixtures.

 Never used by the frozen baseline/post-training evaluation scene.
 """
 simplify_scene(env);sim=env.env.sim;positions={}
 for name in ['wooden_cabinet_1','wine_rack_1']:
  body=sim.model.body_name2id(name+'_main');sim.model.body_pos[body]+=np.array([5.,5.,0.])
  positions[name]=sim.model.body_pos[body].tolist()
 sim.forward();env.raw=env.env.env._get_observations(force_update=True);return positions

class ContactMonitor:
 def __init__(self,sim):
  self.sim=sim;self.original=sim.step;self.unwanted=self.allowed=self.table=0;self.events={}
  self.names=[sim.model.geom_id2name(i) or '' for i in range(sim.model.ngeom)];sim.step=self.step
 def step(self):
  self.original()
  for i in range(self.sim.data.ncon):
   c=self.sim.data.contact[i];pair=[self.names[c.geom1],self.names[c.geom2]]
   robot=[n for n in pair if n.startswith(('robot0_','gripper0_'))]
   others=[n for n in pair if n not in robot]
   if not robot or not others:continue
   base=any('link0' in n or 'base' in n for n in robot)
   tool_plate=any(n.startswith('gripper0_') for n in robot) and any(n.startswith('plate_1') for n in others)
   finger_table=all(n.startswith('gripper0_finger') for n in robot) and all(n=='table_collision' for n in others)
   bad=not base and not tool_plate and not finger_table
   self.table+=int(finger_table);self.allowed+=int(tool_plate);self.unwanted+=int(bad)
   if bad or tool_plate or finger_table:
    key=' | '.join(sorted(pair));self.events[key]=self.events.get(key,0)+1
 def close(self):self.sim.step=self.original
