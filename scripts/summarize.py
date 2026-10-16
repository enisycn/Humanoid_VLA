from pathlib import Path
import argparse,csv,json
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();rows=[]
for f in sorted(a.input.rglob('actual-native-trace.json')):
 m=json.loads(f.read_text());r=m.get('result')
 if r is None:continue
 policy=[v for v in m['commands']if v['phase']=='policy'];holds=[v for v in m['commands']if v['phase']=='hold'];n=len(policy)
 if not n:continue
 z=f.with_suffix('.npz')
 with np.load(z)as arrays:own=arrays['observation.state'][:n+1,:3].astype(float)
 points=np.array([m['case']['plate_start_xy_m']]+[v['scoring_only']['plate_xyz'][:2]for v in policy]);samples=[policy[-1]['scoring_only']]+[v['scoring_only']for v in holds];native=all(v['native_success']for v in samples);near=all(v['centre_error_mm']<=50 for v in samples)and r['maximum_lift_mm']<=20
 rows.append(dict(episode=f.parent.name,scenario=m['case']['episode_id'],functional_success=bool((native or near)and r['route_complete']and r['added_obstacle_collision_free']),native_goal_and_holds=native,centre_error_mm=r['centre_error_mm'],simulated_time_s=n*.05,gripper_path_cm=float(np.linalg.norm(np.diff(own,axis=0),axis=1).sum()*100),plate_path_cm=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()*100),contact_breaks=sum(x['kind']=='contact_loss'for x in r.get('contact_loss_recontact_events',[])),plate_obstacle_samples=r['plate_obstacle_contact_samples'],robot_obstacle_samples=r['robot_obstacle_contact_samples']))
if not rows:raise SystemExit('No complete traces found')
with a.out.open('w',newline='')as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(len(rows),'complete traces')
