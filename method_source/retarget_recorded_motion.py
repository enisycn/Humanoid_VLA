"""Replay the measured hand motion with an open Panda gripper.

The measured hand marker supplies a relative planar trajectory. Static native
tool geometry maps the initial contact to the smaller simulator plate. This
does not assert full human fingertip reconstruction. Object pixels define scene
alignment offline and are QA only during replay. No model is run here.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys
import numpy as np
R=Path('/home/enis/humanoid_isaac_lab');D=Path(__file__).resolve().parent
F=R/'work/challenge2026/learned_improvement_20261014_v1/final-contact-generalization-20261015-v1'
S=R/'work/challenge2026/final_recovery_worldmodel_20261015_v2'
NATIVE=('observation.images.image','observation.images.image2')
def read(p):return json.loads(Path(p).read_text())
def bound(p):
    p=Path(p).resolve()
    with p.open('rb')as f:return dict(path=str(p),sha256=hashlib.file_digest(f,'sha256').hexdigest())
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x')as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def capture(index):
    p=read(D/'retarget-plan.json');assert p['status']=='frozen_after_motion_qualification'
    assert p['implementation']==bound(__file__)
    for item in p['bindings']:assert bound(item['path'])==item
    entry=p['episodes'][index];assert entry['source_motion_qualified']
    case=entry['case'];out=Path(p['output'])/f'episode-{index:02d}';out.mkdir(parents=True,exist_ok=False)
    with np.load(entry['trajectory']['path'])as z:times=z['times'].copy();grips=z['grip_xy_m'].copy();frames=z['source_frame_indices'].copy()
    assert len(times)==len(grips)==len(frames)and np.all(np.diff(times)>0)and np.isfinite(grips).all()
    os.environ.update(MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='0',OMP_NUM_THREADS='1')
    for k in ['LIBGL_ALWAYS_SOFTWARE','GALLIUM_DRIVER','__EGL_VENDOR_LIBRARY_FILENAMES']:os.environ.pop(k,None)
    sys.path[:0]=[str(S/'contact'),str(F),str(F.parent),str(R)]
    import cv2,torch,study
    from contact_support_v2 import Logger,STATE_COLUMNS,CONTACT_COLUMNS
    from work.challenge2026.resfit_paper_v1.episode_isolation_proposal_v1.native_adapter import EpisodeTraceSink
    from work.phone_video_prep.three_scenario_retarget_v4.geometry import rotation
    torch.set_num_threads(1);env=logger=sink=None;records=[];trace_started=False;failure=None;tracking=[]
    fixed=np.array(entry['fixed_rotation_matrix']);eef_z=entry['fixed_EEF_Z_m'];grip=entry['gripper_action'];assert grip==-1.
    assert .90<eef_z<.94 and np.allclose(fixed.T@fixed,np.eye(3),atol=1e-10)
    try:
        env=study.environment(case,out,[e['case']for e in p['episodes']]);obs=env.observation();own=obs['observation.state'][0].numpy().copy()
        from OpenGL import GL
        assert 'NVIDIA' in GL.glGetString(GL.GL_RENDERER).decode()
        scorer=study.scorer(env);logger=Logger(env);sink=EpisodeTraceSink(out);sink.reset(obs,case,float(env.sim.data.time));trace_started=True
        initial_z=float(env.initial_xyz[2]);physics={k:np.asarray(getattr(env.sim.model,k)).copy()for k in ['body_mass','body_inertia','geom_size','geom_pos','geom_quat','geom_friction','cam_pos','cam_quat']}
        def command(target,free=False,orient=True):
            cap=np.array([.6,.6,.4]if free else[.22,.22,.06]);delta=np.clip(4*(target-own[:3])/.05,-cap,cap)
            angular=cv2.Rodrigues(fixed@rotation(own[3:6]).T)[0].ravel()if orient else np.zeros(3)
            return np.r_[delta,np.clip(angular/.5,-.6,.6),grip]
        def tick(action,role,target=None,source=None):
            nonlocal obs,own
            k=len(records);assert k<460 and(role=='measurement_hold'or k<450)
            action=np.asarray(action,np.float32).astype(np.float64);assert np.max(np.abs(action))<=1
            clock=float(env.sim.data.time);logger.command=k;logger.begin_command();info=env.step(action);logger.finish_command(k,'hold'if role=='measurement_hold'else'policy')
            assert np.array_equal(action,np.array(info['action']))and abs(float(env.sim.data.time)-clock-.05)<1e-7
            sample=env.score_sample();phase='hold'if role=='measurement_hold'else'training_teacher';scorer.update_route(sample,k,phase)
            obs=env.observation();own=obs['observation.state'][0].numpy().copy()
            sink.step(obs,info['action'],dict(command_index=k,phase=phase,label_role=role,simulation_time_before_s=clock,simulation_time_after_s=float(env.sim.data.time),native_success=bool(sample['native_success']),scoring_only=sample))
            err=None if target is None else float(np.linalg.norm(own[:2]-target[:2])*1000)
            records.append(dict(index=k,phase=phase,label_role=role,actually_issued7=action.tolist(),source_frame=None if source is None else int(frames[source]),source_time_s=None if source is None else float(times[source]),desired_grip_site_xyz=None if target is None else target.tolist(),tracking_XY_error_mm=err,object_truth_used_for_actor=False))
            if role=='decoded_video_contact':tracking.append(err)
            return sample
        def move(target,budget,orient=True):
            for _ in range(budget):
                tick(command(target,True,orient),'robot_executor_approach',target)
                angle=np.linalg.norm(cv2.Rodrigues(fixed@rotation(own[3:6]).T)[0])if orient else 0
                if np.linalg.norm(own[:3]-target)<.003 and angle<.04:return
            raise RuntimeError('bounded_own_proprio_approach_not_reached')
        move(np.r_[own[:2],1.00],50,False);move(np.r_[own[:2],1.00],50);move(np.r_[grips[0],1.00],65);move(np.r_[grips[0],eef_z],65)
        for j in range(len(times)):
            target=np.r_[grips[j],eef_z]
            while True:
                assert len(records)<430,'450-command budget exceeded'
                tick(command(target),'decoded_video_contact',target,j)
                if np.linalg.norm(own[:2]-target[:2])<=.003:break
        for _ in range(20):tick(np.r_[np.zeros(6),grip],'desired_stop')
        stop=env.score_sample();scorer.set_stop(stop)
        for _ in range(10):scorer.add_hold(tick(np.r_[np.zeros(6),grip],'measurement_hold'))
        unchanged=all(np.array_equal(v,np.asarray(getattr(env.sim.model,k)))for k,v in physics.items());assert unchanged
        result=scorer.result(logger);sink.finish(result)
    except BaseException as e:
        failure=dict(type=type(e).__name__,message=str(e))
        if trace_started and not sink._persisted:sink.finish_partial('run_close',logger.metrics())
    finally:
        save(out/'issued-provenance.json',records)
        np.savez_compressed(out/'contact-physical-audit.npz',substeps=np.asarray(logger.state_samples)if logger else np.empty((0,18)),substep_columns=np.array(STATE_COLUMNS),contacts=np.asarray(logger.contact_samples)if logger else np.empty((0,15)),contact_columns=np.array(CONTACT_COLUMNS))
        save(out/'capture-status.json',dict(status='failed_partial_preserved'if failure else'completed',failure=failure,plan=bound(D/'retarget-plan.json'),entry=entry,source_is_actual_applicant_video=True,source_kind='sim_retargeted_from_personal_hand_video',object_truth_used_for_actor=False,no_model_run=True,tracking_median_mm=float(np.median(tracking))if tracking else None,tracking_max_mm=max(tracking)if tracking else None,native_physics_unchanged=unchanged if 'unchanged'in locals()else False,initial_z_scoring_only=initial_z if 'initial_z'in locals()else None,policy_commands=sum(r['label_role']!='measurement_hold'for r in records)))
        if logger:logger.close()
        if env:env.close()
    if failure:raise RuntimeError(failure)
    meta=read(out/'actual-native-trace.json');meta.update(source_is_actual_applicant_video=True,source_kind='sim_retargeted_from_personal_hand_video',object_truth_used_for_actor=False,actor_kind='personal_marker_video_retarget_executor',no_personal_data=False,deployed_student_inputs=['RGB2','measured own8','language'])
    (out/'actual-native-trace.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(episode=index,status='capture_completed_needs_admission',commands=len(records))),flush=True)
def admit(index):
    p=read(D/'retarget-plan.json');out=Path(p['output'])/f'episode-{index:02d}';s=read(out/'capture-status.json');checks={}
    if s['status']=='completed':
        meta=read(out/'actual-native-trace.json');r=meta['result'];rows=read(out/'issued-provenance.json');n=s['policy_commands']
        with np.load(out/'actual-native-trace.npz')as z:
            assert np.array_equal(z['issued_native_actions'],np.array([v['actually_issued7']for v in rows]))
            assert all(len(z[k])==len(rows)+1 for k in (*NATIVE,'observation.state'))
        with np.load(out/'contact-physical-audit.npz')as z:sub=z['substeps']
        scored=[meta['commands'][n-1]['scoring_only'],*[x['scoring_only']for x in meta['commands'][n:]]]
        functional=all(q['native_success']or(q['centre_error_mm']<=50 and abs(q['plate_xyz'][2]-s['initial_z_scoring_only'])<=.02)for q in scored)
        checks=dict(functional=bool(functional),route_complete=bool(r['route_complete']),no_added_wall_contact=bool(r['added_obstacle_collision_free']),stopped_drift=r['stopped_3d_drift_mm']<=5,peak_lift=max(0,float(sub[:,4].max()-s['initial_z_scoring_only']))<=.020,maximum_penetration=max(0,float(-sub[:,16].min()))<=3,video_reference_tracking_median=s['tracking_median_mm']<=5,video_reference_tracking_max=s['tracking_max_mm']<=15,native_physics_unchanged=s['native_physics_unchanged'],exact_ten_measurement_holds=len(rows)==n+10 and all(v['label_role']=='measurement_hold'for v in rows[n:]))
    passed=bool(checks)and all(checks.values());save(out/'admission.json',dict(passed=passed,checks=checks,capture_status=bound(out/'capture-status.json'),personal_source=True))
    if passed:
        source=dict(name='personal-video-'+str(index),source_kind='sim_retargeted_from_personal_hand_video',source_is_actual_applicant_video=True,physical_TRAIN_admission_passed=True,object_truth_used_for_actor=False,case=p['episodes'][index]['case'],language=p['episodes'][index]['case']['language'],rows=n,arrays=bound(out/'actual-native-trace.npz'),metadata=bound(out/'actual-native-trace.json'),provenance=bound(out/'issued-provenance.json'),admission=bound(out/'admission.json'),capture_status=bound(out/'capture-status.json'),video=p['source_video'])
        save(out/'source.json',source)
    print(json.dumps(dict(episode=index,passed=passed,checks=checks)),flush=True)
def load_training_records(s):
    for key in ['arrays','metadata','provenance','admission','capture_status','video']:assert bound(s[key]['path'])==s[key]
    assert read(s['admission']['path'])['passed']and s['source_is_actual_applicant_video']and s['physical_TRAIN_admission_passed']
    n=s['rows'];rows=read(s['provenance']['path']);assert len(rows)==n+10
    assert all(v['label_role']in ['robot_executor_approach','decoded_video_contact','desired_stop']for v in rows[:n])
    with np.load(s['arrays']['path'])as z:
        images=np.stack([z[k][:n].transpose(0,2,3,1)for k in NATIVE],axis=1);own=z['observation.state'][:n].copy();actions=z['issued_native_actions'][:n].copy()
    assert np.array_equal(actions,np.array([v['actually_issued7']for v in rows[:n]]))and np.array_equal(actions,actions.astype(np.float32).astype(np.float64))
    return[dict(source=s,images=images,own=own,actions=actions.astype(np.float32),label_roles=np.array([v['label_role']for v in rows[:n]]),original_command_indices=np.arange(n))]
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['capture','admit']);parser.add_argument('--episode',type=int,required=True);args=parser.parse_args();globals()[args.stage](args.episode)
