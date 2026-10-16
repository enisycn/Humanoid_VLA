"""Continue the exact saved step200 optimizer, RNG and expert to step1000."""
from pathlib import Path
import datetime, importlib.util, json, os, random, sys, time
import numpy as np
R=Path('/home/enis/humanoid_isaac_lab');D=Path(__file__).resolve().parent
PREVIOUS=R/'work/challenge2026/personal_video_diverse_B_fast_20261016_v1'
spec=importlib.util.spec_from_file_location('previous_joint',PREVIOUS/'run.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
read,bound,save=previous.read,previous.bound,previous.save

def prepare():
    assert not(D/'training-plan.json').exists()
    p=read(PREVIOUS/'training-plan.json');report=read(PREVIOUS/'training/report.json')
    assert report['updates']==200 and report['status']=='completed'
    for key in ['checkpoint','resume']:assert report[key]==bound(report[key]['path'])
    h,records,old_schedule,n=previous.sources_and_records()
    assert len(old_schedule)>=1000
    rng=np.random.default_rng(p['seed']);schedule=np.empty((1000,24,2),np.int32)
    for u in range(1000):
        for j in range(6):
            ep=(6*u+j)%n;length=len(records[ep]['actions'])
            schedule[u,j]=ep,0 if u%10==0 else int(rng.integers(length))
        columns=np.r_[np.arange(12),np.arange(12,18)+6*(u%2)]
        schedule[u,6:]=old_schedule[u,columns];schedule[u,6:,0]+=n
    with np.load(PREVIOUS/'training/schedule.npz',allow_pickle=False)as z:
        assert np.array_equal(schedule[:200],z['contexts'])
    (D/'training').mkdir(exist_ok=False)
    np.savez_compressed(D/'training/schedule.npz',contexts=schedule)
    p.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='authorized_continuation_after_200_update_checks',implementation=bound(__file__),
        original_implementation=bound(PREVIOUS/'run.py'),previous_plan=bound(PREVIOUS/'training-plan.json'),
        resume_checkpoint=report['checkpoint'],resume_optimizer_rng=report['resume'],
        start_update=200,updates=1000,additional_updates=800,
        output=str(R/'runs/challenge2026-personal-video-diverse-B-1000-20261016-v1/training'),
        checkpoints=[600,1000],fixed_selected_step=1000,
        learning_rate_schedule='Continue unchanged1e-6; initial10update warmup was already completed',
        previous_checks_are_exposed=True,no_new_data_or_method=True,
        first200_sampling_prefix_exact=True,schedule=bound(D/'training/schedule.npz'))
    save(D/'training-plan.json',p)
    cases=read(PREVIOUS/'all-policy-checks-before-query.json')
    cases['cases'][3]['reference']=str(R/'runs/challenge2026-personal-video-diverse-B-fast-checks-20261016-v1/B/episode-000003')
    cases.update(fixed_step=1000,no_policy_results_seen=False,all_cases_previously_exposed=True)
    save(D/'all-policy-checks-before-query.json',cases)
    print(json.dumps(dict(ready=True,total_updates=1000,resume_from=200,additional=800,labels=12837)),flush=True)

def train():
    import torch
    from safetensors import safe_open
    from common import set_runtime
    from train_joint_balanced_replay5k_v1 import load_exact
    from train_sft_retention import unchanged
    from lerobot.policies.factory import make_pre_post_processors
    p=read(D/'training-plan.json')
    assert p['implementation']==bound(__file__)
    for key in ['original_implementation','previous_plan','resume_checkpoint','resume_optimizer_rng','initialization_checkpoint','initialization_expert','personal_manifest','personal_loader','old_training_plan','helper','schedule']:
        assert p[key]==bound(p[key]['path'])
    assert __import__('shutil').disk_usage(R).free>22_000_000_000
    h,records,old_schedule,n=previous.sources_and_records()
    with np.load(D/'training/schedule.npz',allow_pickle=False)as z:schedule=z['contexts']
    assert schedule.shape==(1000,24,2) and len(records)==64
    set_runtime(p['seed']);student,names=load_exact(p['parent_model'])
    h.base.prior.apply_expert_overlay(student,p['parent_model'],previous.B)
    unchanged(student,p['parent_model'],names)
    reference=h.FrozenExpertReference(student,names)
    def verify_reference():
        with safe_open(p['initialization_expert']['path'],framework='pt',device='cpu')as f:
            for k,v in reference.parameters.items():assert torch.equal(v.cpu(),f.get_tensor(k))
    verify_reference()
    # The generic overlay loader accepts only a pristine parent. The student
    # already holds B while the frozen reference is copied, so restore the
    # saved step200 expert tensors directly and verify every value and dtype.
    resumed_manifest=read(p['resume_checkpoint']['path'])
    assert resumed_manifest['expert_weight']==bound(resumed_manifest['expert_weight']['path'])
    parameters_by_name=dict(student.named_parameters())
    with safe_open(resumed_manifest['expert_weight']['path'],framework='pt',device='cpu')as f:
        assert set(f.keys())==set(names)
        with torch.no_grad():
            for name in f.keys():
                tensor=f.get_tensor(name);parameter=parameters_by_name[name]
                assert parameter.dtype==tensor.dtype
                parameter.copy_(tensor.to(parameter.device))
                assert torch.equal(parameter.cpu(),tensor)
    unchanged(student,p['parent_model'],names)
    parameters=[v for k,v in student.named_parameters()if k in names]
    order=[k for k,v in student.named_parameters()if k in names]
    pre,_=make_pre_post_processors(student.config,pretrained_path=p['parent_model'],preprocessor_overrides={'device_processor':{'device':'cuda'}})
    optimizer=torch.optim.AdamW(parameters,lr=p['learning_rate'],betas=(.9,.95),eps=1e-8,weight_decay=1e-10)
    state=torch.load(p['resume_optimizer_rng']['path'],map_location='cpu',weights_only=False)
    assert state['step']==200 and state['checkpoint']==p['resume_checkpoint'] and state['parameter_order']==order
    optimizer.load_state_dict(state['optimizer'])
    assert len(optimizer.state)==299
    assert all(int(v['step'])==200 for v in optimizer.state.values())
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
    torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    del state
    save(D/'training/resume-QA.json',dict(passed=True,expert_exact_to_saved200=True,optimizer299_at200=True,
        parameter_order_exact=True,all_saved_rng_restored=True,frozen489_exact=True,reference_still_original_B=True,
        first200_sampling_prefix_exact=True))
    began=time.monotonic();history=[];checkpoints=[]
    for step in range(201,1001):
        if step%25==1:assert __import__('shutil').disk_usage(R).free>=20_100_000_000
        value=h.make_value(student,pre,records,schedule[step-1]);optimizer.zero_grad(set_to_none=True)
        total,bc,kd,losses=h.forward_objective(student,reference,value,1.)
        total.backward();assert all(v.grad is not None and torch.isfinite(v.grad).all()for v in parameters)
        assert all(v.grad is None for k,v in student.named_parameters()if k not in names)
        torch.nn.utils.clip_grad_norm_(parameters,10.)
        for group in optimizer.param_groups:group['lr']=p['learning_rate']
        optimizer.step()
        if step==201 or step%25==0:
            row=dict(step=step,total=1000,additional_step=step-200,seconds=time.monotonic()-began,
                BC24=float(bc.detach()),KD_old18=float(kd.detach()),new_library_BC=float(losses[:6].mean().detach()),lr=p['learning_rate'])
            history.append(row);save(D/'training/progress.json',row);print(json.dumps(row),flush=True)
        if step in p['checkpoints']:
            unchanged(student,p['parent_model'],names);verify_reference()
            target=Path(p['output'])/f'checkpoint-{step}'
            manifest=h.base.prior.save_expert_checkpoint(student,names,p['parent_model'],target,D/'training-plan.json','personal_B_joint')
            manifest.update(update=step,initialization_checkpoint=p['initialization_checkpoint'],
                continuation_from=p['resume_checkpoint'],personal_data_manifest=p['personal_manifest'],
                no_reference_at_deployment=True,no_reward_learning=True,reference_during_training='frozen B',fixed_selected_step=1000,
                complete_robot_demonstrations=13,new_library_labels=3990,old_robot_training_records=45)
            save(target/'expert-overlay.json',manifest);checkpoints.append(bound(target/'expert-overlay.json'))
    def cpu(value):
        if torch.is_tensor(value):return value.detach().cpu().clone()
        if isinstance(value,dict):return{k:cpu(v)for k,v in value.items()}
        if isinstance(value,list):return[cpu(v)for v in value]
        if isinstance(value,tuple):return tuple(cpu(v)for v in value)
        return value
    state=dict(schema='B_personal_joint_optimizer_rng_v1',step=1000,checkpoint=checkpoints[-1],plan=bound(D/'training-plan.json'),
        optimizer=cpu(optimizer.state_dict()),parameter_order=order,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    resume=D/'training/resume-1000.pt';torch.save(state,resume)
    restored=torch.load(resume,map_location='cpu',weights_only=False)
    assert len(restored['optimizer']['state'])==299 and all(int(v['step'])==1000 for v in restored['optimizer']['state'].values())
    save(D/'training/report.json',dict(status='completed',updates=1000,additional_updates=800,seconds=time.monotonic()-began,
        original_initialization=p['initialization_checkpoint'],resumed_from=p['resume_checkpoint'],checkpoint=checkpoints[-1],
        checkpoints=checkpoints,resume=bound(resume),all489_frozen_unchanged=True,reference_B_exact=True,
        new_library_records=n,new_library_labels=3990,old_records=45,old_labels=8847,mixed_sources=True,history=history))

if __name__=='__main__':globals()[sys.argv[1]]()
