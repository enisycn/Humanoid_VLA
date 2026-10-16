"""Predeclared object/target anchor layouts; modifications at reset only."""
import json
import numpy as np
from work.challenge2026 import plate_only

def prepare_variant(env, out, case):
    sim = env.env.sim
    body = sim.model.body_name2id('plate_1_main')
    joint = sim.model.body_jntadr[body]
    q, v = sim.model.jnt_qposadr[joint], sim.model.jnt_dofadr[joint]
    before = sim.data.qpos.copy()
    before_vel = sim.data.qvel.copy()
    initial_native = before[q:q+7].copy()
    start = case.get('object_start_xy_m')
    if start is not None:
        start = np.asarray(start, dtype=float)
        assert start.shape == (2,) and np.isfinite(start).all()
        sim.data.qpos[q:q+2] = start
        sim.data.qvel[v:v+6] = 0.
    qmask, vmask = np.ones(len(before),bool), np.ones(len(before_vel),bool)
    qmask[q:q+2] = False
    vmask[v:v+6] = False
    assert np.array_equal(before[qmask], sim.data.qpos[qmask])
    assert np.array_equal(before_vel[vmask], sim.data.qvel[vmask])
    sim.forward()
    env.raw = env.env.env._get_observations(force_update=True)
    fixed_goal = plate_only.GOAL
    try:
        # Process-local configuration only. The render helper captures its own
        # immutable coordinate copy. No source/checkpoint file is modified.
        plate_only.GOAL = np.asarray(case['goal_xy_m'], dtype=float).copy()
        scene = plate_only.prepare(env, out)
    finally:
        plate_only.GOAL = fixed_goal
    scene.update(layout_family_id=case['layout_family_id'],
                 layout_fingerprint=case['layout_fingerprint'],
                 native_plate_pose_before_layout_reset=initial_native.tolist(),
                 object_initial_XY_from_predeclared_reset=start.tolist() if start is not None else None,
                 plate_geometry_material_mass_inertia_and_pose_unchanged=start is None,
                 plate_geometry_material_mass_inertia_Z_and_yaw_unchanged=True,
                 only_plate_XY_and_clutter_changed_at_reset=True,
                 canonical_robot_joint_position_velocity_unchanged=True,
                 physics_after_reset='actual native OSC commands only; no object state injection during motion',
                 claim='unseen positions using the same task/assets/cameras; not new-object or new-task generalization')
    return scene
