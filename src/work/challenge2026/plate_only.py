"""Authorized basic scene: original plate, table, robot and fixed visual target.

Scene changes happen at reset only. The VLA never receives object truth.
"""
import hashlib
import json
import numpy as np
from liberopush.fixed_green_point import add_fixed_point
from liberopush.mujoco_scene import ContactMonitor

GOAL = np.array([.05, .09])
LANGUAGE = 'push the plate to the green target'

class PlateContactMonitor(ContactMonitor):
    """Score contacts and plate centre lift at every physics substep."""
    def __init__(self, sim):
        self.plate_id = sim.model.body_name2id('plate_1_main')
        self.initial_z = float(sim.data.body_xpos[self.plate_id, 2])
        self.maximum_lift_mm = 0.
        super().__init__(sim)

    def step(self):
        super().step()
        lift = (float(self.sim.data.body_xpos[self.plate_id, 2])-self.initial_z)*1000
        self.maximum_lift_mm = max(self.maximum_lift_mm, lift)

def prepare(env, out):
    sim = env.env.sim
    model = sim.model._model
    original_xml = sim.model.get_xml()
    before = {n: getattr(model, n).copy() for n in dir(model)
              if isinstance(getattr(model, n), np.ndarray)}
    plate_body = sim.model.body_name2id('plate_1_main')
    plate_joint = sim.model.body_jntadr[plate_body]
    adr = sim.model.jnt_qposadr[plate_joint]
    plate_qpos = sim.data.qpos[adr:adr+7].copy()
    qpos_before, qvel_before = sim.data.qpos.copy(), sim.data.qvel.copy()
    unchanged_qpos = np.ones(sim.model.nq, dtype=bool)
    unchanged_qvel = np.ones(sim.model.nv, dtype=bool)
    robot = env.observation()['observation.state'].clone()
    moved = {}
    for i, name in enumerate(['cream_cheese_1', 'akita_black_bowl_1', 'wine_bottle_1']):
        body = sim.model.body_name2id(name+'_main')
        joint = sim.model.body_jntadr[body]
        q = sim.model.jnt_qposadr[joint]
        v = sim.model.jnt_dofadr[joint]
        unchanged_qpos[q:q+7] = False
        unchanged_qvel[v:v+6] = False
        sim.data.qpos[q:q+3] = [5.+i, 5., .95]
        sim.data.qvel[v:v+6] = 0
        moved[name] = sim.data.qpos[q:q+3].tolist()
    for name in ['wooden_cabinet_1', 'wine_rack_1', 'flat_stove_1']:
        body = sim.model.body_name2id(name+'_main')
        sim.model.body_pos[body] += np.array([5., 5., 0.])
        moved[name] = sim.model.body_pos[body].tolist()
    sim.forward()
    env.raw = env.env.env._get_observations(force_update=True)
    assert np.array_equal(plate_qpos, sim.data.qpos[adr:adr+7])
    after_robot = env.observation()['observation.state']
    audit = dict(robot_and_plate_qpos_bitwise_unchanged=bool(np.array_equal(qpos_before[unchanged_qpos], sim.data.qpos[unchanged_qpos])),
                 robot_and_plate_qvel_bitwise_unchanged=bool(np.array_equal(qvel_before[unchanged_qvel], sim.data.qvel[unchanged_qvel])),
                 own_proprio_before=robot[0].tolist(), own_proprio_after=after_robot[0].tolist(),
                 own_proprio_max_absolute_change=float((robot-after_robot).abs().max()),
                 reason='mj_forward refreshes derived kinematics and cached observables at unchanged qpos; no physics integration or robot command')
    out.mkdir(parents=True, exist_ok=True)
    (out/'reset-audit-before-marker.json').write_text(json.dumps(audit, indent=2)+'\n')
    assert audit['robot_and_plate_qpos_bitwise_unchanged']
    assert audit['robot_and_plate_qvel_bitwise_unchanged']
    changed = [n for n, v in before.items() if not np.array_equal(v, getattr(model, n))]
    assert changed == ['body_pos'], changed
    marker = add_fixed_point(env, out, GOAL, LANGUAGE, outline=True)
    marker.update(scene='custom plate-only LIBERO scene: table, original plate, green target',
                  no_removed_objects=False, removed_from_workspace_at_reset=moved,
                  original_full_scene_xml_sha256=hashlib.sha256(original_xml.encode()).hexdigest(),
                  plate_geometry_material_mass_inertia_and_pose_unchanged=True,
                  canonical_robot_start_unchanged=True,
                  changed_model_arrays_from_original=changed,
                  all_native_model_arrays_bitwise_unchanged=False,
                  all_arrays_unchanged_by_visual_marker=True,
                  initial_plate_xyz_m=env.raw['plate_1_pos'].tolist(),
                  displacement_m=float(np.linalg.norm(env.raw['plate_1_pos'][:2]-GOAL)),
                  goal_reason='fixed straight short push before testing; no policy endpoint used',
                  selection='predeclared basic target; dynamic reference feasibility checked separately',
                  native_broad_goal_valid_for_this_task=False,
                  physics_after_reset='only actual native OSC commands; no object teleportation')
    return marker
