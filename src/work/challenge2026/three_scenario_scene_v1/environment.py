"""Native plate/stove, reset-only clutter parking and declared physical boxes.

ROOT owns all simulator execution. Importing this module is not a preflight.
Native RGB2 + own8 + original language remains the complete VLA input. Native
object poses are exposed only to scoring, geometry QA and later reward code.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from work.challenge2026.scenario_scene import make_env as _compile_native_with_boxes
from work.challenge2026.plate_only import PlateContactMonitor

HERE = Path(__file__).resolve().parent
LANGUAGE = 'push the plate to the front of the stove'


def read_contract():
    return json.loads((HERE/'contract.json').read_text())


def case_by_id(case_id):
    return next(c for c in read_contract()['cases'] if c['episode_id'] == case_id)


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


class ThreeScenarioEnvironment:
    """Composition around unchanged NativePanda; no runtime object adjustment."""
    def __init__(self, case, out, horizon=300, evaluation_tail=10):
        self.case = copy.deepcopy(case)
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        if horizon != 300 or evaluation_tail != 10:
            raise ValueError('Same 300-command + 10-hold protocol required')
        if case not in read_contract()['cases']:
            raise ValueError('Case must equal a complete predeclared contract case')
        # This previously audited helper adds boxes to ORIGINAL precompile XML,
        # remotely during native settling; it never round-trips compiled XML.
        self.native = _compile_native_with_boxes(case, horizon, evaluation_tail)
        self.env = self.native.env
        self.sim = self.env.sim
        try:
            self._prepare()
        except BaseException:
            self.native.close()
            raise

    @property
    def raw(self):
        return self.native.raw

    @raw.setter
    def raw(self, value):
        self.native.raw = value

    @property
    def language(self):
        return self.native.language

    @property
    def contract(self):
        return self.native.contract

    @property
    def horizon(self):
        return self.native.horizon

    def observation(self):
        return self.native.observation()

    def step(self, action):
        return self.native.step(action)

    def close(self):
        self.native.close()

    def _prepare(self):
        sim, model, case = self.sim, self.sim.model._model, self.case
        assert self.language == LANGUAGE
        original_raw = copy.deepcopy(self.raw)
        original_obs = self.observation()
        original_own = original_obs['observation.state'][0].numpy().copy()
        qpos, qvel = sim.data.qpos.copy(), sim.data.qvel.copy()
        initial_time = float(sim.data.time)
        # Native reset may leave derived camera/body transforms cached just
        # before its last integration. Match historical ClutterOnly: refresh
        # unchanged kinematics BEFORE capturing pose-protection comparisons.
        # Do not replace the actor's original raw own8 while doing this.
        sim.forward()
        assert np.array_equal(qpos,sim.data.qpos) and np.array_equal(qvel,sim.data.qvel)
        assert float(sim.data.time) == initial_time
        neutral = self.env.env._get_observations(force_update=True)
        neutral_mae = [float(np.abs(np.asarray(neutral[k],float)-np.asarray(original_raw[k],float)).mean())
                       for k in ['agentview_image','robot0_eye_in_hand_image']]
        self.raw = original_raw
        before = {n: getattr(model,n).copy() for n in dir(model)
                  if isinstance(getattr(model,n),np.ndarray)}
        # Retained in RAM for ROOT's preflight against a separate untouched
        # NativePanda. Never an actor feature and never used to drive actions.
        self.native_pre_scene_arrays = before
        self.native_pre_scene_qpos = qpos.copy()
        self.native_pre_scene_qvel = qvel.copy()
        self.plate_id = sim.model.body_name2id('plate_1_main')
        self.body_id = self.plate_id
        self.stove_id = sim.model.body_name2id('flat_stove_1_main')
        self.goal_site_id = sim.model.site_name2id('main_table_stove_front_region')
        goal_before = sim.data.site_xpos[self.goal_site_id].copy()
        goal_rot_before = sim.data.site_xmat[self.goal_site_id].copy()
        camera_xyz = sim.data.cam_xpos.copy()
        camera_rot = sim.data.cam_xmat.copy()
        protected = [i for i in range(sim.model.nbody)
                     if (sim.model.body_id2name(i) or '').startswith(
                         ('robot0_','gripper0_','flat_stove_1'))]
        protected_xyz = sim.data.body_xpos[protected].copy()
        protected_rot = sim.data.body_xmat[protected].copy()
        changed_qp, changed_qv = np.zeros(sim.model.nq,bool), np.zeros(sim.model.nv,bool)
        fixed, parked = [], []
        for i,name in enumerate(['akita_black_bowl_1','wine_bottle_1','cream_cheese_1']):
            bid = sim.model.body_name2id(name+'_main')
            joint = sim.model.body_jntadr[bid]
            assert sim.model.body_jntnum[bid] == 1 and sim.model.jnt_type[joint] == 0
            qa, va = sim.model.jnt_qposadr[joint], sim.model.jnt_dofadr[joint]
            xyz = [5.+i,5.,.95]
            sim.data.qpos[qa:qa+3] = xyz
            sim.data.qvel[va:va+6] = 0.
            changed_qp[qa:qa+3] = True
            changed_qv[va:va+6] = True
            parked.append(dict(name=name,kind='free_joint',parked_xyz_m=xyz))
        for name in ['wooden_cabinet_1','wine_rack_1']:
            bid = sim.model.body_name2id(name+'_main')
            assert sim.model.body_jntnum[bid] == 0
            sim.model.body_pos[bid] += np.array([5.,5.,0.])
            fixed.append(bid)
            parked.append(dict(name=name,kind='fixed_body',parked_body_pos_m=sim.model.body_pos[bid].tolist()))
        joint = sim.model.body_jntadr[self.plate_id]
        qa = sim.model.jnt_qposadr[joint]
        assert sim.model.body_jntnum[self.plate_id] == 1 and sim.model.jnt_type[joint] == 0
        native_plate_pose = qpos[qa:qa+7].copy()
        if case['plate_start_xy_m'] is not None:
            sim.data.qpos[qa:qa+2] = case['plate_start_xy_m']
            changed_qp[qa:qa+2] = True
            # Preserve native plate velocity, Z and quaternion even for shifted
            # tests. Reset-only translation is the only additional state change.
        self.obstacle_body_ids = []
        for wall in case['walls']:
            bid = sim.model.body_name2id('scenario_wall_'+wall['name'])
            assert sim.model.body_jntnum[bid] == 0
            sim.model.body_pos[bid] = wall['center_xyz_m']
            fixed.append(bid)
            self.obstacle_body_ids.append(bid)
        expected_body_pos = before['body_pos'].copy()
        expected_body_pos[fixed] = model.body_pos[fixed]
        assert np.array_equal(expected_body_pos,model.body_pos)
        assert all(np.array_equal(v,getattr(model,n)) for n,v in before.items() if n!='body_pos')
        assert np.array_equal(qpos[~changed_qp],sim.data.qpos[~changed_qp])
        assert np.array_equal(qvel[~changed_qv],sim.data.qvel[~changed_qv])
        sim.forward()
        assert np.array_equal(camera_xyz,sim.data.cam_xpos)
        assert np.array_equal(camera_rot,sim.data.cam_xmat)
        assert np.array_equal(protected_xyz,sim.data.body_xpos[protected])
        assert np.array_equal(protected_rot,sim.data.body_xmat[protected])
        assert np.array_equal(goal_before,sim.data.site_xpos[self.goal_site_id])
        assert np.array_equal(goal_rot_before,sim.data.site_xmat[self.goal_site_id])
        assert float(sim.data.time) == initial_time
        fresh = self.env.env._get_observations(force_update=True)
        self.raw = original_raw
        for key in ['agentview_image','robot0_eye_in_hand_image']:
            self.raw[key] = np.asarray(fresh[key]).copy()
        if case['plate_start_xy_m'] is not None:
            self.raw['plate_1_pos'] = np.asarray(fresh['plate_1_pos']).copy()
        own_after = self.observation()['observation.state'][0].numpy().copy()
        assert np.array_equal(own_after,original_own)
        assert np.array_equal(sim.data.qpos[qa+2:qa+7],native_plate_pose[2:])
        assert np.array_equal(qpos[~changed_qp],sim.data.qpos[~changed_qp])
        assert np.array_equal(qvel[~changed_qv],sim.data.qvel[~changed_qv])
        assert np.array_equal(expected_body_pos,model.body_pos)
        assert all(np.array_equal(v,getattr(model,n)) for n,v in before.items() if n!='body_pos')
        self.names = [sim.model.geom_id2name(i) or '' for i in range(sim.model.ngeom)]
        self.plate_geoms = {i for i,n in enumerate(self.names) if n.startswith('plate_1') and sim.model.geom_group[i]==0}
        self.obstacle_geoms = {i for i,n in enumerate(self.names) if n.startswith('scenario_wall_')}
        self.initial_plate_xyz = sim.data.body_xpos[self.plate_id].copy()
        self.initial_xyz = self.initial_plate_xyz.copy()
        self.goal_xyz = sim.data.site_xpos[self.goal_site_id].copy()
        self.goal_size = sim.model.site_size[self.goal_site_id].copy()
        self.goal_xmat = sim.data.site_xmat[self.goal_site_id].reshape(3,3).copy()
        assert np.allclose(self.goal_xyz,[-.05,.21,.895],atol=1e-12,rtol=0)
        assert np.allclose(self.goal_size,[.04,.04,.007],atol=1e-12,rtol=0)
        self.setup_contract = dict(
            schema='three_scenario_native_plate_reset_v1',case=case,
            scene='native plate/table/Panda/stove; parked clutter; declared fixed obstacle boxes',
            language=LANGUAGE,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            compile_audit=self.native.scenario_compile_audit,parked_clutter=parked,
            no_green_marker=True,no_robot_preposition=True,no_runtime_teleport=True,
            no_physics_step_during_scene_changes=True,simulation_time_before_s=initial_time,
            simulation_time_after_s=float(sim.data.time),own8_bitwise_unchanged=True,
            neutral_kinematic_refresh_before_protected_pose_capture=True,
            neutral_refresh_qpos_qvel_time_bitwise_unchanged=True,
            neutral_refresh_raw_camera_MAE=neutral_mae,
            robot_stove_camera_goal_pose_bitwise_unchanged=True,
            plate_native_Z_quaternion_velocity_unchanged=True,
            native_plate_pose_before_reset=native_plate_pose.tolist(),
            plate_start_xyz_m=self.initial_plate_xyz.tolist(),
            qpos_changed_indices=np.flatnonzero(sim.data.qpos!=qpos).tolist(),
            qvel_changed_indices=np.flatnonzero(sim.data.qvel!=qvel).tolist(),
            qpos_allowed_indices=np.flatnonzero(changed_qp).tolist(),
            qvel_allowed_indices=np.flatnonzero(changed_qv).tolist(),
            only_declared_fixed_body_positions_changed=True,
            other_model_arrays_bitwise_unchanged=True,goal_site_xyz_m=self.goal_xyz.tolist(),
            goal_site_size_m=self.goal_size.tolist(),goal_site_xmat=self.goal_xmat.tolist(),
            controller=self.contract,actor_object_truth=False,official_LIBERO_benchmark=False,
            initial_observation_nonimage_keys='preserved except translated plate pose for scoring only')
        _write(self.out/'setup-audit.json',self.setup_contract)
        _write(self.out/'compile-audit.json',self.native.scenario_compile_audit)

    def score_sample(self):
        xyz = self.sim.data.body_xpos[self.plate_id].copy()
        local = self.goal_xmat.T @ (xyz-self.goal_xyz)
        contact_now = False
        for c in self.sim.data.contact[:self.sim.data.ncon]:
            pair = [self.names[int(c.geom1)],self.names[int(c.geom2)]]
            contact_now |= any(n.startswith('plate_1') for n in pair) and any(n.startswith('gripper0_') for n in pair)
        return dict(plate_xyz=xyz.tolist(),native_success=bool(self.env.check_success()),
                    centre_error_mm=float(np.linalg.norm(xyz[:2]-self.goal_xyz[:2])*1000),
                    region_distance_mm=float(np.linalg.norm(np.maximum(np.abs(local[:2])-self.goal_size[:2],0.))*1000),
                    gripper_plate_contact_at_boundary=bool(contact_now),object_truth_scoring_only=True)


class SceneContactMonitor(PlateContactMonitor):
    """Pair samples at every physics substep; contact gaps at command boundaries."""
    def __init__(self, env):
        self.owner = env
        self.plate_obstacle = self.robot_obstacle = 0
        self.obstacle_events = {}
        self.boundary_contacts = []
        self.boundary_contact_events = []
        self._ever_contact = False
        self._previous_contact = False
        self._command_start = None
        super().__init__(env.sim)

    def begin_command(self):
        self._command_start = dict(allowed=self.allowed,table=self.table,unwanted=self.unwanted,
                                   plate_obstacle=self.plate_obstacle,robot_obstacle=self.robot_obstacle)

    def step(self):
        super().step()
        for c in self.sim.data.contact[:self.sim.data.ncon]:
            ids = [int(c.geom1),int(c.geom2)]
            if not any(i in self.owner.obstacle_geoms for i in ids):
                continue
            pair = [self.names[i] for i in ids]
            plate = any(n.startswith('plate_1') for n in pair)
            robot = any(n.startswith(('robot0_','gripper0_')) for n in pair)
            self.plate_obstacle += int(plate)
            self.robot_obstacle += int(robot)
            if plate or robot:
                key = ' | '.join(sorted(pair))
                self.obstacle_events[key] = self.obstacle_events.get(key,0)+1

    def finish_command(self, index, phase='policy'):
        if self._command_start is None:
            raise ValueError('begin_command must precede each actual command')
        current = dict(allowed=self.allowed,table=self.table,unwanted=self.unwanted,
                       plate_obstacle=self.plate_obstacle,robot_obstacle=self.robot_obstacle)
        delta = {k:current[k]-self._command_start[k] for k in current}
        has_contact = delta['allowed'] > 0
        if phase == 'policy':
            self.boundary_contacts.append(dict(command_index=int(index),has_substep_gripper_plate_contact=has_contact))
            if has_contact and not self._previous_contact:
                kind = 'recontact' if self._ever_contact else 'first_contact'
                self.boundary_contact_events.append(dict(kind=kind,command_index=int(index),simulation_time_s=float(self.sim.data.time)))
                self._ever_contact = True
            elif self._previous_contact and not has_contact:
                self.boundary_contact_events.append(dict(kind='contact_loss',command_index=int(index),simulation_time_s=float(self.sim.data.time)))
            self._previous_contact = has_contact
        self._command_start = None
        return dict(contact_delta=delta,has_substep_gripper_plate_contact=has_contact)

    def metrics(self):
        intervals = []
        for row in self.boundary_contacts:
            if not row['has_substep_gripper_plate_contact']:
                continue
            i = row['command_index']
            if intervals and intervals[-1][1]+1 == i:
                intervals[-1][1] = i
            else:
                intervals.append([i,i])
        return dict(allowed_tool_plate_samples=self.allowed,allowed_finger_table_samples=self.table,
                    unwanted_contact_samples=self.unwanted,plate_obstacle_contact_samples=self.plate_obstacle,
                    robot_obstacle_contact_samples=self.robot_obstacle,maximum_lift_mm=self.maximum_lift_mm,
                    obstacle_contact_pairs=self.obstacle_events,contact_intervals_inclusive=intervals,
                    contact_loss_recontact_events=self.boundary_contact_events,
                    contact_count_definition='all physics-substep pair samples, not distinct collisions',
                    contact_gap_definition='a policy command without any gripper-plate contact after a command with contact')


class EpisodeScorer:
    """Native goal and task-specific passage checks; never supplies actor inputs."""
    def __init__(self, env):
        self.env = env
        self.previous = env.initial_plate_xyz.copy()
        self.crossings = []
        self.holds = []
        self.stop = None

    def update_route(self, sample, command_index, phase='policy'):
        xyz = np.asarray(sample['plate_xyz'],float)
        gate = self.env.case.get('passage_gate')
        if gate and not self.crossings and self.previous[1] < gate['y_m'] <= xyz[1]:
            a = (gate['y_m']-self.previous[1])/(xyz[1]-self.previous[1])
            crossing = self.previous+a*(xyz-self.previous)
            lo,hi = gate['plate_centre_x_interval_m']
            if lo <= crossing[0] <= hi and abs(crossing[2]-self.env.initial_plate_xyz[2]) <= gate['maximum_height_change_m']:
                self.crossings.append(dict(command_index=int(command_index),phase=phase,crossing_xyz_m=crossing.tolist()))
        self.previous = xyz

    def set_stop(self, sample):
        self.stop = copy.deepcopy(sample)

    def add_hold(self, sample):
        self.holds.append(copy.deepcopy(sample))

    def result(self, monitor):
        if self.stop is None or len(self.holds) != 10:
            raise ValueError('Stop sample and all ten actual hold samples required')
        native = bool(self.stop['native_success'] and all(s['native_success'] for s in self.holds))
        route = bool(self.crossings) if self.env.case.get('passage_gate') else True
        collision_free = monitor.plate_obstacle == 0 and monitor.robot_obstacle == 0
        final = self.holds[-1]
        drift = max(np.linalg.norm(np.asarray(s['plate_xyz'])-self.stop['plate_xyz']) for s in self.holds)*1000
        return dict(native_goal_success=native,primary_success=native,
                    scenario_success=bool(native and route and collision_free),route_complete=route,
                    added_obstacle_collision_free=collision_free,passage_crossings=self.crossings,
                    centre_error_mm=final['centre_error_mm'],region_distance_mm=final['region_distance_mm'],
                    stopped_3d_drift_mm=float(drift),final_plate_xyz=final['plate_xyz'],
                    native_success_at_stop=self.stop['native_success'],
                    native_success_all_ten_holds=all(s['native_success'] for s in self.holds),
                    external_hold_commands=10,object_truth_scoring_only=True,**monitor.metrics())
