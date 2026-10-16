"""Reset-only, explicitly declared physical corridors for unchanged VLA tests.

Append static boxes to the ORIGINAL pre-compilation XML. Do not round-trip a
compiled MuJoCo model. Boxes start far away so native reset/settling remains
unchanged; all workspace layout changes occur before policy command zero.
"""
import hashlib
import json

import mujoco
import numpy as np

from liberopush.native import NativePanda
from work.challenge2026.plate_only import PlateContactMonitor
from work.challenge2026.scene_variants import prepare_variant


def wall_xml(walls):
    fragments = []
    for i, wall in enumerate(walls):
        size = ' '.join(map(str, wall['half_size_xyz_m']))
        rgba = ' '.join(map(str, wall['rgba']))
        fragments.append(f'<body name="scenario_wall_{wall["name"]}" pos="{8+i} 8 .925">'
                         f'<geom name="scenario_wall_{wall["name"]}_geom" type="box" size="{size}" '
                         f'rgba="{rgba}" group="1" contype="1" conaffinity="1" condim="3" friction=".6 .005 .0001"/>'
                         '</body>')
    return ''.join(fragments)


def native_model_audit(original, actual):
    """Compare native physical/visual parameters by name despite new IDs."""
    fields = {
        mujoco.mjtObj.mjOBJ_BODY: ['body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat'],
        mujoco.mjtObj.mjOBJ_GEOM: ['geom_type','geom_size','geom_pos','geom_quat','geom_friction','geom_contype','geom_conaffinity','geom_condim','geom_rgba'],
        mujoco.mjtObj.mjOBJ_CAMERA: ['cam_pos','cam_quat','cam_fovy'],
        mujoco.mjtObj.mjOBJ_JOINT: ['jnt_type','jnt_pos','jnt_axis','jnt_range','jnt_limited','jnt_stiffness'],
    }
    counts = {}
    for kind, names in fields.items():
        count = {mujoco.mjtObj.mjOBJ_BODY:original.nbody, mujoco.mjtObj.mjOBJ_GEOM:original.ngeom,
                 mujoco.mjtObj.mjOBJ_CAMERA:original.ncam, mujoco.mjtObj.mjOBJ_JOINT:original.njnt}[kind]
        checked = 0
        for ident in range(count):
            name = mujoco.mj_id2name(original, kind, ident)
            if name is None: continue
            other = mujoco.mj_name2id(actual, kind, name)
            assert other >= 0, name
            for field in names:
                assert np.array_equal(getattr(original,field)[ident],getattr(actual,field)[other]), (name,field)
            checked += 1
        counts[str(kind)] = checked
    for field in ['dof_damping','dof_frictionloss','dof_armature','actuator_gainprm','actuator_biasprm',
                  'actuator_ctrlrange','actuator_forcerange']:
        assert np.array_equal(getattr(original,field),getattr(actual,field)), field
    assert original.nq == actual.nq and original.nv == actual.nv
    return dict(native_named_physical_camera_parameters_bitwise_equal=True, checked_named_elements=counts,
                nq_nv_unchanged=True, native_dof_actuator_arrays_bitwise_equal=True,
                added_fixed_bodies=int(actual.nbody-original.nbody), added_box_geoms=int(actual.ngeom-original.ngeom))


def make_env(case, horizon=300, evaluation_tail=10):
    walls = case.get('walls', [])
    if not walls:
        env = NativePanda(case['init_index'],case['seed'],horizon,evaluation_tail=evaluation_tail)
        env.scenario_compile_audit = dict(no_extra_physics=True)
        return env
    from robosuite.environments.base import MujocoEnv
    original_initialize = MujocoEnv._initialize_sim
    original_xmls = []
    compile_audits = []
    def initialize(instance, xml_string=None):
        prior = instance._xml_processor
        def process(xml):
            xml = prior(xml) if prior is not None else xml
            assert xml.count('</worldbody>') == 1 and 'scenario_wall_' not in xml
            original_xmls.append(xml)
            return xml.replace('</worldbody>',wall_xml(walls)+'</worldbody>')
        instance._xml_processor = process
        try:
            result = original_initialize(instance,xml_string)
            native = mujoco.MjModel.from_xml_string(original_xmls[-1])
            actual = instance.sim.model._model
            compile_audits.append(native_model_audit(native,actual))
            # Do this before native reset/render. Native fixed-object placement
            # subsequently changes body_pos through its usual reset routine.
            for field in ['center','extent','meansize','meanmass','meaninertia']:
                value = getattr(native.stat,field)
                if isinstance(value,np.ndarray): getattr(actual.stat,field)[:] = value
                else: setattr(actual.stat,field,value)
            return result
        finally:
            instance._xml_processor = prior
    MujocoEnv._initialize_sim = initialize
    try:
        env = NativePanda(case['init_index'],case['seed'],horizon,evaluation_tail=evaluation_tail)
    finally:
        MujocoEnv._initialize_sim = original_initialize
    try:
        env.scenario_compile_audit = compile_audits[-1]
        env.scenario_compile_audit.update(original_precompile_xml_sha256=hashlib.sha256(original_xmls[-1].encode()).hexdigest(),
            added_to_original_precompile_XML=True, compiled_model_not_round_tripped=True,
            wall_positions_during_native_settling='remote8m+',native_render_statistics_restored=True,
            native_parameter_comparison_time='immediately after compile, before standard native object-placement reset',
            native_fixed_fixture_placement_after_compile='standard native reset; removed only by declared scene preparation')
        return env
    except BaseException:
        env.close()
        raise


def prepare_scenario(env,out,case):
    sim = env.env.sim
    qpos, qvel = sim.data.qpos.copy(),sim.data.qvel.copy()
    for wall in case.get('walls',[]):
        ident = sim.model.body_name2id('scenario_wall_'+wall['name'])
        sim.model.body_pos[ident] = wall['center_xyz_m']
    sim.forward()
    assert np.array_equal(qpos,sim.data.qpos) and np.array_equal(qvel,sim.data.qvel)
    scene = prepare_variant(env,out,case)
    env.language = case['language']
    scene.update(scene=f'custom LIBERO {case["scenario_id"]}: original plate/Panda/table, declared physical walls',
                 language=case['language'],scenario_id=case['scenario_id'],walls=case.get('walls',[]),
                 route_gates=case.get('route_gates',[]),compile_audit=env.scenario_compile_audit,
                 baseline_weights_unchanged=True,scene_is_original_LIBERO=False,
                 wall_changes_reset_only=True,robot_preposition=False,actor_object_truth=False)
    (out/'compile-audit.json').write_text(json.dumps(env.scenario_compile_audit,indent=2)+'\n')
    return scene


class ScenarioContactMonitor(PlateContactMonitor):
    def __init__(self,env,case):
        self.plate_wall = 0
        self.robot_wall = 0
        super().__init__(env.env.sim)
    def step(self):
        super().step()
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            pair = [self.names[c.geom1],self.names[c.geom2]]
            if not any(n.startswith('scenario_wall_') for n in pair): continue
            self.plate_wall += int(any(n.startswith('plate_1') for n in pair))
            self.robot_wall += int(any(n.startswith(('robot0_','gripper0_')) for n in pair))
    def extra_metrics(self):
        return dict(plate_wall_contact_samples=self.plate_wall,robot_wall_contact_samples=self.robot_wall,
                    wall_contact_definition='physics-substep contact samples, not distinct events; diagnostic, not primary gate')


class RouteScorer:
    """Scoring only: ordered, causal plane crossings through visible openings."""
    def __init__(self,initial,case):
        self.initial_z = float(initial[2])
        self.gates = case.get('route_gates',[])
        self.passed = []
        self.steps = 0
    @property
    def complete(self): return len(self.passed) == len(self.gates)
    def update(self,previous,next_xyz):
        self.steps += 1
        if self.complete: return
        before,after = np.asarray(previous,float),np.asarray(next_xyz,float)
        gate = self.gates[len(self.passed)]
        axis = gate['axis']; position = gate['position_m']; direction = gate['direction']
        assert direction == 1 and axis in (0,1)
        if not (before[axis] < position <= after[axis]): return
        fraction = (position-before[axis])/(after[axis]-before[axis])
        crossing = before+fraction*(after-before)
        low,high = gate['other_axis_interval_m']
        if low <= crossing[1-axis] <= high and abs(crossing[2]-self.initial_z) <= .01:
            self.passed.append(dict(name=gate['name'],command_boundary=self.steps,crossing_xyz_m=crossing.tolist()))
    def metrics(self):
        return dict(route_complete=self.complete,route_gates_required=len(self.gates),
                    route_gates_passed=len(self.passed),route_crossings=self.passed,
                    route_scorer_object_truth=True,route_object_truth_in_policy=False)
