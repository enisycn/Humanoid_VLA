"""Additional native tasks; frozen push-task interface remains unchanged."""
from pathlib import Path
import hashlib
import numpy as np
from liberopush.native import NativePanda, ROOT

TASKS = {
    'drawer': dict(name='open_the_middle_drawer_of_the_cabinet',
                   joint='wooden_cabinet_1_middle_level', threshold=-.14,
                   direction=-1, unit='mm', scale=1000., drift_limit=.005),
    'stove': dict(name='turn_on_the_stove', joint='flat_stove_1_button',
                  threshold=.5, direction=1, unit='rad', scale=1., drift_limit=.05),
}


class NativeTaskPanda(NativePanda):
    def __init__(self, task_key, init_index, seed, clutter):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        self.spec = TASKS[task_key]
        suite = benchmark.get_benchmark_dict()['libero_goal'](task_order_index=0)
        task_id = next(i for i in range(suite.n_tasks)
                       if suite.get_task(i).name == self.spec['name'])
        task = suite.get_task(task_id)
        self.language = task.language
        bddl = Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
        self.env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256,
            camera_widths=256, control_freq=20, horizon=320, hard_reset=True)
        self.env.seed(seed)
        self.env.reset()
        starts = suite.get_task_init_states(task_id)
        if not 0 <= init_index < len(starts):
            raise ValueError('Initial state outside native task suite')
        self.raw = self.env.set_init_state(starts[init_index])
        for _ in range(10):
            self.raw, _, _, _ = self.env.step(np.r_[np.zeros(6), -1.])
        self.steps = 0
        self.horizon = 300
        self.done = False
        sim = self.env.sim
        self.initial_physics_sha256 = hashlib.sha256(sim.data.qpos.tobytes()).hexdigest()
        self.removed = []
        if not clutter:
            for name in ['plate_1', 'cream_cheese_1', 'akita_black_bowl_1', 'wine_bottle_1']:
                body = sim.model.body_name2id(name + '_main')
                joint = sim.model.body_jntadr[body]
                adr = sim.model.jnt_qposadr[joint]
                vel = sim.model.jnt_dofadr[joint]
                sim.data.qpos[adr:adr+3] = [5., 5., .95]
                sim.data.qvel[vel:vel+6] = 0
                self.removed.append(name)
        # Refresh both branches at the same physical boundary. A refresh in
        # only one branch introduces a tiny proprioceptive sensor-time offset.
        sim.forward()
        self.raw = self.env.env._get_observations(force_update=True)
        self.joint_id = sim.model.joint_name2id(self.spec['joint'])
        self.target_body = int(sim.model.jnt_bodyid[self.joint_id])
        self.target_geom_ids = set()
        for geom in range(sim.model.ngeom):
            body = int(sim.model.geom_bodyid[geom])
            while body:
                if body == self.target_body:
                    self.target_geom_ids.add(geom)
                    break
                body = int(sim.model.body_parentid[body])
        if not self.target_geom_ids:
            raise RuntimeError('No collision geometry attached to target articulation')
        if self.env.check_success():
            raise ValueError('Task already successful at start')
        assert self.env.robots[0].controller.use_delta

    def target_value(self):
        return float(self.env.sim.data.get_joint_qpos(self.spec['joint']))

    def deficit(self):
        return max(0., self.spec['direction'] *
                   (self.spec['threshold'] - self.target_value())) * self.spec['scale']


class ArticulationContactMonitor:
    """Permit gripper/target articulation and finger/table support only."""
    def __init__(self, env):
        self.sim = env.env.sim
        self.target_geoms = env.target_geom_ids
        self.original = self.sim.step
        self.allowed = self.table = self.unwanted = 0
        self.events = {}
        self.names = [self.sim.model.geom_id2name(g) or ''
                      for g in range(self.sim.model.ngeom)]
        self.sim.step = self.step

    def step(self):
        self.original()
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            ids = [contact.geom1, contact.geom2]
            robot = [g for g in ids if self.names[g].startswith(('robot0_', 'gripper0_'))]
            other = [g for g in ids if g not in robot]
            if not robot or not other:
                continue
            names = [self.names[g] for g in robot]
            base = any('link0' in n or 'base' in n for n in names)
            target = all(n.startswith('gripper0_') for n in names) and all(g in self.target_geoms for g in other)
            table = all(n.startswith('gripper0_finger') for n in names) and all(self.names[g] == 'table_collision' for g in other)
            bad = not base and not target and not table
            self.allowed += int(target)
            self.table += int(table)
            self.unwanted += int(bad)
            if target or table or bad:
                key = ' | '.join(sorted(self.names[g] for g in ids))
                self.events[key] = self.events.get(key, 0) + 1

    def close(self):
        self.sim.step = self.original
