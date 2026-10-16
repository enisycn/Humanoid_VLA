"""Native RGB+robot-proprio LIBERO interface; object truth is scoring/QA only."""
import os
from pathlib import Path
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault('LIBERO_CONFIG_PATH', str(ROOT/'.cache/libero-config'))
os.environ.setdefault('HF_HOME', str(ROOT/'.cache/huggingface'))
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')

TASK_NAME = 'push_the_plate_to_the_front_of_the_stove'


def model_observation(raw, language):
    """Explicit whitelist: two RGB images and the robot's own measured state.

    Object pose, object velocity and hidden simulator properties are discarded.
    The official environment processor handles image rotation and quaternion
    conversion. These are the benchmark-trained camera/proprio conventions.
    """
    from lerobot.processor.env_processor import LiberoProcessorStep
    images = {}
    for camera, key in [('agentview_image', 'image'), ('robot0_eye_in_hand_image', 'image2')]:
        rgb = np.asarray(raw[camera])
        assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
        images['observation.images.'+key] = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float()/255
    def tensor(name):
        return torch.as_tensor(np.asarray(raw[name]).copy(), dtype=torch.float32)[None]
    images['observation.robot_state'] = {
        'eef': {'pos': tensor('robot0_eef_pos'), 'quat': tensor('robot0_eef_quat')},
        'gripper': {'qpos': tensor('robot0_gripper_qpos')},
    }
    result = LiberoProcessorStep().observation(images)
    result['task'] = [language]
    assert result['observation.state'].shape == (1, 8)
    return result


class NativePanda:
    def __init__(self, init_index=0, seed=7101, horizon=300, evaluation_tail=0):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        suite = benchmark.get_benchmark_dict()['libero_goal'](task_order_index=0)
        self.task_id = next(i for i in range(suite.n_tasks) if suite.get_task(i).name == TASK_NAME)
        task = suite.get_task(self.task_id)
        self.language = task.language
        bddl = Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
        self.env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256,
                                      control_freq=20, horizon=horizon+10+evaluation_tail, hard_reset=True)
        self.env.seed(seed)
        self.env.reset()
        starts = suite.get_task_init_states(self.task_id)
        if not 0 <= init_index < len(starts):
            raise ValueError('initial-state index out of range')
        self.raw = self.env.set_init_state(starts[init_index])
        self.initial_state_count = len(starts)
        for _ in range(10):
            self.raw, _, _, _ = self.env.step(np.r_[np.zeros(6), -1.])
        self.steps = 0; self.horizon = horizon; self.done = False
        controller = self.env.robots[0].controller
        assert controller.use_delta
        self.contract = {'controller': type(controller).__name__, 'use_delta': bool(controller.use_delta),
            'control_frequency_hz': 20, 'action_duration_s': .05, 'action_dim': 7,
            'input_min': np.asarray(controller.input_min).tolist(),
            'input_max': np.asarray(controller.input_max).tolist(),
            'output_min': np.asarray(controller.output_min).tolist(),
            'output_max': np.asarray(controller.output_max).tolist(),
            'camera_transform': 'official LiberoProcessorStep flips both H and W',
            'proprio': 'eef position3 + axis-angle3 + gripper qpos2',
            'object_ground_truth_used_for_policy': False}

    def observation(self):
        return model_observation(self.raw, self.language)

    def step(self, action):
        a = np.asarray(action, dtype=float)
        if a.shape != (7,) or not np.isfinite(a).all():
            raise ValueError('native action must contain seven finite values')
        # LIBERO's native action box; this same boundary applies to all methods.
        a = np.clip(a, -1, 1)
        self.raw, reward, native_done, info = self.env.step(a)
        self.steps += 1
        success = bool(self.env.check_success())
        self.done = success or bool(native_done) or self.steps >= self.horizon
        return {'success': success, 'native_done': bool(native_done), 'reward': float(reward),
                'timeout': self.steps >= self.horizon and not success, 'action': a.tolist()}

    def close(self):
        self.env.close()

