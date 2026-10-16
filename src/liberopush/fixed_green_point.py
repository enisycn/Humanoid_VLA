"""A fixed, physically validated target marker in the unchanged native scene."""
import hashlib

import mujoco
import numpy as np


def add_fixed_point(env, out, goal, language, outline=True):
    sim = env.env.sim
    model = sim.model._model
    before = {name: getattr(model, name).copy() for name in dir(model)
              if isinstance(getattr(model, name), np.ndarray)}
    qpos, qvel = sim.data.qpos.copy(), sim.data.qvel.copy()
    native_success = bool(env.env.check_success())
    xml = sim.model.get_xml()
    goal = np.asarray(goal, dtype=float).copy()
    top_id = sim.model.site_name2id('table_top')
    centre = np.r_[goal, sim.data.site_xpos[top_id, 2] + .0007]
    angles = np.linspace(0, 2*np.pi, 33)
    vertices = centre + np.c_[.085*np.cos(angles), .085*np.sin(angles), np.zeros(33)]
    context = sim._render_context_offscreen
    native_render = context.render
    color = np.array([.02, 1., .02, 1.], np.float32)

    def render(width, height, camera_id=None, segmentation=False):
        native_render(width, height, camera_id, segmentation)
        if segmentation:
            raise ValueError('This point experiment records RGB only')
        assert context.scn.ngeom + (33 if outline else 1) <= context.scn.maxgeom
        geom = context.scn.geoms[context.scn.ngeom]
        mujoco.mjv_initGeom(geom, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           size=np.array([.009, .0001, 0.]), pos=centre,
                           mat=np.eye(3).ravel(), rgba=color)
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        context.scn.ngeom += 1
        if outline:
            for start, end in zip(vertices[:-1], vertices[1:]):
                geom = context.scn.geoms[context.scn.ngeom]
                mujoco.mjv_initGeom(geom, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                                   size=np.ones(3)*.0008, pos=centre,
                                   mat=np.eye(3).ravel(), rgba=color)
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, .0008, start, end)
                geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                context.scn.ngeom += 1
        mujoco.mjr_render(mujoco.MjrRect(0, 0, width, height), context.scn, context.con)

    context.render = render
    env.language = language
    env.raw = env.env.env._get_observations(force_update=True)
    assert all(np.array_equal(value, getattr(model, name)) for name, value in before.items())
    assert np.array_equal(qpos, sim.data.qpos) and np.array_equal(qvel, sim.data.qvel)
    assert bool(env.env.check_success()) == native_success
    out.mkdir(parents=True, exist_ok=True)
    (out/'native-scene.xml').write_text(xml)
    return dict(scene='full original LIBERO plus a fixed green point target',
                goal_xy_m=goal.tolist(), language=language,
                marker_world_xyz_m=centre.tolist(), marker_radius_mm=9,
                outline_radius_mm=85 if outline else None,
                outline_is_not_a_success_region=True, point_success_tolerance_mm=10,
                selection='fixed geometry-only straight-lane preflight; not policy endpoints',
                original_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
                all_native_model_arrays_bitwise_unchanged=True, checked_model_arrays=len(before),
                qpos_qvel_bitwise_unchanged=True, native_goal_region_unmodified=True,
                native_success_unchanged_at_reset=True, no_xml_reload=True,
                no_robot_preposition=True, no_removed_objects=True, no_object_replacement=True)
