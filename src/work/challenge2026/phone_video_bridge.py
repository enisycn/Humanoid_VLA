"""One synthetic 3D hand-video -> measured metric path -> physical Panda pilot.

No training or model loading is implemented. The main renderer owner must run
generate/retarget serially. Assets/track/qa/describe are CPU-only. Generated hand
video is explicitly synthetic, never personal footage. The same tracker can
use a measured real-phone calibration contract. Tracking consumes only
compressed video and declared marker/camera calibration, never source motion.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET

import av
import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PREP = ROOT / 'work/phone_video_prep'
SOURCE_KIND = 'sim_retargeted_from_generated_hand_video'
PERSONAL_SOURCE_KIND = 'sim_retargeted_from_personal_hand_video'
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def new_dir(path, root=ROOT):
    path = Path(path).resolve()
    require(path.is_relative_to(root), f'Output must be inside {root}')
    path.mkdir(parents=True, exist_ok=False)
    return path


def detector():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(DICT, params)


def detect(rgb):
    corners, ids, _ = detector().detectMarkers(rgb)
    if ids is None:
        return {}
    return {int(i): np.asarray(c[0], float) for i, c in zip(ids.ravel(), corners)}


def asset_contract():
    height, width, fovy = 1080, 1440, 55.
    focal = height/(2*math.tan(math.radians(fovy)/2))
    return dict(
        schema='synthetic_hand_phone_pilot_v3_independent_XY_first', synthetic_not_personal=True,
        source_kind=SOURCE_KIND, personal_phone_episodes=0,
        table_top_z_m=.9, calibration_marker_plane_z_m=.9006,
        table_friction=[.6,.005,.0001],table_friction_basis='native LIBERO table_collision friction .6 with MuJoCo default secondary coefficients',
        dictionary='DICT_4X4_50', calibration_marker_ids=[0, 1, 2, 3],
        calibration_marker_black_side_m=.035,
        calibration_marker_centres_world_xy_m=[[-.18, -.22], [.18, -.22], [.18, .22], [-.18, .22]],
        fingertip_marker_id=10, fingertip_marker_black_side_m=.020,
        fingertip_marker_world_z_m=.9252,
        fingertip_marker_to_contact_xy_m=[0., .020],
        fingertip_contact_world_z_m=.914,
        physical_index_pad=dict(body_local_centre_m=[0.,0.,-.011],half_size_m=[.008,.020,.0085],
            world_Z_interval_m=[.9055,.9225],full_XY_size_m=[.016,.040],
            contact_reference='front face centre at worldZ .914',
            fixture_reason='restore original v1 pad after one unsuccessful upper-rim retry; stop physical fixture tuning',
            previous_pilots_preserved_and_excluded=True),
        native_plate_debug_collision_rendering='only copied native plate group0 collision RGBA alpha0; all collision/mass/geometry parameters unchanged',
        fingertip_heading='fixed WORLD+Y, rigid procedural hand; heading changes unsupported in this one pilot',
        object_marker_id=11, object_marker_black_side_m=.020,
        object_marker_local_z_m=.017,
        object_marker_world_z_m=.919506338529,
        object_marker_height_basis='native flat plate centre at 2.506338529mm above table plus 17mm known marker support; constant-height assumption audited separately',
        marker_texture_pixels=640, marker_black_pixels=512,
        marker_texture_physical_side='black_side*640/512, including white quiet border',
        plate='exact native LIBERO plate visual mesh and collision boxes, compiler inertia group0 only; 146mm is the older conservative external proxy, not asserted native visual diameter',
        conservative_external_recording_proxy_diameter_m=.146,
        phone=dict(width=width, height=height, fps=30, fovy_deg=fovy,
                   intrinsics=[[focal, 0., width/2-.5], [0., focal, height/2-.5], [0., 0., 1.]],
                   intrinsics_basis='declared fixed renderer camera, not fitted to source trajectory',
                   position_world_m=[.20, -.35, 1.65], look_at_world_m=[.05, 0., .91]),
        pilot=dict(canonical_anchor_id='train-A',object_start_xy_m=[-.005, -.025], intended_goal_xy_m=[-.005, .095],
                   nominal_collision_rear_radius_m=.067, initial_tip_gap_m=.015,
                   contact_displacement_m=.135, hand_speed_m_s=.020, start_hold_s=1., end_hold_s=1.),
        tracking=dict(inputs_only=['compressed video bytes', 'declared camera intrinsics', 'known four marker world coordinates',
                                   'known marker planes/heights', 'fixed fingertip-to-contact offset'],
                      source_motion_truth_for_tracker=False, source_camera_extrinsic_for_tracker=False,
                      method='four decoded ArUco centres -> table homography -> camera extrinsic from known K -> separate known-height homographies',
                      maximum_reference_motion_px=3., minimum_visible_fraction=.95, maximum_gap_s=.10,
                      qualification_revision='v3 fresh independent clip: XY-first validation with predeclared height envelope; v1/v2 verdicts unchanged and excluded',
                      qualification_change_reason='v1 actual XY maximum2.270mm met the measurement requirements while the internal2mm constant-height assumption failed; no GT fitting or retrospective requalification',
                      object_marker_height_envelope_mm=5.,constant_height_diagnostic_mm=2.,
                      robot_reference_sampling='20Hz previous decoded valid frame only, zero-order hold; no future-frame interpolation',
                      synthetic_QA_bounds=dict(median_XY_error_mm=5.,p95_XY_error_mm=10.,maximum_XY_error_mm=15.,
                                               maximum_marker_height_error_mm=5.,maximum_source_body_lift_mm=5.,
                                               maximum_image_derived_XY_uncertainty_mm=5.,minimum_detection_coverage=.95,
                                               maximum_source_timestamp_match_error_s=.001)),
        limits=['Procedural rigid hand, one colliding fingertip pad; not a biomechanical human hand.',
                'Synthetic imagery and physically executed robot RGB are not personal human RGB.',
                'Real phone lens intrinsics and physical marker heights require measurement before using this height correction.',
                'One straight short push validates a pipeline only; no VLA ability or improvement claim.'])


def prepare_assets(out):
    out = new_dir(out, PREP)
    contract = asset_contract()
    checks = []
    for ident in [0, 1, 2, 3, 10, 11]:
        marker = cv2.aruco.generateImageMarker(DICT, ident, 512)
        texture = np.full((640, 640), 255, np.uint8)
        texture[64:576, 64:576] = marker
        path = out/f'aruco-{ident}.png'
        require(cv2.imwrite(str(path), texture), 'Marker texture write failed')
        detected = detect(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB))
        require(ident in detected, f'CPU marker decode failed for ID{ident}')
        checks.append(dict(id=ident, path=str(path), sha256=sha(path), decoded_from_actual_PNG=True))
    contract['textures'] = checks
    contract['opencv_version'] = cv2.__version__
    contract['bridge_source_sha256'] = sha(__file__)
    save(out/'scale-contract.json', contract)
    save(out/'asset-checks.json', dict(passed=True, checks=checks, renderer_started=False, simulator_started=False))
    print(json.dumps(dict(status='CPU_assets_ready', contract=str(out/'scale-contract.json'), marker_ids=[r['id'] for r in checks])), flush=True)


def vector_text(value):
    return ' '.join(str(float(x)) for x in value)


def scene_xml(contract, assets):
    """Build source XML; model/renderer construction happens only in generate."""
    root = ET.Element('mujoco', model='synthetic_hand_phone_pilot')
    ET.SubElement(root, 'compiler', angle='radian', inertiagrouprange='0 0')
    ET.SubElement(root, 'option', timestep='.002', gravity='0 0 -9.81', impratio='20', cone='elliptic', density='1.2', viscosity='.00002')
    visual = ET.SubElement(root, 'visual')
    ET.SubElement(visual, 'global', offwidth=str(contract['phone']['width']), offheight=str(contract['phone']['height']))
    ET.SubElement(visual, 'map', znear='.001')
    asset = ET.SubElement(root, 'asset')
    for ident in [0, 1, 2, 3, 10, 11]:
        ET.SubElement(asset, 'texture', name=f'marker{ident}', type='2d', file=str((assets/f'aruco-{ident}.png').resolve()))
        ET.SubElement(asset, 'material', name=f'marker{ident}', texture=f'marker{ident}', texrepeat='1 1', texuniform='false', emission='1', specular='0', reflectance='0')
    spec = importlib.util.find_spec('libero')
    plate_dir = Path(next(iter(spec.submodule_search_locations)))/'libero/assets/stable_scanned_objects/plate'
    plate_source = ET.parse(plate_dir/'plate.xml').getroot()
    source_asset = copy.deepcopy(plate_source.find('asset'))
    for node in source_asset.iter():
        for field in ['name', 'texture', 'material', 'mesh']:
            if field in node.attrib:
                node.attrib[field] = 'phone_plate_'+node.attrib[field]
        if 'file' in node.attrib:
            node.attrib['file'] = str((plate_dir/node.attrib['file']).resolve())
    asset.extend(list(source_asset))
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -.3 2', dir='0 .2 -1', diffuse='.8 .8 .8')
    ET.SubElement(world, 'geom', name='table', type='box', pos='0 0 .875', size='.42 .40 .025', rgba='.58 .52 .44 1', friction=vector_text(contract['table_friction']))
    position = np.asarray(contract['phone']['position_world_m']); target = np.asarray(contract['phone']['look_at_world_m'])
    z = position-target; z /= np.linalg.norm(z)
    x = np.cross([0., 1., 0.], z); x /= np.linalg.norm(x); y = np.cross(z, x)
    ET.SubElement(world, 'camera', name='phone', pos=vector_text(position), xyaxes=vector_text(np.r_[x, y]), fovy=str(contract['phone']['fovy_deg']))

    def marker(parent, ident, position, black_side):
        half = black_side*640/512/2
        # A finite thin box gives MuJoCo a nondegenerate visual texture surface.
        ET.SubElement(parent, 'geom', name=f'marker{ident}', type='box', pos=vector_text(position),
                      size=f'{half} {half} .0001', material=f'marker{ident}', contype='0', conaffinity='0', group='1', density='0')
    for ident, xy in enumerate(contract['calibration_marker_centres_world_xy_m']):
        marker(world, ident, [*xy, contract['calibration_marker_plane_z_m']-.0001], .035)
    plate = copy.deepcopy(plate_source.find('./worldbody/body/body'))
    plate.attrib.update(name='phone_plate', pos=vector_text([*contract['pilot']['object_start_xy_m'], .902506338529]), quat='.7071067811865475 0 0 .7071067811865476')
    for index, node in enumerate(plate.iter()):
        if node is plate:
            continue
        for field in ['name', 'material', 'mesh']:
            if field in node.attrib:
                node.attrib[field] = 'phone_plate_'+node.attrib[field]
        if node.tag == 'geom':
            node.attrib['name'] = f'phone_plate_g{index}'
            if node.attrib.get('group') == '0':
                rgba=np.fromstring(node.attrib.get('rgba','.5 .5 .5 1'),sep=' ');rgba[3]=0.
                node.attrib['rgba']=vector_text(rgba)
    ET.SubElement(plate, 'joint', name='plate_free', type='free', damping='.0005')
    ET.SubElement(plate, 'geom', name='marker11_visual_support', type='cylinder',
                  pos='0 0 .0084', size='.012 .0084', rgba='.9 .9 .9 1',
                  contype='0', conaffinity='0', group='1', density='0')
    marker(plate, 11, [0., 0., contract['object_marker_local_z_m']-.0001], .020)
    world.append(plate)
    pilot = contract['pilot']; start = np.asarray(pilot['object_start_xy_m'])
    first_marker = start-[0., pilot['nominal_collision_rear_radius_m']+pilot['initial_tip_gap_m']+.020]
    hand = ET.SubElement(world, 'body', name='procedural_hand', mocap='true', pos=vector_text([*first_marker, .925]))
    ET.SubElement(hand, 'geom', name='palm_visual', type='ellipsoid', pos='0 -.083 .015', size='.036 .043 .015', rgba='.68 .40 .26 1', contype='0', conaffinity='0', group='1')
    for name, x0, y_end, radius in [('thumb', -.043, -.023, .008), ('index', 0., .009, .007),
                                    ('middle', .019, .001, .007), ('ring', .037, -.011, .0065), ('little', .052, -.026, .006)]:
        start_y = -.067 if name != 'thumb' else -.080
        for segment in range(3):
            ys = np.linspace(start_y, y_end, 4)
            ET.SubElement(hand, 'geom', name=f'{name}_segment{segment}_visual', type='capsule',
                          fromto=vector_text([x0, ys[segment], -.008, x0, ys[segment+1], -.008]),
                          size=str(radius), rgba='.72 .44 .29 1', contype='0', conaffinity='0', group='1')
    pad=contract['physical_index_pad']
    ET.SubElement(hand, 'geom', name='index_contact_pad', type='box', pos=vector_text(pad['body_local_centre_m']),size=vector_text(pad['half_size_m']),
                  rgba='.72 .44 .29 1', friction='1 .005 .0001', condim='4')
    marker(hand, 10, [0., 0., .0001], .020)
    goal = contract['pilot']['intended_goal_xy_m']
    ET.SubElement(world, 'geom', name='green_goal_visual', type='cylinder', pos=vector_text([*goal, .9003]), size='.009 .0001', rgba='.05 .9 .05 1', contype='0', conaffinity='0', group='1')
    return ET.tostring(root, encoding='unicode'), plate_dir/'plate.xml', first_marker


def generate(assets, out):
    import mujoco  # No import/model/renderer above the explicit generation entry.
    assets = Path(assets).resolve(); contract = json.loads((assets/'scale-contract.json').read_text())
    out = new_dir(out, ROOT/'runs')
    xml, plate_asset, first = scene_xml(contract, assets)
    (out/'scene.xml').write_text(xml)
    save(out/'design-before-run.json', dict(source='synthetic_3D_hand', synthetic_not_personal=True,
        bridge_source_sha256=sha(__file__), scale_contract_sha256=sha(assets/'scale-contract.json'),
        native_plate_asset_sha256=sha(plate_asset), actor='predeclared hand mocap position only',
        no_object_state_injection_during_motion=True, physics='MuJoCo native plate collision asset, only actual mocap fingertip pushes',
        hand_model='palm+five3segment visual fingers; original index pad is sole physical hand collider',
        physical_index_pad=contract['physical_index_pad'],contact_reference=contract['physical_index_pad']['contact_reference'],
        collision_visualization=contract['native_plate_debug_collision_rendering'],
        start_hold_s=1., end_hold_s=1., hand_ground_truth_not_used_by_tracker=True))
    model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
    model.opt.timestep = .002
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'phone_plate')
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'procedural_hand')
    mocap_id = int(model.body_mocapid[hand_id])
    for _ in range(250):
        mujoco.mj_step(model, data)
    motion = contract['pilot']['contact_displacement_m']/contract['pilot']['hand_speed_m_s']
    duration = 1.+motion+1.; fps = contract['phone']['fps']
    writer = cv2.VideoWriter(str(out/'phone.raw.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                             (contract['phone']['width'], contract['phone']['height']))
    require(writer.isOpened(), 'Phone video writer unavailable')
    renderer = None; truth = []; contacts = 0; initial = data.xpos[body].copy(); time0 = float(data.time); maximum_body_lift=0.
    try:
        renderer = mujoco.Renderer(model, height=contract['phone']['height'], width=contract['phone']['width'])
        for frame_index, t in enumerate(np.arange(0., duration+1e-9, 1/fps)):
            while data.time-time0 < t-1e-9:
                elapsed = float(data.time-time0)
                dy = min(max(elapsed-1., 0.)*contract['pilot']['hand_speed_m_s'], contract['pilot']['contact_displacement_m'])
                data.mocap_pos[mocap_id] = [first[0], first[1]+dy, .925]
                mujoco.mj_step(model, data)
                maximum_body_lift=max(maximum_body_lift,float(data.xpos[body,2]-initial[2])*1000)
                for contact in data.contact[:data.ncon]:
                    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or '' for g in [contact.geom1, contact.geom2]]
                    contacts += int('index_contact_pad' in names and any(n.startswith('phone_plate_g') for n in names))
            renderer.update_scene(data, camera='phone')
            rgb = renderer.render()
            if frame_index == 0:
                cv2.imwrite(str(out/'first-frame.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            plate_marker = data.xpos[body]+data.xmat[body].reshape(3,3)@np.array([0.,0.,.017])
            hand_marker = data.mocap_pos[mocap_id]+[0.,0.,.0002]
            truth.append([float(t), *plate_marker.tolist(), *hand_marker.tolist(), *data.xpos[body].tolist()])
            if frame_index % 60 == 0:
                print(json.dumps(dict(stage='generate', frame=frame_index, seconds=float(t), physical_contact_samples=contacts)), flush=True)
    finally:
        writer.release()
        if renderer is not None:
            renderer.close()
        np.savez_compressed(out/'source-QA-truth.npz', rows=np.asarray(truth))
    video = out/'synthetic-hand-phone.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', '-i', str(out/'phone.raw.mp4'),
                    '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)], check=True)
    video_id = 'synthetic-hand-'+sha(video)[:16]
    contract['generated_video_id'] = video_id
    contract['video_sha256'] = sha(video)
    save(out/'scale-contract.json', contract)
    save(out/'generation-report.json', dict(synthetic_not_personal=True, generated_video_id=video_id,
        video_sha256=sha(video), source_kind=SOURCE_KIND, personal_video_id=None, personal_phone_episodes=0,
        contact_samples=contacts, initial_plate_xyz=initial.tolist(), final_plate_xyz=data.xpos[body].tolist(),
        actual_plate_displacement_mm=float(np.linalg.norm(data.xpos[body][:2]-initial[:2])*1000),
        source_QA_all_substep_maximum_body_lift_mm=maximum_body_lift,
        source_truth_for_QA_only=str(out/'source-QA-truth.npz'), tracker_has_not_run=True,
        plate_mass_kg=float(model.body_mass[body]), plate_body_inertia_kg_m2=model.body_inertia[body].tolist(),
        native_plate_compiler_inertiagrouprange=[0,0], no_training=True))
    print(json.dumps(dict(status='synthetic_video_generated', video=str(video), video_id=video_id)), flush=True)


def homographies(reference_pixels, contract):
    """Extrinsic is estimated from decoded reference centres, not renderer pose."""
    world = np.asarray(contract['calibration_marker_centres_world_xy_m'], float)
    H0 = cv2.getPerspectiveTransform(world.astype(np.float32), np.asarray(reference_pixels, np.float32))
    K = np.asarray(contract['phone']['intrinsics'], float)
    h = np.linalg.inv(K)@H0
    scale = .5*(np.linalg.norm(h[:,0])+np.linalg.norm(h[:,1]))
    if h[2,2] < 0:
        h = -h
    r1, r2, t = h[:,0]/scale, h[:,1]/scale, h[:,2]/scale
    approximate = np.column_stack([r1,r2,np.cross(r1,r2)])
    u, _, vt = np.linalg.svd(approximate); R = u@vt
    if np.linalg.det(R) < 0:
        u[:,-1] *= -1; R = u@vt
    result = {}
    for ident, z in [(10,contract['fingertip_marker_world_z_m']), (11,contract['object_marker_world_z_m'])]:
        relative = z-contract['calibration_marker_plane_z_m']
        H = K@np.column_stack([R[:,0],R[:,1],t+R[:,2]*relative])
        result[ident] = np.linalg.inv(H)
    return result, np.linalg.inv(H0), R, t


def project(H, pixel):
    p = H@np.r_[pixel,1.]
    return p[:2]/p[2]


def height_envelope_homographies(contract, R, t):
    """Pixel-only uncertainty planes; no generator pose or motion truth."""
    K=np.asarray(contract['phone']['intrinsics'],float)
    envelope=contract['tracking']['object_marker_height_envelope_mm']/1000
    relative=contract['object_marker_world_z_m']-contract['calibration_marker_plane_z_m']
    return [np.linalg.inv(K@np.column_stack([R[:,0],R[:,1],t+R[:,2]*(relative+delta)]))
            for delta in [-envelope,envelope]]


def causal_samples(rows, period=.05, maximum_age=.10):
    """Each reference uses only a valid decoded frame at or before its tick."""
    valid=rows[:,5].astype(bool)
    require(valid[0], 'First decoded frame must provide contact and object; no initial future-frame filling')
    times=np.arange(rows[0,0],rows[-1,0]+1e-9,period)
    valid_indices=np.flatnonzero(valid);selected=np.searchsorted(rows[valid,0],times,side='right')-1
    require(np.all(selected>=0), 'No past decoded frame for an initial 20Hz sample')
    source_frames=valid_indices[selected];source_times=rows[source_frames,0];ages=times-source_times
    require(np.all(ages>=0) and np.max(ages)<=maximum_age+1e-6, 'Past-frame sample age exceeds 100ms')
    return times,rows[source_frames,1:5].copy(),source_frames,source_times,ages


def track(video, contract_path, out):
    video, contract_path = Path(video).resolve(), Path(contract_path).resolve()
    contract = json.loads(contract_path.read_text())
    identity = capture_identity(contract)
    require(contract.get('dictionary')=='DICT_4X4_50', 'Only the declared existing DICT_4X4_50 marker decoder is implemented')
    require(contract.get('video_sha256') == sha(video), 'Compressed video identity differs from contract')
    out = new_dir(out, ROOT/'runs')
    save(out/'design-before-tracking.json', dict(video=str(video), video_sha256=sha(video),
        scale_contract_sha256=sha(contract_path), bridge_source_sha256=sha(__file__),
        opencv_version=cv2.__version__,
        inputs=contract['tracking']['inputs_only'], source_truth_or_generator_motion_loaded=False,
        source_camera_pose_loaded_for_calibration=False, **identity))
    rows, shifts, calibration_rows, detected_corners, uncertainty = [], [], [], [], []
    H = Hflat = first_refs = None; first_time = None; R = t = None
    with av.open(str(video)) as container:
        for frame in container.decode(video=0):
            require(frame.time is not None, 'Actual compressed video timestamps required')
            now = float(frame.time)
            if first_time is None:
                first_time = now
            rgb = frame.to_ndarray(format='rgb24'); markers = detect(rgb)
            detected_corners.append(np.array([markers.get(i,np.full((4,2),np.nan)) for i in [0,1,2,3,10,11]]))
            refs = np.array([markers[i].mean(axis=0) for i in range(4)]) if all(i in markers for i in range(4)) else None
            if H is None:
                cv2.imwrite(str(out/'first-decoded-frame.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                require(refs is not None, 'First decoded frame must show all four calibration markers')
                H, Hflat, R, t = homographies(refs, contract); first_refs = refs.copy()
                envelope_H=height_envelope_homographies(contract,R,t)
            if refs is not None:
                shifts.append(float(np.linalg.norm(refs-first_refs,axis=1).max()))
                calibration_rows.append(now-first_time)
            valid = 10 in markers and 11 in markers
            obj = project(H[11],markers[11].mean(axis=0)) if valid else np.full(2,np.nan)
            finger = project(H[10],markers[10].mean(axis=0)) if valid else np.full(2,np.nan)
            flat_obj = project(Hflat,markers[11].mean(axis=0)) if valid else np.full(2,np.nan)
            flat_finger = project(Hflat,markers[10].mean(axis=0)) if valid else np.full(2,np.nan)
            uncertainty.append(max(float(np.linalg.norm(project(h,markers[11].mean(axis=0))-obj)*1000)
                                   for h in envelope_H) if valid else np.nan)
            rows.append([now-first_time,*obj,*finger,float(valid),*flat_obj,*flat_finger])
    a = np.asarray(rows,float)
    np.savez_compressed(out/'decoded-pixels-and-metric-rows.npz', rows=a,
                        calibration_frame_times=np.asarray(calibration_rows), reference_motion_px=np.asarray(shifts),
                        marker_corners_px=np.asarray(detected_corners), marker_ids=np.array([0,1,2,3,10,11]))
    require(len(a)>0, 'No video frames decoded')
    valid = a[:,5].astype(bool)
    require(len(a)>=90 and np.all(np.diff(a[:,0])>0), 'At least three seconds of chronological video required')
    require(valid.mean()>=.95 and len(shifts)>=.8*len(a) and max(shifts)<=3., 'Marker visibility or fixed-camera check failed')
    require(np.diff(a[valid,0]).max()<=.10+1e-6, 'Contact/object marker occlusion exceeds100ms')
    uncertainty=np.asarray(uncertainty,float);maximum_uncertainty=float(max(uncertainty[valid]))
    uncertainty_path=out/'height-envelope-uncertainty.json'
    uncertainty_pass=maximum_uncertainty<=contract['tracking']['synthetic_QA_bounds']['maximum_image_derived_XY_uncertainty_mm']
    save(uncertainty_path,dict(video_sha256=sha(video),scale_contract_sha256=sha(contract_path),
        bridge_source_sha256=sha(__file__),decoded_pixels_sha256=sha(out/'decoded-pixels-and-metric-rows.npz'),
        qualification_revision=contract['tracking']['qualification_revision'],
        declared_object_marker_height_envelope_mm=contract['tracking']['object_marker_height_envelope_mm'],
        maximum_object_XY_uncertainty_mm=maximum_uncertainty,uncertainty_pass=uncertainty_pass,
        valid_frame_indices=np.flatnonzero(valid).tolist(),valid_frame_uncertainty_mm=uncertainty[valid].tolist(),
        method='max distance from nominal XY to decoded object ray intersections with nominal markerZ plus/minus5mm; estimated camera from decoded table markers',
        uncertainty_scope='height interval only, conditional on the same decoded ray and estimated calibration; pixel/calibration error validated separately by actual XY QA',
        source_truth_loaded=False,renderer_pose_used=False,measured_path_modified=False))
    require(uncertainty_pass,'Declared height-envelope image-derived XY uncertainty exceeds5mm')
    times,state,source_frames,source_times,sample_ages=causal_samples(a)
    start = np.median(state[times<=times[0]+1.,:2],axis=0)
    end = np.median(state[times>=times[-1]-1.,:2],axis=0)
    drift = float(np.linalg.norm(state[times>=times[-1]-1.,:2]-end,axis=1).max()*1000)
    contact = state[:,2:4]+np.asarray(contract['fingertip_marker_to_contact_xy_m'])
    measured_motion = float(np.linalg.norm(end-start)*1000)
    require(measured_motion>=30., 'Decoded object has no meaningful physical push')
    np.savez_compressed(out/'measured.npz', times=times, state=state, contact_xy=contact,
                        decoded_frame_rows=a, calibration_frame_times=np.asarray(calibration_rows),
                        reference_source_frame_indices=source_frames,reference_source_frame_times_s=source_times,
                        reference_sample_age_s=sample_ages,object_XY_height_envelope_uncertainty_mm=uncertainty)
    report = dict(**identity, video_sha256=sha(video), opencv_version=cv2.__version__,
        video_path=str(video), scale_contract_path=str(contract_path), scale_contract_sha256=sha(contract_path),
        whole_video_lineage_id=('generated-video-' if identity['synthetic_not_personal'] else 'personal-video-')+sha(video),
        measured_npz_sha256=sha(out/'measured.npz'),
        height_uncertainty_report_path=str(uncertainty_path),height_uncertainty_report_sha256=sha(uncertainty_path),
        maximum_image_derived_object_XY_uncertainty_mm=maximum_uncertainty,
        qualification_revision=contract['tracking']['qualification_revision'],
        marker_visible_fraction=float(valid.mean()), reference_visible_fraction=len(shifts)/len(a),
        maximum_reference_motion_px=max(shifts), frames=len(a), duration_s=float(a[-1,0]),
        metric_object_start_xy_m=start.tolist(), metric_object_end_xy_m=end.tolist(),
        measured_object_motion_mm=measured_motion, end_hold_max_radius_mm=drift,
        goal_for_retarget='decoded object end-hold median fixed before Panda reset; not chosen from robot outcomes',
        height_correction=True, calibration_camera_R_estimated_from_pixels=R.tolist(),
        calibration_camera_t_relative_reference_plane_m=t.tolist(),
        height_correction_max_difference_from_flat_table_homography_mm=float(max(
            np.linalg.norm(a[valid,1:3]-a[valid,6:8],axis=1).max(),
            np.linalg.norm(a[valid,3:5]-a[valid,8:10],axis=1).max())*1000),
        source_truth_loaded=False, renderer_extrinsic_used=False,
        calibration_basis=contract['tracking']['method'], robot_action_labels_available=False,
        actually_decoded_compressed_video=True, trajectory_columns=['object_x','object_y','fingertip_marker_x','fingertip_marker_y'],
        measurement_passed=True, model_or_training_run=False)
    report.update(reference_sampling='20Hz zero-order hold from previous decoded valid frame only',
        reference_uses_future_valid_frames=False,maximum_reference_sample_age_ms=float(sample_ages.max()*1000),
        decoded_valid_frames=int(valid.sum()),decoded_missing_frames=int((~valid).sum()),
        missing_frame_indices=np.flatnonzero(~valid).tolist(),all_decoded_frames_saved=True,
        sample_source_frame_indices=source_frames.tolist(),sample_source_frame_times_s=source_times.tolist(),sample_age_s=sample_ages.tolist())
    save(out/'report.json',report)
    print(json.dumps(report),flush=True)
    return report


def capture_identity(contract):
    """Same pixel tracker supports measured real captures without relabeling one."""
    synthetic = contract.get('synthetic_not_personal')
    require(isinstance(synthetic, bool), 'Contract must explicitly distinguish synthetic from real capture')
    if synthetic:
        ident = contract.get('generated_video_id')
        require(isinstance(ident, str) and bool(ident), 'Generated video identity required')
        require(not contract.get('personal_video_id'), 'Synthetic capture cannot claim a personal video identity')
        return dict(source_kind=SOURCE_KIND, synthetic_not_personal=True, generated_video_id=ident,
                    personal_video_id=None, personal_phone_episodes=0)
    ident = contract.get('personal_video_id')
    require(isinstance(ident, str) and bool(ident), 'Real capture requires its actual declared personal video identity')
    require(contract.get('real_phone_calibration_measured') is True,
            'Real capture requires measured intrinsics, marker coordinates, heights and contact offset')
    require(not contract.get('generated_video_id'), 'Real capture cannot claim a generated video identity')
    return dict(source_kind=PERSONAL_SOURCE_KIND, synthetic_not_personal=False, generated_video_id=None,
                personal_video_id=ident, personal_phone_episodes=1)


def qa(generated, decoded, out):
    """Separate comparison only: never repair or replace measured video paths."""
    generated, decoded = Path(generated).resolve(), Path(decoded).resolve()
    report=json.loads((decoded/'report.json').read_text())
    generation=json.loads((generated/'generation-report.json').read_text())
    require(report['synthetic_not_personal'] is True, 'Generator truth QA only applies to synthetic capture')
    require(report['video_sha256']==generation['video_sha256']==sha(generated/'synthetic-hand-phone.mp4'),
            'QA sources do not bind the same whole compressed video')
    require(report['measured_npz_sha256']==sha(decoded/'measured.npz'), 'Measured trajectory changed')
    with np.load(generated/'source-QA-truth.npz',allow_pickle=False) as f:truth=f['rows']
    with np.load(decoded/'measured.npz',allow_pickle=False) as f:
        rows=f['decoded_frame_rows'];times=f['times'];contact=f['contact_xy'];source_frames=f['reference_source_frame_indices']
    contract=json.loads(Path(report['scale_contract_path']).read_text())
    uncertainty_path=Path(report['height_uncertainty_report_path']);uncertainty_report=json.loads(uncertainty_path.read_text())
    require(sha(uncertainty_path)==report['height_uncertainty_report_sha256']
            and uncertainty_report['video_sha256']==report['video_sha256'], 'Uncertainty artifact changed')
    valid=rows[:,5].astype(bool);valid_rows=rows[valid]
    # Timestamp matching only, not source-path interpolation or trajectory fitting.
    def nearest(times_to_match):
        right=np.clip(np.searchsorted(truth[:,0],times_to_match),0,len(truth)-1)
        left=np.maximum(right-1,0)
        return np.where(abs(truth[right,0]-times_to_match)<abs(truth[left,0]-times_to_match),right,left)
    matches=nearest(rows[:,0]);timestamp_errors=abs(truth[matches,0]-rows[:,0])
    object_errors=np.linalg.norm(valid_rows[:,1:3]-truth[matches[valid],1:3],axis=1)*1000
    finger_errors=np.linalg.norm(valid_rows[:,3:5]-truth[matches[valid],4:6],axis=1)*1000
    truth_contact=truth[matches[source_frames],4:6]+contract['fingertip_marker_to_contact_xy_m']
    contact_errors=np.linalg.norm(contact-truth_contact,axis=1)*1000
    object_height=np.abs(truth[:,3]-contract['object_marker_world_z_m'])*1000
    finger_height=np.abs(truth[:,6]-contract['fingertip_marker_world_z_m'])*1000
    frame_body_lift=float(max(0.,max(truth[:,9]-truth[0,9])*1000))
    body_lift=max(frame_body_lift,float(generation['source_QA_all_substep_maximum_body_lift_mm']))
    out=new_dir(out,ROOT/'runs')
    bounds=contract['tracking']['synthetic_QA_bounds']
    def stats(values):return dict(median_mm=float(np.median(values)),p95_mm=float(np.quantile(values,.95)),maximum_mm=float(max(values)))
    complete=bool(len(rows)==len(truth) and len(np.unique(matches))==len(truth))
    passed=bool(complete and timestamp_errors.max()<=bounds['maximum_source_timestamp_match_error_s']
        and valid.mean()>=bounds['minimum_detection_coverage'] and max(object_height)<=bounds['maximum_marker_height_error_mm']
        and max(finger_height)<=bounds['maximum_marker_height_error_mm']
        and body_lift<=bounds['maximum_source_body_lift_mm'] and uncertainty_report['uncertainty_pass']
        and uncertainty_report['maximum_object_XY_uncertainty_mm']<=bounds['maximum_image_derived_XY_uncertainty_mm'] and all(
        np.median(v)<=bounds['median_XY_error_mm'] and np.quantile(v,.95)<=bounds['p95_XY_error_mm']
        and max(v)<=bounds['maximum_XY_error_mm'] for v in [object_errors,finger_errors,contact_errors]))
    save(out/'report.json',dict(qa_only=True,measured_trajectory_not_modified=True,
        tracker_source_truth_loaded=False,retarget_actor_uses_QA_truth=False,
        video_sha256=report['video_sha256'],measurement_report_sha256=sha(decoded/'report.json'),
        source_truth_sha256=sha(generated/'source-QA-truth.npz'),
        generation_report_sha256=sha(generated/'generation-report.json'),
        object_XY_error=stats(object_errors),fingertip_XY_error=stats(finger_errors),resampled_contact_XY_error=stats(contact_errors),
        max_declared_object_marker_height_error_mm=float(max(object_height)),
        max_declared_fingertip_marker_height_error_mm=float(max(finger_height)),
        source_body_maximum_lift_mm=body_lift,video_frame_body_maximum_lift_mm=frame_body_lift,
        source_body_lift_basis='maximum at every source physics substep from hash-bound generator; decoded-frame truth checked separately',
        constant_height_2mm_diagnostic_pass=bool(max(object_height)<=contract['tracking']['constant_height_diagnostic_mm']
                                               and max(finger_height)<=contract['tracking']['constant_height_diagnostic_mm']),
        constant_height_2mm_is_primary_gate=False,
        height_uncertainty_report_sha256=sha(uncertainty_path),
        maximum_image_derived_object_XY_uncertainty_mm=uncertainty_report['maximum_object_XY_uncertainty_mm'],
        qualification_revision=contract['tracking']['qualification_revision'],
        source_truth_frames=len(truth),decoded_frames=len(rows),valid_marker_frames=int(valid.sum()),
        missing_marker_frames=int((~valid).sum()),missing_marker_frame_indices=np.flatnonzero(~valid).tolist(),
        detection_coverage=float(valid.mean()),whole_video_no_frame_skip=complete,
        maximum_timestamp_match_error_ms=float(timestamp_errors.max()*1000),predeclared_bounds=bounds,
        metric_measurement_QA_pass=passed,
        gate='QA diagnostic only; no motion correction, no bypass of tracker, no real-video truth assumed',
        simulator_or_model_started=False))
    print((out/'report.json').read_text(),flush=True)


def retarget(decoded, out, index=0, seed=889100):
    """Own-proprio tracking of current decoded contact samples; no object feedback."""
    import torch
    from liberopush.native import NativePanda
    from liberopush.contact_frame import contact_point, geometry_contract, rotation
    from work.challenge2026.plate_only import PlateContactMonitor
    from work.challenge2026.scene_variants import prepare_variant
    from tools.mujoco_pilot import rgb
    decoded = Path(decoded).resolve(); measurement = json.loads((decoded/'report.json').read_text())
    require(measurement.get('measurement_passed') is True, 'Validated video measurement required')
    require(index not in {23,24,25,32,33,34,44,45}, 'Do not collect reserved development/final robot starts')
    with np.load(decoded/'measured.npz',allow_pickle=False) as f:
        times=f['times']; contacts=f['contact_xy']; measured_state=f['state']
    contract_path=Path(measurement['scale_contract_path']);contract=json.loads(contract_path.read_text())
    require(sha(contract_path)==measurement['scale_contract_sha256'], 'Measured contract changed')
    identity=capture_identity(contract)
    require(all(measurement.get(k)==v for k,v in identity.items()), 'Measured capture identity changed')
    require(sha(decoded/'measured.npz')==measurement['measured_npz_sha256'], 'Decoded metric samples changed')
    require(sha(measurement['video_path'])==measurement['video_sha256']==contract['video_sha256'], 'Whole source video changed')
    uncertainty_sha=sha(measurement['height_uncertainty_report_path'])
    require(uncertainty_sha==measurement['height_uncertainty_report_sha256'], 'Decoded height uncertainty changed')
    out=new_dir(out,ROOT/'runs');torch.set_num_threads(2)
    anchor_id=contract['pilot']['canonical_anchor_id']
    require(isinstance(anchor_id,str) and bool(anchor_id),'Declared anchor grouping string required')
    case=dict(episode_id=f"retarget-{identity['generated_video_id'] or identity['personal_video_id']}-N{index:02d}-S{seed}",split='train',init_index=index,native_init_index=index,seed=seed,
        language='push the plate to the green target',
        object_start_xy_m=measurement['metric_object_start_xy_m'],goal_xy_m=measurement['metric_object_end_xy_m'],
        layout_family_id=anchor_id,layout_fingerprint=hashlib.sha256(json.dumps(
            [measurement['metric_object_start_xy_m'],measurement['metric_object_end_xy_m']],separators=(',',':')).encode()).hexdigest(),
        lineage_id=measurement['whole_video_lineage_id'])
    files=['work/challenge2026/phone_video_bridge.py','work/challenge2026/scene_variants.py',
           'work/challenge2026/plate_only.py','liberopush/native.py','liberopush/contact_frame.py','liberopush/fixed_green_point.py',
           'liberopush/mujoco_scene.py','tools/mujoco_pilot.py']
    sources={p:sha(ROOT/p) for p in files}
    save(out/'design-before-run.json',dict(case,**identity,camera_transform='official_LIBERO_already_applied',
        measurement_report_sha256=sha(decoded/'report.json'),measured_npz_sha256=sha(decoded/'measured.npz'),
        height_uncertainty_report_sha256=uncertainty_sha,qualification_revision=measurement['qualification_revision'],
        source_sha256=sources,actor_object_truth=False,actor_inputs='own robot8, current decoded contactXY, fixed Z/orientation/gripper and static mesh contact geometry',
        no_recorded_robot_reference_actions=True,no_source_QA_truth_loaded=True,
        no_human_finger_motion_as_joint_labels=True,approach='fixed waypoints relative to first decoded contact; robot own-proprio control',
        playback='current20Hz sample from independently decoded video; no simulated future object information',
        contact_height_m=.914,tracking_gain=4.,native_action_cap=.20,maximum_commands=700,
        desired_expert_stop_commands=20,external_hold_commands=10,
        labels='actual Panda commands including disclosed robot executor approach; not original human RGB or human joint/action labels',
        functional_primary='finalXY<=50mm +abs(finalZ-initialZ)<=10mm heightproxy',
        retarget_tracking_gate='median contactXYerror<=5mm andmax<=15mm',training_started=False))
    env=monitor=writer=None;images=[];states=[];actions=[];after_states=[];objects=[];trace=[];sim_times=[];wall_times=[]
    phases=[];tracking_errors=[];tail=[];stop=None;tic=time.perf_counter();initial=None

    def preserve(partial=False):
        np.savez_compressed(out/('partial-actual-observations.npz' if partial else 'actual-robot-observations-actions.npz'),
            images=np.asarray(images,np.uint8),states=np.asarray(states,np.float32),actions=np.asarray(actions,np.float32),
            next_states=np.asarray(after_states,np.float32),phases=np.asarray(phases),objects_after_scoring=np.asarray(objects),
            simulation_times=np.asarray(sim_times),wall_observation_command_step_times=np.asarray(wall_times),
            tail_objects=np.asarray(tail),stop_object=np.asarray([] if stop is None else stop))
        save(out/'trace.json',trace)
        if monitor is not None:save(out/'contact-pairs.json',monitor.events)
    try:
        env=NativePanda(index,seed,700,evaluation_tail=10);scene=prepare_variant(env,out,case);save(out/'scene-contract.json',scene)
        require(env.language==scene['language']==case['language'],'Exact retarget task language differs from scene/environment')
        monitor=PlateContactMonitor(env.env.sim);initial=env.raw['plate_1_pos'].copy()
        tool=geometry_contract(front=True,mesh_front=True)
        save(out/'tool-geometry-contract.json',tool)
        design=json.loads((out/'design-before-run.json').read_text())
        design.update(tool_geometry_contract_sha256=sha(out/'tool-geometry-contract.json'),
            tool_asset_sha256=tool['asset_sha256'],tool_collision_mesh_sha256=tool['collision_mesh_sha256'])
        save(out/'design-before-run.json',design)
        state0=env.observation()['observation.state'][0].numpy()
        forward=rotation(state0[3:6])@np.array([0.,-1.,0.]);yaw=np.pi/2-np.arctan2(forward[1],forward[0])
        orientation=rotation([0.,0.,yaw])@rotation(state0[3:6])
        writer=cv2.VideoWriter(str(out/'robot-retarget.raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(512,300));require(writer.isOpened(),'Robot video writer unavailable')
        def tick(target,phase,video_time=None):
            require(len(actions)<700,'Retarget command budget exhausted')
            obs=env.observation();ims=rgb(obs);state=obs['observation.state'][0].numpy().copy()
            action=np.r_[np.zeros(6),-1.]
            if target is not None:
                action[:3]=np.clip(4.*(np.asarray(target)-contact_point(state,tool))/.05,-.20,.20)
                angular=cv2.Rodrigues(orientation@rotation(state[3:6]).T)[0].ravel();action[3:6]=np.clip(angular/.5,-.20,.20)
            sim0=float(env.env.sim.data.time);wall0=time.perf_counter()-tic;sent=time.perf_counter()-tic
            info=env.step(action);wall1=time.perf_counter()-tic;after=env.observation()['observation.state'][0].numpy().copy();obj=env.raw['plate_1_pos'].copy()
            images.append(ims);states.append(state);actions.append(info['action']);after_states.append(after);objects.append(obj);phases.append(phase)
            sim_times.append([sim0,float(env.env.sim.data.time)]);wall_times.append([wall0,sent,wall1])
            tracking=float(np.linalg.norm(contact_point(after,tool)[:2]-np.asarray(target)[:2])*1000) if target is not None else None
            trace.append(dict(step=len(actions)-1,phase=phase,state=state.tolist(),state_after=after.tolist(),action=info['action'],
                simulation_time_before_s=sim0,simulation_time_after_s=sim_times[-1][1],wall_observation_command_step_s=wall_times[-1],
                observation_rgb_sha256=hashlib.sha256(ims.tobytes()).hexdigest(),object_scoring_only=obj.tolist(),
                contact_target_xyz_m=None if target is None else np.asarray(target).tolist(),video_time_s=video_time,
                contact_tracking_error_mm=tracking,unwanted_contact_samples_cumulative=monitor.unwanted,
                maximum_lift_mm_so_far=monitor.maximum_lift_mm,actor_object_truth=False,VLA_command=False))
            image=np.zeros((300,512,3),np.uint8);image[44:]=cv2.cvtColor(np.concatenate(ims,axis=1),cv2.COLOR_RGB2BGR)
            capture_label='GENERATED HAND' if identity['synthetic_not_personal'] else 'PERSONAL PHONE'
            cv2.putText(image,capture_label+' VIDEO -> PHYSICAL PANDA | NOT VLA',(4,16),cv2.FONT_HERSHEY_SIMPLEX,.34,(255,255,255),1)
            cv2.putText(image,f'{phase} | {len(actions)}',(4,35),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1);writer.write(image)
            if len(actions)%25==0:print(json.dumps(dict(stage='retarget',commands=len(actions),phase=phase,unwanted_contacts=monitor.unwanted)),flush=True)
            return after,tracking
        first=contacts[0];behind=first-[0.,.080]
        for target,phase,budget,tolerance in [(np.r_[behind,1.04],'decoded-start approach above',150,.015),
                (np.r_[behind,.914],'decoded-start descend behind',100,.006),
                (np.r_[first,.914],'decoded-start contact approach',80,.006)]:
            for _ in range(budget):
                after,_=tick(target,phase)
                if np.linalg.norm(contact_point(after,tool)-target)<tolerance:break
            else:raise RuntimeError(phase+' failed to converge')
        for j,t in enumerate(times):
            _,error=tick(np.r_[contacts[j],.914],'decoded video contact',float(t));tracking_errors.append(error)
        for _ in range(20):tick(None,'expert stop demonstration')
        stop=env.raw['plate_1_pos'].copy()
        for _ in range(10):tick(None,'external stopped hold');tail.append(env.raw['plate_1_pos'].copy())
        tail=np.asarray(tail);final_error=float(np.linalg.norm(tail[-1,:2]-case['goal_xy_m'])*1000)
        drift=float(np.linalg.norm(tail-stop,axis=1).max()*1000);height=abs(float(tail[-1,2]-initial[2]))*1000
        tracking_pass=bool(np.median(tracking_errors)<=5 and max(tracking_errors)<=15)
        functional=bool(final_error<=50 and height<=10)
        forbidden=np.array([t['unwanted_contact_samples_cumulative']-(trace[i-1]['unwanted_contact_samples_cumulative'] if i else 0)>0 for i,t in enumerate(trace)])
        external=np.array([p=='external stopped hold' for p in phases]);allowed=~external&~forbidden
        save(out/'label-masks.json',dict(label_allowed=allowed.tolist(),forced=[False]*len(actions),scripted_setup=[False]*len(actions),
            external_hold=external.tolist(),invalid_target=forbidden.tolist(),
            robot_executor_approach=[p.startswith('decoded-start ') for p in phases],
            decoded_video_contact=[p=='decoded video contact' for p in phases],
            desired_expert_stop=[p=='expert stop demonstration' for p in phases],
            roles='separate executor approach, measured contact playback, desired stop, and excluded external hold'))
        preserve()
        report=dict(case,status='completed',**identity,whole_video_lineage_id=measurement['whole_video_lineage_id'],
            task_language=env.language,camera_transform='official_LIBERO_already_applied',
            original_human_RGB_training=False,
            robot_RGB_source='actual native Panda simulator cameras',robot_state_source='actual own-proprio8',robot_action_source='actually issued native OSC7',
            actor_object_truth=False,no_teacher_object_feedback=True,no_robot_reference_actions=True,
            functional_success_50mm=functional,centre_error_mm=final_error,post_stop_drift_mm=drift,
            precision_success_10mm=bool(final_error<=10),
            strict_success_10mm=bool(final_error<=10 and height<=10 and drift<=5 and monitor.maximum_lift_mm<=5 and monitor.unwanted==0),
            final_absolute_height_change_mm=height,object_on_table_height_proxy=bool(height<=10),maximum_lift_mm=monitor.maximum_lift_mm,
            unwanted_contact_samples=monitor.unwanted,allowed_tool_plate_samples=monitor.allowed,allowed_finger_table_samples=monitor.table,
            median_contact_tracking_error_mm=float(np.median(tracking_errors)),max_contact_tracking_error_mm=float(max(tracking_errors)),
            retarget_tracking_pass=tracking_pass,training_eligible=bool(functional and tracking_pass),
            steps=len(actions),desired_expert_stop_commands=20,external_hold_commands=10,
            measured_video_sha256=measurement['video_sha256'],measured_npz_sha256=sha(decoded/'measured.npz'),source_sha256=sources,
            height_uncertainty_report_sha256=uncertainty_sha,qualification_revision=measurement['qualification_revision'],
            tool_geometry_contract_sha256=sha(out/'tool-geometry-contract.json'),
            tool_asset_sha256=tool['asset_sha256'],tool_collision_mesh_sha256=tool['collision_mesh_sha256'],
            source_truth_loaded=False,actual_training_run=False,
            limits='single video-retarget pipeline pilot; no direct human-RGB learning, VLA success or improvement claim')
        save(out/'report.json',report);print(json.dumps(report),flush=True)
    except BaseException as error:
        preserve(partial=True);save(out/'runtime-failure.json',dict(error=type(error).__name__,message=str(error),actual_commands=len(actions),training_eligible=False,excluded=True));raise
    finally:
        if writer is not None:writer.release()
        if monitor is not None:monitor.close()
        if env is not None:env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('describe')
    assets=commands.add_parser('assets');assets.add_argument('--out',type=Path,default=PREP/'assets-v1')
    gen=commands.add_parser('generate');gen.add_argument('--assets',type=Path,default=PREP/'assets-v1');gen.add_argument('--out',type=Path,required=True)
    tracking=commands.add_parser('track');tracking.add_argument('--video',type=Path,required=True);tracking.add_argument('--contract',type=Path,required=True);tracking.add_argument('--out',type=Path,required=True)
    checking=commands.add_parser('qa');checking.add_argument('--generated',type=Path,required=True);checking.add_argument('--decoded',type=Path,required=True);checking.add_argument('--out',type=Path,required=True)
    robot=commands.add_parser('retarget');robot.add_argument('--decoded',type=Path,required=True);robot.add_argument('--out',type=Path,required=True);robot.add_argument('--index',type=int,default=0);robot.add_argument('--seed',type=int,default=889100)
    args=parser.parse_args();out=getattr(args,'out',None);existed=out is not None and out.exists()
    try:
        if args.command=='describe':print(json.dumps(asset_contract(),indent=2))
        elif args.command=='assets':prepare_assets(args.out)
        elif args.command=='generate':generate(args.assets,args.out)
        elif args.command=='track':track(args.video,args.contract,args.out)
        elif args.command=='qa':qa(args.generated,args.decoded,args.out)
        else:retarget(args.decoded,args.out,args.index,args.seed)
    except BaseException as error:
        if out is not None and not existed and out.is_dir() and out.resolve().is_relative_to(ROOT):
            save(out/'command-failure.json',dict(command=args.command,error=type(error).__name__,message=str(error),
                bridge_source_sha256=sha(__file__),training_started=False,partial_files_preserved=True))
        raise


if __name__=='__main__':main()
