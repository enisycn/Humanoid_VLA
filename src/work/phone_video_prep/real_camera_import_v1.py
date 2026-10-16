#!/usr/bin/env python3
"""Strict CPU import of measured camera evidence; no video/robot/training run.

Does not create a calibration, personal identity, or actual footage. Camera and
physical-measurement provenance remain source-bound; the later clip and Panda
audits are still required. Only the five-coefficient OpenCV pinhole model is
supported. Frozen bridge geometry is reused for independent static checkpoints.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
BRIDGE=ROOT/'work/challenge2026/phone_video_bridge.py'
BRIDGE_SHA='c36d602aa45b2d5bd9ce7af400e8afe9faf7f3a85ac72314a109677d14f7a50d'
MODEL='opencv_pinhole_brown5'
BOUNDS=dict(median_XY_error_mm=5.,p95_XY_error_mm=10.,maximum_XY_error_mm=15.,
    maximum_marker_height_error_mm=5.,maximum_source_body_lift_mm=5.,
    maximum_image_derived_XY_uncertainty_mm=5.,minimum_detection_coverage=.95,
    maximum_source_timestamp_match_error_s=.001)
sys.dont_write_bytecode=True


def require(ok,message):
    if not ok: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def bind(item, bindings, name):
    require(isinstance(item,dict) and set(item)=={'path','sha256'}, 'Missing exact path/hash binding: '+name)
    p=Path(item['path']); require(p.is_absolute() and p.is_file(), 'Missing evidence file: '+name)
    require(isinstance(item['sha256'],str) and len(item['sha256'])==64 and sha(p)==item['sha256'], 'Evidence hash differs: '+name)
    bindings[name]=dict(path=str(p.resolve()),sha256=item['sha256']); return p


def array(value,shape,label):
    import numpy as np
    a=np.asarray(value,float); require(a.shape==shape and np.isfinite(a).all(), 'Invalid '+label); return a


def camera_profile(report):
    import cv2
    import numpy as np
    require(report.get('schema')=='opencv_pinhole_calibration_report_v1'
        and report.get('model')==MODEL and report.get('method')=='cv2.calibrateCamera'
        and report.get('corner_detector')=='findChessboardCornersSB' and report.get('calibration_flags')==0,
        'Unsupported calibration model/method/flags; measured=true is insufficient')
    size=report.get('image_size_wh'); require(isinstance(size,list) and len(size)==2
        and all(type(v) is int and v>0 for v in size), 'Invalid decoded width/height')
    w,h=size; K=array(report.get('K'),(3,3),'K'); D=array(report.get('D'),(5,),'Brown5 distortion coefficients')
    require(np.array_equal(K[2],[0.,0.,1.]) and K[0,1]==K[1,0]==0 and K[0,0]>0 and K[1,1]>0
        and 0<=K[0,2]<w and 0<=K[1,2]<h, 'Invalid pinhole K/image geometry')
    settings=report.get('capture_settings'); fields={'camera_id','lens_id','capture_mode','orientation',
        'crop_xywh','optical_zoom','digital_zoom','focus_mode','focus_setting','stabilization','mirror'}
    require(isinstance(settings,dict) and set(settings)==fields, 'All capture-mode settings must be explicit')
    require(all(isinstance(settings[k],str) and settings[k] for k in ['camera_id','lens_id','capture_mode','focus_setting'])
        and settings['orientation']=='decoded_top_left_unrotated' and settings['crop_xywh']==[0,0,w,h]
        and settings['focus_mode']=='locked' and settings['stabilization']=='off' and settings['mirror'] is False
        and all(type(settings[k]) in [int,float] and math.isfinite(settings[k]) and settings[k]>0
            for k in ['optical_zoom','digital_zoom']), 'Unsupported/missing fixed capture geometry')
    grid=np.array([[x,y] for x in np.linspace(0,w-1,9) for y in np.linspace(0,h-1,9)],np.float64)
    rays=cv2.undistortPoints(grid[:,None,:],K,D).reshape(-1,2)
    projected=cv2.projectPoints(np.c_[rays,np.ones(len(rays))],np.zeros(3),np.zeros(3),K,D)[0].reshape(-1,2)
    roundtrip=float(np.linalg.norm(projected-grid,axis=1).max())
    r2=np.sum(rays*rays,axis=1); radial=1+3*D[0]*r2+5*D[1]*r2**2+7*D[4]*r2**3
    require(np.isfinite(rays).all() and np.isfinite(projected).all() and roundtrip<=.1 and (radial>0).all(),
        'Distortion is inconsistent/noninvertible on sampled image grid')
    return dict(model=MODEL,K=K.tolist(),D=D.tolist(),image_size_wh=size,capture_settings=settings,
        sampled_distortion_roundtrip_max_px=roundtrip,sampled_grid_only_not_global_invertibility_proof=True)


def image(path,profile):
    import cv2
    from PIL import Image
    with Image.open(path) as source:
        require(source.getexif().get(274,1)==1, 'Unsupported nonidentity EXIF orientation; no silent rotation')
    rgb=cv2.imread(str(path),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION); w,h=profile['image_size_wh']
    require(rgb is not None and rgb.shape==(h,w,3), 'Evidence image decoded dimensions differ from measured K')
    return cv2.cvtColor(rgb,cv2.COLOR_BGR2RGB)


def heading_geometry(heading,side):
    import numpy as np
    require(isinstance(heading,dict), 'Measured marker orientation required')
    local=array(heading.get('marker10_corners_contact_frame_xy_m'),(4,2),'measured marker10 corner geometry')
    tail,head,limit=[heading.get(k) for k in ['tail_corner_indices','head_corner_indices','maximum_deviation_deg']]
    adjacent=[{0,1},{1,2},{2,3},{0,3}]
    require(isinstance(tail,list) and isinstance(head,list) and len(tail)==len(head)==2
        and sorted(tail+head)==[0,1,2,3] and set(tail) in adjacent and set(head) in adjacent
        and heading.get('expected_forward_native_xy')==[0.,1.]
        and type(limit) in [int,float] and math.isfinite(limit) and 0<limit<=5,
        'Fixed WORLD+Y heading and opposing adjacent corner edges required')
    edges=np.roll(local,-1,axis=0)-local; area=abs(float(np.sum(local[:,0]*np.roll(local[:,1],-1)-local[:,1]*np.roll(local[:,0],-1))/2))
    diagonals=np.linalg.norm(local[:2]-local[2:],axis=1)
    forward=local[head].mean(axis=0)-local[tail].mean(axis=0)
    require(len(np.unique(local,axis=0))==4 and area>=.9*side**2 and np.linalg.norm(local.mean(axis=0))<=1e-6
        and np.allclose(np.linalg.norm(edges,axis=1),side,atol=.001,rtol=0)
        and np.allclose(diagonals,math.sqrt(2)*side,atol=.001,rtol=0)
        and np.linalg.norm(forward)>0 and np.allclose(forward/np.linalg.norm(forward),[0.,1.],atol=1e-6,rtol=0),
        'Marker10 must have distinct nondegenerate square corners and a forward axis')
    return tail,head,limit


def heading_deviation(world_corners,heading,side):
    import numpy as np
    tail,head,limit=heading_geometry(heading,side)
    corners=array(world_corners,(4,2),'observed marker10 world corners')
    forward=corners[head].mean(axis=0)-corners[tail].mean(axis=0)
    require(.5*side<=np.linalg.norm(forward)<=1.5*side, 'Observed marker10 forward axis is degenerate/incompatible with measured side')
    angle=math.degrees(math.atan2(abs(float(forward[0])),float(forward[1])))
    require(angle<=limit, 'Independent fixed-heading checkpoint fails'); return angle


def validate_manifest(manifest_path):
    """Return reusable calibration fields only; never a personal clip identity."""
    import cv2
    import numpy as np
    manifest_path=Path(manifest_path).resolve(); m=read(manifest_path); bindings={}
    require(m.get('schema')=='real_camera_calibration_import_v1'
        and m.get('source_kind')=='measured_real_camera_calibration' and m.get('synthetic') is False,
        'Actual measured calibration provenance required; synthetic fixtures are not real calibration')
    required={'calibration_report','board_measurement','geometry_measurement','independent_validation'}
    require(required.issubset(m), 'Missing calibration/board/geometry/independent-validation evidence')
    reports={key:read(bind(m[key],bindings,key)) for key in required}
    calibration,board,geometry,validation=[reports[k] for k in
        ['calibration_report','board_measurement','geometry_measurement','independent_validation']]
    profile=camera_profile(calibration); K=np.asarray(profile['K']); D=np.asarray(profile['D'])
    require(board.get('schema')=='measured_chessboard_v1' and board.get('measurement_source')=='physical_measurement'
        and isinstance(board.get('measurement_method'),str) and board['measurement_method'], 'Measured board definition required')
    wh=board.get('inner_corners_wh'); side=board.get('square_size_m'); uncertainty=board.get('measurement_uncertainty_m')
    require(isinstance(wh,list) and len(wh)==2 and all(type(x) is int and x>=3 for x in wh)
        and type(side) in [int,float] and math.isfinite(side) and side>0
        and type(uncertainty) in [int,float] and math.isfinite(uncertainty) and 0<uncertainty<side,
        'Measured board size/uncertainty required')
    for key in ['board_measurement','geometry_measurement']:
        evidence=reports[key].get('measurement_evidence')
        require(isinstance(evidence,list) and evidence, 'Missing physical-measurement evidence: '+key)
        for i,item in enumerate(evidence): bind(item,bindings,f'{key}_evidence_{i}')
    require(calibration.get('board_sha256')==m['board_measurement']['sha256'], 'Calibration uses a different measured board')
    views=calibration.get('views'); require(isinstance(views,list) and len(views)>=3, 'At least three distinct calibration views required')
    hashes=[]; errors=[]; poses=[]
    object_points=np.zeros((wh[0]*wh[1],3),np.float64); object_points[:,:2]=np.mgrid[0:wh[0],0:wh[1]].T.reshape(-1,2)*side
    for i,view in enumerate(views):
        p=bind(view['image'],bindings,f'calibration_image_{i}'); hashes.append(sha(p))
        ok,corners=cv2.findChessboardCornersSB(cv2.cvtColor(image(p,profile),cv2.COLOR_RGB2GRAY),tuple(wh))
        require(ok, 'Measured calibration chessboard not detected: '+str(i))
        r=array(view.get('rvec'),(3,),'calibration rvec'); t=array(view.get('tvec'),(3,),'calibration tvec')
        projected=cv2.projectPoints(object_points,r,t,K,D)[0]; error=float(np.sqrt(np.mean(np.sum((projected-corners)**2,axis=2))))
        require(abs(error-float(view['rms_reprojection_error_px']))<=.001, 'Calibration reprojection evidence differs')
        errors.append(error); poses.append(tuple(np.r_[r,t]))
    require(len(set(hashes))==len(views) and len(set(poses))>=3, 'Calibration views/poses must not be duplicated')
    require(abs(float(np.sqrt(np.mean(np.square(errors))))-float(calibration['rms_reprojection_error_px']))<=.001,
        'Calibration aggregate reprojection error differs')
    require(geometry.get('schema')=='measured_marker_contact_geometry_v1'
        and geometry.get('measurement_source')=='physical_measurement' and geometry.get('dictionary')=='DICT_4X4_50',
        'Measured marker/contact geometry required')
    refs=array(geometry.get('reference_centres_native_xy_m'),(4,2),'reference coordinates')
    require(len(np.unique(refs,axis=0))==4 and abs(cv2.contourArea(refs.astype(np.float32)))>1e-6
        and geometry.get('reference_ids')==[0,1,2,3], 'Four ordered nondegenerate measured references required')
    plane=geometry.get('planes_above_table_m'); require(isinstance(plane,dict)
        and set(plane)=={'reference','object_marker','fingertip_marker','contact'}
        and all(type(v) in [int,float] and math.isfinite(v) and 0<=v<=.15 for v in plane.values()), 'Measured marker/contact planes required')
    require(geometry.get('native_table_top_z_m')==.9 and geometry.get('native_xy_mapping')=='measured_rigid_board_axes_no_scale',
        'Declare measured rigid XY correspondence to native table; no scale fitting')
    plane_uncertainty=geometry.get('plane_measurement_uncertainty_m')
    require(isinstance(plane_uncertainty,dict) and set(plane_uncertainty)==set(plane)
        and all(type(v) in [int,float] and math.isfinite(v) and 0<v<=.005 for v in plane_uncertainty.values()),
        'Physical plane-measurement uncertainty must be explicit and within total5mm envelope')
    offset=array(geometry.get('fingertip_marker_to_contact_xy_m'),(2,),'contact offset')
    sizes=geometry.get('marker_black_side_m'); require(isinstance(sizes,dict) and set(sizes)=={'10','11'}
        and all(type(v) in [int,float] and math.isfinite(v) and .005<=v<=.10 for v in sizes.values()), 'Measured marker black-side lengths required')
    heading=geometry.get('fixed_heading_admission'); heading_geometry(heading,sizes['10'])
    bridge_fields=dict(real_phone_calibration_measured=True,dictionary='DICT_4X4_50',table_top_z_m=.9,
        phone=dict(width=profile['image_size_wh'][0],height=profile['image_size_wh'][1],intrinsics=profile['K'],
            distortion_coefficients=profile['D'],distortion_model=MODEL,capture_settings=profile['capture_settings']),
        calibration_marker_centres_world_xy_m=refs.tolist(),calibration_marker_plane_z_m=.9+plane['reference'],
        fingertip_marker_world_z_m=.9+plane['fingertip_marker'],fingertip_marker_to_contact_xy_m=offset.tolist(),
        fingertip_contact_world_z_m=.9+plane['contact'],object_marker_world_z_m=.9+plane['object_marker'],
        tracking=dict(inputs_only=['actual compressed video bytes','measured calibration and geometry'],
            method='decoded undistorted ArUco corners; frozen height homographies',qualification_revision='personal_measured_calibration_v1',
            object_marker_height_envelope_mm=5.,constant_height_diagnostic_mm=2.,synthetic_QA_bounds=BOUNDS,
            height_bound_scope='total5mm includes measured plane uncertainty; dynamic height/tilt not validated by static calibration'))
    require(sha(BRIDGE)==BRIDGE_SHA, 'Frozen geometry implementation changed')
    spec=importlib.util.spec_from_file_location('_real_calibration_frozen_bridge',BRIDGE)
    bridge=importlib.util.module_from_spec(spec); spec.loader.exec_module(bridge)
    require(validation.get('schema')=='independent_measured_marker_checkpoints_v1'
        and validation.get('capture_settings')==profile['capture_settings'] and validation.get('used_for_calibration') is False
        and validation.get('calibration_report_sha256')==m['calibration_report']['sha256']
        and validation.get('geometry_measurement_sha256')==m['geometry_measurement']['sha256'], 'Independent checkpoint provenance/settings differ')
    checkpoints=validation.get('checkpoints'); require(isinstance(checkpoints,list) and len(checkpoints)>=3, 'Three independent measured checkpoint positions required')
    metric_errors={key:[] for key in ['object','fingertip','contact']}; headings=[]; measured_positions=[]; validation_hashes=[]
    for i,checkpoint in enumerate(checkpoints):
        p=bind(checkpoint['image'],bindings,f'validation_image_{i}'); require(sha(p) not in hashes, 'Validation image was used for calibration')
        require(sha(p) not in validation_hashes, 'Validation checkpoint images must be distinct'); validation_hashes.append(sha(p))
        require(checkpoint.get('measurement_source')=='physical_measurement', 'Checkpoint must be independently measured')
        evidence=checkpoint.get('measurement_evidence'); require(isinstance(evidence,list) and evidence, 'Checkpoint measurement evidence missing')
        for j,item in enumerate(evidence): bind(item,bindings,f'validation_measurement_{i}_{j}')
        raw=bridge.detect(image(p,profile)); require(all(k in raw for k in [0,1,2,3,10,11]), 'Static validation markers missing')
        corrected={k:cv2.undistortPoints(v[:,None,:],K,D,P=K).reshape(4,2) for k,v in raw.items()}
        H,_,_,_=bridge.homographies(np.array([corrected[k].mean(axis=0) for k in range(4)]),bridge_fields)
        finger=bridge.project(H[10],corrected[10].mean(axis=0)); obj=bridge.project(H[11],corrected[11].mean(axis=0))
        expected={key:array(checkpoint.get(key+'_native_xy_m'),(2,),key+' checkpoint') for key in metric_errors}
        require(all(np.min(np.linalg.norm(refs-v,axis=1))>1e-9 for v in expected.values()),
            'Independent checkpoints cannot equal homography-fitting reference centres')
        measured_positions.append(tuple(np.r_[expected['object'],expected['fingertip']]))
        for key,actual in dict(object=obj,fingertip=finger,contact=finger+offset).items(): metric_errors[key].append(float(np.linalg.norm(actual-expected[key])*1000))
        world_corners=np.array([bridge.project(H[10],v) for v in corrected[10]])
        headings.append(heading_deviation(world_corners,heading,sizes['10']))
    require(len(set(measured_positions))>=3, 'Independent checkpoints must vary measured positions')
    metrics={key:dict(median_mm=float(np.median(v)),p95_mm=float(np.percentile(v,95)),maximum_mm=float(max(v))) for key,v in metric_errors.items()}
    require(all(v['median_mm']<=5 and v['p95_mm']<=10 and v['maximum_mm']<=15 for v in metrics.values()), 'Independent metric checkpoint accuracy fails')
    for item in bindings.values(): require(sha(item['path'])==item['sha256'], 'Evidence changed during import')
    require('torch' not in sys.modules and 'mujoco' not in sys.modules, 'Importer loaded model/simulator')
    return dict(schema='validated_real_camera_calibration_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        importer_sha256=sha(__file__),manifest_sha256=sha(manifest_path),camera_profile=profile,bridge_contract_fields=bridge_fields,
        geometry_measurement=geometry,fixed_heading_admission=dict(heading,static_checkpoint_maximum_deviation_deg=max(headings),
            marker_id=10,derive_heading='corrected marker corners -> frozen marker10 height homography -> head-edge minus tail-edge',
            observed_forward_length_interval_m=[.5*sizes['10'],1.5*sizes['10']],
            per_clip_heading_rejection_still_required=True),independent_checkpoint_metrics=metrics,artifact_bindings=bindings,
        frozen_geometry_source_sha256=BRIDGE_SHA,personal_clip_identity_supplied=False,personal_video_imported=False,
        actual_personal_pipeline_ready=False,training_eligible=False,
        validation_scope='hash integrity, recorded calibration reprojection, sampled distortion consistency and independently measured static image checkpoints; physical measurements/settings provenance source-bound',
        remaining_gates=['actual clip dimensions/orientation/mode equality','raw+corrected corner and fixed-heading audit',
            'dynamic marker-height/lift/tilt uncertainty qualification','contact-height mapping compatibility with unchanged native14mm contact',
            'one actual Panda retarget and independent personal-source admission; no automatic training'])


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--manifest',type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True); group.add_argument('--validate-only',action='store_true'); group.add_argument('--out',type=Path)
    args=parser.parse_args(); result=validate_manifest(args.manifest)
    if args.out:
        out=args.out.resolve(); require(out.is_relative_to(ROOT) and not out.exists(), 'Fresh contract inside project required')
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open('x') as stream: stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='calibration_artifact_validated',actual_personal_pipeline_ready=False,
        personal_video_imported=False,training_eligible=False,out=str(args.out) if args.out else None)))


if __name__=='__main__': main()
