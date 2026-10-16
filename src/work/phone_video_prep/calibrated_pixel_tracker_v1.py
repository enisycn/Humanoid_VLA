"""Thin CPU pixel correction around the frozen tracker; no personal admission.

Real capture sidecar: schema personal_hand_capture_v1, source_kind
actual_personal_hand_video, synthetic_not_personal=false, personal_video_id,
video{path,sha256}, calibration_manifest{path,sha256}, image_size_wh,
capture_settings, pixel_transform='none'. Settings provenance is source-bound.
"""
import argparse, copy, hashlib, importlib.util, json, math, sys, traceback
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
BRIDGE=ROOT/'work/challenge2026/phone_video_bridge.py'
BRIDGE_SHA='c36d602aa45b2d5bd9ce7af400e8afe9faf7f3a85ac72314a109677d14f7a50d'
IMPORTER=HERE/'real_camera_import_v1.py'
IMPORTER_SHA='1b49c65549579fc199f1baf4de759a66ee32c7724bbeb6cbab502922b26e2099'
TRAIN_A=ROOT/'runs/challenge2026-hand-video-pilot-v3'
A_CONTRACT_SHA='ff83866e4a04f70ef3b6a8872df4d23f3aca21d03082e9d6f9dd66ed9068f0a8'
A_VIDEO_SHA='f2f2de25df7ee8b2c4f9dbf843fbf3028ccafbcc9f0dc03081b90cbabedec31b'
A_TRACK=ROOT/'runs/challenge2026-hand-video-tracking-v3'
A_REPORT_SHA='32403921db6a0de08f0932d54c42c2d2ae57562cd448f647f1072aabff4a2040'
METHOD_SOURCES=['https://github.com/PyAV-Org/PyAV/blob/main/av/video/frame.py','https://pyav.basswood.io/docs/stable/api/video.html']
IDS=[0,1,2,3,10,11]; sys.dont_write_bytecode=True

def require(ok,why):
    if not ok: raise ValueError(why)
def read(path): return json.loads(Path(path).read_text())
def save(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def load(path,digest,name):
    require(sha(path)==digest,'Source changed: '+str(path))
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def bound(item):
    p=Path(item['path']);require(p.is_absolute() and p.is_file() and sha(p)==item['sha256'],'Missing/changed bound input');return p.resolve()

def check_pixels(size,profile,metadata=None,rotation=0,side_types=()):
    require(list(size)==profile['image_size_wh'],'Full decoded dimensions differ from calibrated pixels')
    require(not any('rotat' in str(k).lower() or 'displaymatrix' in str(k).lower() for k in (metadata or {})), 'Rotation/display metadata unsupported')
    require(float(rotation)==0 and not any('DISPLAYMATRIX' in str(k).upper().replace('_','') for k in side_types), 'Rotation/display matrix unsupported')

def prepare(mode,capture=None,calibration=None):
    """Real mode revalidates evidence, not a cached validated=true profile."""
    bridge=load(BRIDGE,BRIDGE_SHA,'_calibrated_frozen_bridge')
    bindings=dict(bridge=dict(path=str(BRIDGE),sha256=BRIDGE_SHA),importer=dict(path=str(IMPORTER),sha256=IMPORTER_SHA),adapter=dict(path=__file__,sha256=sha(__file__)))
    if mode=='synthetic-regression':
        contract_path=TRAIN_A/'scale-contract.json';video=TRAIN_A/'synthetic-hand-phone.mp4'
        require(sha(contract_path)==A_CONTRACT_SHA and sha(video)==A_VIDEO_SHA,'Only existing whole TRAIN-A is a regression input')
        contract=read(contract_path);require(contract['synthetic_not_personal'] and contract['pilot']['canonical_anchor_id']=='train-A','Synthetic identity required')
        profile=dict(K=contract['phone']['intrinsics'],D=[0.]*5,image_size_wh=[contract['phone']['width'],contract['phone']['height']],
            capture_settings=dict(orientation='decoded_top_left_unrotated',crop_xywh=[0,0,1440,1080],mirror=False),distortion_basis='declared zero-D synthetic renderer')
        heading=dict(head_corner_indices=[0,1],tail_corner_indices=[2,3],maximum_deviation_deg=5.,expected_forward_native_xy=[0.,1.],
            convention='predeclared marker texture top edge WORLD+Y; no edge selection from decoded trajectory')
        bindings['synthetic_contract']=dict(path=str(contract_path),sha256=A_CONTRACT_SHA)
        require(sha(A_TRACK/'report.json')==A_REPORT_SHA,'Historical TRAIN-A tracking report changed')
        bindings.update(regression_report=dict(path=str(A_TRACK/'report.json'),sha256=A_REPORT_SHA),
            regression_measurement=dict(path=str(A_TRACK/'measured.npz'),sha256=read(A_TRACK/'report.json')['measured_npz_sha256']))
        contract=copy.deepcopy(contract);contract['phone']['distortion_coefficients']=[0.]*5
        validation=None
    else:
        require(capture is not None and calibration is not None,'Actual capture sidecar and calibration manifest required')
        capture,calibration=Path(capture).resolve(),Path(calibration).resolve();clip=read(capture)
        require(clip.get('schema')=='personal_hand_capture_v1' and clip.get('source_kind')=='actual_personal_hand_video'
            and clip.get('synthetic_not_personal') is False and isinstance(clip.get('personal_video_id'),str) and clip['personal_video_id'], 'Actual declared personal capture identity required')
        require(bound(clip['calibration_manifest'])==calibration,'Capture binds a different calibration manifest')
        importer=load(IMPORTER,IMPORTER_SHA,'_calibrated_real_importer');validation=importer.validate_manifest(calibration)
        profile=validation['camera_profile'];heading=validation['fixed_heading_admission'];video=bound(clip['video'])
        require(clip.get('capture_settings')==profile['capture_settings'] and clip.get('image_size_wh')==profile['image_size_wh']
            and clip.get('pixel_transform')=='none','Capture settings/orientation/crop must exactly match calibration; no transformations')
        contract=copy.deepcopy(validation['bridge_contract_fields']);contract.update(synthetic_not_personal=False,personal_video_id=clip['personal_video_id'],generated_video_id=None,personal_phone_episodes=0)
        bindings.update(importer=dict(path=str(IMPORTER),sha256=IMPORTER_SHA),capture=dict(path=str(capture),sha256=sha(capture)),
            calibration_manifest=dict(path=str(calibration),sha256=sha(calibration)))
        bindings.update({'calibration_'+k:v for k,v in validation['artifact_bindings'].items()})
    contract['video_sha256']=sha(video);bindings['video']=dict(path=str(video),sha256=sha(video))
    for item in bindings.values():require(sha(item['path'])==item['sha256'],'Source/input binding failed before decoding')
    require(0<float(heading['maximum_deviation_deg'])<=5 and heading['expected_forward_native_xy']==[0.,1.],'Only declared fixed WORLD+Y up to5degrees')
    return bridge,video,contract,profile,heading,bindings,validation

def track(mode,out,capture=None,calibration=None):
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    bridge,video,contract,profile,heading,bindings,validation=prepare(mode,capture,calibration)
    out=Path(out).resolve();require(out.is_relative_to(ROOT/'runs') and not out.exists(),'Fresh runs/ output required');out.mkdir(parents=True)
    save(out/'design-before-tracking.json',dict(mode=mode,bindings=bindings,camera_profile=profile,fixed_heading_admission=heading,
        correction='original detect -> undistortPoints every raw corner K,D,P=K -> unchanged centres/height homographies/causal sampling',
        frozen_track_verbatim=True,no_transform_resize_reencode=True,source_truth_loaded=False,personal_source_admitted=False,method_sources=METHOD_SOURCES,
        real_dynamic_height_lift_tilt_validated=False,retarget_allowed=False,training_eligible=False))
    save(out/'tracker-contract.json',contract)
    if validation is not None:save(out/'revalidated-calibration.json',validation)
    raw_rows=[];corrected_rows=[];times=[];missing=[];K=np.asarray(profile['K'],float);D=np.asarray(profile['D'],float)
    original_detect,original_av=bridge.detect,bridge.av;original_open=original_av.open
    def corrected_detect(rgb):
        require(rgb.shape==(profile['image_size_wh'][1],profile['image_size_wh'][0],3),'Decoded RGB dimension mismatch')
        raw=original_detect(rgb);fixed={k:cv2.undistortPoints(v[:,None,:],K,D,P=K).reshape(4,2) for k,v in raw.items()}
        raw_rows.append(np.array([raw.get(k,np.full((4,2),np.nan)) for k in IDS]));corrected_rows.append(np.array([fixed.get(k,np.full((4,2),np.nan)) for k in IDS]))
        missing.append(dict(frame_index=len(raw_rows)-1,missing_marker_ids=[k for k in IDS if k not in raw],detected_marker_ids=sorted(raw)))
        return fixed
    class CheckedContainer:
        def __init__(self,*args,**kwargs):self.container=original_open(*args,**kwargs)
        def __enter__(self):
            c=self.container
            try:
                require(len(c.streams.video)==1,'Exactly one video stream required');s=c.streams.video[0]
                check_pixels([s.codec_context.width,s.codec_context.height],profile,dict(c.metadata,**s.metadata));return self
            except BaseException:c.close();raise
        def __exit__(self,*args):self.container.close()
        def decode(self,*args,**kwargs):
            for frame in self.container.decode(*args,**kwargs):
                check_pixels([frame.width,frame.height],profile,frame.metadata,frame.rotation,[v.type.name for v in frame.side_data])
                require(frame.time is not None,'Actual decoded timestamp required');times.append(float(frame.time));yield frame
    failure=None;report=None
    try:
        bridge.detect=corrected_detect;bridge.av=SimpleNamespace(open=CheckedContainer)
        report=bridge.track(video,out/'tracker-contract.json',out/'frozen-tracker')
    except BaseException as exc:
        failure=dict(error=type(exc).__name__,message=str(exc));(out/'failure-traceback.txt').write_text(traceback.format_exc())
    finally:
        bridge.detect=original_detect;bridge.av=original_av
        raw_report=out/'frozen-tracker/report.json'
        if raw_report.exists():raw_report.rename(out/'frozen-tracker-report.json')
        np.savez_compressed(out/'raw-and-corrected-corners.npz',raw_corners_px=np.asarray(raw_rows),corrected_corners_px=np.asarray(corrected_rows),
            frame_indices=np.arange(len(raw_rows)),decoded_frame_times_s=np.asarray(times[:len(raw_rows)]),marker_ids=np.asarray(IDS))
        save(out/'missing-markers.json',missing)
    try:
        require(failure is None,'Frozen tracker failed: '+str(failure));require(len(raw_rows)==len(times)==report['frames'],'Frame/corner/timestamp count differs')
        first=corrected_rows[0];H,_,_,_=bridge.homographies(np.array([first[i].mean(axis=0) for i in range(4)]),contract)
        heading_rows=[]
        heading_importer=load(IMPORTER,IMPORTER_SHA,'_calibrated_heading_importer') if mode=='real' else None
        for i,frame in enumerate(corrected_rows):
            corners=frame[IDS.index(10)]
            if not np.isfinite(corners).all():continue
            world=np.array([bridge.project(H[10],p) for p in corners]);forward=world[heading['head_corner_indices']].mean(axis=0)-world[heading['tail_corner_indices']].mean(axis=0)
            if mode=='real':angle=heading_importer.heading_deviation(world,heading,validation['geometry_measurement']['marker_black_side_m']['10'])
            else:
                require(np.linalg.norm(forward)>0,'Degenerate marker10 heading');angle=math.degrees(math.atan2(abs(float(forward[0])),float(forward[1])))
            heading_rows.append(dict(frame_index=i,decoded_time_s=times[i]-times[0],marker10_world_corners_xy_m=world.tolist(),deviation_deg=angle))
        heading_pass=bool(heading_rows) and max(r['deviation_deg'] for r in heading_rows)<=heading['maximum_deviation_deg']
        save(out/'fixed-heading.json',dict(passed=heading_pass,admission=heading,maximum_deviation_deg=max(r['deviation_deg'] for r in heading_rows) if heading_rows else None,rows=heading_rows,source_truth_loaded=False))
        regression=None
        if mode=='synthetic-regression':
            require(np.allclose(raw_rows,corrected_rows,rtol=0,atol=1e-9,equal_nan=True),'Declared zero-D changed decoded corners')
            with np.load(A_TRACK/'measured.npz',allow_pickle=False) as historical, np.load(out/'frozen-tracker/measured.npz',allow_pickle=False) as current:
                equal={k:bool(historical[k].shape==current[k].shape and np.allclose(historical[k],current[k],rtol=0,atol=1e-9,equal_nan=True)) for k in historical.files}
            regression=dict(passed=all(equal.values()),fields=equal,absolute_tolerance_native_units=1e-9,zero_D_corner_maximum_difference_px=float(np.nanmax(np.abs(np.asarray(raw_rows)-corrected_rows))),synthetic_only=True)
            save(out/'synthetic-regression.json',regression);require(regression['passed'],'Zero-D tracking differs from historical TRAIN-A; no repair')
        for item in bindings.values():require(sha(item['path'])==item['sha256'],'Source/input changed during tracking')
        require(bridge.detect is original_detect and bridge.av is original_av,'Frozen module wrappers were not restored')
        final=dict(report);final.update(status='image_tracking_completed' if heading_pass else 'fixed_heading_failed',
            image_measurement_passed=bool(report['measurement_passed'] and heading_pass),measurement_passed=False,
            synthetic_regression_only=mode=='synthetic-regression',personal_source_admitted=False,retarget_allowed=False,training_eligible=False,
            dynamic_height_lift_tilt_qualified=False,personal_pipeline_ready=False,robot_episode_count=0,robot_action_labels_available=False,
            source_kind=report['source_kind'] if mode=='synthetic-regression' else 'calibrated_personal_video_image_measurement_only',
            artifact_bindings=bindings,opencv_version=cv2.__version__,pyav_version=original_av.__version__,
            corners_sha256=sha(out/'raw-and-corrected-corners.npz'),missing_markers_sha256=sha(out/'missing-markers.json'),
            frozen_report_sha256=sha(out/'frozen-tracker-report.json'),fixed_heading_sha256=sha(out/'fixed-heading.json'),
            raw_frozen_tracker_report=str(out/'frozen-tracker-report.json'),measured_trajectory=str(out/'frozen-tracker/measured.npz'),
            synthetic_regression=regression,method_sources=METHOD_SOURCES,
            unsupported_rotation_or_display_matrix_rejected=True,no_crop_resize_rotation_reencode=True,
            limitations=['Calibration/settings and physical measurements remain source-bound.','Static/image checks do not validate dynamic marker height, source lift/tilt or native contact compatibility.','No personal Panda retarget/admission or training is implemented; frozen raw tracker measurement_passed is not admission.'])
        save(out/'report.json',final);require(heading_pass,'Decoded marker10 heading exceeds the declared fixed WORLD+Y limit')
        require('torch' not in sys.modules and 'mujoco' not in sys.modules,'CPU wrapper loaded model/simulator')
        return final
    except BaseException as exc:
        save(out/'failure.json',dict(status='failed_preserved',error=type(exc).__name__,message=str(exc),bindings=bindings,measurement_passed=False,personal_source_admitted=False,retarget_allowed=False,training_eligible=False));raise

def numerical_checks():
    """Prepared synthetic projection checks only; never actual camera evidence."""
    import cv2
    import numpy as np
    bridge=load(BRIDGE,BRIDGE_SHA,'_calibrated_numeric_bridge');K=np.array([[700.,0,319.5],[0,700.,239.5],[0,0,1.]])
    D=np.array([-.08,.01,.001,-.001,0.]);r=np.array([.25,-.1,.05]);t=np.array([0.,0.,1.2]);refs=np.array([[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]])
    contract=dict(phone=dict(intrinsics=K.tolist()),calibration_marker_centres_world_xy_m=refs.tolist(),calibration_marker_plane_z_m=.9,fingertip_marker_world_z_m=.925,object_marker_world_z_m=.918)
    def pixels(xyz):return cv2.projectPoints(np.asarray(xyz,float),r,t,K,D)[0].reshape(-1,2)
    refpx=cv2.undistortPoints(pixels(np.c_[refs,np.zeros(4)])[:,None,:],K,D,P=K).reshape(-1,2)
    H,flat,_,_=bridge.homographies(refpx,contract);point=np.array([.03,.04]);px=pixels([[*point,.025]])
    fixed=cv2.undistortPoints(px[:,None,:],K,D,P=K).reshape(2);error=float(np.linalg.norm(bridge.project(H[10],fixed)-point)*1000)
    require(error<.01,'Distortion/known-height geometry check failed');checks=['distorted projection -> corrected pixels -> frozen known-height homography']
    require(np.linalg.norm(bridge.project(flat,fixed)-point)*1000>.1,'Height correction check is not meaningful')
    profile=dict(image_size_wh=[640,480])
    for name,args in [('dimension',([480,640],profile)),('rotation',([640,480],profile,{},90)),('displaymatrix',([640,480],profile,{},0,['DISPLAYMATRIX']))]:
        try:check_pixels(*args)
        except ValueError:checks.append('reject '+name)
        else:raise AssertionError('Missing rejection: '+name)
    return dict(passed=True,checks=checks,metric_error_mm=error,synthetic_numeric_fixture_only=True,actual_video_decoded=False,personal_calibration_validated=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['check','numerical-checks','synthetic-regression','real'])
    parser.add_argument('--out',type=Path);parser.add_argument('--capture-manifest',type=Path);parser.add_argument('--calibration-manifest',type=Path);args=parser.parse_args()
    require(sha(BRIDGE)==BRIDGE_SHA and sha(IMPORTER)==IMPORTER_SHA,'Pinned sources changed')
    if args.mode=='check':print(json.dumps(dict(status='CPU_source_prepared',adapter_sha256=sha(__file__),bridge_sha256=BRIDGE_SHA,importer_sha256=IMPORTER_SHA,actual_video_decoded=False,personal_pipeline_ready=False)));return
    if args.mode=='numerical-checks':print(json.dumps(numerical_checks()));return
    require(args.out is not None,'Fresh output required');result=track(args.mode,args.out,args.capture_manifest,args.calibration_manifest)
    print(json.dumps(dict(status=result['status'],image_measurement_passed=result['image_measurement_passed'],measurement_passed=False,personal_source_admitted=False,training_eligible=False)))

if __name__=='__main__':main()
