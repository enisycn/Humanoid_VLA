"""Read-only first look at applicant video: metadata, sparse marker/board audit.

Not a calibrated trajectory, robot retarget, or training admission. Original
video bytes remain unchanged; previews are separate labelled derivatives.
"""
from pathlib import Path
import av,cv2,hashlib,json,math,time
import numpy as np
R=Path('/home/enis/humanoid_isaac_lab');D=R/'data/personal_video_inbox/recording-20261016-v1'
VIDEO=D/'original.mov';OUT=D/'first-look';OUT.mkdir(exist_ok=False)
manifest=json.loads((D/'download.json').read_text());assert manifest['status']=='download_completed'
with VIDEO.open('rb')as f:assert hashlib.file_digest(f,'sha256').hexdigest()==manifest['sha256']
cv2.setNumThreads(1);det=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
IDS=[0,1,2,3,10,11,12,13];rows=[];corners=[];board=[];preview=[];start=time.monotonic()
with av.open(str(VIDEO))as container:
    stream=container.streams.video[0];stream.thread_type='AUTO';stream.codec_context.thread_count=4
    duration=float(stream.duration*stream.time_base)if stream.duration else float(container.duration/av.time_base)
    metadata=dict(codec=stream.codec_context.name,coded_width=stream.width,coded_height=stream.height,
        average_rate=str(stream.average_rate),duration_s=duration,declared_frames=stream.frames,
        has_audio=bool(container.streams.audio),source_sha256=manifest['sha256'],source_kind='applicant_recorded_real_video',synthetic=False)
    targets=np.linspace(0,max(0,duration-.5),20);next_preview=0;sample_clock=0.;decoded=0;rotations=set()
    for frame in container.decode(stream):
        decoded+=1
        if frame.time is None:continue
        t=float(frame.time);rotation=getattr(frame,'rotation',None)
        if rotation is not None:rotations.add(float(rotation))
        if t+1e-6<sample_clock:continue
        rgb=frame.to_ndarray(format='bgr24');gray=cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)
        cc,ii,_=det.detectMarkers(gray);found={}if ii is None else {int(i):np.asarray(c[0],float)for i,c in zip(ii.ravel(),cc)}
        rows.append([t,*[i in found for i in IDS]])
        corners.append(np.array([found.get(i,np.full((4,2),np.nan))for i in IDS]))
        if t<60:
            ok,points=cv2.findChessboardCorners(gray,(7,5),cv2.CALIB_CB_FAST_CHECK)
            if ok:
                board.append(dict(time_s=t,corners_px=points[:,0,:].tolist()))
                if len(board)<=8:cv2.imwrite(str(OUT/f'calibration-{len(board):02d}.jpg'),rgb)
        if next_preview<len(targets)and t>=targets[next_preview]-.001:
            shown=rgb
            if rotation is not None and abs(rotation)%360==90:shown=cv2.rotate(rgb,cv2.ROTATE_90_COUNTERCLOCKWISE if rotation==90 else cv2.ROTATE_90_CLOCKWISE)
            elif rotation is not None and abs(rotation)%360==180:shown=cv2.rotate(rgb,cv2.ROTATE_180)
            scale=min(470/shown.shape[1],310/shown.shape[0]);small=cv2.resize(shown,None,fx=scale,fy=scale)
            canvas=np.full((350,490,3),245,np.uint8);canvas[0:small.shape[0],0:small.shape[1]]=small
            cv2.putText(canvas,f'{t:.1f}s IDs: '+','.join(map(str,sorted(found))), (8,335),cv2.FONT_HERSHEY_SIMPLEX,.48,(25,25,25),1,cv2.LINE_AA)
            preview.append(canvas);cv2.imwrite(str(OUT/f'view-{next_preview:02d}.jpg'),shown);next_preview+=1
        sample_clock+=.5
        if len(rows)%100==0:print(json.dumps(dict(sampled=len(rows),at_s=t)),flush=True)
metadata.update(decoded_frames=decoded,frame_rotations_deg=sorted(rotations))
a=np.asarray(rows,float);c=np.asarray(corners)
np.savez_compressed(OUT/'sparse-marker-pixels.npz',rows=a,corners_px=c,marker_ids=np.array(IDS))
calibration_samples=[]
if len(preview):
    while len(preview)%4:preview.append(np.full_like(preview[0],245))
    sheet=np.vstack([np.hstack(preview[i:i+4])for i in range(0,len(preview),4)]);cv2.imwrite(str(OUT/'contact-sheet.jpg'),sheet)
refs_valid=np.isfinite(c[:,:4]).all(axis=(1,2,3));ref_shift=None
if refs_valid.any():
    centres=c[refs_valid,:4].mean(axis=2);median=np.median(centres,axis=0)
    dist=np.linalg.norm(centres-median,axis=2)
    ref_shift=dict(samples_with_all_four=int(refs_valid.sum()),median_reference_shift_px=float(np.median(dist)),p95_reference_shift_px=float(np.quantile(dist,.95)),maximum_reference_shift_px=float(dist.max()),uncalibrated_image_diagnostic_only=True)
report=dict(metadata=metadata,sparse_sample_period_s=.5,sampled_frames=len(a),
    visibility_fraction_whole_video={str(i):float(a[:,j+1].mean())for j,i in enumerate(IDS)},
    detected_full_chessboard_samples_first_minute=len(board),board_views=board,
    reference_motion=ref_shift,seconds=time.monotonic()-start,
    original_not_modified=True,preview_derivatives_only=True,calibrated=False,robot_retarget_run=False,training_eligible=False,
    interpretation='Whole-video visibility includes calibration/reset intervals. Sparse samples cannot establish contact or dynamic hand-height accuracy.')
(OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items()if k!='board_views'}),flush=True)
