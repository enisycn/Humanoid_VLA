"""Pixel-detected ID12 boundaries for one-file demonstrations.

ID12 is a data-collection intervention flag, never a robot input. Only closed
gaps between confirmed reset intervals are candidates. No actions are created
and no segment is automatically admitted to training.
"""
from pathlib import Path
import argparse, hashlib, json
import av, cv2, numpy as np

IDS=[0,1,2,3,10,11]
CONFIG=dict(reset_id=12,minimum_reset_visible_s=.25,bridge_dropout_s=.15,
    guard_after_removal_s=.50,guard_before_appearance_s=.25,
    minimum_segment_s=1.,minimum_core_marker_fraction=.95)

def intervals(mask):
    edges=np.diff(np.r_[False,mask,False].astype(np.int8))
    return list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))

def segment(times,visible):
    times=np.asarray(times,float);visible=np.asarray(visible,bool)
    assert visible.shape==(len(times),7)and len(times)>2 and np.all(np.diff(times)>0)
    dt=float(np.median(np.diff(times)));raw=visible[:,6];closed=raw.copy()
    for lo,hi in intervals(~raw):
        if lo>0 and hi<len(raw)and times[hi]-times[lo]<=CONFIG['bridge_dropout_s']+1e-8:
            closed[lo:hi]=True
    confirmed=[];ambiguous=[]
    for lo,hi in intervals(closed):
        span=dict(first_frame=int(lo),last_frame_exclusive=int(hi),start_s=float(times[lo]),end_s=float(times[hi-1]+dt))
        (confirmed if span['end_s']-span['start_s']>=CONFIG['minimum_reset_visible_s']else ambiguous).append(span)
    accepted=[];rejected=[]
    for previous,next_reset in zip(confirmed,confirmed[1:]):
        start=previous['end_s']+CONFIG['guard_after_removal_s']
        end=next_reset['start_s']-CONFIG['guard_before_appearance_s']
        lo=int(np.searchsorted(times,start));hi=int(np.searchsorted(times,end))
        if hi<=lo:continue
        core=visible[lo:hi,:6].all(axis=1)
        # Choose the first real valid frame; no future fill of missing markers.
        if np.any(core):lo+=int(np.flatnonzero(core)[0]);core=visible[lo:hi,:6].all(axis=1)
        row=dict(first_frame=lo,last_frame_exclusive=hi,start_s=float(times[lo]),end_s=float(times[hi-1]+dt),
            core_marker_fraction=float(core.mean()),reset_seen_inside=bool(raw[lo:hi].any()),
            preceding_reset=previous,following_reset=next_reset)
        row['checks']=dict(duration=row['end_s']-row['start_s']>=CONFIG['minimum_segment_s'],
            core_markers=bool(core.mean()>=CONFIG['minimum_core_marker_fraction']),no_reset=not row['reset_seen_inside'])
        (accepted if all(row['checks'].values())else rejected).append(row)
    return dict(config=CONFIG,confirmed_reset_intervals=confirmed,ambiguous_short_reset_intervals=ambiguous,
        candidate_demonstrations=accepted,rejected_gaps=rejected,
        boundaries_usable=bool(len(confirmed)>=2 and accepted and not ambiguous and not rejected),
        trailing_after_last_reset_excluded=True,frames_before_first_reset_excluded=True,
        reset_and_guard_frames_excluded=True,no_action_chunk_may_cross_a_candidate_boundary=True,
        training_eligible=False,only_segmentation_checked=True,
        remaining_checks=['actual calibration','marker/contact geometry','dynamic tracking','Panda retarget/admission','intervention-label exclusion'])

def scan(video,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    params=cv2.aruco.DetectorParameters();params.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
    detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),params)
    cv2.setNumThreads(1);times=[];visibility=[];stove_corners=[]
    with av.open(str(video))as container:
        for f in container.decode(video=0):
            assert f.time is not None
            corners,ids,_=detector.detectMarkers(f.to_ndarray(format='gray'))
            found=set()if ids is None else set(ids.ravel().tolist())
            by_id={}if ids is None else {int(i):np.asarray(c[0],float)for i,c in zip(ids.ravel(),corners)}
            times.append(float(f.time));visibility.append([i in found for i in IDS+[12]])
            stove_corners.append(by_id.get(13,np.full((4,2),np.nan)))
    times=np.array(times);times-=times[0];visibility=np.array(visibility)
    result=segment(times,visibility)
    with Path(video).open('rb')as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    result.update(video=str(Path(video).resolve()),video_sha256=digest,frames=len(times),
        actually_decoded_compressed_video=True,marker_ids=IDS+[12],synthetic_or_personal_not_inferred_from_pixels=True,
        optional_stove_anchor_id=13,stove_anchor_visible_frames=int(np.isfinite(stove_corners).all(axis=(1,2)).sum()),
        stove_anchor_is_not_plate_goal=True,stove_anchor_used_for_segmentation=False,
        stove_goal_metric_mapping_not_implemented_here=True)
    np.savez_compressed(out/'detected-frame-flags.npz',times=times,visible=visibility,marker_ids=np.array(IDS+[12]))
    np.savez_compressed(out/'stove-anchor-pixels.npz',times=times,marker_id=np.array(13),corners_px=np.asarray(stove_corners))
    (out/'segments.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();scan(a.video,a.out)
