#!/usr/bin/env python3
"""Generate a same-backend, non-ROS reference for ROS transport regression.

Keep the preserved Windows reference unchanged. This isolates ROS message/node
effects from differences between OpenCV/NumPy/LAPACK versions. It is not a new
independent ground truth and does not relax the ROS comparison tolerances.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
import cv2
import numpy as np
from dolbat_core.runtime import Runtime

ROOT=Path(__file__).resolve().parents[1]
FIELDS=['frame','time_s','valid','status','target_x_m','target_y_m','steering_rad',
        'steering_deg','curvature_per_m','heading_error_rad','segment_index',
        'segment_fraction','candidate_count']
NUMERIC=['target_x_m','target_y_m','steering_rad','steering_deg',
         'curvature_per_m','heading_error_rad','segment_fraction']


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))


def generate(video,output,windows_reference):
    video,output,windows_reference=Path(video),Path(output),Path(windows_reference)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Use a new or empty native reference directory.')
    provenance=json.loads((windows_reference/'reference_provenance.json').read_text(encoding='utf-8'))
    source_hash=hashlib.sha256(video.read_bytes()).hexdigest()
    if source_hash!=provenance['source_video_sha256']:
        raise ValueError('Use the original calibrated dolbat_toyproject.mp4 recording.')
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise ValueError('Cannot open original video')
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if fps!=float(provenance['fps']) or size!=(640,480):
        raise ValueError('Source metadata mismatch')
    cv2.setNumThreads(2)
    runtime=Runtime(fps=fps,image_size=size)
    rows=[];paths=[];started=time.perf_counter()
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            index=len(rows)
            p=runtime.pre.process(frame)
            lanes=runtime.detector.detect(p.binary,runtime.pre.paint,runtime.pre.suppression.crosswalk)
            plan=runtime.plan(lanes)
            rear=runtime.rear_path(plan.path_cm)
            control=runtime.control(plan.path_cm)
            row={key:getattr(control,key,None) for key in FIELDS}
            row.update(frame=index,time_s=index/fps,
                       target_x_m=control.target_m[0] if control.valid else None,
                       target_y_m=control.target_m[1] if control.valid else None)
            rows.append(row);paths.append(rear)
            if len(rows)%100==0:print(f'Native reference {len(rows)}/{provenance["frames"]}',flush=True)
    finally:cap.release()
    if len(rows)!=int(provenance['frames']):raise AssertionError('Incomplete original recording')
    native_paths=np.stack(paths)
    if not np.isfinite(native_paths).all():raise AssertionError('Nonfinite native paths')
    windows_rows=read_csv(windows_reference/'reference_control.csv')
    with np.load(windows_reference/'reference_paths.npz',allow_pickle=False) as archive:
        windows_paths=archive['paths_rear_m'].copy()
    if len(windows_rows)!=len(rows) or windows_paths.shape!=native_paths.shape:
        raise AssertionError('Native/Windows frame or path shape mismatch')
    comparison=[];maxima={key:0. for key in NUMERIC};status_changes=[]
    for native,windows in zip(rows,windows_rows):
        index=native['frame']
        changed=(str(native['valid']).lower()!=windows['valid'].lower() or native['status']!=windows['status'])
        if changed:status_changes.append(index)
        entry={'frame':index,'native_status':native['status'],'windows_status':windows['status'],
               'validity_or_status_changed':changed,
               'path_max_absolute_difference_m':float(np.max(abs(native_paths[index]-windows_paths[index])))}
        for key in NUMERIC:
            if native[key] is not None and windows[key]!='':
                delta=abs(float(native[key])-float(windows[key]))
                maxima[key]=max(maxima[key],delta);entry[key+'_absolute_difference']=delta
        entry['segment_index_changed']=str(native['segment_index'])!=windows['segment_index']
        entry['candidate_count_changed']=str(native['candidate_count'])!=windows['candidate_count']
        comparison.append(entry)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'reference_control.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
    np.savez_compressed(output/'reference_paths.npz',paths_rear_m=native_paths)
    backend={'platform':platform.platform(),'python':platform.python_version(),
             'numpy':np.__version__,'opencv':cv2.__version__,'runtime_module':__import__('dolbat_core.runtime',fromlist=['__file__']).__file__}
    report={'status':'generated','scope':'Same-backend serial core reference, without ROS/DDS. Not independent lane ground truth.',
            'frames':len(rows),'fps':fps,'source_video_sha256':source_hash,'backend':backend,
            'valid_frames':sum(bool(r['valid']) for r in rows),'elapsed_seconds':time.perf_counter()-started,
            'windows_status_changed_frames':status_changes,'windows_max_absolute_control_differences':maxima,
            'windows_max_absolute_path_difference_m':float(np.max(abs(native_paths-windows_paths))),
            'per_frame_windows_comparison':comparison,
            'tolerance_note':'Differences are reported without pass/fail thresholds. ROS-versus-native comparison retains strict tolerances.'}
    (output/'windows_comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    native_provenance={'source_video_sha256':source_hash,'frames':len(rows),'fps':fps,
                      'source_run':'Serial native core on the same backend as the ROS run','backend':backend,
                      'scope':report['scope'],'reference_sha256':{
                          name:hashlib.sha256((output/name).read_bytes()).hexdigest()
                          for name in ('reference_control.csv','reference_paths.npz')}}
    (output/'reference_provenance.json').write_text(json.dumps(native_provenance,indent=2),encoding='utf-8')
    print(json.dumps({key:value for key,value in report.items() if key!='per_frame_windows_comparison'},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,default=ROOT/'data/dolbat_toyproject.mp4')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/native_reference')
    parser.add_argument('--windows-reference-dir',type=Path,default=ROOT/'validation')
    args=parser.parse_args()
    generate(args.video,args.output,args.windows_reference_dir)
