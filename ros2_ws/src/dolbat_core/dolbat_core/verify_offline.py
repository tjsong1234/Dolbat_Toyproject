"""Compare the ROS-independent port against the preserved Windows regression.

This does not validate DDS, executors or ROS message transport. The ROS launch
regression performs that separate check after colcon build.
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
from .runtime import Runtime


def verify(video,reference,output=None,path_atol_m=1e-5,steering_atol_deg=1e-4):
    video,reference=Path(video),Path(reference)
    provenance=json.loads((reference/'reference_provenance.json').read_text(encoding='utf-8'))
    source_hash=hashlib.sha256(video.read_bytes()).hexdigest()
    if source_hash!=provenance['source_video_sha256']:
        raise ValueError('The original calibrated dolbat_toyproject.mp4 is required.')
    with (reference/'reference_control.csv').open(encoding='utf-8-sig',newline='') as handle:
        expected_control=list(csv.DictReader(handle))
    with np.load(reference/'reference_paths.npz',allow_pickle=False) as data:
        expected_paths=data['paths_rear_m'].copy()
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise ValueError('Cannot open the original recording')
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    if abs(fps-provenance['fps'])>1e-6:raise ValueError('Source FPS mismatch')
    cv2.setNumThreads(2)
    runtime=Runtime(fps=fps)
    index=0;max_path=0.;max_steering=0.;valid=0;started=time.perf_counter()
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            if index>=len(expected_control):raise AssertionError('Unexpected extra source frame')
            p=runtime.pre.process(frame)
            lanes=runtime.detector.detect(p.binary,runtime.pre.paint,runtime.pre.suppression.crosswalk)
            plan=runtime.plan(lanes);rear=runtime.rear_path(plan.path_cm);control=runtime.control(plan.path_cm)
            if rear.shape!=expected_paths[index].shape:raise AssertionError(f'Path shape changed at frame {index}')
            error=float(np.max(abs(rear-expected_paths[index])))
            max_path=max(max_path,error)
            if error>path_atol_m:raise AssertionError(f'Path changed at frame {index}: {error:g} m')
            expected=expected_control[index]
            if control.valid!=(expected['valid'].lower()=='true') or control.status!=expected['status']:
                raise AssertionError(f'Control validity changed at frame {index}')
            if control.valid:
                valid+=1
                steering_error=abs(control.steering_deg-float(expected['steering_deg']))
                max_steering=max(max_steering,steering_error)
                if steering_error>steering_atol_deg:
                    raise AssertionError(f'Steering changed at frame {index}: {steering_error:g} deg')
                np.testing.assert_allclose(control.target_m,[float(expected['target_x_m']),float(expected['target_y_m'])],atol=path_atol_m,rtol=0)
            index+=1
    finally:cap.release()
    if index!=provenance['frames'] or index!=len(expected_control):raise AssertionError('Incomplete original video')
    result={'status':'passed','scope':'ROS-independent core only; no ROS/DDS validation',
            'platform':platform.system(),'python':platform.python_version(),'numpy':np.__version__,'opencv':cv2.__version__,
            'frames':index,'valid_control_frames':valid,'source_sha256':source_hash,
            'max_path_difference_m':max_path,'max_steering_difference_deg':max_steering,
            'path_atol_m':path_atol_m,'steering_atol_deg':steering_atol_deg,
            'elapsed_seconds':time.perf_counter()-started}
    if output:
        output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    print(json.dumps(verify(args.video,args.reference,args.output),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
