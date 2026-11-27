#!/usr/bin/env python3
"""Validate the real ROS2 run against the preserved Windows numeric reference."""
import argparse,csv,hashlib,json,math,subprocess,sys
from pathlib import Path
import cv2
import numpy as np


ROOT=Path(__file__).resolve().parents[1]


def csv_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def verify(output,reference_dir=None):
    output=Path(output).resolve()
    if reference_dir is None:
        reference_dir=output/'records/native_reference'
        if not (reference_dir/'reference_control.csv').exists():
            subprocess.run([sys.executable,str(ROOT/'scripts/generate_native_reference.py'),
                            '--output',str(reference_dir)],check=True)
    reference_dir=Path(reference_dir).resolve()
    if (output/'failure.txt').exists():raise AssertionError((output/'failure.txt').read_text())
    summary=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    assert summary['frames']==564 and summary['fps']==24
    rows=csv_rows(output/'records/control.csv');reference=csv_rows(reference_dir/'reference_control.csv')
    assert len(rows)==len(reference)==564
    max_differences={key:0. for key in ('target_x_m','target_y_m','steering_rad','steering_deg','curvature_per_m')}
    # Cross-platform LAPACK/OpenCV roundoff is allowed, never missing frames.
    tolerances={'target_x_m':1e-5,'target_y_m':1e-5,'steering_rad':2e-6,'steering_deg':1e-4,'curvature_per_m':1e-5}
    circle_error=0.
    for i,(actual,expected) in enumerate(zip(rows,reference)):
        assert int(actual['frame'])==i
        assert actual['valid'].lower()==expected['valid'].lower()=='true'
        assert actual['status']==expected['status']=='OK'
        assert int(actual['candidate_count'])==int(expected['candidate_count'])==1
        for key in max_differences:
            delta=abs(float(actual[key])-float(expected[key]));max_differences[key]=max(max_differences[key],delta)
            assert delta<=tolerances[key],f'Frame {i}: {key} drift {delta} exceeds {tolerances[key]}'
        x,y=float(actual['target_x_m']),float(actual['target_y_m'])
        radius=math.hypot(x,y);circle_error=max(circle_error,abs(radius-1.7))
        angle=math.atan(.55*2*math.sin(math.atan2(y,x))/radius)
        assert abs(math.degrees(angle)-float(actual['steering_deg']))<1e-9
    assert circle_error<1e-10
    blocks={}
    for row in csv_rows(output/'records/rear_paths.csv'):
        blocks.setdefault(int(row['frame']),[]).append([float(row['forward_X_m']),float(row['left_Y_m'])])
    paths=np.array([blocks[i] for i in range(564)],float)
    with np.load(reference_dir/'reference_paths.npz',allow_pickle=False) as reference_paths:
        golden=reference_paths['paths_rear_m']
    assert paths.shape==golden.shape==(564,160,2) and np.isfinite(paths).all()
    max_path_error=float(np.max(np.abs(paths-golden)))
    assert max_path_error<=1e-5,f'Path drift {max_path_error}m exceeds 1e-5m'
    for i,row in enumerate(rows):
        k=int(row['segment_index']);t=float(row['segment_fraction'])
        assert 0<=k<159 and 0<=t<=1
        lp=np.array([float(row['target_x_m']),float(row['target_y_m'])])
        assert np.linalg.norm(paths[i,k]+t*(paths[i,k+1]-paths[i,k])-lp)<1e-9
    axes=csv_rows(output/'records/plot_axes.csv')
    assert len(axes)==564 and len({tuple(r[k] for k in r if k!='frame') for r in axes})==1
    videos={}
    for name in ('01_final_four_panels.mp4','02_green_path_control.mp4'):
        cap=cv2.VideoCapture(str(output/name));assert cap.isOpened()
        count=0;fps=cap.get(cv2.CAP_PROP_FPS)
        while True:
            ok,image=cap.read()
            if not ok:break
            assert image.std()>2;count+=1
        cap.release();assert count==564 and abs(fps-24)<1e-6
        videos[name]={'decoded_frames':count,'fps':fps}
    assert summary['writer']['written_pairs']==564 and summary['writer']['frames_dropped']==0
    report={'status':'passed','frames':564,'all_frame_indexes_present':True,'path_points_per_frame':160,
        'path_max_absolute_error_m':max_path_error,'control_max_absolute_differences':max_differences,
        'circle_max_radius_error_m':circle_error,'fixed_plot_transforms':1,'videos':videos,
        'ros2_measured_throughput_fps':summary['pipeline_fps'],
        'performance_note':'Measured on this ROS2 host only; Windows throughput is not a ROS2 throughput claim.',
        'regression_tolerances':tolerances,'path_tolerance_m':1e-5,
        'reference_backend':str(reference_dir),
        'source_video_sha256':hashlib.sha256((ROOT/'data/dolbat_toyproject.mp4').read_bytes()).hexdigest()}
    assert report['source_video_sha256']=='c99bd02d43f12dbcd093b808ef7bdc6c85d17c3b5620808213c5e2e2e14cc66f'
    (output/'records/ros_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-dir',type=Path)
    args=p.parse_args();verify(args.output,args.reference_dir)
