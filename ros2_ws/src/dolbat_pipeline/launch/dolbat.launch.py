"""Launch all stages as separate processes, with bounded frame credits."""
from datetime import datetime
from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,EmitEvent,LogInfo,RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def default_video():
    roots=[Path.cwd(),Path.cwd().parent,*Path(__file__).resolve().parents]
    for root in roots:
        candidate=root/'data/dolbat_toyproject.mp4'
        if candidate.is_file():return str(candidate)
    return str(Path.cwd()/'data/dolbat_toyproject.mp4')


def exited(event,context,recorder=False):
    if recorder:
        return [LogInfo(msg=f'Recording process exited with code {event.returncode}'),
                EmitEvent(event=Shutdown(reason='Synchronized recorder finished or failed'))]
    if event.returncode!=0:
        return [LogInfo(msg=f'Pipeline process failed with code {event.returncode}'),
                EmitEvent(event=Shutdown(reason='A pipeline stage failed'))]
    return []


def generate_launch_description():
    video=LaunchConfiguration('video');output=LaunchConfiguration('output')
    display=LaunchConfiguration('display');replay=LaunchConfiguration('replay_rate');window=LaunchConfiguration('max_in_flight')
    common={'package':'dolbat_pipeline','output':'screen','emulate_tty':True}
    recorder=Node(executable='visualization',parameters=[{'output':output,'display':ParameterValue(display,value_type=bool)}],**common)
    stages=[recorder,Node(executable='pure_pursuit',**common),Node(executable='path_planning',**common),
            Node(executable='sliding_window',**common),Node(executable='preprocessing',**common),
            Node(executable='video_source',parameters=[{'video':video,
                'replay_rate':ParameterValue(replay,value_type=float),
                'max_in_flight':ParameterValue(window,value_type=int)}],**common)]
    arguments=[DeclareLaunchArgument('video',default_value=default_video(),description='Original calibrated MP4'),
               DeclareLaunchArgument('output',default_value=str(Path.cwd()/'outputs'/datetime.now().strftime('ros2_%Y%m%d_%H%M%S'))),
               DeclareLaunchArgument('display',default_value='false',description='Show OpenCV window; false works headlessly'),
               DeclareLaunchArgument('replay_rate',default_value='1.0',description='1=original FPS; 0=as fast as credits allow'),
               DeclareLaunchArgument('max_in_flight',default_value='4',description='1..4; full queues wait instead of dropping frames')]
    handlers=[RegisterEventHandler(OnProcessExit(target_action=node,
              on_exit=lambda event,context,is_recorder=node is recorder:exited(event,context,is_recorder))) for node in stages]
    return LaunchDescription(arguments+handlers+stages)
