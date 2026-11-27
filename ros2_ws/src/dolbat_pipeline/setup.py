from glob import glob
from setuptools import find_packages,setup

package_name='dolbat_pipeline'
setup(name=package_name,version='0.1.0',packages=find_packages(exclude=['test']),
      data_files=[('share/ament_index/resource_index/packages',['resource/'+package_name]),
                  ('share/'+package_name,['package.xml']),
                  ('share/'+package_name+'/launch',glob('launch/*.launch.py'))],
      install_requires=['setuptools'],zip_safe=True,
      maintainer='tjsong1234',maintainer_email='tjsong1234@users.noreply.github.com',
      description='Six ordered ROS2 processes for the four Dolbat study stages.',
      license='UNLICENSED',tests_require=['pytest'],
      entry_points={'console_scripts':[
          'video_source = dolbat_pipeline.video_source:main',
          'preprocessing = dolbat_pipeline.preprocessing:main',
          'sliding_window = dolbat_pipeline.sliding_window:main',
          'path_planning = dolbat_pipeline.path_planning:main',
          'pure_pursuit = dolbat_pipeline.pure_pursuit:main',
          'visualization = dolbat_pipeline.visualizer:main',
      ]})
