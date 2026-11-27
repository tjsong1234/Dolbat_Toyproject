from setuptools import setup, find_packages

setup(name='dolbat_core',version='0.1.0',packages=find_packages(),
      data_files=[('share/ament_index/resource_index/packages',['resource/dolbat_core']),
                  ('share/dolbat_core',['package.xml'])],
      package_data={'dolbat_core':['resources/*.json','resources/config/*.yaml']},
      install_requires=['setuptools'],zip_safe=False,
      maintainer='tjsong1234',maintainer_email='tjsong1234@users.noreply.github.com',
      description='Verified CPU perception, B-spline planning and Pure Pursuit core.',
      license='UNLICENSED',tests_require=['pytest'],
      entry_points={'console_scripts':['dolbat_verify_offline = dolbat_core.verify_offline:main']})
