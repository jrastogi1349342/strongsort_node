import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'strongsort_node'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(include=['strong_sort', 'yolov7']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Jai Rastogi',
    maintainer_email='jai1rastogi@gmail.com',
    description='ROS2 Humble Version of StrongSORT Object Tracking Algorithm',
    license='Apache License 2.0',
    # tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'strongsort_entry = strongsort_node.track:main'
        ],
    },
)
