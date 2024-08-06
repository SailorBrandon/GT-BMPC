"""
Setup for simulator package
"""
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['motion_primitive_planner'],
    package_dir={'': 'scripts'}
)

setup(**d)
