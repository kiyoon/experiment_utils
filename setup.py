"""setup script for experiment_utils package."""

from setuptools import setup

setup(
    name = "experiment_utils",
    version="0.1",
    author = "Kiyoon Kim",
    author_email='kiyoon.kim@ed.ac.uk',
    description = "A python 3 module to construct deep learning experiments (dirs, files, stats, plots, etc.)",
    url = "https://github.com/kiyoon/experiment_utils",
    packages=['experiment_utils'],
    #package_dir={'experiment_utils': 'src'},
    python_requires='>=3.6',
    install_requires=['numpy>=1.16.0'],
)
