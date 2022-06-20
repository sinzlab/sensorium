#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name="sensorium",
    version="0.0",
    description="Code base for SENSORIUM challenge.",
    author="sinzlab",
    # author_email="sinzlab@gmail.com",
    packages=find_packages(exclude=[]),
    install_requires=[
        "neuralpredictors==0.3.0",
        "nnfabrik==0.2.1",
        "scikit-image>=0.19.1",
        "lipstick",
        "numpy>=1.22.0",
    ],
)
