#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name="cascade",
    version="0.0",
    description="Code base for neural prediction challenge.",
    author="sinzlab",
    # author_email="sinzlab@gmail.com",
    packages=find_packages(exclude=[]),
    install_requires=[
        "nnfabrik==0.1.0",
        "scikit-image>=0.19.1",
        "lipstick",
        "numpy>=1.22.0",
    ],
)
