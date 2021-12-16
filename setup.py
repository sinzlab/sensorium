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
        "sphinx",
        "pytorch_sphinx_theme",
        "recommonmark",
        "nnfabrik==0.1.0",
    ],
)
