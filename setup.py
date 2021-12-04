#!/usr/bin/env python
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

setup(
    name="cascade_",
    version="0.1",
    description="",
    author="Konstantin Willeke",
    author_email="konstantin.willeke@gmail.com",
    packages=find_packages(exclude=[]),
    install_requires=[
        "sphinx",
        "pytorch_sphinx_theme",
        "recommonmark",
    ],
)
