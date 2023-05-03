#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pathlib

from setuptools import setup, find_packages


HERE = pathlib.Path(__file__).parent
README = (HERE / 'README.md').read_text()


def read_requirements(reqs_path):
    with open(reqs_path, encoding='utf8') as f:
        reqs = [
            line.strip()
            for line in f
            if (not line.strip().startswith('#')
                and not line.strip().startswith('--'))
        ]
    return reqs


setup(
    name="wav2vec2_hebrew",
    version="0.0.1",
    description="Get link (URL) preview",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Vladimir Gurevich",
    license="",
    keywords="",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Unix",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries",
    ],
    packages=find_packages(exclude=['tests*', 'scripts', 'utils']),
    include_package_data=True,
    install_requires=read_requirements(HERE / 'requirements.txt'),
)