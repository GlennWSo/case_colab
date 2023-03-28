#!/usr/bin/env python

from setuptools import setup

with open("VERSION", mode="r") as file:
    version = file.read()

setup(
    name="lungdata",
    version=version,
    description="",
    author="Glenn",
    author_email="gward@python.net",
    packages=[
        "lungdata",
        "lungdata.meta",
    ],
    # tell setup that the root python source is inside py folder
    # package_dir={
    #     "lungdata": "src",
    # },
    include_package_data=True,
    install_requires=[
        "numpy",
        "librosa",
        "pandas",
        "soundfile",
        "toml",
    ],
    entry_points={
        "console_scripts": ["pickle_db=lungdata.cli:make_dataset"],
    },
    zip_safe=False,
)
