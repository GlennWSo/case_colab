#!/usr/bin/env python


from setuptools import setup

setup(
    name="lungdata",
    version="0.0.1",
    description="",
    author="Glenn",
    author_email="gward@python.net",
    packages=[
        "lungdata",
    ],
    # tell setup that the root python source is inside py folder
    # package_dir={
    #     "lungdata": "src",
    # },
    # install_requires=["pyvista", "numpy"],
    entry_points={
        "console_scripts": ["pickle_db=lungdata.cli:make_dataset"],
    },
    zip_safe=False,
)
