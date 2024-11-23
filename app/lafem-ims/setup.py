
import os
import pathlib
from setuptools import setup, find_packages

from lafemims import __version__


setup(
    name="lafemims",
    version=__version__,
    description="LaFEM-IMS: Learn-automated FEM  Inverse Medium Scattering",
    url="",
    author="Chaos",
    author_email="",
    license="GNU",
    packages=find_packages(),
    python_requires=">=3.8",
)
