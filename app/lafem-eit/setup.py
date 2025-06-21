
import os
import pathlib
from setuptools import setup, find_packages

from lafemeit import __version__


setup(
    name="lafemeit",
    version=__version__,
    description="LaFEM-EIT: Learn-automated FEM Electrical Impedence Tomography",
    url="",
    author="Albert",
    author_email="",
    license="GNU",
    packages=find_packages(),
    python_requires=">=3.8",
)
