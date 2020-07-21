import os
import pathlib
from setuptools import setup, find_packages
from fealpy import __version__


here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")


def load_requirements(path_dir=here, comment_char="#"):
    with open(os.path.join(path_dir, "requirements.txt"), "r") as file:
        lines = [line.strip() for line in file.readlines()]
    requirements = []
    for line in lines:
        # filer all comments
        if comment_char in line:
            line = line[: line.index(comment_char)]
        if line:  # if requirement is not empty
            requirements.append(line)
    return requirements


setup(
    name="fealpy",
    version=__version__,
    description="FEALPy: Finite Element Analysis Library in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/weihuayi/fealpy",
    author="Huayi Wei",
    author_email="weihuayi@xtu.edu.cn",
    license="GNU",
    packages=find_packages(),
    install_requires=load_requirements(),
    zip_safe=False,
    extras_require={
        "doc": ["sphinx", "recommonmark", "sphinx-rtd-theme"],
        "dev": ["pytest>=3.6", "pytest-cov", "codecov", "bump2version"],
    },
    include_package_data=True,
    python_requires=">=3.6",
)
