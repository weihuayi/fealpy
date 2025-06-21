"""
Machine Learning
================
The machine leaning module in FEALPy for solving PDEs.

Contents
--------

1. Sampler classes
2. Module classes
3. Operator classes
4. PDE preset classes

"""

from .solvertools import rescale, ridge
from .hyperparams import AutoTest, timer
from .tools import mkfs, use_mkfs, proj, as_tensor_func
