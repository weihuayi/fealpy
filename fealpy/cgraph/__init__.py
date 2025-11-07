"""
FEALPy Computing Graph
============
"""

from .core import *
from .nodetype import CNodeType, search, create, from_dict, to_dict
from .registry import *

__nodes__ = [
    "const",
    "ops",
    "model",
    "mesh",
    "functionspace",
    "solver",
    "postprocess",
    "fem",
    "cfd",
    "opt",
    "pathplanning",
    "sampling",
    "material",
    "postreport"
]

register_all_nodes()
