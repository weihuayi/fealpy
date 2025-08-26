"""
FEALPy Computing Graph
============
"""

from .core import WORLD_GRAPH, Graph
from .nodetype import CNodeType, search, create, from_dict, to_dict
from .registry import *

__nodes__ = [
    "model",
    "mesh",
    "functionspace",
    "solver",
    "fem",
    "cfd"
]

register_all_nodes()
