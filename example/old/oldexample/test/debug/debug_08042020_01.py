#!/usr/bin/env python3
#

"""
"""

import sys
import numpy as np

from fealpy.decorator import cartesian, barycentric
from fealpy.mesh import MeshFactory
from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.functionspace import RaviartThomasFiniteElementSpace2d


@cartesian
def f(p):
    return p

mf = MeshFactory()
mesh = mf.boxmesh2d([0, 1, 0, 1], nx=2, ny=2, meshtype='tri')
space = RaviartThomasFiniteElementSpace2d(mesh, p=0)
F = space.source_vector(f)
print(F)





