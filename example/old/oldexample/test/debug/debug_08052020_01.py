#!/usr/bin/env python3
#

from fealpy.functionspace.ScaledMonomialSpace2d import ScaledMonomialSpace2d
import numpy as np
from fealpy.mesh import TriangleMesh
from fealpy.decorator import cartesian

# --- mesh
n = 0
node = np.array([
    (0, 0),
    (1, 0),
    (1, 1),
    (0, 1)], dtype=np.float)
cell = np.array([
    (1, 2, 0),
    (3, 0, 2)], dtype=np.int)
mesh = TriangleMesh(node, cell)
mesh.uniform_refine(n)

p = 1
smspace = ScaledMonomialSpace2d(mesh, p)

uh = smspace.function()

@cartesian
def f1(x, index=np.s_[:]):
    return uh.value(x, index)

S = smspace.integralalg.integral(f1, celltype=True, barycenter=False)
