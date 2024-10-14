import numpy as np
import pytest

from fealpy.mesh import TriangleMesh
from fealpy.fem import SourceVectorIntegrator 
from fealpy.fem import LinearForm
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import LagrangeFiniteElementSpace

def f2d(p):
    x = p[..., 0]
    y = p[..., 1]
    return x+y

def test_one_triangle_mesh():
    mesh = TriangleMesh.from_one_triangle(meshtype='iso')
    space0 = LagrangeFESpace(mesh, p=1)
    space1 = LagrangeFiniteElementSpace(mesh, p=1)

    F = space1.source_vector(f2d)
    integrator = SourceIntegrator(f2d)

