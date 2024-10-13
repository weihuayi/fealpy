
import sys
import pytest
import numpy as np

from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace, BernsteinFESpace

def u(p):
    x = p[..., 0]
    y = p[..., 1]
    return np.sin(np.pi*x)*np.sin(np.pi*y)

def grad_u(p):
    x = p[..., 0]
    y = p[..., 1]
    val = np.zeros(p.shape, dtype=np.float_)
    val[..., 0] = np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)
    val[..., 1] = np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)
    return val

@pytest.mark.parametrize("p", range(1, 6))
def test_interpolation_fe_function(p):
    error = np.zeros((2, 4), dtype=np.float_)
    for i in range(4):
        N = 2**(i+2)
        mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=N, ny=N)
        space = BernsteinFESpace(mesh, p=p)

        uI = space.interpolate(u)
        error[0, i] = mesh.error(u, uI)
        error[1, i] = mesh.error(grad_u, uI.grad_value)
    order = np.log2(error[:, :-1]/error[:, 1:])
    assert(np.all(np.abs(order[0, -2:]-p-1)<0.2))
    assert(np.all(np.abs(order[1, -2:]-p)<0.2))




