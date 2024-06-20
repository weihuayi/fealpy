import pytest
import ipdb

import numpy as np

from fealpy.mesh import TriangleMesh as Mesh

from fealpy.np import logger
from fealpy.np.mesh.utils import *
from fealpy.np.mesh.functional import *


def test_simplex_shape_function():

    q = 3
    p = 2

    mesh = Mesh.from_box(nx=2, ny=2)
    qf = mesh.integrator(q)
    bcs, ws = qf.get_quadrature_points_and_weights()
    #ipdb.set_trace()
    phi0 = mesh.shape_function(bcs, p)
    gphi0 = mesh.grad_shape_function(bcs, p)

    TD = mesh.top_dimension()
    mi = mesh.multi_index_matrix(p, TD)
    phi = simplex_shape_function(bcs, p, mi)
    gphi = simplex_grad_shape_function(bcs, p, mi)

    err0 = np.abs(phi0 - phi)
    assert(np.max(err0 < 1e-12))

    print(gphi0.shape)
    print(gphi.shape)
    print(TD)
    #err1 = np.abs(gphi0 - gphi)
    #assert(np.max(err1 < 1e-12))

if __name__ == "__main__":
    test_simplex_shape_function()
