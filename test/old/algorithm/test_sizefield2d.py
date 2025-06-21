import numpy as np
import pytest

from fealpy.mesh import UniformMesh3d, MeshFactory
from fealpy.algorithm import BackgroundMeshInterpolationAlg2D
import time

def exu(x):
    return np.sin(np.pi*x[..., 0])*np.sin(np.pi*x[..., 1])

@pytest.mark.parametrize("N", [10])
def test_interpolation(N): 
    alg   = BackgroundMeshInterpolationAlg2D([0, 1, 0, 1], N, N)
    mesht = MeshFactory.boxmesh2d([0, 1, 0, 1], 4, 4, meshtype='tri')
    tnode = mesht.entity('node')
    tval  = exu(tnode)
    f     = alg.interpolation_with_sample_points(tnode, tval)
    pnode = alg.mesh.node
    maxerr = np.max(np.abs(f.f[:-1, :-1] - exu(pnode[:-1, :-1])))
    assert(np.abs(maxerr-0.025728070205245035)<1e-5)


