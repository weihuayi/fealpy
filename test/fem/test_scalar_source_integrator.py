import pytest

import numpy as np

from fealpy.mesh import TriangleMesh

from fealpy.functionspace import LagrangeFESpace

from fealpy.decorator import cartesian

from fealpy.fem import LinearForm
from fealpy.fem import BilinearForm
from fealpy.fem import ScalarMassIntegrator
from fealpy.fem import ScalarSourceIntegrator

@pytest.fixture
def setup_fem():
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx = 40, ny = 40)
    space = LagrangeFESpace(mesh, p=2)
    return space

@cartesian
def source(p):
    """
    @brief 返回给定点的源项值 f
    @param[in] p 一个表示空间点坐标的数组
    @return 返回源项值
    """
    x = p[..., 0]
    y = p[..., 1]
    val = 35/13*y - 35/13*y**2 + 10/13*x - 10/13*x**2
    return val

def test_scalar_source_integrator(setup_fem):
    space = setup_fem

    # Method 1: Using ScalarSourceIntegrator
    lform = LinearForm(space)
    lform.add_domain_integrator(ScalarSourceIntegrator(f=source, q=3))
    b = lform.assembly()

    # Method 2: Using interpolation and mass matrix
    bform = BilinearForm(space)
    bform.add_domain_integrator(ScalarMassIntegrator(q=3))
    M = bform.assembly()
    ipoint = space.interpolation_points()
    bb = source(ipoint)
    interpolated_b = M @ bb

    # Assert the results are close
    assert np.allclose(b, interpolated_b, atol=1e-10)
