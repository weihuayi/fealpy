import numpy as np
import pytest
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import LinearElasticityOperatorIntegrator

@pytest.fixture
def mesh_and_space():
    mesh = TriangleMesh.from_one_triangle()
    p = 1
    space = LagrangeFESpace(mesh, p=p, doforder='vdims')
    GD = 2
    vspace = GD*(space, )
    return mesh, vspace

def test_assembly_cell_matrix_fast(mesh_and_space):
    mesh, vspace = mesh_and_space
    p = vspace[0].p
    lam = 1.0
    mu = 1.0

    # 测试 c 为 None
    mi = LinearElasticityOperatorIntegrator(lam=lam, mu=mu, q=p+2)
    FM = mi.assembly_cell_matrix_fast(space=vspace)
    M = mi.assembly_cell_matrix(space=vspace)
    assert np.allclose(FM, M)

    # 测试 c 为标量
    scalar_coef = 2.0
    mi = LinearElasticityOperatorIntegrator(lam=lam, mu=mu, q=p+2, c=scalar_coef)
    FM = mi.assembly_cell_matrix_fast(space=vspace)
    M = mi.assembly_cell_matrix(space=vspace)
    assert np.allclose(FM, M)

    # 测试 c 为函数
    from fealpy.decorator import cartesian
    @cartesian
    def func_coef(p):
        x = p[..., 0]
        y = p[..., 1]
        return x + y

    mi = LinearElasticityOperatorIntegrator(lam=lam, mu=mu, q=p+2, c=func_coef)
    FM = mi.assembly_cell_matrix_fast(space=vspace)
    M = mi.assembly_cell_matrix(space=vspace)
    assert np.allclose(FM, M)

