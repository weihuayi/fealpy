import numpy as np
import pytest
from fealpy.decorator import cartesian
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import ScalarMassIntegrator 

@pytest.mark.parametrize('p, mtype', 
        [(p, mtype) for p in range(1, 7) for mtype in ('equ', 'iso')])
def test_one_triangle_mesh(p, mtype):
    mesh = TriangleMesh.from_one_triangle(meshtype=mtype)
    space = LagrangeFiniteElementSpace(mesh, p=p)
    mi = ScalarMassIntegrator(q=p+3)
    M = mi.assembly_cell_matrix(space, space)
    
    cell2dof = space.cell_to_dof()
    M0 = space.mass_matrix().toarray()
    M0 = M0[cell2dof[0], :][:, cell2dof[0]]
    assert np.allclose(M[0], M0)

@pytest.mark.parametrize('p, mtype', 
        [(p, mtype) for p in range(1, 7) for mtype in ('equ', 'iso')])
def test_one_triangle_mesh_with_scalar_coef(p, mtype):
    @cartesian
    def coef(p):
        x = p[..., 0]
        return x**2 + 1
    mesh = TriangleMesh.from_one_triangle(meshtype=mtype)
    space = LagrangeFiniteElementSpace(mesh, p=p)
    mi = ScalaMassIntegrator(coef, q=p+3)
    M = mi.assembly_cell_matrix(space, space)
    cell2dof = space.cell_to_dof()
    M0 = space.mass_matrix(c=coef).toarray()
    M0 = M0[cell2dof[0], :][:, cell2dof[0]]
    assert np.allclose(M[0], M0)

def test_fast_assembel():
    @cartesian
    def coef(p):
        x = p[..., 0]
        return x**2 + 1
    mesh = TriangleMesh.from_one_triangle()
    space = LagrangeFESpace(mesh, p=2)
    mi = ScalarMassIntegrator(q=3)
    FM = mi.assembly_cell_matrix_fast(space)
    M = mi.assembly_cell_matrix(space)
    print(FM-M)
    assert np.allclose(FM, M)
