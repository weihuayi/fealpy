import numpy as np
import pytest
from fealpy.decorator import cartesian
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.fem import ConvectionIntegrator 

@cartesian
def u(p):
    x = p[...,0]
    y = p[...,1]
    u = np.zeros(p.shape)
    u[...,0] = np.sin((np.pi*x))**2 * np.sin(2*np.pi*y)
    u[...,1] = -np.sin((np.pi*y))**2 * np.sin(2*np.pi*x)
    return u

@pytest.mark.parametrize('p, mtype', 
        [(p, mtype) for p in range(1, 7) for mtype in ('equ', 'iso')])
def test_one_triangle_mesh(p, mtype):
    mesh = TriangleMesh.from_one_triangle(meshtype=mtype)
    space = LagrangeFiniteElementSpace(mesh, p=p)
    di = ConvectionIntegrator(q=p+3, c=u)
    C = di.assembly_cell_matrix(space)
    cell2dof = space.cell_to_dof()
    C0 = space.convection_matrix(u).toarray()
    C0 = C0[cell2dof[0], :][:, cell2dof[0]]
    assert np.allclose(C[0], C)

if __name__ == '__main__':
    test_one_triangle_mesh(1, 'equ')    
