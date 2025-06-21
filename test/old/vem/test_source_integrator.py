import numpy as np
from fealpy.mesh import PolygonMesh 
from fealpy.mesh import TriangleMesh

from fealpy.functionspace import ConformingScalarVESpace2d 
from fealpy.vem import ConformingVEMScalarSourceIntegrator2d
from fealpy.vem import LinearForm
from fealpy.vem import ScaledMonomialSpaceMassIntegrator2d
from fealpy.vem import ConformingVEMDoFIntegrator2d
from fealpy.vem import ConformingScalarVEMH1Projector2d
from fealpy.vem import ConformingScalarVEMLaplaceIntegrator2d
from fealpy.vem import ConformingScalarVEML2Projector2d 


def test_source_integrator(p):
    nx = 4
    ny = 4
    domain = [0, 1, 0, 1]
    def f(p):
        x = p[...,0]
        y = p[...,1]
        val = x**2+y**2
        return val
    tmesh = TriangleMesh.from_box(domain, nx=nx, ny=ny)
    mesh = PolygonMesh.from_triangle_mesh_by_dual(tmesh)
    space =  ConformingScalarVESpace2d(mesh, p=p)
    #构造PI0
    m = ScaledMonomialSpaceMassIntegrator2d()
    M = m.assembly_cell_matrix(space.smspace)

    d = ConformingVEMDoFIntegrator2d()
    D = d.assembly_cell_matrix(space, M)


    projector = ConformingScalarVEMH1Projector2d(D)
    PI1 = projector.assembly_cell_matrix(space)


    a = ConformingScalarVEML2Projector2d(M, PI1)
    C = a.assembly_cell_right_hand_side(space)
    PI0 = a.assembly_cell_matrix(space)


    b = ConformingVEMScalarSourceIntegrator2d(f,PI0)
    a = LinearForm(space)
    a.add_domain_integrator(b)
    F = a.assembly()
if __name__ == "__main__":
    test_source_integrator(2)
