import numpy as np
import matplotlib.pyplot as plt

from fealpy.functionspace import ConformingScalarVESpace2d 
import ipdb

from fealpy.vem import ScaledMonomialSpaceMassIntegrator2d
from fealpy.vem import ConformingVEMDoFIntegrator2d
from fealpy.vem import ConformingScalarVEMH1Projector2d
from fealpy.vem import ConformingScalarVEMLaplaceIntegrator2d
from fealpy.vem import BilinearForm

from fealpy.mesh import TriangleMesh
from fealpy.mesh import PolygonMesh

def test_assembly_cell_matrix(p, plot=False):
    nx = 4
    ny = 4
    domain = np.array([0, 1, 0, 1])

    tmesh = TriangleMesh.from_box(domain, nx=nx, ny=ny)
    mesh = PolygonMesh.from_triangle_mesh_by_dual(tmesh)
    space =  ConformingScalarVESpace2d(mesh, p=p)
    m = ScaledMonomialSpaceMassIntegrator2d()
    M = m.assembly_cell_matrix(space.smspace)

    d = ConformingVEMDoFIntegrator2d()
    D = d.assembly_cell_matrix(space, M)

    projector = ConformingScalarVEMH1Projector2d(D)
    PI1 = projector.assembly_cell_matrix(space)


    a = BilinearForm(space)

    I = ConformingScalarVEMLaplaceIntegrator2d(projector)
    a.add_domain_integrator(I)
    stiff = a.assembly()

    if plot:
        fig ,axes = plt.subplots()
        mesh.add_plot(axes)
        mesh.find_node(axes, showindex=True)
        mesh.find_cell(axes, showindex=True)
        mesh.find_edge(axes, showindex=True)
        plt.show()
if __name__ == "__main__":
    test_assembly_cell_matrix(2)

