import numpy as np
import matplotlib.pyplot as plt
from fealpy.mesh import StructureQuadMesh
from fealpy.pde.poisson_2d import CosCosData
from fealpy.tools.show import showmultirate

class StructureQuadMeshTest:
    def __init__(self):
        box = [0.0, 1.0, 0.0, 1.0]
        nx = 3
        ny = 3
        self.mesh = StructureQuadMesh(box, nx, ny)

    def test_cell_location(self):
        mesh = self.mesh
        p = np.random.rand(10, 2)
        cidx = mesh.cell_location(p)
        cell = mesh.entity('cell')
        fig = plt.figure()
        axes = fig.gca()
        mesh.add_plot(axes)
        mesh.find_cell(axes, showindex=True)
        mesh.find_node(axes, node=p, showindex=True)
        plt.show()

    def test_interpolation(self):
        pde = CosCosData()
        F0 = self.mesh.interpolation(pde.solution, intertype='node')
        print(F0.shape)
        F1 = self.mesh.interpolation(pde.solution, intertype='edge')
        print(F1.shape)
        F2 = self.mesh.interpolation(pde.solution, intertype='edgex')
        print(F2.shape)
        F3 = self.mesh.interpolation(pde.solution, intertype='edgey')
        print(F3.shape)
        F4 = self.mesh.interpolation(pde.solution, intertype='cell')
        print(F4.shape)

    def test_polation_interoperator(self):
        pde = CosCosData()
        maxit = 4
        errorType = ['$|u_I - u_h|_{max}$']
        Maxerror = np.zeros((len(errorType), maxit), dtype=np.float)
        NE = np.zeros(maxit,)

        for i in range(maxit):
            mesh = self.mesh
            isBDEdge = mesh.ds.boundary_edge_flag()
            NE[i] = mesh.number_of_edges()
            bc = mesh.entity_barycenter('cell')
            ec = mesh.entity_barycenter('edge')

            uI = mesh.polation_interoperator(pde.solution(bc))
            uh = pde.solution(ec)
            Maxerror[0, i] = max(abs(uh - uI))

            if i < maxit - 1:
                mesh.uniform_refine()

        showmultirate(plt, 0, NE, Maxerror, errorType)
        print('Maxerror', Maxerror)
        plt.show()







test = StructureQuadMeshTest()
#test.test_cell_location()
#test.test_interpolation()
test.test_polation_interoperator()
