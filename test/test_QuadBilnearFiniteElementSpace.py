import numpy as np
from fealpy.functionspace import QuadBilinearFiniteElementSpace
from fealpy.pde.poisson_2d import CosCosData
from fealpy.mesh import QuadrangleMesh


class QuadBilinearFiniteElementSpaceTest:
    def __init__(self):
        self.pde = CosCosData()

        node = 
        cell = 
        self.mesh = QuadrangleMesh(node, cell)

        self.space = QuadBilinearFiniteElementSpace(self.mesh)

    def test_intetpolation(self):
        pass

    def test_projection(self):
        pass

    def test_sovle_poisson_equation(self):
        pass

test = QuadBilinearFiniteElementSpaceTest()
