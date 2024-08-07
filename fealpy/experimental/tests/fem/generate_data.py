import numpy as np
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
import matplotlib.pyplot as plt
from fealpy.decorator import barycentric, cartesian
from fealpy.fem import ScalarMassIntegrator




mesh = TriangleMesh.from_box([0,1,0,1], 1, 1)
space = LagrangeFESpace(mesh, 2)
integrator = ScalarMassIntegrator(1, 3)
assembly_cell_matrix = integrator.assembly_cell_matrix(space)
print(assembly_cell_matrix)





