from fealpy.backend import backend_manager as bm
bm.set_backend('numpy')

import matplotlib.pyplot as plt
from fealpy.mesh import UniformMesh
from fealpy.model.poisson import get_example
from fealpy.fdm import LaplaceOperator
from fealpy.fdm import DirichletBC
# example = get_example()
# pde = get_example('sin', flag = True)
domain = [0 ,1, 0, 1, 0, 1] 
extent = [0, 5, 0, 5, 0, 5]
mesh = UniformMesh(domain, extent)
K = mesh.linear_index_map('node')
print(K)
# A = LaplaceOperator(mesh=mesh).assembly()
# print(A.to_dense())

# node = mesh.entity('node')
# F = pde.source(node)

# from fealpy.fdm import DirichletBC
# bc = DirichletBC(mesh=mesh, gd=pde.dirichlet)
# A, F = bc.apply(A, F)
# print(A.to_dense())

# from 