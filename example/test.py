# from fealpy.mesh import UniformMesh
# from fealpy.fdm import LaplaceOperator

# from fealpy.model import PDEDataManager
# from fealpy.fdm.diffusion_operator import DiffusionOperator
# from fealpy.fdm.convection_operator import ConvectionOperator

# pde = PDEDataManager('elliptic').get_example('sinsin')
# domain = pde.domain()  
# GD = pde.geo_dimension()
# extent = [0, 2] * GD
# mesh = UniformMesh(domain, extent)

# from fealpy.fdm.convection_operator import ConvectionOperator
# C = ConvectionOperator(mesh=mesh, convection_coef=pde.convection_coef,
#                        method='upwind').assembly()
# # print(C.to_dense())

# from fealpy.fdm.reaction_operator import ReactionOperator
# R = ReactionOperator(mesh=mesh, reaction_coef=pde.reaction_coef).assembly()
# print(R.to_dense())
# pde = PDEDataManager('poisson').get_example('sinsin')
# domain = pde.domain()  
# GD = pde.geo_dimension()
# extent = [0, 5] * GD
# mesh = UniformMesh(domain, extent)

# A = LaplaceOperator(mesh=mesh).assembly()
# # print(A.to_dense())

# # node = mesh.entity('node')
# # f = pde.source(node)
# # print(f)

# import matplotlib.pyplot as plt

# uI = mesh.interpolate(pde.solution, 'node')
# fig = plt.figure()
# axes = fig.add_subplot(111, projection='3d')
# # show_function 函数在网格上绘制插值函数
# mesh.show_function(axes, uI)
# plt.show()

from fealpy.backend import backend_manager as bm
from fealpy.mesh import UniformMesh

bm.set_backend('pytorch')
domain = [0, 1, 0, 1]
nx = 3
ny = 3
extent = [0, nx, 0, ny]
mesh = UniformMesh(extent, (1/nx, 1/ny), (0, 0))

