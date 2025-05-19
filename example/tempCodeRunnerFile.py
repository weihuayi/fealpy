from fealpy.mesh import UniformMesh
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