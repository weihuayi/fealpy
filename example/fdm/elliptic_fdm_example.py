from fealpy.backend import backend_manager as bm

from fealpy.model import PDEDataManager
from fealpy.mesh import UniformMesh
from fealpy.fdm import (
        EllipticOperator, 
        DiffusionOperator, 
        ConvectionOperator,
        ReactionOperator)

def diffusion_coef(p):
    return bm.array([[2, 0], [0, 2]])

def convection_coef(p):
    return bm.array([1, 1])

def reaction_coef(p):
    return 2.0

pde = PDEDataManager('elliptic').get_example('coscos')

domain = pde.domain()

extent = [0, 2, 0, 2]
mesh = UniformMesh(domain, extent)

#op = EllipticOperator(mesh, diffusion_coef, convection_coef, reaction_coef)

A = DiffusionOperator(mesh, diffusion_coef).assembly()
B = ConvectionOperator(mesh, convection_coef).assembly() 
C = ReactionOperator(mesh, reaction_coef).assembly()
print(C.to_dense())

