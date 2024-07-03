import torch
from torch import Tensor

CONTEXT = 'torch'

from fealpy.mesh import TriangleMesh as TMD
from fealpy.utils import timer

from fealpy.torch.mesh import TriangleMesh
from fealpy.torch.functionspace import LagrangeFESpace, TensorFunctionSpace, utils
from fealpy.torch.fem import (
    BilinearForm, LinearForm,
    LinearElasticityIntegrator
)
from fealpy.torch.solver import sparse_cg

from torch import cos, pi, tensordot

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

tmr = timer()

NX = 4
NY = 4
mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=NX, ny=NY, device=device)
NC = mesh.number_of_cells()
print("NC:", NC)
NN = mesh.number_of_nodes()
print("NN:", NN)

space = LagrangeFESpace(mesh, p=1, ctype='C')
mesh = space.mesh
print("mesh:", mesh)

ldof = space.number_of_local_dofs()
tensor_space = TensorFunctionSpace(space, shape=(2, ), dof_last=True)
mesh_tensor = tensor_space.mesh
print("mesh_tensor:", mesh_tensor)




integrator = LinearElasticityIntegrator(e=1.0, nu=0.3)
# bcs, ws, gphi, cm, index = integrator.fetch(space=space) # bcs-(NQ, ldof), gphi-(NC, NQ, ldof, GD)
# print("gphi:", gphi.shape)
# GD = space.GD
# print("GD:", GD)
# cm = mesh.entity_measure('cell', index=slice(None))
# qf = mesh.integrator(3, 'cell')
# bcs, ws = qf.get_quadrature_points_and_weights()
# gphi = space.grad_basis(bcs, index=slice(None), variable='x')
# print("bcs:", bcs.shape)

# strain = tensor_space.strain(p=bcs, variable='x') 
# print("strian:", strain.shape)

test = integrator.assembly(space=tensor_space)
#test1 = integrator.assembly(space=space)




uh = space.function(dim=2)