import pytest
import torch
from fealpy.torch.mesh import TriangleMesh
from fealpy.torch.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.torch.fem import LinearElasticityIntegrator, BilinearForm


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

NX = 1
NY = 1
mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=NX, ny=NY, device=device)
space = LagrangeFESpace(mesh, p=1, ctype='C')
tensor_space = TensorFunctionSpace(space, shape=(2, -1))
# tensor_space = TensorFunctionSpace(space, shape=(-1, 2))

integrator = LinearElasticityIntegrator(E=1.0, nu=0.3, \
                                        device=device, method='fast_strain')
KK_torch = integrator.assembly(space=tensor_space)
bform = BilinearForm(tensor_space)
K = bform.assembly()

test = integrator.fast_assembly_strain_constant(space=tensor_space)
print("test:", test.shape, "\n", test)


integrator1 = LinearElasticityIntegrator(E=1.0, nu=0.3, 
                                         device=device, method='fast_stress')
test1 = integrator1.fast_assembly_stress_constant(space=tensor_space)
print("test1:", test1.shape, "\n", test1)

