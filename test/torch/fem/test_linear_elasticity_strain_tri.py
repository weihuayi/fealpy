import pytest
import torch
from fealpy.torch.mesh import TriangleMesh
from fealpy.torch.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.torch.fem import LinearElasticityIntegrator
from fealpy.fem import LinearElasticityOperatorIntegrator, BilinearForm
from fealpy.functionspace import LagrangeFESpace as LFS
from fealpy.mesh import TriangleMesh as TMD

from fealpy.torch.fem.integrator import (
    CellOperatorIntegrator,
    enable_cache,
    assemblymethod,
    _S, Index, CoefLike
)

@pytest.fixture
def device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def test_assembly(device):
    NX = 4
    NY = 4
    mesh_torch = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=NX, ny=NY, device=device)
    space_torch = LagrangeFESpace(mesh_torch, p=1, ctype='C')
    qf = mesh_torch.integrator(3, 'cell')
    bcs, ws = qf.get_quadrature_points_and_weights()
    gphi = space_torch.grad_basis(bcs, index=_S, variable='u')
    print("gphi:", gphi.shape)
    tensor_space = TensorFunctionSpace(space_torch, shape=(2, -1))

    integrator_torch = LinearElasticityIntegrator(E=1.0, nu=0.3,\
                elasticity_type='strain', device=device, method='fast_strain')
    KK_torch = integrator_torch.assembly(space=tensor_space)

    mesh = TMD.from_box(box=[0, 1, 0, 1], nx=NX, ny=NY)
    p = 1
    space = LFS(mesh, p=p, ctype='C', doforder='sdofs')

    GD = 2

    # Material parameters
    E0 = 1.0  # Elastic modulus
    nu = 0.3  # Poisson's ratio
    lambda_ = (E0 * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E0 / (2 * (1 + nu))
    integrator = LinearElasticityOperatorIntegrator(lam=lambda_, mu=mu, q=3)
    vspace = GD * (space,)
    bform = BilinearForm(vspace)
    bform.add_domain_integrator(integrator)
    KK_expected = integrator.assembly_cell_matrix(space=vspace)

    assert torch.allclose(KK_torch, torch.tensor(KK_expected), atol=1e-9)
