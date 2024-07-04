import pytest
import torch
from fealpy.torch.mesh import QuadrangleMesh
from fealpy.torch.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.torch.fem import LinearElasticityIntegrator
from fealpy.fem import LinearElasticityOperatorIntegrator, BilinearForm
from fealpy.functionspace import LagrangeFESpace as LFS
from fealpy.mesh import TriangleMesh as TMD

@pytest.fixture
def device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def test_assembly(device):
    NX = 4
    NY = 4
    mesh_torch = QuadrangleMesh.from_box(box=[0, 1, 0, 1], nx=NX, ny=NY, device=device)
    space_torch = LagrangeFESpace(mesh_torch, p=1, ctype='C')
    tensor_space = TensorFunctionSpace(space_torch, shape=(2, ), dof_last=True)

    integrator_torch = LinearElasticityIntegrator(e=1.0, nu=0.3, elasticity_type='strain', device=device)
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
