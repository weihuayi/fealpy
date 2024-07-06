import pytest
import torch
from fealpy.torch.mesh import TetrahedronMesh
from fealpy.torch.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.torch.fem import LinearElasticityIntegrator
from fealpy.fem import LinearElasticityOperatorIntegrator, BilinearForm
from fealpy.functionspace import LagrangeFESpace as LFS
from fealpy.mesh import TetrahedronMesh as TMD

from fealpy.torch.fem.integrator import (
    CellOperatorIntegrator,
    enable_cache,
    assemblymethod,
    _S, Index, CoefLike
)

@pytest.fixture
def device():
    """
    Returns the device to be used for computations.

    If a CUDA-enabled GPU is available, the function returns 'cuda',
    otherwise it returns 'cpu'.

    Returns:
        torch.device: The device to be used for computations.
    """
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def test_assembly(device):
    """
    Test if the 3D shapes of KK_torch and KK_expected are consistent.

    Args:
        device (str): The device to run the test on.

    Returns:
        None
    """
    NX = 4
    NY = 4
    NZ = 4  

    # Rest of the code...
def test_assembly(device):
    NX = 4
    NY = 4
    NZ = 4  

    # Create a triangle mesh
    mesh_torch = TetrahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=NX, ny=NY, nz=NZ, device=device)

    # Create a Lagrange finite element space
    space_torch = LagrangeFESpace(mesh_torch, p=1, ctype='C')

    # Get quadrature points and weights
    qf = mesh_torch.integrator(3, 'cell')
    bcs, ws = qf.get_quadrature_points_and_weights()

    # Compute the gradient of basis functions
    gphi = space_torch.grad_basis(bcs, index=_S, variable='u')
    print("gphi:", gphi.shape)

    # Create a tensor function space
    tensor_space = TensorFunctionSpace(space_torch, shape=(2, -1))

    # Create a linear elasticity integrator
    integrator_torch = LinearElasticityIntegrator(E=1.0, nu=0.3, device=device, method='fast_3d')

    # Assemble the strain constant matrix
    KK_torch = integrator_torch.fast_assembly_constant(space=tensor_space)
    print("KK_torch:", KK_torch)

    # Create a triangle mesh
    mesh = TMD.from_box(box=[0, 1, 0, 1, 0, 1], nx=NX, ny=NY, nz=NZ)

    # Create a Lagrange finite element space
    p = 1
    space = LFS(mesh, p=p, ctype='C', doforder='sdofs')

    GD = 2

    # Material parameters
    E0 = 1.0  # Elastic modulus
    nu = 0.3  # Poisson's ratio
    lambda_ = (E0 * nu) / ((1 + nu) * (1 - 2 * nu))
    mu = E0 / (2 * (1 + nu))

    # Create a linear elasticity operator integrator
    integrator = LinearElasticityOperatorIntegrator(lam=lambda_, mu=mu, q=3)

    # Create a vector-valued finite element space
    vspace = GD * (space,)

    # Create a bilinear form
    bform = BilinearForm(vspace)
    bform.add_domain_integrator(integrator)

    # Assemble the cell matrix
    KK_expected = integrator.assembly_cell_matrix(space=vspace)

    # Check if the computed matrix is close to the expected matrix
    assert KK_torch.shape==KK_expected.shape

    