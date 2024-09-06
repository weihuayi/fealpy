from fealpy.experimental.backend import backend_manager as bm
# bm.set_backend('numpy')
bm.set_backend('pytorch')
# bm.set_backend('jax')

from fealpy.experimental.mesh import UniformMesh2d
from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.experimental.typing import TensorLike

# Half-MBB 
def source(points: TensorLike) -> TensorLike:
    
    val = bm.zeros(points.shape, dtype=points.dtype)
    val[ny, 1] = -1
    
    return val

def dirichlet(points: TensorLike) -> TensorLike:

    return bm.ones(points.shape, dtype=points.dtype)

def is_dirichlet_boundary(points: TensorLike) -> TensorLike:
    """
    Determine which boundary edges satisfy the given property.

    Args:
        points (TensorLike): The coordinates of the points defining the edges.

    Returns:
        TensorLike: A boolean array indicating which boundary edges satisfy the property. 
        The length of the array is NBE, which represents the number of boundary edges.
    """
    temp1 = bm.abs(points[:, 0]) < 1e-12
    temp2 = (bm.arange(len(points)) % 2 == 0)
    temp = temp1 & temp2

    return temp

def is_dirichlet_boundary_edge(points: TensorLike) -> TensorLike:
    """
    Determine which boundary edges satisfy the given property.

    Args:
        points (TensorLike): The coordinates of the points defining the edges.

    Returns:
        TensorLike: A boolean array indicating which boundary edges satisfy the property. 
        The length of the array is NBE, which represents the number of boundary edges.
    """
    temp = bm.abs(points[:, 0]) < 1e-12

    return temp

def is_dirichlet_direction_0() -> TensorLike:
    temp = bm.tensor([True, False])

    return temp

def is_dirichlet_direction_1() -> TensorLike:
    temp = bm.tensor([1, 0])

    return temp


# Default input parameters
nx = 2
ny = 2

extent = [0, nx, 0, ny]
h = [1, 1]
origin = [0, 0]
mesh = UniformMesh2d(extent, h, origin)
NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()

space = LagrangeFESpace(mesh, p=2, ctype='C')
ldof = space.number_of_local_dofs()
gdof = space.number_of_global_dofs()

tensor_space = TensorFunctionSpace(space, shape=(-1, 2))
tldof = tensor_space.number_of_local_dofs()
tgdof = tensor_space.number_of_global_dofs()
uh = tensor_space.function()

space_d = LagrangeFESpace(mesh, p=2, ctype='D')
ldof_d = space_d.number_of_local_dofs()
gdof_d = space_d.number_of_global_dofs()
rho = space_d.function()
print("uh:", uh)