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
NC = mesh.number_of_cells()
p = 1
space = LagrangeFESpace(mesh, p=p, ctype='C')
tensor_space = TensorFunctionSpace(space, shape=(-1, 2))
# tensor_space = TensorFunctionSpace(space, shape=(2, -1))

F = tensor_space.interpolate(source)
# isDDof = tensor_space.is_boundary_dof(threshold=is_dirichlet_boundary_edge, direction=is_dirichlet_direction)
isDDof_0 = tensor_space.is_boundary_dof(threshold=(is_dirichlet_boundary_edge, is_dirichlet_direction_0))
isDDof_1 = tensor_space.is_boundary_dof(threshold=(is_dirichlet_boundary_edge, is_dirichlet_direction_1))

isDDof_test = tensor_space.is_boundary_dof(threshold=is_dirichlet_boundary)
uh = tensor_space.function()
uh = tensor_space.boundary_interpolate(gD=dirichlet, uh=uh, 
                                           threshold=is_dirichlet_boundary_edge)
uh[1::2] = 0
uh_test = tensor_space.function()
uh_test = tensor_space.boundary_interpolate(gD=dirichlet, uh=uh_test, 
                                           threshold=is_dirichlet_boundary)
print("uh:", uh)