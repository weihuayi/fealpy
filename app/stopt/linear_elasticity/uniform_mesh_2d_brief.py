from fealpy.experimental.mesh import UniformMesh2d, QuadrangleMesh

from fealpy.experimental.fem import LinearElasticityIntegrator, \
                                    BilinearForm, LinearForm, \
                                    VectorSourceIntegrator
from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.experimental.typing import TensorLike
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.sparse.linalg import sparse_cg
from fealpy.experimental.fem import DirichletBC as DBC
from fealpy.experimental.sparse import COOTensor

bm.set_backend('numpy')

def source(points: TensorLike) -> TensorLike:
    x = points[..., 0]
    y = points[..., 1]
    
    val = bm.zeros(points.shape, dtype=points.dtype)
    val[..., 0] = 35/13 * y - 35/13 * y**2 + 10/13 * x - 10/13 * x**2
    val[..., 1] = -25/26 * (-1 + 2 * y) * (-1 + 2 * x)
    
    return val

def solution(points: TensorLike) -> TensorLike:
    x = points[..., 0]
    y = points[..., 1]
    
    val = bm.zeros(points.shape, dtype=points.dtype)
    val[..., 0] = x * (1 - x) * y * (1 - y)
    val[..., 1] = 0
    
    return val

def dirichlet(points: TensorLike) -> TensorLike:

    return solution(points)

extent = [0, 2, 0, 2]
h = [1, 1]
origin = [0, 0]
mesh = UniformMesh2d(extent, h, origin)

maxit = 4
errorMatrix = bm.zeros((2, maxit), dtype=bm.float64)
for i in range(maxit):

    space = LagrangeFESpace(mesh, p=1, ctype='C')
    tensor_space = TensorFunctionSpace(space, shape=(2, -1))

    uh = tensor_space.function()

    integrator_bi = LinearElasticityIntegrator(E=1.0, nu=0.3, 
                                            elasticity_type='stress', q=5)
    KK = integrator_bi.assembly(space=tensor_space)
    bform = BilinearForm(tensor_space)
    bform.add_integrator(integrator_bi)
    K = bform.assembly()

    integrator_li = VectorSourceIntegrator(source=source, q=5)
    FF = integrator_li.assembly(space=tensor_space)
    lform = LinearForm(tensor_space)
    lform.add_integrator(integrator_li)
    F = lform.assembly()

    dbc = DBC(space=tensor_space, gd=dirichlet, left=False)
    isDDof = tensor_space.is_boundary_dof(threshold=None)
    K = dbc.check_matrix(K)
    kwargs = K.values_context()
    indices = K.indices()
    new_values = bm.copy(K.values())
    IDX = isDDof[indices[0, :]] | isDDof[indices[1, :]]
    new_values[IDX] = 0
    K = COOTensor(indices, new_values, K.sparse_shape)
    index, = bm.nonzero(isDDof, as_tuple=True)
    one_values = bm.ones(len(index), **kwargs)
    one_indices = bm.stack([index, index], axis=0)
    K1 = COOTensor(one_indices, one_values, K.sparse_shape)
    K = K.add(K1).coalesce()

    F = dbc.check_vector(F)
    tensor_space.boundary_interpolate(gD=dirichlet, uh=uh)
    F = F - K.matmul(uh[:])
    F[isDDof] = uh[isDDof]

    uh[:] = sparse_cg(K, F, maxiter=5000, atol=1e-14, rtol=1e-14)

    ipoints = tensor_space.interpolation_points()
    u_exact = solution(ipoints)
    errorMatrix[0, i] = bm.max(bm.abs(bm.array(uh) - u_exact.reshape(-1)))
    errorMatrix[1, i] = bm.max(bm.abs(bm.array(uh) - u_exact.reshape(-1))[isDDof])

    if i < maxit-1:
        mesh.uniform_refine(n=1)

print("errorMatrix:", errorMatrix)
