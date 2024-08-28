from fealpy.experimental.backend import backend_manager as bm

bm.set_backend('numpy')
# bm.set_backend('pytorch')
# bm.set_backend('jax')

from fealpy.experimental.mesh import TriangleMesh

from fealpy.experimental.fem import LinearElasticityIntegrator, \
                                    BilinearForm, LinearForm, \
                                    VectorSourceIntegrator

from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.experimental.typing import TensorLike

from fealpy.experimental.solver import cg
from fealpy.experimental.fem import DirichletBC as DBC
from fealpy.experimental.sparse import COOTensor

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from fealpy.utils import timer


# 平面应变问题定义
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

nx = 2
ny = 2
mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=nx, ny=ny)
maxit = 5
errorMatrix = bm.zeros((3, maxit), dtype=bm.float64)
for i in range(maxit):
    p = 2
    space = LagrangeFESpace(mesh, p=p, ctype='C')
    tensor_space = TensorFunctionSpace(space, shape=(2, -1))

    # (tgdof, )
    uh_cg = tensor_space.function()
    uh_spsolve = tensor_space.function()

    # 与单元有关的组装方法
    integrator_bi = LinearElasticityIntegrator(E=1.0, nu=0.3, 
                                            elasticity_type='strain', q=5)
    
    # 与单元有关的组装方法
    KK = integrator_bi.assembly(space=tensor_space)

    # 与单元有关的组装方法 
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

    F = dbc.check_vector(F)

    uh_cg = tensor_space.boundary_interpolate(gD=dirichlet, uh=uh_cg)
    uh_spsolve = tensor_space.boundary_interpolate(gD=dirichlet, uh=uh_spsolve)
    F = F - K.matmul(uh_cg[:])
    F[isDDof] = uh_cg[isDDof]

    K = dbc.check_matrix(K)
    kwargs = K.values_context()
    indices = K.indices()
    new_values = bm.copy(K.values())
    IDX = isDDof[indices[0, :]] | isDDof[indices[1, :]]
    new_values[IDX] = 0
    K = COOTensor(indices, new_values, K.sparse_shape)
    index, = bm.nonzero(isDDof)
    one_values = bm.ones(len(index), **kwargs)
    one_indices = bm.stack([index, index], axis=0)
    K1 = COOTensor(one_indices, one_values, K.sparse_shape)
    K = K.add(K1).coalesce()

    tmr = timer()
    next(tmr)
    uh_cg[:] = cg(K, F, maxiter=5000, atol=1e-14, rtol=1e-14)
    tmr.send(f'第 {i} 次的 cg 时间')

    temp1 = csr_matrix((K.values(), K.indices()), shape=K.shape)
    temp2 = bm.to_numpy(F)
    tmr.send(f'第 {i} 次转换类型时间')
    
    uh_spsolve[:] = spsolve(temp1, temp2)
    tmr.send(f'第 {i} 次的 spsolve 时间')
    next(tmr)

    u_exact = tensor_space.interpolate(solution)
    errorMatrix[0, i] = bm.max(bm.abs(uh_cg[:] - u_exact))
    errorMatrix[1, i] = bm.max(bm.abs(uh_cg[:] - u_exact)[isDDof])
    errorMatrix[2, i] = bm.max(bm.abs(uh_cg[:] - uh_spsolve[:]))
    
    if i < maxit-1:
        mesh.uniform_refine()

print("errorMatrix:\n", errorMatrix)
print("order:\n ", bm.log2(errorMatrix[0,:-1]/errorMatrix[0,1:]))