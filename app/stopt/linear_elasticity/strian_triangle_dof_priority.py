from fealpy.experimental.mesh import TriangleMesh
from fealpy.mesh import TriangleMesh as TriangleMesh_old

from fealpy.experimental.fem import LinearElasticityIntegrator, \
                                    BilinearForm, LinearForm, \
                                    VectorSourceIntegrator

from fealpy.fem import LinearElasticityOperatorIntegrator as LinearElasticityIntegrator_old
from fealpy.fem import VectorSourceIntegrator as VectorSourceIntegrator_old
from fealpy.fem import BilinearForm as BilinearForm_old
from fealpy.fem import LinearForm as LinearForm_old
from fealpy.fem import DirichletBC as DirichletBC_old

from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.functionspace import LagrangeFESpace as LagrangeFESpace_old

from fealpy.experimental.typing import TensorLike
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.solver import cg
from fealpy.experimental.fem import DirichletBC as DBC
from fealpy.experimental.sparse import COOTensor


# bm.set_backend('numpy')
# bm.set_backend('pytorch')
# bm.set_backend('jax')

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
    uh_dependent = tensor_space.function()
    uh_independent = tensor_space.function()

    # 与单元有关的组装方法
    integrator_bi_dependent = LinearElasticityIntegrator(E=1.0, nu=0.3, 
                                            elasticity_type='strain', q=5)
    # 与单元无关的组装方法
    integrator_bi_independent = LinearElasticityIntegrator(E=1.0, nu=0.3, 
                                           method='fast_strain', q=5)
    
    # 与单元有关的组装方法
    KK_dependent = integrator_bi_dependent.assembly(space=tensor_space)
    # print("KK_dependent:\n", KK_dependent[0])
    # 与单元无关的组装方法
    KK_independent = integrator_bi_independent.fast_assembly_strain_constant(space=tensor_space)
    # print("KK_independent:\n", KK_independent[0])

    # 与单元有关的组装方法 
    bform_dependent = BilinearForm(tensor_space)
    bform_dependent.add_integrator(integrator_bi_dependent)
    K_dependent = bform_dependent.assembly()
    # 与单元无关的组装方法
    bform_independent = BilinearForm(tensor_space)
    bform_independent.add_integrator(integrator_bi_independent)
    K_independent = bform_independent.assembly()

    integrator_li = VectorSourceIntegrator(source=source, q=5)

    FF = integrator_li.assembly(space=tensor_space)

    lform = LinearForm(tensor_space)
    lform.add_integrator(integrator_li)
    F = lform.assembly()
    
    dbc = DBC(space=tensor_space, gd=dirichlet, left=False)
    isDDof = tensor_space.is_boundary_dof(threshold=None)

    F = dbc.check_vector(F)

    uh_dependent = tensor_space.boundary_interpolate(gD=dirichlet, uh=uh_dependent)
    F_dependent = F - K_dependent.matmul(uh_dependent[:])
    F_dependent[isDDof] = uh_dependent[isDDof]

    uh_independent = tensor_space.boundary_interpolate(gD=dirichlet, uh=uh_independent)
    F_independent = F - K_independent.matmul(uh_independent[:])
    F_independent[isDDof] = uh_independent[isDDof]

    K_dependent = dbc.check_matrix(K_dependent)
    kwargs = K_dependent.values_context()
    indices = K_dependent.indices()
    new_values = bm.copy(K_dependent.values())
    IDX = isDDof[indices[0, :]] | isDDof[indices[1, :]]
    new_values[IDX] = 0
    K_dependent = COOTensor(indices, new_values, K_dependent.sparse_shape)
    index, = bm.nonzero(isDDof, as_tuple=True)
    one_values = bm.ones(len(index), **kwargs)
    one_indices = bm.stack([index, index], axis=0)
    K1_dependent = COOTensor(one_indices, one_values, K_dependent.sparse_shape)
    K_dependent = K_dependent.add(K1_dependent).coalesce()

    K_independent = dbc.check_matrix(K_independent)
    kwargs = K_independent.values_context()
    indices = K_independent.indices()
    new_values = bm.copy(K_independent.values())
    IDX = isDDof[indices[0, :]] | isDDof[indices[1, :]]
    new_values[IDX] = 0
    K_independent = COOTensor(indices, new_values, K_independent.sparse_shape)
    index, = bm.nonzero(isDDof, as_tuple=True)
    one_values = bm.ones(len(index), **kwargs)
    one_indices = bm.stack([index, index], axis=0)
    K1_independent = COOTensor(one_indices, one_values, K_independent.sparse_shape)
    K_independent = K_independent.add(K1_independent).coalesce()

    uh_independent = tensor_space.boundary_interpolate(gD=dirichlet, uh=uh_independent)
    F_independent = F - K_independent.matmul(uh_independent[:])
    F_independent[isDDof] = uh_independent[isDDof]

    uh_dependent[:] = cg(K_dependent, F_dependent, maxiter=5000, atol=1e-14, rtol=1e-14)
    uh_independent[:] = cg(K_independent, F_independent, maxiter=5000, atol=1e-14, rtol=1e-14)

    u_exact = tensor_space.interpolate(solution)
    errorMatrix[0, i] = bm.max(bm.abs(bm.array(uh_dependent) - u_exact))
    errorMatrix[1, i] = bm.max(bm.abs(bm.array(uh_dependent) - u_exact)[isDDof])
    errorMatrix[2, i] = bm.max(bm.abs(bm.array(uh_independent) - uh_dependent))
    
    if i < maxit-1:
        mesh.uniform_refine()

print("errorMatrix:\n", errorMatrix)
print("order:\n ", bm.log2(errorMatrix[0,:-1]/errorMatrix[0,1:]))