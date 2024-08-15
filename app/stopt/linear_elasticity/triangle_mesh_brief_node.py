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
from fealpy.experimental.sparse.linalg import sparse_cg
from fealpy.experimental.fem import DirichletBC as DBC
from fealpy.experimental.sparse import COOTensor

import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve


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

nx = 1
ny = 1
mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=nx, ny=ny)
mesh_old = TriangleMesh_old.from_box(box = [0, 1, 0, 1], nx=nx, ny=ny)

maxit = 6
errorMatrix = bm.zeros((2, maxit), dtype=bm.float64)
errorMatrix_old = np.zeros((2, maxit), dtype=np.float64)
errorMatrix_new_old = np.zeros((5, maxit), dtype=np.float64)
for i in range(maxit):

    p = 1
    space = LagrangeFESpace(mesh, p=p, ctype='C')
    space_old = LagrangeFESpace_old(mesh_old, p=p, ctype='C', doforder='vdims')
    # space_old = LagrangeFESpace_old(mesh_old, p=p, ctype='C', doforder='sdofs')
    # tensor_space = TensorFunctionSpace(space, shape=(2, -1))
    tensor_space = TensorFunctionSpace(space, shape=(-1, 2))
    tensor_space_old = 2*(space_old, )

    # (tgdof, )
    uh = tensor_space.function()
    # (ldof, GD)
    uh_old = space_old.function(dim=2)

    integrator_bi = LinearElasticityIntegrator(E=1.0, nu=0.3, 
                                            elasticity_type='strain', q=5)
    E = 1.0
    nu = 0.3
    lambda_ = nu * E / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    integrator_bi_old = LinearElasticityIntegrator_old(lam=lambda_, mu=mu, q=5)
    
    KK = integrator_bi.assembly(space=tensor_space)
    KK_strain_old = integrator_bi_old.assembly_cell_matrix(space=tensor_space_old)

    bform = BilinearForm(tensor_space)
    bform.add_integrator(integrator_bi)
    K = bform.assembly()

    bform_old = BilinearForm_old(tensor_space_old)
    bform_old.add_domain_integrator(integrator_bi_old)
    K_old = bform_old.assembly()
    errorMatrix_new_old[0, i] = np.max(np.abs(bm.to_numpy(K.to_dense()) - K_old.toarray()))

    integrator_li = VectorSourceIntegrator(source=source, q=5)

    integrator_li_old = VectorSourceIntegrator_old(f=source, q=5)

    FF = integrator_li.assembly(space=tensor_space)
    FF_old = integrator_li_old.assembly_cell_vector(space=tensor_space_old)

    lform = LinearForm(tensor_space)
    lform.add_integrator(integrator_li)
    F = lform.assembly()
    lform_old = LinearForm_old(tensor_space_old)
    lform_old.add_domain_integrator(integrator_li_old)
    F_old = lform_old.assembly()
    errorMatrix_new_old[1, i] = np.max(np.abs(bm.to_numpy(F) - F_old))
    
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
    dflag_old = space_old.boundary_interpolate(gD=dirichlet, uh=uh_old)
    # bdIdx = np.zeros(K.shape[0], dtype=np.int32)
    # bdIdx[dflag_old.flat] = 1
    # D0 = spdiags(1-bdIdx, 0, K_old.shape[0], K_old.shape[0])
    # D1 = spdiags(bdIdx, 0, K_old.shape[0], K_old.shape[0])
    # K_old = D0@K_old@D0 + D1

    F = dbc.check_vector(F)
    uh = tensor_space.boundary_interpolate(gD=dirichlet, uh=uh)
    F = F - K.matmul(uh[:])
    F[isDDof] = uh[isDDof]

    # ipoints_old = space_old.interpolation_points()
    # uh[dflag] = dirichlet(ipoints[dflag])
    # F_old -= K_old@uh_old.flat
    # F[dflag_old.flat] = uh_old.ravel()[dflag_old.flat]
    K_old, F_old = DirichletBC_old(space=space_old, gD=dirichlet).apply(K_old, F_old)

    errorMatrix_new_old[2, i] = np.max(np.abs(bm.to_numpy(K.to_dense()) - K_old.toarray()))
    errorMatrix_new_old[3, i] = np.max(np.abs(bm.to_numpy(F) - F_old))

    uh[:] = sparse_cg(K, F, maxiter=5000, atol=1e-14, rtol=1e-14)
    uh_old.flat[:] = spsolve(K_old, F_old)
    # print("uh_diff:", np.max(np.abs(bm.to_numpy(uh[:]) - uh_old.ravel())))

    ipoints = tensor_space.interpolation_points()
    u_exact = solution(ipoints)
    errorMatrix[0, i] = bm.max(bm.abs(bm.array(uh) - u_exact.reshape(-1)))
    errorMatrix[1, i] = bm.max(bm.abs(bm.array(uh) - u_exact.reshape(-1))[isDDof])
    errorMatrix_old[0, i] = np.max(np.abs(uh_old.ravel() - u_exact.reshape(-1)))
    errorMatrix_old[1, i] = np.max(np.abs(uh_old.ravel() - u_exact.reshape(-1))[dflag_old.flat])
    errorMatrix_new_old[4, i] = np.max(np.abs(bm.to_numpy(uh[:]) - uh_old.ravel()))
    
    

    if i < maxit-1:
        mesh.uniform_refine(n=1)
        mesh_old.uniform_refine(n=1)

print("errorMatrix:\n", errorMatrix)
print("order:\n ", bm.log2(errorMatrix[0,:-1]/errorMatrix[0,1:]))
print("errorMatrix_old:\n", errorMatrix_old)
print("order_old:\n", np.log2(errorMatrix_old[0,:-1]/errorMatrix_old[0,1:]))
print("errorMatrix_new_old:\n", errorMatrix_new_old)