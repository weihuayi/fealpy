from fealpy.experimental.mesh import TriangleMesh
from fealpy.mesh import TriangleMesh as TriangleMesh_old

from fealpy.experimental.fem import LinearElasticityIntegrator, \
                                    BilinearForm, LinearForm, \
                                    VectorSourceIntegrator
from fealpy.fem import LinearElasticityOperatorIntegrator
from fealpy.fem import BilinearForm as BilinearForm_old
from fealpy.fem import LinearForm as LinearForm_old
from fealpy.fem import VectorSourceIntegrator as VectorSourceIntegrator_old

from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.functionspace import LagrangeFESpace as LagrangeFESpace_old

from fealpy.experimental.typing import TensorLike
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.sparse.linalg import sparse_cg

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
# mesh.uniform_refine(n=2)

import matplotlib.pyplot as plt
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_edge(axes, showindex=True)
mesh.find_cell(axes, showindex=True)
plt.show()

NN = mesh.number_of_nodes()
print("NN:", NN)
NC = mesh.number_of_cells()
print("NC:", NC)
node = mesh.entity('node')
edge = mesh.entity('edge')
cell = mesh.entity('cell')
print("cell:", cell.shape, "\n", cell)

qf = mesh.quadrature_formula(3, 'cell')
# bcs-(NQ, BC), ws-(NQ, )
bcs, ws = qf.get_quadrature_points_and_weights()

space = LagrangeFESpace(mesh, p=1, ctype='C')
space_old = LagrangeFESpace_old(mesh_old, p=1, ctype='C', doforder='sdofs')
# space_old = LagrangeFESpace_old(mesh_old, p=1, ctype='C', doforder='vdims')
ldof = space.number_of_local_dofs()
print("ldof:", ldof)
gdof = space.number_of_global_dofs()
print("gdof:", gdof)
cell2dof = space.cell_to_dof()
print("cell2dof:", cell2dof.shape, "\n", cell2dof)
cell2dof_old = space_old.cell_to_dof()
print("cell2dof_old:", cell2dof_old.shape, "\n", cell2dof_old)
phi = space.basis(bc=bcs)
print("phi:", phi.shape, "\n", phi)

tensor_space = TensorFunctionSpace(space, shape=(2, -1))
# tensor_space = TensorFunctionSpace(space, shape=(-1, 2))
tensor_space_old = 2*(space_old, )
tldof = tensor_space.number_of_local_dofs()
print("tldof:", tldof)
tgdof = tensor_space.number_of_global_dofs()
print("tgdof:", tgdof)
cell2tldof = tensor_space.cell_to_dof()
print("cell2tldof:", cell2tldof.shape, "\n", cell2tldof)
tensor_phi = tensor_space.basis(p=bcs)
print("tensor_phi:", tensor_phi.shape, "\n", tensor_phi)

# integrator_bi = LinearElasticityIntegrator(E=1.0, nu=0.3, 
#                                            method='fast_strain', q=5)
# integrator_bi = LinearElasticityIntegrator(E=1.0, nu=0.3, 
#                                            elasticity_type='stress', q=5)
integrator_bi = LinearElasticityIntegrator(E=1.0, nu=0.3, 
                                           elasticity_type='strain', q=5)
E = 1.0
nu = 0.3
lambda_ = nu * E / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))
integrator_bi_old = LinearElasticityOperatorIntegrator(lam=lambda_, mu=mu, q=5)

# KK_tri_strain - (NC, TLDOF, TLDOF)
# KK_strain = integrator_bi.fast_assembly_strain_constant(space=tensor_space)
KK_strain = integrator_bi.assembly(space=tensor_space)
print("KK_strain:", KK_strain.shape, "\n", KK_strain[0])
KK_strain_old = integrator_bi_old.assembly_cell_matrix(space=tensor_space_old)
print("KK_strain_old:", KK_strain_old.shape, "\n", KK_strain_old[0])
 
bform = BilinearForm(tensor_space)
bform.add_integrator(integrator_bi)
K = bform.assembly()
print("K:", K.shape, "\n", K.to_dense().round(4))

bform_old = BilinearForm_old(tensor_space_old)
bform_old.add_domain_integrator(integrator_bi_old)
K_old = bform_old.assembly()
print("K_old:", K_old.shape, "\n", K_old.toarray().round(4))

integrator_li = VectorSourceIntegrator(source=source, q=5)
# FF_tri - (NC, TLDOF)
FF = integrator_li.assembly(space=tensor_space)
print("FF:", FF.shape, "\n", FF[0])

integrator_li_old = VectorSourceIntegrator_old(f=source, q=5)
FF_old = integrator_li_old.assembly_cell_vector(space=tensor_space_old)
print("FF_old:", FF_old.shape, "\n", FF_old)

lform = LinearForm(tensor_space)
lform.add_integrator(integrator_li)
# F_tri - (TGDOF)
F = lform.assembly()
print("F:", F.shape, "\n", F)

lform_old = LinearForm_old(tensor_space_old)
lform_old.add_domain_integrator(integrator_li_old)
F_old = lform_old.assembly()
print("F_old:", F_old.shape, "\n", F_old)
asd

uh = tensor_space.function()
print("uh:", uh.shape, "\n", uh)

from fealpy.experimental.fem import DirichletBC as DBC
from fealpy.experimental.sparse import COOTensor
dbc = DBC(space=tensor_space, gd=dirichlet, left=False)
isDDof = tensor_space.is_boundary_dof(threshold=None)
print("isDDof:", isDDof.shape, "\n", isDDof)

K = dbc.check_matrix(K)
#print("K1:", K.to_dense().round(4))
kwargs = K.values_context()
# 获取矩阵 K 的非 0 元素的索引
indices = K.indices()
# 获取矩阵 K 的非 0 元素的值
new_values = bm.copy(K.values())
# 将 Dirichlet 边界条件对应的元素置 0:
IDX = isDDof[indices[0, :]] | isDDof[indices[1, :]] # 根据索引获取 K 的非 0 元素的行列索引
new_values[IDX] = 0 # 将所有需要被置零的矩阵元素的值设为 0
# 用更新后的值重新构建一个新的稀疏矩阵
K = COOTensor(indices, new_values, K.sparse_shape)
# 将 Dirichlet 自由度的对角线上的元素置为 1:
index, = bm.nonzero(isDDof, as_tuple=True) # 找到所有属于 Dirichlet 边界条件的自由度索引, 些索引表示需要在对角线处设置为 1 的位置
one_values = bm.ones(len(index), **kwargs) # 创建一个与 Dirichlet 自由度数量相同的全为 1 的数组
one_indices = bm.stack([index, index], axis=0) # 创建对应的对角线索引，确保在这些位置设置 1
K1 = COOTensor(one_indices, one_values, K.sparse_shape) # 构造一个新的稀疏矩阵 A1, 其中仅在 Dirichlet 边界条件位置的对角线上设置为 1, 其余元素为 0
# 将调整后的矩阵 K 和 K1 相加
K = K.add(K1).coalesce()
print("K_bc:", K.shape, "\n", K.to_dense().round(4))


F = dbc.check_vector(F)
tensor_space.boundary_interpolate(gD=dirichlet, uh=uh)
print("uh_bi:", uh.shape, "\n", uh)
F = F - K.matmul(uh[:])
F[isDDof] = uh[isDDof]
print("F_bc:", F.shape, "\n", F)

ipoints = tensor_space.interpolation_points()
print("ipoints:", ipoints.shape, "\n", ipoints)
u_exact = solution(ipoints)
print("u_exact:", u_exact.shape, "\n", u_exact)

from scipy.linalg import solve 
uh[:] = sparse_cg(K, F, maxiter=5000, atol=1e-14, rtol=1e-14)
# uh[:] = solve(bm.to_numpy(K.to_dense()), F)
print("uh:", uh.shape, "\n", uh)
uhdense = uh[:]

errorMatrix = bm.max(bm.abs(bm.array(uh) - u_exact.reshape(-1)))
print("errorMatrix:", errorMatrix)
errorMatrix = bm.max(bm.abs(bm.array(uh) - u_exact.reshape(-1))[isDDof])
print("errorMatrix:", errorMatrix)

import matplotlib.pyplot as plt
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
mesh.find_node(axes, node=ipoints, showindex=True)
# mesh.find_edge(axes, showindex=True)
# mesh.find_cell(axes, showindex=True)
plt.show()






# uh_interpolation = tensor_space.interpolation_points()
# uh[dflag] = dirichlet(uh_interpolation[dflag])

# def fixeddofs(points: TensorLike) -> TensorLike:
#     x = points[..., 0]
#     return x == 0

# isDDof = space.is_boundary_dof(threshold=fixeddofs)
# isDDof2 = tensor_space.is_boundary_dof()
# print("is_DDof:", isDDof.shape, "\n", isDDof)
# print("isDDof2:", isDDof2.shape, "\n", isDDof2)

# import numpy as np
# from scipy.sparse import spdiags
# bdIdx = np.zeros(K.shape[0], dtype=np.int32)
# bdIdx[isDDof] = 1
# D0 = spdiags(1-bdIdx, 0, K.shape[0], K.shape[0])
# D1 = spdiags(bdIdx, 0, K.shape[0], K.shape[0])
# K = D0@K@D0 + D1



# uh_interpolation = tensor_space.interpolation_points()
# print("uh_interpolation:", uh_interpolation.shape, "\n", uh_interpolation)
# uh_tri[isDDof] = solution(uh_tri_interpolate[isDDof])
# print("uh_tri:", uh_tri.shape, "\n", uh_tri)


# ipoints = mesh.interpolation_points(p=2)
