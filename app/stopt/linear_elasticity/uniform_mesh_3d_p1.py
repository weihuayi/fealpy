from fealpy.experimental.mesh import UniformMesh3d

from fealpy.experimental.fem import LinearElasticityIntegrator, \
                                    BilinearForm, LinearForm, \
                                    VectorSourceIntegrator
from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace

from fealpy.experimental.typing import TensorLike
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.sparse.linalg import sparse_cg

bm.set_backend('numpy')

extent = [0, 2, 0, 2, 0, 2]
h = [1, 1]
origin = [0, 0]
mesh = UniformMesh3d(extent, h, origin)

import matplotlib.pyplot as plt
fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_edge(axes, showindex=True)
mesh.find_cell(axes, showindex=True)
plt.show()


NN = mesh.number_of_nodes()
print("NN:", NN)
NC = mesh.number_of_cells()
print("NC:", NC)
GD = mesh.geo_dimension()
print("GD:", GD)

qf = mesh.quadrature_formula(3, 'cell')
# bcs-(NQ, BC), ws-(NQ, )
bcs, ws = qf.get_quadrature_points_and_weights()

space = LagrangeFESpace(mesh, p=2, ctype='C')
ldof = space.number_of_local_dofs()
print("ldof:", ldof)
gdof = space.number_of_global_dofs()
print("gdof:", gdof)
cell2dof = space.cell_to_dof()
print("cell2dof:", cell2dof.shape, "\n", cell2dof)
phi = space.basis(bc=bcs)
print("phi:", phi.shape, "\n", phi)

tensor_space = TensorFunctionSpace(space, shape=(2, -1))
#tensor_space = TensorFunctionSpace(space, shape=(-1, 2))
tldof = tensor_space.number_of_local_dofs()
print("tldof:", tldof)
tgdof = tensor_space.number_of_global_dofs()
print("tgdof:", tgdof)
cell2tldof = tensor_space.cell_to_dof()
print("cell2tldof:", cell2tldof.shape, "\n", cell2tldof)
tensor_phi = tensor_space.basis(p=bcs)
print("tensor_phi:", tensor_phi.shape, "\n", tensor_phi)

#integrator = LinearElasticityIntegrator(E=1.0, nu=0.3, method='fast_strain')
integrator_bi = LinearElasticityIntegrator(E=1.0, nu=0.3, elasticity_type='stress')
# KK_tri_strain - (NC, TLDOF, TLDOF)
#KK = integrator.fast_assembly_strain_constant(space=tensor_space)
KK = integrator_bi.assembly(space=tensor_space)
print("KK:", KK.shape, "\n", KK[0])
 
bform = BilinearForm(tensor_space)
bform.add_integrator(integrator_bi)
K = bform.assembly()
print("K:", K.shape, "\n", K.to_dense().round(4))

integrator_li = VectorSourceIntegrator(source=source)
# FF_tri - (NC, TLDOF)
FF = integrator_li.assembly(space=tensor_space)
#print("FF:", FF.shape, "\n", FF)
lform = LinearForm(tensor_space)
lform.add_integrator(integrator_li)
# F_tri - (TGDOF)
F = lform.assembly()
print("F:", F.shape, "\n", F)

uh = tensor_space.function()
print("uh:", uh.shape, "\n", uh)

## 边界条件处理
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
ipoints = tensor_space.interpolation_points()
print("ipoints:", ipoints.shape, "\n", ipoints)
gD = dirichlet(ipoints.reshape(-1)[isDDof].reshape(-1, 2))
print("gD:", gD.shape, "\n", gD)
uh[isDDof] = gD.reshape(-1)
print("uh_bc:", uh.shape, "\n", uh)
F = F - K.matmul(uh[:])
print("F_bc:", F.shape, "\n", F)

u_exact = solution(ipoints)
print("u_exact:", u_exact.shape, "\n", u_exact)

uh[:] = sparse_cg(K, F, maxiter=5000, atol=1e-14, rtol=1e-14)
print("uh:", uh.shape, "\n", uh)

# ps = mesh.bc_to_point(bcs=bcs)
# print("ps:", ps.shape, "\n", ps)

errorMatrix = bm.max(bm.abs(uh - u_exact.reshape(-1)))
print("errorMatrix:", errorMatrix)


# ipoints = mesh.interpolation_points(p=2)
# fig = plt.figure()
# axes = fig.gca()
# mesh.add_plot(axes)
# mesh.find_node(axes, showindex=True)
# mesh.find_edge(axes, showindex=True)
# mesh.find_cell(axes, showindex=True)
# plt.show()