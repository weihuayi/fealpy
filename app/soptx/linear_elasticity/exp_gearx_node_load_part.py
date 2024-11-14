from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.sparse import COOTensor
from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem.dirichlet_bc import DirichletBC
from fealpy.typing import TensorLike
from fealpy.solver import cg, spsolve

import pickle
from app.gearx.gear import ExternalGear, InternalGear
from app.gearx.utils import *

from fealpy.mesh import HexahedronMesh


with open('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/external_gear_data_part.pkl', 'rb') as f:
    data = pickle.load(f)

hex_mesh = data['hex_mesh']
helix_node = data['helix_node']
# inner_node_idx = data['inner_node_idx']


target_cell_idx = data['target_cell_idx']
parameters = data['parameters']
is_inner_node = data['is_inner_node']

hex_cell = hex_mesh.cell
hex_node = hex_mesh.node

mesh = HexahedronMesh(hex_node, hex_cell)

GD = mesh.geo_dimension()   
NC = mesh.number_of_cells()
NN = mesh.number_of_nodes()
node = mesh.entity('node')
cell = mesh.entity('cell')

load_values = bm.array([50.0, 60.0, 79.0, 78.0, 87.0, 95.0, 102.0, 109.0, 114.0,
                        119.0, 123.0, 127.0, 129.0, 130.0, 131.0], dtype=bm.float64)

u = parameters[..., 0]
v = parameters[..., 1]
w = parameters[..., 2]
bcs_list = [
    (
        bm.tensor([[1 - u, u]]),
        bm.tensor([[1 - v, v]]),
        bm.tensor([[1 - w, w]])
    )
    for u, v, w in parameters
]
for idx, (u_tensor, v_tensor, w_tensor) in enumerate(bcs_list):
    u_values = u_tensor.flatten()
    v_values = v_tensor.flatten()
    w_values = w_tensor.flatten()
    print(f"载荷点 {idx + 1} 的重心坐标:\n u = {u_values}, v = {v_values}, w = {w_values}")
space = LagrangeFESpace(mesh, p=1, ctype='C')
scalar_gdof = space.number_of_global_dofs()
tensor_space = TensorFunctionSpace(space, shape=(3, -1))
tgdof = tensor_space.number_of_global_dofs()
tldof = tensor_space.number_of_local_dofs()
cell2tdof = tensor_space.cell_to_dof()

scalar_phi_loads = []
for bcs in bcs_list:
    scalar_phi = space.basis(bcs)
    scalar_phi_loads.append(scalar_phi)

for idx, scalar_phi in enumerate(scalar_phi_loads):
    print(f"载荷点 {idx + 1} 处的基函数值:\n", scalar_phi.flatten())

phi_loads = []
for bcs in bcs_list:
    phi = tensor_space.basis(bcs)
    phi_loads.append(phi)


phi_loads_array = bm.concatenate(phi_loads, axis=1) # (1, NP, tldof, GD)

FE_load = bm.einsum('p, cpld -> pl', load_values, phi_loads_array) # (NP, tldof)

FE = bm.zeros((NC, tldof), dtype=bm.float64)
FE[target_cell_idx, :] = FE_load[:, :] # (NC, tldof)

F = COOTensor(indices = bm.empty((1, 0), dtype=bm.int32, device=bm.get_device(space)),
            values = bm.empty((0, ), dtype=bm.float64, device=bm.get_device(space)),
            spshape = (tgdof, ))
indices = cell2tdof.reshape(1, -1)
F = F.add(COOTensor(indices, FE.reshape(-1), (tgdof, ))).to_dense() # (tgdof, )


linear_elastic_material = LinearElasticMaterial(name='lam1_mu1', 
                                                lame_lambda=1, shear_modulus=1, 
                                                hypo='3D', device=bm.get_device(mesh))
integrator_K = LinearElasticIntegrator(material=linear_elastic_material, q=2)

KE = integrator_K.assembly(space=tensor_space)
bform = BilinearForm(tensor_space)
bform.add_integrator(integrator_K)
K = bform.assembly(format='csr')

# 矩阵和载荷向量的范数
values = K.values()
K_norm = bm.sqrt(bm.sum(values * values))
F_norm = bm.sqrt(bm.sum(F * F))   
print(f"Matrix norm: {K_norm:.6f}")
print(f"Load vector norm: {F_norm:.6f}")

def dirichlet(points: TensorLike) -> TensorLike:
    return bm.zeros(points.shape, dtype=points.dtype, device=bm.get_device(points))

scalar_is_bd_dof = is_inner_node
tensor_is_bd_dof = tensor_space.is_boundary_dof(
        threshold=(scalar_is_bd_dof, scalar_is_bd_dof, scalar_is_bd_dof), 
        method='interp')

dbc = DirichletBC(space=tensor_space, 
                    gd=dirichlet, 
                    threshold=tensor_is_bd_dof, 
                    method='interp')
K, F = dbc.apply(A=K, f=F, check=True)

# 1. 检查矩阵对称性
def check_symmetry(K):
    """
    检查稀疏矩阵的对称性，使用CSRTensor类提供的方法
    """
    max_error = 0.0
    # 获取非零元素的行索引和列索引
    row_indices = K.row()  # 使用CSRTensor提供的row()方法
    col_indices = K.col()  # 使用CSRTensor提供的col()方法
    values = K.values()    # 使用CSRTensor提供的values()方法
    
    # 创建一个非零元素的位置字典
    nz_dict = {}
    for i, (r, c, v) in enumerate(zip(row_indices, col_indices, values)):
        nz_dict[(r, c)] = v
    
    # 检查对称性
    for r, c, v in zip(row_indices, col_indices, values):
        if r <= c:  # 只检查上三角部分
            # 检查对称位置的元素
            sym_val = nz_dict.get((c, r), 0.0)
            error = abs(v - sym_val)
            max_error = max(max_error, error)
    
    return max_error

# 输出诊断信息
print("\n7. Solving linear system...")

# 1. 检查对称性
max_symmetry_error = check_symmetry(K)
print(f"Max symmetry error: {max_symmetry_error:.6e}")

# 2. 输出矩阵的一些基本信息
# 矩阵规模
nrow, ncol = K.shape
print(f"Matrix size: {nrow}x{ncol}")

# # 非零元素个数
# nnz = K.nnz
# print(f"Matrix non-zeros: {nnz}")
# sparsity = (nnz / (nrow * ncol)) * 100
# print(f"Matrix sparsity: {sparsity:.2f}%")

# 矩阵和载荷向量的范数
values = K.values()
K_norm = bm.sqrt(bm.sum(values * values))
F_norm = bm.sqrt(bm.sum(F * F))   
print(f"Matrix norm_after: {K_norm:.6f}")
print(f"Load vector norm_after: {F_norm:.6f}")

# 载荷向量的范围
F_min = bm.min(F)
F_max = bm.max(F)
print(f"F min: {F_min:.6f}")
print(f"F max: {F_max:.6f}")

from fealpy import logger
logger.setLevel('INFO')

uh = tensor_space.function()
uh[:] = cg(K, F, maxiter=10000, atol=1e-6, rtol=1e-6)
# uh[:] = spsolve(K, F, solver='mumps')

# 计算残差向量和范数
residual = K.matmul(uh[:]) - F  # 使用 CSRTensor 的 matmul 方法
residual_norm = bm.sqrt(bm.sum(residual * residual))
print(f"Final residual norm: {residual_norm:.6e}")

uh = uh.reshape(GD, NN).T

mesh.nodedata['deform'] = uh[:]
mesh.to_vtk('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/gearx_part_cg.vtu')
print("-----------")