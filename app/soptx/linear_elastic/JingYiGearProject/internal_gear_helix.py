"""
内齿轮 15 个载荷点的算例
"""
from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.sparse import COOTensor
from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem.dirichlet_bc import DirichletBC
from fealpy.solver import cg, spsolve
from fealpy.mesh import HexahedronMesh

from soptx.utils import timer
from app.soptx.linear_elasticity.JingYiGearProject.utils import export_to_inp

import pickle

def compute_strain_stress(tensor_space, uh, B_BBar, D):
    cell2tdof = tensor_space.cell_to_dof()
    cuh = uh[cell2tdof]  # (NC, TLDOF) 
    strain = bm.einsum('cqil, cl -> cqi', B_BBar, cuh) # (NC, NQ, 6)
    stress = bm.einsum('cqij, cqi -> cqj', D, strain) # (NC, NQ, 6)
    
    return strain, stress

bm.set_backend('numpy')

with open('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/JingYiGearProject/pkl/internal_gear.pkl', 'rb') as f:
    data = pickle.load(f)

internal_gear = data['internal_gear']
hex_mesh = data['hex_mesh']
is_outer_node = data['is_outer_node']

hex_cell = hex_mesh.cell
hex_node = hex_mesh.node
mesh = HexahedronMesh(hex_node, hex_cell)

GD = mesh.geo_dimension()   
NC = mesh.number_of_cells()
print(f"NC: {NC}")
NN = mesh.number_of_nodes()
print(f"NN: {NN}")
node = mesh.entity('node')
cell = mesh.entity('cell')

# 节点载荷
load_values = 100 * bm.array([50.0, 60.0, 79.0, 78.0, 87.0, 95.0, 102.0, 109.0, 114.0,
                        119.0, 123.0, 127.0, 129.0, 130.0, 131.0], dtype=bm.float64)   # (15, )

n = 15
helix_d = bm.linspace(internal_gear.d, internal_gear.d_a, n)
helix_width = bm.linspace(0, internal_gear.tooth_width, n)
helix_node = internal_gear.cylindrical_to_cartesian(helix_d, helix_width)

target_cell_idx = bm.zeros(n, bm.int32)
face_normal = bm.zeros((n, 3), bm.float64)
parameters = bm.zeros((n, 3), bm.float64)
for i, t_node in enumerate(helix_node):
    target_cell_idx[i], face_normal[i], parameters[i] = internal_gear.find_node_location_kd_tree(t_node)
print(f"target_cell_idx: {target_cell_idx}")
average_normal = bm.mean(face_normal, axis=0)
average_normal /= bm.linalg.norm(average_normal)

threshold = 0.1
for i in range(len(face_normal)):
    deviation = bm.linalg.norm(face_normal[i] - average_normal)
    if deviation > threshold:
        face_normal[i] = average_normal

P = bm.einsum('p, pd -> pd', load_values, face_normal)  # (15, GD)
print(f"P:", P)

u = parameters[..., 0]
v = parameters[..., 1]
w = parameters[..., 2]

bcs_list = [
    (
        bm.tensor([[u, 1 - u]]),
        bm.tensor([[v, 1 - v]]),
        bm.tensor([[w, 1 - w]])
    )
    for u, v, w in zip(u, v, w)
]
p = 1
q = 2
space = LagrangeFESpace(mesh, p=p, ctype='C')
scalar_gdof = space.number_of_global_dofs()
print(f"gdof: {scalar_gdof}")
cell2dof = space.cell_to_dof()
tensor_space = TensorFunctionSpace(space, shape=(-1, 3)) # gd_priority
cell2tdof = tensor_space.cell_to_dof()
tgdof = tensor_space.number_of_global_dofs()
print(f"tgdof: {tgdof}")
tldof = tensor_space.number_of_local_dofs()

# 节点载荷的索引（去重）
load_node_indices0 = cell[target_cell_idx].flatten() # (15*8, )
unique_nodes, first_indices = bm.unique(load_node_indices0, return_index=True)
sort1_indices = bm.sort(first_indices)
load_node_indices = load_node_indices0[sort1_indices] # (15*8, )
# 带有载荷的节点对应的全局自由度编号（跟顺序有关）
if tensor_space.dof_priority:
    dof_indices = bm.stack([scalar_gdof * d + load_node_indices for d in range(GD)], axis=1)  # (15*8, GD)
else:
    dof_indices = bm.stack([load_node_indices * GD + d for d in range(GD)], axis=1)  # (15*8, GD)


phi_loads = []
for bcs in bcs_list:
    phi = tensor_space.basis(bcs)
    phi_loads.append(phi)

phi_loads_array = bm.concatenate(phi_loads, axis=1) # (1, 15, tldof, GD)

FE_load = bm.einsum('pd, cpld -> pl', P, phi_loads_array) # (15, 24)
print(f"FE_load:\n {FE_load}")

FE = bm.zeros((NC, tldof), dtype=bm.float64)
bm.add.at(FE, target_cell_idx, FE_load)  # (NC, tldof) 这会自动累加重复索引的值
# FE[target_cell_idx, :] = FE_load[:, :] # (NC, tldof)
# FE[target_cell_idx, :] = FE_load # (NC, tldof)
non_zero_rows = bm.any(FE != 0, axis=1)
non_zero_indices = bm.where(non_zero_rows)[0]
print(f"non_zero_indices: {non_zero_indices}")
unit_max = bm.max(FE, axis=1)
unit_with_FE_max = bm.argmax(unit_max)
FE_max = unit_max[unit_with_FE_max]

print(f"最大的载荷值为 {FE_max}，位于第 {unit_with_FE_max} 个单元")
print(f"FE:, {FE[14311, :]}")

F = COOTensor(indices = bm.empty((1, 0), dtype=bm.int32, device=bm.get_device(space)),
            values = bm.empty((0, ), dtype=bm.float64, device=bm.get_device(space)),
            spshape = (tgdof, ))
indices = cell2tdof.reshape(1, -1)
F = F.add(COOTensor(indices, FE.reshape(-1), (tgdof, ))).to_dense() # (tgdof, )

# 从全局载荷向量中提取有载荷节点处的值
F_load_nodes = F[dof_indices] # (15*8, GD)

fixed_node_index = bm.where(is_outer_node)[0]
export_to_inp(filename='/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/JingYiGearProject/inp/internal_gear_abaqus.inp', 
              nodes=node, elements=cell, 
              fixed_nodes=fixed_node_index, load_nodes=load_node_indices, loads=F_load_nodes, 
              young_modulus=206e3, poisson_ratio=0.3, density=7.85e-9, 
              used_app='abaqus', mesh_type='hex')

# 创建计时器
t = timer(f"Timing_{i}")
next(t)  # 启动计时器
E = 206e3
nu = 0.3
lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
linear_elastic_material = LinearElasticMaterial(name='E_nu', 
                                                elastic_modulus=E, poisson_ratio=nu, 
                                                hypo='3D', device=bm.get_device(mesh))

# B-Bar 修正的刚度矩阵
integrator_K = LinearElasticIntegrator(material=linear_elastic_material, 
                                       q=q, method='C3D8_BBar')
integrator_K.keep_data(True)
_, _, D, B_BBar = integrator_K.fetch_c3d8_bbar_assembly(tensor_space)
KE = integrator_K.c3d8_bbar_assembly(tensor_space)
print(f"KE: {KE[0]}")
bform = BilinearForm(tensor_space)
bform.add_integrator(integrator_K)
K = bform.assembly(format='csr')

# 处理 Dirichlet 边界条件
scalar_is_bd_dof = bm.zeros(scalar_gdof, dtype=bm.bool)
scalar_is_bd_dof[:NN] = is_outer_node
tensor_is_bd_dof = tensor_space.is_boundary_dof(
                                threshold=(scalar_is_bd_dof, scalar_is_bd_dof, scalar_is_bd_dof), 
                                method='interp')
dbc = DirichletBC(space=tensor_space, 
                    gd=bm.zeros(tgdof), 
                    threshold=tensor_is_bd_dof, 
                    method='interp')
K, F = dbc.apply(K, F)
t.send('准备时间')
from fealpy import logger
logger.setLevel('INFO')
uh = tensor_space.function()
# uh[:] = cg(K, F, maxiter=10000, atol=1e-8, rtol=1e-8)
uh[:] = spsolve(K, F, solver="mumps")
t.send('求解时间')
t.send(None)

# 计算残差向量和范数
residual = K.matmul(uh[:]) - F  
residual_norm = bm.sqrt(bm.sum(residual * residual))
print(f"Final residual norm: {residual_norm:.6e}")

if tensor_space.dof_priority:
    uh_show = uh.reshape(GD, NN).T
else:
    uh_show = uh.reshape(NN, GD)

uh_x = uh_show[:, 0]
uh_y = uh_show[:, 1]
uh_z = uh_show[:, 2]

uh_magnitude = bm.linalg.norm(uh_show, axis=1)

mesh.nodedata['uh'] = uh_show[:]
mesh.nodedata['uh_magnitude'] = uh_magnitude[:]

# ------------------------------------------------------------------------------
# 计算单元积分点处的应变和应力
# ------------------------------------------------------------------------------
# 1. 计算单元积分点处应变和应力
strain, stress = compute_strain_stress(tensor_space, uh, B_BBar, D) # (NC, NQ, 6)

extrapolation_matrix = bm.tensor([
            [-0.0490381057,  0.1830127019,  0.1830127019, -0.6830127019,  0.1830127019, -0.6830127019, -0.6830127019,  2.5490381057],
            [ 0.1830127019, -0.0490381057, -0.6830127019,  0.1830127019, -0.6830127019,  0.1830127019,  2.5490381057, -0.6830127019],
            [ 0.1830127019, -0.6830127019, -0.0490381057,  0.1830127019, -0.6830127019,  2.5490381057,  0.1830127019, -0.6830127019],
            [-0.6830127019,  0.1830127019,  0.1830127019, -0.0490381057,  2.5490381057, -0.6830127019, -0.6830127019,  0.1830127019],
            [ 0.1830127019, -0.6830127019, -0.6830127019,  2.5490381057, -0.0490381057,  0.1830127019,  0.1830127019, -0.6830127019],
            [-0.6830127019,  0.1830127019,  2.5490381057, -0.6830127019,  0.1830127019, -0.0490381057, -0.6830127019,  0.1830127019],
            [-0.6830127019,  2.5490381057,  0.1830127019, -0.6830127019,  0.1830127019, -0.6830127019, -0.0490381057,  0.1830127019],
            [ 2.5490381057, -0.6830127019, -0.6830127019,  0.1830127019, -0.6830127019,  0.1830127019,  0.1830127019, -0.0490381057]
            ], dtype=bm.float64) # (NQ, 8)

# 计算外插后的单元节点应变和应力
cell2dof_map = [0, 4, 6, 2, 1, 5, 7, 3]
extrapolation_map = [7, 3, 1, 5, 6, 2, 0, 4 ]
strain_extrapolation = bm.einsum('lq, cqj -> clj', extrapolation_matrix, strain)
stress_extrapolation = bm.einsum('lq, cqj -> clj', extrapolation_matrix, stress)
strain_extrapolation_maps = strain_extrapolation[:, extrapolation_map, :]
stress_extrapolation_maps = stress_extrapolation[:, extrapolation_map, :]

# 使用直接平均法计算节点应变和应力
cell2dof_maps = cell2dof[:, cell2dof_map] 
nstrain = bm.zeros((NN, 6), dtype=bm.float64)
nstress = bm.zeros((NN, 6), dtype=bm.float64)
nc = bm.zeros(NN, dtype=bm.int32)
bm.add_at(nc, cell2dof_maps, 1)
for i in range(6):
    bm.add_at(nstrain[:, i], cell2dof_maps.flatten(), strain_extrapolation_maps[:, :, i].flatten())
    nstrain[:, i] /= nc
    bm.add_at(nstress[:, i], cell2dof_maps.flatten(), stress_extrapolation_maps[:, :, i].flatten())
    nstress[:, i] /= nc

mesh.nodedata['nstrain'] = nstrain
mesh.nodedata['nstress'] = nstress

# ------------------------------------------------------------------------------
# 计算 Mises 应力（Von Mises Stress）
# ------------------------------------------------------------------------------
# 计算单元节点处的 Mises 应力
# 1. 获取外插后的单元节点处的应力分量
# sigma_xx_node = stress_extrapolation[..., 0]
# sigma_yy_node = stress_extrapolation[..., 1]
# sigma_zz_node = stress_extrapolation[..., 2]
# tau_xy_node = stress_extrapolation[..., 3]
# tau_xz_node = stress_extrapolation[..., 4]
# tau_yz_node = stress_extrapolation[..., 5]
sigma_xx_node = stress_extrapolation_maps[..., 0]
sigma_yy_node = stress_extrapolation_maps[..., 1]
sigma_zz_node = stress_extrapolation_maps[..., 2]
tau_xy_node = stress_extrapolation_maps[..., 3]
tau_xz_node = stress_extrapolation_maps[..., 4]
tau_yz_node = stress_extrapolation_maps[..., 5]
# 2. 计算单元节点处的 Mises 应力
mises_elem_node = bm.sqrt(0.5 * (
                    (sigma_xx_node - sigma_yy_node)**2 +
                    (sigma_xx_node - sigma_zz_node)**2 +
                    (sigma_yy_node - sigma_zz_node)**2 +
                    6 * (tau_xy_node**2 + tau_xz_node**2 + tau_yz_node**2)
                )) # (NC, NCN)
# mises_elem_node_map = [7, 3, 1, 5, 6, 2, 0, 4]
# print(f"mises_node_0: {cell2dof[0][cell2dof_map]}\n, {mises_elem_node[0][mises_elem_node_map]}")
# print(f"mises_node_1: {cell2dof[1][cell2dof_map]}\n, {mises_elem_node[1][mises_elem_node_map]}")
# print(f"mises_node_1363: {cell2dof[1363][cell2dof_map]}\n, {mises_elem_node[1363][mises_elem_node_map]}")
# print(f"mises_node_1364: {cell2dof[1364][cell2dof_map]}\n, {mises_elem_node[1364][mises_elem_node_map]}")
# 3. 平均分配到节点
mises_node = bm.zeros(NN, dtype=bm.float64)
nc = bm.zeros(NN, dtype=bm.int32)
bm.add_at(nc, cell2dof_maps, 1)
# bm.add_at(mises_node, cell2dof_maps.flatten(), mises_elem_node[:, mises_elem_node_map].flatten())
bm.add_at(mises_node, cell2dof_maps.flatten(), mises_elem_node.flatten())
mises_node /= nc

mesh.nodedata['mises'] = mises_node
mesh.to_vtk('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/JingYiGearProject/vtu/internal_gear_fealpy.vtu')
print("-----------")