"""
外齿轮 15 个载荷点的算例
"""
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

def compute_strain_stress_1(space, uh, mu, lam):
    """在积分点处计算应变和应力"""
    mesh = space.mesh
    cell = mesh.entity('cell')
    cell2dof = space.cell_to_dof()
    NC = mesh.number_of_cells()

    # 计算积分点处的基函数梯度
    qf2 = mesh.quadrature_formula(2)
    bcs_q2, ws = qf2.get_quadrature_points_and_weights()
    gphix_q2 = space.grad_basis(bcs_q2, variable='x')  # (NC, NQ, LDOF, GD)

    cuh = uh[cell2dof]  # (NC, LDOF, GD)

    # 计算应变
    NQ = len(ws)  # 积分点个数
    strain = bm.zeros((NC, NQ, 6), dtype=bm.float64)
    
    # 计算正应变和剪切应变
    strain[:, :, 0:3] = bm.einsum('cid, cqid -> cqd', cuh, gphix_q2)  # (NC, NQ, 3)
    for i in range(NQ):  
        # strain[:, i, 3] = bm.sum(
        #         cuh[:, :, 2]*gphix_q2[:, i, :, 1] + cuh[:, :, 1]*gphix_q2[:, i, :, 2], 
        #         axis=-1) / 2.0  # (NC,)

        # strain[:, i, 4] = bm.sum(
        #         cuh[:, :, 2]*gphix_q2[:, i, :, 0] + cuh[:, :, 0]*gphix_q2[:, i, :, 2], 
        #         axis=-1) / 2.0  # (NC,)

        # strain[:, i, 5] = bm.sum(
        #         cuh[:, :, 1]*gphix_q2[:, i, :, 0] + cuh[:, :, 0]*gphix_q2[:, i, :, 1], 
        #         axis=-1) / 2.0  # (NC,)
        strain[:, i, 3] = bm.sum(
                cuh[..., 1]*gphix_q2[:, i, :, 0] + cuh[..., 0]*gphix_q2[:, i, :, 1], axis=-1)  # (NC,)
        strain[:, i, 4] = bm.sum(
                cuh[..., 2]*gphix_q2[:, i, :, 1] + cuh[..., 1]*gphix_q2[:, i, :, 2], axis=-1)  # (NC,)
        strain[:, i, 5] = bm.sum(
                cuh[..., 2]*gphix_q2[:, i, :, 0] + cuh[..., 0]*gphix_q2[:, i, :, 2], axis=-1)  # (NC,)

    # 计算应力
    val = 2*mu + lam
    stress = bm.zeros((NC, NQ, 6), dtype=bm.float64)
    stress[:, :, 0] = val * strain[:, :, 0] + lam * (strain[:, :, 1] + strain[:, :, 2])
    stress[:, :, 1] = val * strain[:, :, 1] + lam * (strain[:, :, 2] + strain[:, :, 0])
    stress[:, :, 2] = val * strain[:, :, 2] + lam * (strain[:, :, 0] + strain[:, :, 1])
    # stress[:, :, 3] = 2*mu * strain[:, :, 3]
    # stress[:, :, 4] = 2*mu * strain[:, :, 4]
    # stress[:, :, 5] = 2*mu * strain[:, :, 5]
    stress[:, :, 3] = mu * strain[:, :, 3]
    stress[:, :, 4] = mu * strain[:, :, 4]
    stress[:, :, 5] = mu * strain[:, :, 5]

    return strain, stress

def compute_strain_stress_2(tensor_space, uh, B, D):
    cell2tdof = tensor_space.cell_to_dof()
    cuh = uh[cell2tdof]  # (NC, TLDOF) 
    strain = bm.einsum('cqil, cl -> cqi', B, cuh) # (NC, NQ, 6)
    stress = bm.einsum('cqij, cqi -> cqj', D, strain) # (NC, NQ, 6)
    
    return strain, stress

def compute_strain_stress_3(tensor_space, uh, B_BBar, D):
    cell2tdof = tensor_space.cell_to_dof()
    cuh = uh[cell2tdof]  # (NC, TLDOF) 
    strain = bm.einsum('cqil, cl -> cqi', B_BBar, cuh) # (NC, NQ, 6)
    stress = bm.einsum('cqij, cqi -> cqj', D, strain) # (NC, NQ, 6)
    
    return strain, stress

def read_mtx_file(filename):
    """
    读取 mtx 文件并将数据转换为三维数组
        
    返回:
    numpy.ndarray
        形状为 (2, 24, 24) 的三维数组
    """
    result = bm.zeros((2, 24, 24))
    
    # 读取文件
    with open(filename, 'r') as file:
        for line in file:
            # 跳过空行
            if not line.strip():
                continue
                
            # 将每行分割成数组
            parts = line.strip().split()
            if len(parts) == 4:  # 确保行格式正确
                # 解析数据
                matrix_idx = int(parts[0])     # 矩阵索引 (0-NC-1)
                i = int(parts[1]) - 1          # 行索引 (0-23)
                j = int(parts[2]) - 1          # 列索引 (0-23)
                value = float(parts[3])        # 值
                
                # 将值存入对应位置
                if 0 <= matrix_idx < 2 and 0 <= i < 24 and 0 <= j < 24:
                    result[matrix_idx, i, j] = value
    
    return result

bm.set_backend('numpy')

with open('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/external_gear_data_part.pkl', 'rb') as f:
    data = pickle.load(f)

external_gear = data['external_gear']
hex_mesh = data['hex_mesh']
helix_node = data['helix_node']
is_inner_node = data['is_inner_node']

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
helix_d = bm.linspace(external_gear.d, external_gear.effective_da, n)
helix_width = bm.linspace(0, external_gear.tooth_width, n)
helix_node = external_gear.cylindrical_to_cartesian(helix_d, helix_width)

target_cell_idx = bm.zeros(n, bm.int32)
face_normal = bm.zeros((n, 3), bm.float64)
parameters = bm.zeros((n, 3), bm.float64)
for i, t_node in enumerate(helix_node):
    target_cell_idx[i], face_normal[i], parameters[i] = external_gear.find_node_location_kd_tree(t_node)

average_normal = bm.mean(face_normal, axis=0)
average_normal /= bm.linalg.norm(average_normal)

threshold = 0.1
for i in range(len(face_normal)):
    deviation = np.linalg.norm(face_normal[i] - average_normal)
    if deviation > threshold:
        face_normal[i] = average_normal

P = bm.einsum('p, pd -> pd', load_values, face_normal)  # (15, GD)

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
map = [ 0,  1,  2, 12, 13, 14,  9, 10, 11, 21, 22, 23,  3,  4,  5, 15, 16,
       17,  6,  7,  8, 18, 19, 20]
tgdof = tensor_space.number_of_global_dofs()
print(f"tgdof: {tgdof}")
tldof = tensor_space.number_of_local_dofs()

# 节点载荷的索引 (需要去重)
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

FE = bm.zeros((NC, tldof), dtype=bm.float64)
FE[target_cell_idx, :] = FE_load[:, :] # (NC, tldof)

F = COOTensor(indices = bm.empty((1, 0), dtype=bm.int32, device=bm.get_device(space)),
            values = bm.empty((0, ), dtype=bm.float64, device=bm.get_device(space)),
            spshape = (tgdof, ))
indices = cell2tdof.reshape(1, -1)
F = F.add(COOTensor(indices, FE.reshape(-1), (tgdof, ))).to_dense() # (tgdof, )
non_zero_indices = bm.nonzero(F)[0]
non_zero_values = F[non_zero_indices]
F_non_zero = bm.concatenate([non_zero_indices[:, None], non_zero_values[:, None]], axis=1)
np.savetxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/F_without_dc.csv', 
           F, delimiter=',', fmt='%s')
np.savetxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/F_non_zero_without_dc.csv', 
           F_non_zero, delimiter='', fmt=['%10d', '%20.6f'])


# 从全局载荷向量中提取有载荷节点处的值
F_load_nodes = F[dof_indices] # (15*8, GD)

fixed_node_index = bm.where(is_inner_node)[0]
export_to_inp(filename='/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/inp/external_gear_ansys.inp', 
              nodes=node, elements=cell, 
              fixed_nodes=fixed_node_index, load_nodes=load_node_indices, loads=F_load_nodes, 
              young_modulus=206e3, poisson_ratio=0.3, density=7.85e-9, 
              used_app='ansys', mesh_type='hex')

E = 206e3
nu = 0.3
lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
linear_elastic_material = LinearElasticMaterial(name='E_nu', 
                                                elastic_modulus=E, poisson_ratio=nu, 
                                                hypo='3D', device=bm.get_device(mesh))

# B-Bar 修正的刚度矩阵
integrator_K0 = LinearElasticIntegrator(material=linear_elastic_material, 
                                        q=q, method='voigt')
integrator_K0.keep_data(True)
_, _, _, D, B = integrator_K0.fetch_voigt_assembly(tensor_space)
integrator_K = LinearElasticIntegrator(material=linear_elastic_material, 
                                       q=q, method='C3D8_BBar')
integrator_K.keep_data(True)
_, _, D, B_BBar = integrator_K.fetch_c3d8_bbar_assembly(tensor_space)
bform = BilinearForm(tensor_space)
bform.add_integrator(integrator_K)
K = bform.assembly(format='csr')


# 处理 Dirichlet 边界条件
scalar_is_bd_dof = bm.zeros(scalar_gdof, dtype=bm.bool)
scalar_is_bd_dof[:NN] = is_inner_node
tensor_is_bd_dof = tensor_space.is_boundary_dof(
                                threshold=(scalar_is_bd_dof, scalar_is_bd_dof, scalar_is_bd_dof), 
                                method='interp')
dbc = DirichletBC(space=tensor_space, 
                    gd=bm.zeros(tgdof), 
                    threshold=tensor_is_bd_dof, 
                    method='interp')
K, F = dbc.apply(K, F)

from fealpy import logger
logger.setLevel('INFO')
uh = tensor_space.function()
# uh[:] = cg(K, F, maxiter=10000, atol=1e-8, rtol=1e-8)
uh[:] = spsolve(K, F, solver="mumps")

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

# NOTE 应变
# strain1, stress1 = compute_strain_stress_1(space, uh_show, mu, lam)
# map_abaqus = [0, 4, 6, 2, 1, 5, 7, 3]
# print(f"strain1_0-{strain1[0].shape}, {cell2dof[0]}\n {strain1[0]}")
# print(f"strain1_0_map-{strain1[0].shape}, {cell2dof[0][map_abaqus]}\n {strain1[0][map_abaqus, :]}")

# strain2, stress2 = compute_strain_stress_2(tensor_space, uh, B, D)
# error12 = bm.linalg.norm(strain1 - strain2)
# print(f"error12: {error12:.12e}")

# NQ 轴: z-y-x
strain3, stress3 = compute_strain_stress_3(tensor_space, uh, B_BBar, D) # (NC, NQ, 6)
# print(f"strain_BBar_0: {strain3.shape}, {cell2dof[0]}\n {strain3[0]}")
# qf = mesh.quadrature_formula(2)
# bcs, wq = qf.get_quadrature_points_and_weights()
# ps = mesh.bc_to_point(bcs)
# print(f"ps_1:\n {ps[1]}")
# print(f"strain_BBar_0_map: {strain3.shape}, {cell2dof[0][map_abaqus]}\n {strain3[0][map_abaqus, :]}")
print(f"strain_BBar_1: {strain3.shape}, {cell2dof[1]}\n {strain3[1]}")
# print(f"strain_BBar_1_map: {strain3.shape}, {cell2dof[1][map_abaqus]}\n {strain3[1][map_abaqus, :]}")
# print(f"stess_BBar_0: {stress3.shape}, {cell2dof[0]}\n {stress3[0]}")
# print(f"stess_BBar_0_map: {stress3.shape}, {cell2dof[0][map_abaqus]}\n {stress3[0][map_abaqus, :]}")
# print(f"strain_BBar_1363: {strain3.shape}, {cell2dof[1363]}\n {strain3[1363]}")
# print(f"strain_BBar_1363_map: {strain3.shape}, {cell2dof[1363][map_abaqus]}\n {strain3[1363][map_abaqus, :]}")
# print(f"strain_BBar_1364: {strain3.shape}, {cell2dof[1364]}\n {strain3[1364]}")
# print(f"strain_BBar_1364_map: {strain3.shape}, {cell2dof[1364][map_abaqus]}\n {strain3[1364][map_abaqus, :]}")

# NQ 轴: z-y-x, LDOF 轴 z-y-x
extrapolation_matrix = bm.tensor([
            [-0.0490381057,  0.1830127019,  0.1830127019, -0.6830127019,  0.1830127019, -0.6830127019, -0.6830127019,  2.5490381057],
            [ 0.1830127019, -0.0490381057, -0.6830127019,  0.1830127019, -0.6830127019,  0.1830127019,  2.5490381057, -0.6830127019],
            [ 0.1830127019, -0.6830127019, -0.0490381057,  0.1830127019, -0.6830127019,  2.5490381057,  0.1830127019, -0.6830127019],
            [-0.6830127019,  0.1830127019,  0.1830127019, -0.0490381057,  2.5490381057, -0.6830127019, -0.6830127019,  0.1830127019],
            [ 0.1830127019, -0.6830127019, -0.6830127019,  2.5490381057, -0.0490381057,  0.1830127019,  0.1830127019, -0.6830127019],
            [-0.6830127019,  0.1830127019,  2.5490381057, -0.6830127019,  0.1830127019, -0.0490381057, -0.6830127019,  0.1830127019],
            [-0.6830127019,  2.5490381057,  0.1830127019, -0.6830127019,  0.1830127019, -0.6830127019, -0.0490381057,  0.1830127019],
            [ 2.5490381057, -0.6830127019, -0.6830127019,  0.1830127019, -0.6830127019,  0.1830127019,  0.1830127019, -0.0490381057]
            ], dtype=bm.float64) # (NQ, LDOF)
strain3_extrapolation = bm.einsum('lq, cqj -> clj', extrapolation_matrix, strain3)
# strain3_extrapolation = np.zeros((NC, 8, 6), dtype=np.float64)
# for c in range(NC):
#     for l in range(8):
#         for j in range(6):
#             # 初始化应变值
#             strain_sum = 0.0
#             for q in range(8):
#                 strain_sum += extrapolation_matrix[l, q] * strain3[c, q, j]
#             strain3_extrapolation[c, l, j] = strain_sum
# ip2 = mesh.interpolation_points(p=1)
# print(f"ip2_1:\n {node[cell2dof[1]]}")
print(f"strain_extrapolation_BBar_0: {strain3_extrapolation.shape}, {cell2dof[0]}\n {strain3_extrapolation[0]}")
# print(f"strain_extrapolation_BBar_0_map: {strain3_extrapolation.shape}, {cell2dof[0][map_abaqus]}\n {strain3_extrapolation[0][map_abaqus, :]}")

# print(f"strain_extrapolation_BBar_0_map1: {strain3_extrapolation.shape}, {cell2dof[0][map_abaqus][map_abaqus2]}\n {strain3_extrapolation[0][map_abaqus, :][map_abaqus2]}")

print(f"strain_extrapolation_BBar_1: {strain3_extrapolation.shape}, {cell2dof[1]}\n {strain3_extrapolation[1]}")
# print(f"strain_extrapolation_BBar_1_map: {strain3_extrapolation.shape}, {cell2dof[1][map_abaqus]}\n {strain3_extrapolation[1][map_abaqus, :]}")
# print(f"strain_extrapolation_BBar_1_map1: {strain3_extrapolation.shape}, {cell2dof[1][map_abaqus][map_abaqus2]}\n {strain3_extrapolation[1][map_abaqus, :][map_abaqus2]}")

print(f"strain_extrapolation_BBar_1363: {strain3_extrapolation.shape}, {cell2dof[1363]}\n {strain3_extrapolation[1363]}")
# print(f"strain_extrapolation_BBar_1363_map: {strain3_extrapolation.shape}, {cell2dof[1363][map_abaqus]}\n {strain3_extrapolation[1363][map_abaqus, :]}")

print(f"strain_extrapolation_BBar_1364: {strain3_extrapolation.shape}, {cell2dof[1364]}\n {strain3_extrapolation[1364]}")
# print(f"strain_extrapolation_BBar_1364_map: {strain3_extrapolation.shape}, {cell2dof[1364][map_abaqus]}\n {strain3_extrapolation[1364][map_abaqus, :]}")

# 计算节点应变和应力
nstrain = bm.zeros((NN, 6), dtype=bm.float64)
nstress = bm.zeros((NN, 6), dtype=bm.float64)
nc = bm.zeros(NN, dtype=bm.int32)
bm.add_at(nc, cell2dof, 1)
# bm.add_at(nc, cell, 1)
for i in range(6):
    bm.add_at(nstrain[:, i], cell2dof.flatten(), strain3_extrapolation[..., i].flatten())
    # bm.add_at(nstrain[:, i], cell.flatten(), strain3_extrapolation[..., i].flatten())
    nstrain[:, i] /= nc
    bm.add_at(nstress[:, i], cell2dof.flatten(), strain3_extrapolation[..., i].flatten())
    # bm.add_at(nstress[:, i], cell.flatten(), strain3_extrapolation[..., i].flatten())
    nstress[:, i] /= nc
nstrain24 = nstrain[24]
print(f"nstrain24: {nstrain24}")
mesh.nodedata['nstrain'] = nstrain
mesh.nodedata['nstress'] = nstress
mesh.to_vtk('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/vtu/external_gear_fealpy.vtu')
print("-----------")