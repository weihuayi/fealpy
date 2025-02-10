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

file_name_local = "/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/mtx/external_gear_local_STIF2.mtx"
KE_abaqus_proto = read_mtx_file(file_name_local)
KE_abaqus_utri = bm.triu(KE_abaqus_proto, 1)
KE_abaqus_ltri = bm.transpose(KE_abaqus_utri, (0, 2, 1))
KE_abaqus = KE_abaqus_proto + KE_abaqus_ltri
KE0_abaqus = KE_abaqus[0]
KE29779_abaqus = KE_abaqus[1]

np.savetxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/KE0_abaqus.csv', 
            KE0_abaqus, delimiter=',', fmt='%s')
np.savetxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/KE29779_abaqus.csv',
            KE29779_abaqus, delimiter=',', fmt='%s')

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

KE_abaqus_map = KE_abaqus[:, :, map]
KE_abaqus_map = KE_abaqus_map[:, map, :]
KE0_abaqus_map = KE_abaqus_map[0]
KE29779_abaqus_map = KE_abaqus_map[1]

np.savetxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/KE0_abaqus_map.csv', 
           KE0_abaqus_map, delimiter=',', fmt='%s')
np.savetxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/KE29779_abaqus_map.csv',
              KE29779_abaqus_map, delimiter=',', fmt='%s')

# 节点载荷的索引
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

mask = (cell2tdof == 99338)
rows_99338 = bm.where(mask.any(axis=1))[0]
indices_99338 = np.argwhere(mask)
cols_99338 = indices_99338[:, 1]
FE_99338 = FE[rows_99338]
row_indices = [0, 1, 2, 3, 4, 5, 6, 7]
FE_99338_1 = FE_99338[row_indices, cols_99338].reshape(-1, 1)

# 输出结果

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
integrator_K = LinearElasticIntegrator(material=linear_elastic_material, 
                                       q=q, method='C3D8_BBar')
KE_bbar = integrator_K.c3d8_bbar_assembly(tensor_space)
KE0_bbar = KE_bbar[0]
KE29779_bbar = KE_bbar[29779]
np.savetxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/KE0_bbar.csv', 
           KE0_bbar, delimiter=',', fmt='%s')
np.savetxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/KE29779_bbar.csv', 
           KE29779_bbar, delimiter=',', fmt='%s')


error_KE0 = bm.max(bm.abs(KE0_abaqus_map - KE0_bbar))
error_KE29779 = bm.max(bm.abs(KE29779_abaqus_map - KE29779_bbar))
print(f"error_KE0: {error_KE0}")
print(f"error_KE29779: {error_KE29779}")

bform = BilinearForm(tensor_space)
bform.add_integrator(integrator_K)
K = bform.assembly(format='csr')
Kdense = K.to_dense()[7896, 7896]


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

cuh = uh_show[cell] # (NC, NCN, GD)
print(f"node[cell0]-{node[cell[0]].shape}, {node[cell[0]]}")
print(f"cuh_0-{cuh[0].shape}, {cell[0]}\n {cuh[0]}")
print(f"node[cell20779]-{node[cell[29779]].shape}, {node[cell[29779]]}")
print(f"cuh_20779-{cuh[29779].shape}, {cell[29779]}\n {cuh[29779]}")
uh_x = uh_show[:, 0]
uh_y = uh_show[:, 1]
uh_z = uh_show[:, 2]

uh_magnitude = bm.linalg.norm(uh_show, axis=1)

mesh.nodedata['uh'] = uh_show[:]
mesh.nodedata['uh_magnitude'] = uh_magnitude[:]

mesh.to_vtk('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/vtu/external_gear_fealpy.vtu')
print("-----------")