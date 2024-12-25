from fealpy.backend import backend_manager as bm
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.sparse import COOTensor
from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem.dirichlet_bc import DirichletBC
from fealpy.typing import TensorLike
from fealpy.solver import cg, spsolve

from soptx.utils import timer

import pickle
from app.gearx.gear import ExternalGear, InternalGear
from app.gearx.utils import *

from fealpy.mesh import HexahedronMesh

def compute_strain_stress_at_vertices(space, uh, mu, lam):
    """在网格顶点处计算应变和应力"""
    mesh = space.mesh
    cell = mesh.entity('cell')
    cell2dof = space.cell_to_dof()
    p = space.p
    NC = mesh.number_of_cells()
    NN = mesh.number_of_nodes()
    
    # 插值点的多重指标
    shape = (p+1, p+1, p+1)
    mi = bm.arange(p+1, device=bm.get_device(cell))
    multiIndex0 = bm.broadcast_to(mi[:, None, None], shape).reshape(-1, 1)
    multiIndex1 = bm.broadcast_to(mi[None, :, None], shape).reshape(-1, 1)
    multiIndex2 = bm.broadcast_to(mi[None, None, :], shape).reshape(-1, 1)
    multiIndex = bm.concatenate([multiIndex0, multiIndex1, multiIndex2], axis=-1)
    
    # 多重指标的映射
    multiIndex_map = mesh.multi_index_matrix(p=p, etype=1)
    # 插值点的重心坐标
    barycenter = multiIndex_map[multiIndex].astype(bm.float64)
    bcs = (barycenter[:, 0, :], barycenter[:, 1, :], barycenter[:, 2, :])

    # 计算基函数梯度
    gphix_list = []
    for i in range(barycenter.shape[0]):
        bc_i = (
            bcs[0][i].reshape(1, -1),
            bcs[1][i].reshape(1, -1),
            bcs[2][i].reshape(1, -1)
        )
        gphix_i = space.grad_basis(bc_i, variable='x')
        gphix_list.append(gphix_i)
    
    gphix_i2 = bm.concatenate(gphix_list, axis=1) # (NC, 8, LDOF, GD)
    cuh = uh[cell2dof]                            # (NC, LDOF, GD)

    # 计算应变
    strain = bm.zeros((NC, 8, 6), dtype=bm.float64)
    strain[:, :, 0:3] = bm.einsum('cid, cnid -> cnd', cuh, gphix_i2) # (NC, 8, 3)
    # 计算剪应变，遍历每个节点
    for i in range(8):  # 遍历每个节点
        strain[:, i, 3] = bm.sum(
                cuh[:, :, 2]*gphix_i2[:, i, :, 1] + cuh[:, :, 1]*gphix_i2[:, i, :, 2], 
                axis=-1) / 2.0  # (NC,)

        strain[:, i, 4] = bm.sum(
                cuh[:, :, 2]*gphix_i2[:, i, :, 0] + cuh[:, :, 0]*gphix_i2[:, i, :, 2], 
                axis=-1) / 2.0  # (NC,)

        strain[:, i, 5] = bm.sum(
                cuh[:, :, 1]*gphix_i2[:, i, :, 0] + cuh[:, :, 0]*gphix_i2[:, i, :, 1], 
                axis=-1) / 2.0  # (NC,)

    # 计算应力
    val = 2*mu + lam
    stress = bm.zeros((NC, 8, 6), dtype=bm.float64)
    stress[:, :, 0] = val * strain[:, :, 0] + lam * (strain[:, :, 1] + strain[:, :, 2])
    stress[:, :, 1] = val * strain[:, :, 1] + lam * (strain[:, :, 2] + strain[:, :, 0])
    stress[:, :, 2] = val * strain[:, :, 2] + lam * (strain[:, :, 0] + strain[:, :, 1])
    stress[:, :, 3] = 2*mu * strain[:, :, 3]
    stress[:, :, 4] = 2*mu * strain[:, :, 4]
    stress[:, :, 5] = 2*mu * strain[:, :, 5]

    # 计算节点应变和应力
    nstrain = bm.zeros((NN, 6), dtype=bm.float64)
    nstress = bm.zeros((NN, 6), dtype=bm.float64)
    nc = bm.zeros(NN, dtype=bm.int32)
    bm.add_at(nc, cell, 1)
    for i in range(6):
        bm.add_at(nstrain[:, i], cell.flatten(), strain[:, :, i].flatten())
        nstrain[:, i] /= nc
        bm.add_at(nstress[:, i], cell.flatten(), stress[:, :, i].flatten())
        nstress[:, i] /= nc
        
    return strain, stress, nstrain, nstress

def compute_strain_stress_at_centers(space, uh, mu, lam):
    """在单元中心处计算应变和应力"""
    mesh = space.mesh
    cell = mesh.entity('cell')
    cell2dof = space.cell_to_dof()
    NC = mesh.number_of_cells()
    NN = mesh.number_of_nodes()

    # 计算中心点处的基函数梯度
    qf1 = mesh.quadrature_formula(1)
    bcs_q1, ws = qf1.get_quadrature_points_and_weights()
    gphix_q1 = space.grad_basis(bcs_q1, variable='x') # (NC, 1, LDOF, GD)
    gphix_q1 = gphix_q1.squeeze(axis=1)               # (NC, LDOF, GD)

    cuh = uh[cell2dof]

    # 计算应变
    strain = bm.zeros((NC, 6), dtype=bm.float64)
    strain[:, 0:3] = bm.einsum('cid, cid -> cd', cuh, gphix_q1) # (NC, 3)
    strain[:, 3] = bm.sum(
            cuh[:, :, 2]*gphix_q1[:, :, 1] + cuh[:, :, 1]*gphix_q1[:, :, 2], 
            axis=-1)/2.0 # (NC, )
    strain[:, 4] = bm.sum(
            cuh[:, :, 2]*gphix_q1[:, :, 0] + cuh[:, :, 0]*gphix_q1[:, :, 2], 
            axis=-1)/2.0 # (NC, )
    strain[:, 5] = bm.sum(
            cuh[:, :, 1]*gphix_q1[:, :, 0] + cuh[:, :, 0]*gphix_q1[:, :, 1], 
            axis=-1)/2.0 # (NC, )

    # 计算应力
    val = 2*mu + lam
    stress = bm.zeros((NC, 6), dtype=bm.float64)
    stress[:, 0] = val * strain[:, 0] + lam * (strain[:, 1] + strain[:, 2])
    stress[:, 1] = val * strain[:, 1] + lam * (strain[:, 2] + strain[:, 0])
    stress[:, 2] = val * strain[:, 2] + lam * (strain[:, 0] + strain[:, 1])
    stress[:, 3] = 2*mu * strain[:, 3]
    stress[:, 4] = 2*mu * strain[:, 4]
    stress[:, 5] = 2*mu * strain[:, 5]

    # 计算节点应变和应力
    nstrain = bm.zeros((NN, 6), dtype=bm.float64)
    nstress = bm.zeros((NN, 6), dtype=bm.float64)
    nc = bm.zeros(NN, dtype=bm.int32)
    bm.add_at(nc, cell, 1)
    for i in range(6):
        bm.add_at(nstrain[:, i], cell, strain[:, i, None] * bm.ones_like(cell))
        nstrain[:, i] /= nc
        bm.add_at(nstress[:, i], cell, stress[:, i, None] * bm.ones_like(cell))
        nstress[:, i] /= nc
        
    return strain, stress, nstrain, nstress

def compute_strain_stress_at_quadpoints1(space, uh, mu, lam):
    """在积分点处计算应变和应力"""
    mesh = space.mesh
    cell = mesh.entity('cell')
    cell2dof = space.cell_to_dof()
    NC = mesh.number_of_cells()
    NN = mesh.number_of_nodes()

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
        strain[:, i, 3] = bm.sum(
                cuh[:, :, 2]*gphix_q2[:, i, :, 1] + cuh[:, :, 1]*gphix_q2[:, i, :, 2], 
                axis=-1) / 2.0  # (NC,)

        strain[:, i, 4] = bm.sum(
                cuh[:, :, 2]*gphix_q2[:, i, :, 0] + cuh[:, :, 0]*gphix_q2[:, i, :, 2], 
                axis=-1) / 2.0  # (NC,)

        strain[:, i, 5] = bm.sum(
                cuh[:, :, 1]*gphix_q2[:, i, :, 0] + cuh[:, :, 0]*gphix_q2[:, i, :, 1], 
                axis=-1) / 2.0  # (NC,)

    # 计算应力
    val = 2*mu + lam
    stress = bm.zeros((NC, NQ, 6), dtype=bm.float64)
    stress[:, :, 0] = val * strain[:, :, 0] + lam * (strain[:, :, 1] + strain[:, :, 2])
    stress[:, :, 1] = val * strain[:, :, 1] + lam * (strain[:, :, 2] + strain[:, :, 0])
    stress[:, :, 2] = val * strain[:, :, 2] + lam * (strain[:, :, 0] + strain[:, :, 1])
    stress[:, :, 3] = 2*mu * strain[:, :, 3]
    stress[:, :, 4] = 2*mu * strain[:, :, 4]
    stress[:, :, 5] = 2*mu * strain[:, :, 5]

    # 初始化节点累加器和计数器
    nstrain = bm.zeros((NN, 6), dtype=bm.float64)
    nstress = bm.zeros((NN, 6), dtype=bm.float64)

    map = bm.array([0, 4, 6, 2, 1, 5, 7, 3], dtype=bm.int32)
    strain = strain[:, map, :] # (NC, 8, 6)
    stress = stress[:, map, :] # (NC, 8, 6)

    nc = bm.zeros(NN, dtype=bm.int32)
    bm.add_at(nc, cell, 1)
    for i in range(6):
        bm.add_at(nstrain[:, i], cell.flatten(), strain[:, :, i].flatten())
        nstrain[:, i] /= nc
        bm.add_at(nstress[:, i], cell.flatten(), stress[:, :, i].flatten())
        nstress[:, i] /= nc

    return strain, stress, nstrain, nstress

def compute_strain_stress_at_quadpoints2(space, uh, mu, lam):
    """在积分点处计算应变和应力"""
    mesh = space.mesh
    cell = mesh.entity('cell')
    cell2dof = space.cell_to_dof()
    NC = mesh.number_of_cells()
    NN = mesh.number_of_nodes()

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
        strain[:, i, 3] = bm.sum(
                cuh[:, :, 2]*gphix_q2[:, i, :, 1] + cuh[:, :, 1]*gphix_q2[:, i, :, 2], 
                axis=-1) / 2.0  # (NC,)

        strain[:, i, 4] = bm.sum(
                cuh[:, :, 2]*gphix_q2[:, i, :, 0] + cuh[:, :, 0]*gphix_q2[:, i, :, 2], 
                axis=-1) / 2.0  # (NC,)

        strain[:, i, 5] = bm.sum(
                cuh[:, :, 1]*gphix_q2[:, i, :, 0] + cuh[:, :, 0]*gphix_q2[:, i, :, 1], 
                axis=-1) / 2.0  # (NC,)

    # 计算应力
    val = 2*mu + lam
    stress = bm.zeros((NC, NQ, 6), dtype=bm.float64)
    stress[:, :, 0] = val * strain[:, :, 0] + lam * (strain[:, :, 1] + strain[:, :, 2])
    stress[:, :, 1] = val * strain[:, :, 1] + lam * (strain[:, :, 2] + strain[:, :, 0])
    stress[:, :, 2] = val * strain[:, :, 2] + lam * (strain[:, :, 0] + strain[:, :, 1])
    stress[:, :, 3] = 2*mu * strain[:, :, 3]
    stress[:, :, 4] = 2*mu * strain[:, :, 4]
    stress[:, :, 5] = 2*mu * strain[:, :, 5]

    # 获取积分点重心坐标
    import itertools
    tensor_product = itertools.product(bcs_q2[2], bcs_q2[1], bcs_q2[0])
    bc = bm.tensor([[coord for array in combination for coord in array] for combination in tensor_product])

    # 初始化节点累加器和计数器
    nstrain = bm.zeros((NN, 6), dtype=bm.float64)
    nstress = bm.zeros((NN, 6), dtype=bm.float64)
    nc = bm.zeros(NN, dtype=bm.int32)

    # 对每个单元进行处理
    for c in range(NC):
        for q in range(NQ):
            # 使用重心坐标值直接判断最近的顶点
            # bc[q] = [x1, x2, y1, y2, z1, z2]
            nearest_vertex = 0
            if bc[q][0] < bc[q][1]:  # x2 > x1
                nearest_vertex += 4
            if bc[q][2] < bc[q][3]:  # y2 > y1
                nearest_vertex += 2
            if bc[q][4] < bc[q][5]:  # z2 > z1
                nearest_vertex += 1
            
            # 获取最近节点的全局编号
            global_vertex = cell[c, nearest_vertex]
            
            # 贡献应变和应力
            nstrain[global_vertex] += strain[c, q]
            nstress[global_vertex] += stress[c, q]
            nc[global_vertex] += 1

    # 取平均值
    for i in range(6):
        nstrain[:, i] /= bm.maximum(nc, 1) 
        nstress[:, i] /= bm.maximum(nc, 1)

    return strain, stress, nstrain, nstress

def compute_equivalent_strain(strain, nu):
    exx = strain[..., 0, 0]
    eyy = strain[..., 1, 1]
    ezz = strain[..., 2, 2]
    gamma_xy = strain[..., 0, 1]
    gamma_yz = strain[..., 1, 2]
    gamma_xz = strain[..., 0, 2]
    
    d1 = exx - eyy
    d2 = eyy - ezz
    d3 = ezz - exx
    
    equiv_strain = (d1**2 + d2**2 + d3**2 + 6.0 * (gamma_xy**2 + gamma_yz**2 + gamma_xz**2))
    
    # equiv_strain = bm.sqrt(equiv_strain / 2.0) / (1.0 + nu)
    equiv_strain = bm.sqrt(equiv_strain / 2.0) / (1.0)
    
    return equiv_strain

def compute_equivalent_stress(stress, nu):
    sxx = stress[..., 0, 0]
    syy = stress[..., 1, 1]
    szz = stress[..., 2, 2]
    sxy = stress[..., 0, 1]
    syz = stress[..., 1, 2]
    sxz = stress[..., 0, 2]
    
    d1 = sxx - syy
    d2 = syy - szz
    d3 = szz - sxx
    
    equiv_stress = (d1**2 + d2**2 + d3**2 + 6.0 * (sxy**2 + syz**2 + sxz**2))

    equiv_stress = bm.sqrt(equiv_stress / 2.0)
    
    return equiv_stress

with open('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/external_gear_data_part.pkl', 'rb') as f:
    data = pickle.load(f)

external_gear = data['external_gear']
hex_mesh = data['hex_mesh']
helix_node = data['helix_node']
is_inner_node = data['is_inner_node']

t = timer(f"总计算时间")
next(t)

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
# load_values = 100 * bm.array([131.0, 131.0], dtype=bm.float64) # (2, )
load_values = 100 * bm.array([131.0], dtype=bm.float64) # (2, )

n = 1
helix_d = bm.array([(external_gear.d+external_gear.effective_da)/2], dtype=bm.float64)
helix_width = bm.array([0], dtype=bm.float64)
# helix_width = bm.array([external_gear.tooth_width], dtype=bm.float64)
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

P = bm.einsum('p, pd -> pd', load_values, face_normal)  # (1, GD)

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
space = LagrangeFESpace(mesh, p=p, ctype='C')
scalar_gdof = space.number_of_global_dofs()
print(f"gdof: {scalar_gdof}")
cell2dof = space.cell_to_dof()

q = p+1
tensor_space = TensorFunctionSpace(space, shape=(3, -1)) # dof_priority

tgdof = tensor_space.number_of_global_dofs()
print(f"tgdof: {tgdof}")
tldof = tensor_space.number_of_local_dofs()
cell2tdof = tensor_space.cell_to_dof()

load_node_indices = cell[target_cell_idx].flatten() # (1*8, )
# 带有载荷的节点对应的全局自由度编号
dof_indices = bm.stack([scalar_gdof * d + 
                        load_node_indices for d in range(GD)], axis=1)  # (1*8, GD)

phi_loads = []
for bcs in bcs_list:
    phi = tensor_space.basis(bcs)
    phi_loads.append(phi)

phi_loads_array = bm.concatenate(phi_loads, axis=1) # (1, 1, tldof, GD)

FE_load = bm.einsum('pd, cpld -> pl', P, phi_loads_array) # (1, 24)

FE = bm.zeros((NC, tldof), dtype=bm.float64)
FE[target_cell_idx, :] = FE_load[:, :] # (NC, tldof)

F = COOTensor(indices = bm.empty((1, 0), dtype=bm.int32, device=bm.get_device(space)),
            values = bm.empty((0, ), dtype=bm.float64, device=bm.get_device(space)),
            spshape = (tgdof, ))
indices = cell2tdof.reshape(1, -1)
F = F.add(COOTensor(indices, FE.reshape(-1), (tgdof, ))).to_dense() # (tgdof, )

# 从全局载荷向量中提取有载荷节点处的值
F_load_nodes = F[dof_indices] # (1*8, GD)

# 
# load_node_indices = bm.tensor([20049], dtype=bm.int32)
# F = bm.zeros(tgdof, dtype=bm.float64)
# F = bm.set_at(F, 20049, 1000*-3.6292831049748364)

# # 从全局载荷向量中提取有载荷节点处的值
# F_load_nodes = bm.tensor([[1000*-3.6292831049748364, 0, 0]], dtype=bm.float64)

fixed_node_index = bm.where(is_inner_node)[0]
export_to_inp(filename='/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/gear_F1_anays_fealpy.inp', 
              nodes=node, elements=cell, 
              fixed_nodes=fixed_node_index, load_nodes=load_node_indices, loads=F_load_nodes, 
              young_modulus=206e3, poisson_ratio=0.3, density=7.85e-9,
              used_app='ansys', mesh_type='hex')
export_to_inp(filename='/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/gear_F1_abaqus_fealpy.inp', 
              nodes=node, elements=cell, 
              fixed_nodes=fixed_node_index, load_nodes=load_node_indices, loads=F_load_nodes, 
              young_modulus=206e3, poisson_ratio=0.3, density=7.85e-9,
              used_app='abaqus', mesh_type='hex')

E = 206e3
nu = 0.3
lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
linear_elastic_material = LinearElasticMaterial(name='E_nu', 
                                                elastic_modulus=E, poisson_ratio=nu, 
                                                hypo='3D', device=bm.get_device(mesh))
integrator_K = LinearElasticIntegrator(material=linear_elastic_material, q=q)
bform = BilinearForm(tensor_space)
bform.add_integrator(integrator_K)
K = bform.assembly(format='csr')
values = K.values()
K_norm = bm.sqrt(bm.sum(values * values))
F_norm = bm.sqrt(bm.sum(F * F))   
print(f"Matrix norm after dc: {K_norm:.6f}")
print(f"Load vector norm after dc: {F_norm:.6f}")

scalar_is_bd_dof = bm.zeros(scalar_gdof, dtype=bm.bool)
scalar_is_bd_dof[:NN] = is_inner_node
tensor_is_bd_dof = tensor_space.is_boundary_dof(
        threshold=(scalar_is_bd_dof, scalar_is_bd_dof, scalar_is_bd_dof), 
        method='interp')
dbc = DirichletBC(space=tensor_space, 
                    gd=0, 
                    threshold=tensor_is_bd_dof, 
                    method='interp')
K = dbc.apply_matrix(matrix=K, check=True)
uh_bd = bm.zeros(tensor_space.number_of_global_dofs(), dtype=bm.float64, device=bm.get_device(mesh))
# uh_bd, isDDof = tensor_space.boundary_interpolate(gd=0, 
#                                                 uh=uh_bd, 
#                                                 threshold=tensor_is_bd_dof, 
#                                                 method='interp')
isDDof = tensor_is_bd_dof
F = F - K.matmul(uh_bd)
F = bm.set_at(F, isDDof, uh_bd[isDDof])
values = K.values()
K_norm = bm.sqrt(bm.sum(values * values))
F_norm = bm.sqrt(bm.sum(F * F))   
print(f"Matrix norm after dc: {K_norm:.6f}")
print(f"Load vector norm after dc: {F_norm:.6f}")

from fealpy import logger
logger.setLevel('INFO')
uh = tensor_space.function()
# uh[:] = cg(K, F, maxiter=10000, atol=1e-8, rtol=1e-8)
uh[:] = spsolve(K, F, solver="mumps")

# 计算残差向量和范数
residual = K.matmul(uh[:]) - F  # 使用 CSRTensor 的 matmul 方法
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

# 计算方式一：在顶点处计算
strain1, stress1, nstrain1, nstress1 = compute_strain_stress_at_vertices(space, 
                                                                        uh_show, mu, lam)

# 计算方式二：在中心点处计算
strain2, stress2, nstrain2, nstress2 = compute_strain_stress_at_centers(space, 
                                                                        uh_show, mu, lam)

# 计算方式三：在积分点处计算
strain3, stress3, nstrain3, nstress3 = compute_strain_stress_at_quadpoints1(space, 
                                                                        uh_show, mu, lam)

# 计算方式四：在积分点处计算
strain4, stress4, nstrain4, nstress4 = compute_strain_stress_at_quadpoints2(space, 
                                                                        uh_show, mu, lam)

mesh.nodedata['uh'] = uh_show[:]
mesh.nodedata['uh_magnitude'] = uh_magnitude[:]
mesh.nodedata['strain_vertices'] = nstrain1
mesh.nodedata['stress_vertices'] = nstress1
mesh.nodedata['strian_centers'] = nstrain2
mesh.nodedata['stress_centers'] = nstress2
mesh.nodedata['strain_quadpoints1'] = nstrain3
mesh.nodedata['stress_quadpoints1'] = nstress3
mesh.nodedata['strain_quadpoints2'] = nstrain4
mesh.nodedata['stress_quadpoints2'] = nstress4
mesh.to_vtk('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/gear_1F_fealpy_2.vtu')
print("-----------")