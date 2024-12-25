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

bm.set_backend('numpy')

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

def compute_strain_stress_at_quadpoints3(space, uh, B_BBar, D):
    cell2tdof = space.cell_to_dof()
    cuh = uh[cell2tdof]  # (NC, TLDOF) 
    strain5 = bm.einsum('cqil, cl -> cqi', B_BBar, cuh) # (NC, NQ, 6)
    stress5 = bm.einsum('cqij, cqi -> cqj', D, strain5) # (NC, NQ, 6)

    # 初始化节点累加器和计数器
    mesh = space.mesh
    NN = mesh.number_of_nodes()
    nstrain5 = bm.zeros((NN, 6), dtype=bm.float64)
    nstress5 = bm.zeros((NN, 6), dtype=bm.float64)

    map = bm.array([0, 4, 6, 2, 1, 5, 7, 3], dtype=bm.int32)
    strain5 = strain5[:, map, :] # (NC, 8, 6)
    stress5 = stress5[:, map, :] # (NC, 8, 6)
    nc = bm.zeros(NN, dtype=bm.int32)
    bm.add_at(nc, cell, 1)
    for i in range(6):
        bm.add_at(nstrain5[:, i], cell.flatten(), strain5[:, :, i].flatten())
        nstrain5[:, i] /= nc
        bm.add_at(nstress5[:, i], cell.flatten(), stress5[:, :, i].flatten())
        nstress5[:, i] /= nc
    
    return strain5, stress5, nstrain5, nstress5

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

# Ansys 位移结果
u_x_ansys = bm.tensor(np.loadtxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/u_x.txt', 
                                skiprows=1, usecols=1), dtype=bm.float64)  # (NN, )
u_y_ansys = bm.tensor(np.loadtxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/u_y.txt',
                                skiprows=1, usecols=1), dtype=bm.float64)
u_z_ansys = bm.tensor(np.loadtxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/u_z.txt',
                                skiprows=1, usecols=1), dtype=bm.float64)
uh_ansys_show = bm.stack([u_x_ansys, u_y_ansys, u_z_ansys], axis=1)  # (NN, GD)

# Ansys 应变结果     
strain_xx_ansys = bm.tensor(np.loadtxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/e_x.txt',
                                skiprows=1, usecols=1), dtype=bm.float64) # (NN, )
strain_yy_ansys = bm.tensor(np.loadtxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/e_y.txt',
                                skiprows=1, usecols=1), dtype=bm.float64)
strain_zz_ansys = bm.tensor(np.loadtxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/e_z.txt',
                                skiprows=1, usecols=1), dtype=bm.float64)
strain_xy_ansys = bm.tensor(np.loadtxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/e_xy.txt',
                                skiprows=1, usecols=1), dtype=bm.float64)
strain_yz_ansys = bm.tensor(np.loadtxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/e_yz.txt',
                                skiprows=1, usecols=1), dtype=bm.float64)
strain_xz_ansys = bm.tensor(np.loadtxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/e_xz.txt',
                                skiprows=1, usecols=1), dtype=bm.float64)

# Ansys 应力结果
stress_xx_ansys = bm.tensor(np.loadtxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/s_x.txt',
                                skiprows=1, usecols=1), dtype=bm.float64) # (NN, )
stress_yy_ansys = bm.tensor(np.loadtxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/s_y.txt',
                                skiprows=1, usecols=1), dtype=bm.float64)
stress_zz_ansys = bm.tensor(np.loadtxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/s_z.txt',
                                skiprows=1, usecols=1), dtype=bm.float64)
stress_xy_ansys = bm.tensor(np.loadtxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/s_xy.txt',
                                skiprows=1, usecols=1), dtype=bm.float64)
stress_yz_ansys = bm.tensor(np.loadtxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/s_yz.txt',
                                skiprows=1, usecols=1), dtype=bm.float64)
stress_xz_ansys = bm.tensor(np.loadtxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/s_xz.txt',
                                skiprows=1, usecols=1), dtype=bm.float64)

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
q = p+2
space = LagrangeFESpace(mesh, p=p, ctype='C')
scalar_gdof = space.number_of_global_dofs()
print(f"gdof: {scalar_gdof}")
cell2dof = space.cell_to_dof()
tensor_space = TensorFunctionSpace(space, shape=(3, -1)) # dof_priority

# test = cell[29779]

tgdof = tensor_space.number_of_global_dofs()
print(f"tgdof: {tgdof}")
tldof = tensor_space.number_of_local_dofs()
cell2tdof = tensor_space.cell_to_dof()

load_node_indices = cell[target_cell_idx].flatten() # (15*8, )
# 带有载荷的节点对应的全局自由度编号
dof_indices = bm.stack([scalar_gdof * d + 
                        load_node_indices for d in range(GD)], axis=1)  # (15*8, GD)

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

# 从全局载荷向量中提取有载荷节点处的值
F_load_nodes = F[dof_indices] # (15*8, GD)

fixed_node_index = bm.where(is_inner_node)[0]
export_to_inp(filename='/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/inp/external_gear_abaqus.inp', 
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
integrator_K0 = LinearElasticIntegrator(material=linear_elastic_material, 
                                        q=q, method='voigt')
KE = integrator_K0.voigt_assembly(tensor_space)
KE0 = KE[0]
KE29779 = KE[29779]
np.savetxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/KE0.csv', KE0.round(4), delimiter=',', fmt='%s')
np.savetxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/KE29779.csv', KE29779.round(4), delimiter=',', fmt='%s')

integrator_K = LinearElasticIntegrator(material=linear_elastic_material, 
                                       q=q, method='C3D8_BBar')
KE_bbar = integrator_K.c3d8_bbar_assembly(tensor_space)
KE_bbar0 = KE_bbar[0]
KE_bbar29779 = KE_bbar[29779]
np.savetxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/KE_bbar0.csv', KE_bbar0.round(4), delimiter=',', fmt='%s')
np.savetxt('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/txt/KE_bbar29779.csv', KE_bbar29779.round(4), delimiter=',', fmt='%s')
integrator_K.keep_data(True)
_, _, D, B_BBar = integrator_K.fetch_c3d8_bbar_assembly(tensor_space)
bform = BilinearForm(tensor_space)
bform.add_integrator(integrator_K)
K = bform.assembly(format='csr')
values = K.values()
K_norm = bm.sqrt(bm.sum(values * values))
F_norm = bm.sqrt(bm.sum(F * F))   
print(f"Matrix norm after dc: {K_norm:.6f}")
print(f"Load vector norm after dc: {F_norm:.6f}")

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
uh_bd = bm.zeros(tensor_space.number_of_global_dofs(), dtype=bm.float64, device=bm.get_device(mesh))
uh_bd, isDDof = tensor_space.boundary_interpolate(gd=bm.zeros(tgdof), 
                                                uh=uh_bd, 
                                                threshold=tensor_is_bd_dof, 
                                                method='interp')
# isDDof = tensor_is_bd_dof
# 处理载荷
F = F - K.matmul(uh_bd)
F = bm.set_at(F, isDDof, uh_bd[isDDof])
# 处理刚度
K = dbc.apply_matrix(matrix=K, check=True)
values = K.values()
K_norm = bm.sqrt(bm.sum(values * values))
F_norm = bm.sqrt(bm.sum(F * F))   
print(f"Matrix norm  after dc: {K_norm:.6f}")
print(f"Load vector norm after dc: {F_norm:.6f}")

from fealpy import logger
logger.setLevel('INFO')
uh = tensor_space.function()
# uh[:] = cg(K, F, maxiter=10000, atol=1e-8, rtol=1e-8)
uh[:] = spsolve(K, F, solver="mumps")

uh_ansys = tensor_space.function()
if tensor_space.dof_priority:
    uh_ansys[:] = uh_ansys_show.T.flatten()
else:
    uh_ansys[:] = uh_ansys_show.flatten()

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

# 位移误差
error_u_x = uh_x - u_x_ansys # (NN, )
error_u_y = uh_y - u_y_ansys
error_u_z = uh_z - u_z_ansys
relative_error_u_x = bm.linalg.norm(error_u_x) / (bm.linalg.norm(u_x_ansys)+bm.linalg.norm(uh_x))
relative_error_u_y = bm.linalg.norm(error_u_y) / (bm.linalg.norm(u_y_ansys)+bm.linalg.norm(uh_y))
relative_error_u_z = bm.linalg.norm(error_u_z) / (bm.linalg.norm(u_z_ansys)+bm.linalg.norm(uh_z))
print(f"Relative error_u_x: {relative_error_u_x:.12e}")
print(f"Relative error_u_y: {relative_error_u_y:.12e}")
print(f"Relative error_u_z: {relative_error_u_z:.12e}")

uh_magnitude = bm.linalg.norm(uh_show, axis=1)

mesh.nodedata['uh'] = uh_show[:]
mesh.nodedata['uh_magnitude'] = uh_magnitude[:]

mesh.to_vtk('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/vtu/external_gear_fealpy.vtu')

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

# 计算方式五：使用 B-Bar 修正后的 B 计算
strain5, stress5, nstrain5, nstress5 = compute_strain_stress_at_quadpoints3(tensor_space,
                                                                        uh, B_BBar, D)


# 计算方式一的应变误差
error_e_x_1 = nstrain1[:, 0] - strain_xx_ansys # (NN, )
error_e_y_1 = nstrain1[:, 1] - strain_yy_ansys
error_e_z_1 = nstrain1[:, 2] - strain_zz_ansys
error_e_xy_1 = nstrain1[:, 3] - strain_xy_ansys
error_e_yz_1 = nstrain1[:, 4] - strain_yz_ansys
error_e_xz_1 = nstrain1[:, 5] - strain_xz_ansys
relative_error_e_x_1 = bm.linalg.norm(error_e_x_1) / \
                        (bm.linalg.norm(strain_xx_ansys)+bm.linalg.norm(nstrain1[:, 0]))
print(f"Relative error_e_x_1: {relative_error_e_x_1:.12e}")
relative_error_e_y_1 = bm.linalg.norm(error_e_y_1) / \
                        (bm.linalg.norm(strain_yy_ansys)+bm.linalg.norm(nstrain1[:, 1]))
print(f"Relative error_e_y_1: {relative_error_e_y_1:.12e}")
relative_error_e_z_1 = bm.linalg.norm(error_e_z_1) / \
                        (bm.linalg.norm(strain_zz_ansys)+bm.linalg.norm(nstrain1[:, 2]))
print(f"Relative error_e_z_1: {relative_error_e_z_1:.12e}")
relative_error_e_xy_1 = bm.linalg.norm(error_e_xy_1) / \
                        (bm.linalg.norm(strain_xy_ansys)+bm.linalg.norm(nstrain1[:, 3]))
print(f"Relative error_e_xy_1: {relative_error_e_xy_1:.12e}")
relative_error_e_yz_1 = bm.linalg.norm(error_e_yz_1) / \
                        (bm.linalg.norm(strain_yz_ansys)+bm.linalg.norm(nstrain1[:, 4]))
print(f"Relative error_e_yz_1: {relative_error_e_yz_1:.12e}")
relative_error_e_xz_1 = bm.linalg.norm(error_e_xz_1) / \
                        (bm.linalg.norm(strain_xz_ansys)+bm.linalg.norm(nstrain1[:, 5]))
print(f"Relative error_e_xz_1: {relative_error_e_xz_1:.12e}")
# 计算方式一的应力误差
error_s_x_1 = nstress1[:, 0] - stress_xx_ansys # (NN, )
error_s_y_1 = nstress1[:, 1] - stress_yy_ansys
error_s_z_1 = nstress1[:, 2] - stress_zz_ansys
error_s_xy_1 = nstress1[:, 3] - stress_xy_ansys
error_s_yz_1 = nstress1[:, 4] - stress_yz_ansys
error_s_xz_1 = nstress1[:, 5] - stress_xz_ansys
relative_error_s_x_1 = bm.linalg.norm(error_s_x_1) / \
                        (bm.linalg.norm(stress_xx_ansys)+bm.linalg.norm(nstress1[:, 0]))
print(f"Relative error_s_x_1: {relative_error_s_x_1:.12e}")
relative_error_s_y_1 = bm.linalg.norm(error_s_y_1) / \
                        (bm.linalg.norm(stress_yy_ansys)+bm.linalg.norm(nstress1[:, 1]))
print(f"Relative error_s_y_1: {relative_error_s_y_1:.12e}")
relative_error_s_z_1 = bm.linalg.norm(error_s_z_1) / \
                        (bm.linalg.norm(stress_zz_ansys)+bm.linalg.norm(nstress1[:, 2]))
print(f"Relative error_s_z_1: {relative_error_s_z_1:.12e}")
relative_error_s_xy_1 = bm.linalg.norm(error_s_xy_1) / \
                        (bm.linalg.norm(stress_xy_ansys)+bm.linalg.norm(nstress1[:, 3]))
print(f"Relative error_s_xy_1: {relative_error_s_xy_1:.12e}")
relative_error_s_yz_1 = bm.linalg.norm(error_s_yz_1) / \
                        (bm.linalg.norm(stress_yz_ansys)+bm.linalg.norm(nstress1[:, 4]))
print(f"Relative error_s_yz_1: {relative_error_s_yz_1:.12e}")
relative_error_s_xz_1 = bm.linalg.norm(error_s_xz_1) / \
                        (bm.linalg.norm(stress_xz_ansys)+bm.linalg.norm(nstress1[:, 5]))
print(f"Relative error_s_xz_1: {relative_error_s_xz_1:.12e}")

# 计算方式二的应变误差
error_e_x_2 = nstrain2[:, 0] - strain_xx_ansys # (NN, )
error_e_y_2 = nstrain2[:, 1] - strain_yy_ansys
error_e_z_2 = nstrain2[:, 2] - strain_zz_ansys
error_e_xy_2 = nstrain2[:, 3] - strain_xy_ansys
error_e_yz_2 = nstrain2[:, 4] - strain_yz_ansys
error_e_xz_2 = nstrain2[:, 5] - strain_xz_ansys
relative_error_e_x_2 = bm.linalg.norm(error_e_x_2) / \
                        (bm.linalg.norm(strain_xx_ansys)+bm.linalg.norm(nstrain2[:, 0]))
print(f"Relative error_e_x_2: {relative_error_e_x_2:.12e}")
relative_error_e_y_2 = bm.linalg.norm(error_e_y_2) / \
                        (bm.linalg.norm(strain_yy_ansys)+bm.linalg.norm(nstrain2[:, 1]))
print(f"Relative error_e_y_2: {relative_error_e_y_2:.12e}")
relative_error_e_z_2 = bm.linalg.norm(error_e_z_2) / \
                        (bm.linalg.norm(strain_zz_ansys)+bm.linalg.norm(nstrain2[:, 2]))
print(f"Relative error_e_z_2: {relative_error_e_z_2:.12e}")
relative_error_e_xy_2 = bm.linalg.norm(error_e_xy_2) / \
                        (bm.linalg.norm(strain_xy_ansys)+bm.linalg.norm(nstrain2[:, 3]))
print(f"Relative error_e_xy_2: {relative_error_e_xy_2:.12e}")
relative_error_e_yz_2 = bm.linalg.norm(error_e_yz_2) / \
                        (bm.linalg.norm(strain_yz_ansys)+bm.linalg.norm(nstrain2[:, 4]))
print(f"Relative error_e_yz_2: {relative_error_e_yz_2:.12e}")
relative_error_e_xz_2 = bm.linalg.norm(error_e_xz_2) / \
                        (bm.linalg.norm(strain_xz_ansys)+bm.linalg.norm(nstrain2[:, 5]))
print(f"Relative error_e_xz_2: {relative_error_e_xz_2:.12e}")
# 计算方式二的应力误差
error_s_x_2 = nstress2[:, 0] - stress_xx_ansys # (NN, )
error_s_y_2 = nstress2[:, 1] - stress_yy_ansys
error_s_z_2 = nstress2[:, 2] - stress_zz_ansys
error_s_xy_2 = nstress2[:, 3] - stress_xy_ansys
error_s_yz_2 = nstress2[:, 4] - stress_yz_ansys
error_s_xz_2 = nstress2[:, 5] - stress_xz_ansys
relative_error_s_x_2 = bm.linalg.norm(error_s_x_2) / \
                        (bm.linalg.norm(stress_xx_ansys)+bm.linalg.norm(nstress2[:, 0]))
print(f"Relative error_s_x_2: {relative_error_s_x_2:.12e}")
relative_error_s_y_2 = bm.linalg.norm(error_s_y_2) / \
                        (bm.linalg.norm(stress_yy_ansys)+bm.linalg.norm(nstress2[:, 1]))
print(f"Relative error_s_y_2: {relative_error_s_y_2:.12e}")
relative_error_s_z_2 = bm.linalg.norm(error_s_z_2) / \
                        (bm.linalg.norm(stress_zz_ansys)+bm.linalg.norm(nstress2[:, 2]))
print(f"Relative error_s_z_2: {relative_error_s_z_2:.12e}")
relative_error_s_xy_2 = bm.linalg.norm(error_s_xy_2) / \
                        (bm.linalg.norm(stress_xy_ansys)+bm.linalg.norm(nstress2[:, 3]))
print(f"Relative error_s_xy_2: {relative_error_s_xy_2:.12e}")
relative_error_s_yz_2 = bm.linalg.norm(error_s_yz_2) / \
                        (bm.linalg.norm(stress_yz_ansys)+bm.linalg.norm(nstress2[:, 4]))
print(f"Relative error_s_yz_2: {relative_error_s_yz_2:.12e}")
relative_error_s_xz_2 = bm.linalg.norm(error_s_xz_2) / \
                        (bm.linalg.norm(stress_xz_ansys)+bm.linalg.norm(nstress2[:, 5]))
print(f"Relative error_s_xz_2: {relative_error_s_xz_2:.12e}")

# 计算方式三的应变误差
error_e_x_3 = nstrain3[:, 0] - strain_xx_ansys # (NN, )
error_e_y_3 = nstrain3[:, 1] - strain_yy_ansys
error_e_z_3 = nstrain3[:, 2] - strain_zz_ansys
error_e_xy_3 = nstrain3[:, 3] - strain_xy_ansys
error_e_yz_3 = nstrain3[:, 4] - strain_yz_ansys
error_e_xz_3 = nstrain3[:, 5] - strain_xz_ansys
relative_error_e_x_3 = bm.linalg.norm(error_e_x_3) / \
                        (bm.linalg.norm(strain_xx_ansys)+bm.linalg.norm(nstrain3[:, 0]))
print(f"Relative error_e_x_3: {relative_error_e_x_3:.12e}")
relative_error_e_y_3 = bm.linalg.norm(error_e_y_3) / \
                        (bm.linalg.norm(strain_yy_ansys)+bm.linalg.norm(nstrain3[:, 1]))
print(f"Relative error_e_y_3: {relative_error_e_y_3:.12e}")
relative_error_e_z_3 = bm.linalg.norm(error_e_z_3) / \
                        (bm.linalg.norm(strain_zz_ansys)+bm.linalg.norm(nstrain3[:, 2]))
print(f"Relative error_e_z_3: {relative_error_e_z_3:.12e}")
relative_error_e_xy_3 = bm.linalg.norm(error_e_xy_3) / \
                        (bm.linalg.norm(strain_xy_ansys)+bm.linalg.norm(nstrain3[:, 3]))
print(f"Relative error_e_xy_3: {relative_error_e_xy_3:.12e}")
relative_error_e_yz_3 = bm.linalg.norm(error_e_yz_3) / \
                        (bm.linalg.norm(strain_yz_ansys)+bm.linalg.norm(nstrain3[:, 4]))
print(f"Relative error_e_yz_3: {relative_error_e_yz_3:.12e}")
relative_error_e_xz_3 = bm.linalg.norm(error_e_xz_3) / \
                        (bm.linalg.norm(strain_xz_ansys)+bm.linalg.norm(nstrain3[:, 5]))
print(f"Relative error_e_xz_3: {relative_error_e_xz_3:.12e}")
# 计算方式三的应力误差
error_s_x_3 = nstress3[:, 0] - stress_xx_ansys # (NN, )
error_s_y_3 = nstress3[:, 1] - stress_yy_ansys
error_s_z_3 = nstress3[:, 2] - stress_zz_ansys
error_s_xy_3 = nstress3[:, 3] - stress_xy_ansys
error_s_yz_3 = nstress3[:, 4] - stress_yz_ansys
error_s_xz_3 = nstress3[:, 5] - stress_xz_ansys
relative_error_s_x_3 = bm.linalg.norm(error_s_x_3) / \
                        (bm.linalg.norm(stress_xx_ansys)+bm.linalg.norm(nstress3[:, 0]))
print(f"Relative error_s_x_3: {relative_error_s_x_3:.12e}")
relative_error_s_y_3 = bm.linalg.norm(error_s_y_3) / \
                        (bm.linalg.norm(stress_yy_ansys)+bm.linalg.norm(nstress3[:, 1]))
print(f"Relative error_s_y_3: {relative_error_s_y_3:.12e}")
relative_error_s_z_3 = bm.linalg.norm(error_s_z_3) / \
                        (bm.linalg.norm(stress_zz_ansys)+bm.linalg.norm(nstress3[:, 2]))
print(f"Relative error_s_z_3: {relative_error_s_z_3:.12e}")
relative_error_s_xy_3 = bm.linalg.norm(error_s_xy_3) / \
                        (bm.linalg.norm(stress_xy_ansys)+bm.linalg.norm(nstress3[:, 3]))
print(f"Relative error_s_xy_3: {relative_error_s_xy_3:.12e}")
relative_error_s_yz_3 = bm.linalg.norm(error_s_yz_3) / \
                        (bm.linalg.norm(stress_yz_ansys)+bm.linalg.norm(nstress3[:, 4]))
print(f"Relative error_s_yz_3: {relative_error_s_yz_3:.12e}")
relative_error_s_xz_3 = bm.linalg.norm(error_s_xz_3) / \
                        (bm.linalg.norm(stress_xz_ansys)+bm.linalg.norm(nstress3[:, 5]))
print(f"Relative error_s_xz_3: {relative_error_s_xz_3:.12e}")

# 计算方式五的应变误差
error_e_x_5 = nstrain5[:, 0] - strain_xx_ansys # (NN, )
error_e_y_5 = nstrain5[:, 1] - strain_yy_ansys
error_e_z_5 = nstrain5[:, 2] - strain_zz_ansys
error_e_xy_5 = nstrain5[:, 3] - strain_xy_ansys
error_e_yz_5 = nstrain5[:, 4] - strain_yz_ansys
error_e_xz_5 = nstrain5[:, 5] - strain_xz_ansys
relative_error_e_x_5 = bm.linalg.norm(error_e_x_5) / \
                        (bm.linalg.norm(strain_xx_ansys)+bm.linalg.norm(nstrain5[:, 0]))
print(f"Relative error_e_x_5: {relative_error_e_x_5:.12e}")
relative_error_e_y_5 = bm.linalg.norm(error_e_y_5) / \
                        (bm.linalg.norm(strain_yy_ansys)+bm.linalg.norm(nstrain5[:, 1]))
print(f"Relative error_e_y_5: {relative_error_e_y_5:.12e}")
relative_error_e_z_5 = bm.linalg.norm(error_e_z_5) / \
                        (bm.linalg.norm(strain_zz_ansys)+bm.linalg.norm(nstrain5[:, 2]))
print(f"Relative error_e_z_5: {relative_error_e_z_5:.12e}")
relative_error_e_xy_5 = bm.linalg.norm(error_e_xy_5) / \
                        (bm.linalg.norm(strain_xy_ansys)+bm.linalg.norm(nstrain5[:, 3]))
print(f"Relative error_e_xy_5: {relative_error_e_xy_5:.12e}")
relative_error_e_yz_5 = bm.linalg.norm(error_e_yz_5) / \
                        (bm.linalg.norm(strain_yz_ansys)+bm.linalg.norm(nstrain5[:, 4]))
print(f"Relative error_e_yz_5: {relative_error_e_yz_5:.12e}")
relative_error_e_xz_5 = bm.linalg.norm(error_e_xz_5) / \
                        (bm.linalg.norm(strain_xz_ansys)+bm.linalg.norm(nstrain5[:, 5]))
print(f"Relative error_e_xz_5: {relative_error_e_xz_5:.12e}")
# 计算方式五的应力误差
error_s_x_5 = nstress5[:, 0] - stress_xx_ansys # (NN, )
error_s_y_5 = nstress5[:, 1] - stress_yy_ansys
error_s_z_5 = nstress5[:, 2] - stress_zz_ansys
error_s_xy_5 = nstress5[:, 3] - stress_xy_ansys
error_s_yz_5 = nstress5[:, 4] - stress_yz_ansys
error_s_xz_5 = nstress5[:, 5] - stress_xz_ansys
relative_error_s_x_5 = bm.linalg.norm(error_s_x_5) / \
                        (bm.linalg.norm(stress_xx_ansys)+bm.linalg.norm(nstress5[:, 0]))
print(f"Relative error_s_x_5: {relative_error_s_x_5:.12e}")
relative_error_s_y_5 = bm.linalg.norm(error_s_y_5) / \
                        (bm.linalg.norm(stress_yy_ansys)+bm.linalg.norm(nstress5[:, 1]))
print(f"Relative error_s_y_5: {relative_error_s_y_5:.12e}")
relative_error_s_z_5 = bm.linalg.norm(error_s_z_5) / \
                        (bm.linalg.norm(stress_zz_ansys)+bm.linalg.norm(nstress5[:, 2]))
print(f"Relative error_s_z_5: {relative_error_s_z_5:.12e}")
relative_error_s_xy_5 = bm.linalg.norm(error_s_xy_5) / \
                        (bm.linalg.norm(stress_xy_ansys)+bm.linalg.norm(nstress5[:, 3]))
print(f"Relative error_s_xy_5: {relative_error_s_xy_5:.12e}")
relative_error_s_yz_5 = bm.linalg.norm(error_s_yz_5) / \
                        (bm.linalg.norm(stress_yz_ansys)+bm.linalg.norm(nstress5[:, 4]))
print(f"Relative error_s_yz_5: {relative_error_s_yz_5:.12e}")
relative_error_s_xz_5 = bm.linalg.norm(error_s_xz_5) / \
                        (bm.linalg.norm(stress_xz_ansys)+bm.linalg.norm(nstress5[:, 5]))
print(f"Relative error_s_xz_5: {relative_error_s_xz_5:.12e}")

# 节点应变张量和 Ansys 中节点应变张量的误差
error_s_xx = strain1[:, 0] - strain_xx_ansys
relative_error_s_xx = bm.linalg.norm(error_s_xx) / (bm.linalg.norm(nodal_strain_xx)+bm.linalg.norm(strain_xx_ansys))
print("节点应变张量和 Ansys 中节点应变张量的相对误差")
print(f"Relative error_s_xx: {relative_error_s_xx:.12e}")
print(f"Relative error_s_yy: {relative_error_s_yy:.12e}")
print(f"Relative error_s_zz: {relative_error_s_zz:.12e}")
print(f"Relative error_s_xy: {relative_error_s_xy:.12e}")
print(f"Relative error_s_yz: {relative_error_s_yz:.12e}")
print(f"Relative error_s_xz: {relative_error_s_xz:.12e}")

# # 节点处的位移梯度
# # 方法一
# bcs1 = (bm.tensor([[1, 0]], dtype=bm.float64), 
#        bm.tensor([[1, 0]], dtype=bm.float64), 
#        bm.tensor([[1, 0]], dtype=bm.float64))
# p1 = mesh.bc_to_point(bc=bcs1)
# bcs2 = (bm.tensor([[1, 0]], dtype=bm.float64), 
#        bm.tensor([[1, 0]], dtype=bm.float64), 
#        bm.tensor([[0, 1]], dtype=bm.float64))
# p2 = mesh.bc_to_point(bc=bcs2)
# bcs3 = (bm.tensor([[1, 0]], dtype=bm.float64), 
#        bm.tensor([[0, 1]], dtype=bm.float64), 
#        bm.tensor([[1, 0]], dtype=bm.float64))
# p3 = mesh.bc_to_point(bc=bcs3)
# bcs4 = (bm.tensor([[1, 0]], dtype=bm.float64), 
#        bm.tensor([[0, 1]], dtype=bm.float64), 
#        bm.tensor([[0, 1]], dtype=bm.float64))
# p4 = mesh.bc_to_point(bc=bcs4)
# bcs5 = (bm.tensor([[0, 1]], dtype=bm.float64), 
#        bm.tensor([[1, 0]], dtype=bm.float64), 
#        bm.tensor([[1, 0]], dtype=bm.float64))
# p5 = mesh.bc_to_point(bc=bcs5)
# bcs6 = (bm.tensor([[0, 1]], dtype=bm.float64), 
#        bm.tensor([[1, 0]], dtype=bm.float64), 
#        bm.tensor([[0, 1]], dtype=bm.float64))
# p6 = mesh.bc_to_point(bc=bcs6)
# bcs7 = (bm.tensor([[0, 1]], dtype=bm.float64), 
#        bm.tensor([[0, 1]], dtype=bm.float64), 
#        bm.tensor([[1, 0]], dtype=bm.float64))
# p7 = mesh.bc_to_point(bc=bcs7)
# bcs8 = (bm.tensor([[0, 1]], dtype=bm.float64), 
#        bm.tensor([[0, 1]], dtype=bm.float64), 
#        bm.tensor([[0, 1]], dtype=bm.float64))
# p8 = mesh.bc_to_point(bc=bcs8)
# tgphi1 = tensor_space.grad_basis(bcs1) # (NC, 1, tldof, GD, GD)
# tgphi2 = tensor_space.grad_basis(bcs2) # (NC, 1, tldof, GD, GD)
# tgphi3 = tensor_space.grad_basis(bcs3) # (NC, 1, tldof, GD, GD)
# tgphi4 = tensor_space.grad_basis(bcs4) # (NC, 1, tldof, GD, GD)
# tgphi5 = tensor_space.grad_basis(bcs5) # (NC, 1, tldof, GD, GD)
# tgphi6 = tensor_space.grad_basis(bcs6) # (NC, 1, tldof, GD, GD)
# tgphi7 = tensor_space.grad_basis(bcs7) # (NC, 1, tldof, GD, GD)
# tgphi8 = tensor_space.grad_basis(bcs8) # (NC, 1, tldof, GD, GD)
# tgphi = bm.concatenate([tgphi1, tgphi2, tgphi3, tgphi4, 
#                         tgphi5, tgphi6, tgphi7, tgphi8], axis=1) # (NC, 8, tldof, GD, GD)
# # 方法二
# # 插值点的多重指标
# shape = (p+1, p+1, p+1)
# mi = bm.arange(p+1, device=bm.get_device(cell))
# multiIndex0 = bm.broadcast_to(mi[:, None, None], shape).reshape(-1, 1)
# multiIndex1 = bm.broadcast_to(mi[None, :, None], shape).reshape(-1, 1)
# multiIndex2 = bm.broadcast_to(mi[None, None, :], shape).reshape(-1, 1)
# multiIndex = bm.concatenate([multiIndex0, multiIndex1, multiIndex2], axis=-1)
# # 多重指标的映射
# multiIndex_map = mesh.multi_index_matrix(p=p, etype=1)
# # 插值点的重心坐标
# barycenter = multiIndex_map[multiIndex].astype(bm.float64)
# bcs = (barycenter[:, 0, :], barycenter[:, 1, :], barycenter[:, 2, :])

# tgphi_list = []
# for i in range(barycenter.shape[0]):
#     bc_i = (
#         bcs[0][i].reshape(1, -1),  # (1, 2)
#         bcs[1][i].reshape(1, -1),  # (1, 2)
#         bcs[2][i].reshape(1, -1)   # (1, 2)
#     )
#     tgphi_i = tensor_space.grad_basis(bc_i)  # (NC, 1, tldof, GD, GD)
#     tgphi_list.append(tgphi_i)
# tgphi = bm.concatenate(tgphi_list, axis=1)  # (NC, 8, tldof, GD, GD)


tgrad = bm.einsum('cqimn, ci -> cqmn', tgphi, uh_cell)      # (NC, 8, GD, GD)

# 应变张量
strain = (tgrad + bm.transpose(tgrad, (0, 1, 3, 2))) / 2 # (NC, 8, GD, GD)
strain_xx = strain[..., 0, 0] # (NC, 8)
strain_yy = strain[..., 1, 1] # (NC, 8)
strain_zz = strain[..., 2, 2] # (NC, 8)
strain_xy = strain[..., 0, 1] # (NC, 8)
strain_yz = strain[..., 1, 2] # (NC, 8)
strain_xz = strain[..., 0, 2] # (NC, 8)

# 基于 Ansys 中的位移的应变张量
tgrad_ansys = bm.einsum('cqimn, ci -> cqmn', tgphi, uh_cell_ansys)      # (NC, 8, GD, GD)
strain_ansys_u = (tgrad_ansys + bm.transpose(tgrad_ansys, (0, 1, 3, 2))) / 2 # (NC, 8, GD, GD)
strain_xx_ansys_u = strain_ansys_u[..., 0, 0] # (NC, 8)
strain_yy_ansys_u = strain_ansys_u[..., 1, 1] # (NC, 8)
strain_zz_ansys_u = strain_ansys_u[..., 2, 2] # (NC, 8)
strain_xy_ansys_u = strain_ansys_u[..., 0, 1] # (NC, 8)
strain_yz_ansys_u = strain_ansys_u[..., 1, 2] # (NC, 8)
strain_xz_ansys_u = strain_ansys_u[..., 0, 2] # (NC, 8)

# 节点计算次数
num_count = bm.zeros(NN, dtype=bm.int32)
bm.add_at(num_count, cell2dof.flatten(), 1)

# 节点应变张量
nodal_strain_xx = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_strain_xx, cell2dof.flatten(), strain_xx.flatten())
nodal_strain_xx /= num_count
nodal_strain_yy = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_strain_yy, cell2dof.flatten(), strain_yy.flatten())
nodal_strain_yy /= num_count
nodal_strain_zz = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_strain_zz, cell2dof.flatten(), strain_zz.flatten())
nodal_strain_zz /= num_count
nodal_strain_xy = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_strain_xy, cell2dof.flatten(), strain_xy.flatten())
nodal_strain_xy /= num_count
nodal_strain_yz = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_strain_yz, cell2dof.flatten(), strain_yz.flatten())
nodal_strain_yz /= num_count
nodal_strain_xz = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_strain_xz, cell2dof.flatten(), strain_xz.flatten())
nodal_strain_xz /= num_count

# 基于 Ansys 中的位移的节点应变张量
nodal_strain_xx_ansys_u = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_strain_xx_ansys_u, cell2dof.flatten(), strain_xx_ansys_u.flatten())
nodal_strain_xx_ansys_u /= num_count
nodal_strain_yy_ansys_u = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_strain_yy_ansys_u, cell2dof.flatten(), strain_yy_ansys_u.flatten())
nodal_strain_yy_ansys_u /= num_count
nodal_strain_zz_ansys_u = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_strain_zz_ansys_u, cell2dof.flatten(), strain_zz_ansys_u.flatten())
nodal_strain_zz_ansys_u /= num_count
nodal_strain_xy_ansys_u = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_strain_xy_ansys_u, cell2dof.flatten(), strain_xy_ansys_u.flatten())
nodal_strain_xy_ansys_u /= num_count
nodal_strain_yz_ansys_u = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_strain_yz_ansys_u, cell2dof.flatten(), strain_yz_ansys_u.flatten())
nodal_strain_yz_ansys_u /= num_count
nodal_strain_xz_ansys_u = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_strain_xz_ansys_u, cell2dof.flatten(), strain_xz_ansys_u.flatten())
nodal_strain_xz_ansys_u /= num_count

# 基于 Ansys 中位移的节点应变张量和 Ansys 中节点应变张量的误差
error_s_xx_u = nodal_strain_xx_ansys_u - strain_xx_ansys
error_s_yy_u = nodal_strain_yy_ansys_u - strain_yy_ansys
error_s_zz_u = nodal_strain_zz_ansys_u - strain_zz_ansys
error_s_xy_u = nodal_strain_xy_ansys_u - strain_xy_ansys
error_s_yz_u = nodal_strain_yz_ansys_u - strain_yz_ansys
error_s_xz_u = nodal_strain_xz_ansys_u - strain_xz_ansys
relative_error_s_xx_u = bm.linalg.norm(error_s_xx_u) / (
                            bm.linalg.norm(nodal_strain_xx_ansys_u)+bm.linalg.norm(strain_xx_ansys))
relative_error_s_yy_u = bm.linalg.norm(error_s_yy_u) / (
                            bm.linalg.norm(nodal_strain_yy_ansys_u)+bm.linalg.norm(strain_yy_ansys))
relative_error_s_zz_u = bm.linalg.norm(error_s_zz_u) / (
                            bm.linalg.norm(nodal_strain_zz_ansys_u)+bm.linalg.norm(strain_zz_ansys))
relative_error_s_xy_u = bm.linalg.norm(error_s_xy_u) / (
                            bm.linalg.norm(nodal_strain_xy_ansys_u)+bm.linalg.norm(strain_xy_ansys))
relative_error_s_yz_u = bm.linalg.norm(error_s_yz_u) / (
                            bm.linalg.norm(nodal_strain_yz_ansys_u)+bm.linalg.norm(strain_yz_ansys))
relative_error_s_xz_u = bm.linalg.norm(error_s_xz_u) / (
                            bm.linalg.norm(nodal_strain_xz_ansys_u)+bm.linalg.norm(strain_xz_ansys))
print("基于 Ansys 中位移的节点应变张量和 Ansys 中节点应变张量的相对误差")
print(f"Relative error_s_xx_1: {relative_error_s_xx_u:.12e}")
print(f"Relative error_s_yy_1: {relative_error_s_yy_u:.12e}")
print(f"Relative error_s_zz_1: {relative_error_s_zz_u:.12e}")
print(f"Relative error_s_xy_1: {relative_error_s_xy_u:.12e}")
print(f"Relative error_s_yz_1: {relative_error_s_yz_u:.12e}")
print(f"Relative error_s_xz_1: {relative_error_s_xz_u:.12e}")

# 节点应变张量和 Ansys 中节点应变张量的误差
error_s_xx = nodal_strain_xx - strain_xx_ansys
error_s_yy = nodal_strain_yy - strain_yy_ansys
error_s_zz = nodal_strain_zz - strain_zz_ansys
error_s_xy = nodal_strain_xy - strain_xy_ansys
error_s_yz = nodal_strain_yz - strain_yz_ansys
error_s_xz = nodal_strain_xz - strain_xz_ansys
relative_error_s_xx = bm.linalg.norm(error_s_xx) / (bm.linalg.norm(nodal_strain_xx)+bm.linalg.norm(strain_xx_ansys))
relative_error_s_yy = bm.linalg.norm(error_s_yy) / (bm.linalg.norm(nodal_strain_yy)+bm.linalg.norm(strain_yy_ansys))
relative_error_s_zz = bm.linalg.norm(error_s_zz) / (bm.linalg.norm(nodal_strain_zz)+bm.linalg.norm(strain_zz_ansys))
relative_error_s_xy = bm.linalg.norm(error_s_xy) / (bm.linalg.norm(nodal_strain_xy)+bm.linalg.norm(strain_xy_ansys))
relative_error_s_yz = bm.linalg.norm(error_s_yz) / (bm.linalg.norm(nodal_strain_yz)+bm.linalg.norm(strain_yz_ansys))
relative_error_s_xz = bm.linalg.norm(error_s_xz) / (bm.linalg.norm(nodal_strain_xz)+bm.linalg.norm(strain_xz_ansys))
print("节点应变张量和 Ansys 中节点应变张量的相对误差")
print(f"Relative error_s_xx: {relative_error_s_xx:.12e}")
print(f"Relative error_s_yy: {relative_error_s_yy:.12e}")
print(f"Relative error_s_zz: {relative_error_s_zz:.12e}")
print(f"Relative error_s_xy: {relative_error_s_xy:.12e}")
print(f"Relative error_s_yz: {relative_error_s_yz:.12e}")
print(f"Relative error_s_xz: {relative_error_s_xz:.12e}")

# 应力张量
trace_e = bm.einsum("...ii", strain) # (NC, 8)
I = bm.eye(GD, dtype=bm.float64)
stress = 2 * mu * strain + lam * trace_e[..., None, None] * I # (NC, 8, GD, GD)
stress_xx = stress[..., 0, 0] # (NC, 8)
stress_yy = stress[..., 1, 1] # (NC, 8)
stress_zz = stress[..., 2, 2] # (NC, 8)
stress_xy = stress[..., 0, 1] # (NC, 8)
stress_yz = stress[..., 1, 2] # (NC, 8)
stress_xz = stress[..., 0, 2] # (NC, 8)

# 节点应力张量
nodal_stress_xx = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_stress_xx, cell2dof.flatten(), stress_xx.flatten())
nodal_stress_xx = nodal_stress_xx / num_count
nodal_stress_yy = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_stress_yy, cell2dof.flatten(), stress_yy.flatten())
nodal_stress_yy = nodal_stress_yy / num_count
nodal_stress_zz = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_stress_zz, cell2dof.flatten(), stress_zz.flatten())
nodal_stress_zz = nodal_stress_zz / num_count
nodal_stress_xy = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_stress_xy, cell2dof.flatten(), stress_xy.flatten())
nodal_stress_xy = nodal_stress_xy / num_count
nodal_stress_yz = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_stress_yz, cell2dof.flatten(), stress_yz.flatten())
nodal_stress_yz = nodal_stress_yz / num_count
nodal_stress_xz = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_stress_xz, cell2dof.flatten(), stress_xz.flatten())
nodal_stress_xz = nodal_stress_xz / num_count

# 等效应变
equiv_strain = compute_equivalent_strain(strain, nu) # (NC, 8)
# 等效应力
equiv_stress = compute_equivalent_stress(stress, nu) # (NC, 8)

# 节点等效应变
nodal_equiv_strain = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_equiv_strain, cell2dof.flatten(), equiv_strain.flatten())
nodal_equiv_strain = nodal_equiv_strain / num_count

# 节点等效应力
nodal_equiv_stress = bm.zeros(NN, dtype=bm.float64)
bm.add_at(nodal_equiv_stress, cell2dof.flatten(), equiv_stress.flatten())
nodal_equiv_stress = nodal_equiv_stress / num_count

error_es = nodal_equiv_stress - nodal_equiv_stress_ansys # (NN, )
relative_error_es = bm.linalg.norm(error_es) / (bm.linalg.norm(nodal_equiv_stress)+bm.linalg.norm(nodal_equiv_stress_ansys))
print(f"Relative error_es: {relative_error_es:.12e}")

mesh.nodedata['nodal_equiv_strain'] = nodal_equiv_strain[:]
mesh.nodedata['nodal_equiv_stress'] = nodal_equiv_stress[:]

mesh.nodedata['nodal_strain_xx'] = nodal_strain_xx[:]
mesh.nodedata['nodal_strain_yy'] = nodal_strain_yy[:]
mesh.nodedata['nodal_strain_zz'] = nodal_strain_zz[:]
mesh.nodedata['nodal_strain_xy'] = nodal_strain_xy[:]
mesh.nodedata['nodal_strain_yz'] = nodal_strain_yz[:]
mesh.nodedata['nodal_strain_xz'] = nodal_strain_xz[:]

mesh.nodedata['nodal_stress_xx'] = nodal_stress_xx[:]
mesh.nodedata['nodal_stress_yy'] = nodal_stress_yy[:]
mesh.nodedata['nodal_stress_zz'] = nodal_stress_zz[:]
mesh.nodedata['nodal_stress_xy'] = nodal_stress_xy[:]
mesh.nodedata['nodal_stress_yz'] = nodal_stress_yz[:]
mesh.nodedata['nodal_stress_xz'] = nodal_stress_xz[:]
mesh.to_vtk('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/gear15_fealpy.vtu')
print("-----------")