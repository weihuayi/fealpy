"""精益项目 kxz_5.inp"""
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
import re
from app.gearx.gear import ExternalGear, InternalGear
from app.gearx.utils import *

from fealpy.mesh import HexahedronMesh

def read_inp_mesh_multiple_sections(file_path):
    """
    读取 .inp 文件，处理多个 Nodes 和 Cells 部分，创建 HexahedronMesh 对象。

    参数:
        file_path (str): .inp 文件的路径。

    返回:
        mesh (HexahedronMesh): 创建的网格对象。
        nodes (np.ndarray): 节点坐标数组。
        cells (np.ndarray): 单元节点索引数组。
    """
    # 读取文件内容
    with open(file_path, 'r') as file:
        content = file.read()

    # 提取Nodes部分
    nodes_section = re.search(r'Nodes:\s*(.*?)\s*EndNodes', content, re.DOTALL | re.IGNORECASE)
    nodes_list = []
    if nodes_section:
        nodes_content = nodes_section.group(1).strip()
        # 分割行，支持不同的换行符
        nodes_lines = re.split(r'\r?\n', nodes_content)
        print(f"找到 {len(nodes_lines)} 个节点行。")  # 调试信息
        for line in nodes_lines:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            # 使用正则提取每行的数据，去掉第一个序号
            parts = line.split(',')
            temp_list = []
            temp_list.append(float(parts[1].strip()))
            temp_list.append(float(parts[2].strip()))
            temp_list.append(float(parts[3].strip()))
            nodes_list.append(temp_list)
    else:
        print("未找到Nodes部分。")

    nodes = bm.array(nodes_list, dtype=bm.float64)

    # 提取所有 Cells 部分
    cells_section = re.search(r'Cells:\s*(.*?)\s*EndCells', content, re.DOTALL | re.IGNORECASE)
    cells_list = []
    if cells_section:
        cells_content = cells_section.group(1).strip()
        # 分割行，支持不同的换行符
        cells_lines = re.split(r'\r?\n', cells_content)
        print(f"找到 {len(cells_lines)} 个单元行。")  # 调试信息
        for line in cells_lines:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            # 使用逗号分割并转换为整数，去掉第一个序号
            parts = line.split(',')
            if len(parts) < 2:
                print(f"单元行格式不正确: '{line}'")  # 调试信息
                continue
            try:
                # 转换每个节点ID为整数
                node_ids = [int(part.strip()) for part in parts[1:]]
                cells_list.append(node_ids)
            except ValueError as e:
                print(f"无法解析单元行: '{line}' 错误: {e}")  # 调试信息
    else:
        print("未找到Cells部分。")

    cells = bm.array(cells_list, dtype=np.int32)

    return nodes, cells

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
        nstrain[:, i] /= bm.maximum(nc, 1) 
        bm.add_at(nstress[:, i], cell.flatten(), stress[:, :, i].flatten())
        nstress[:, i] /= bm.maximum(nc, 1)
        
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
        nstrain[:, i] /= bm.maximum(nc, 1) 
        bm.add_at(nstress[:, i], cell, stress[:, i, None] * bm.ones_like(cell))
        nstress[:, i] /= bm.maximum(nc, 1)

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
        nstrain[:, i] /= bm.maximum(nc, 1) 
        bm.add_at(nstress[:, i], cell.flatten(), stress[:, :, i].flatten())
        nstress[:, i] /= bm.maximum(nc, 1)

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

# 指定 .inp 文件的路径
file_path = "/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/kxz_5.inp"

nodes, cells = read_inp_mesh_multiple_sections(file_path)
print(f"cells_max", {bm.max(cells)})
print(f"cells_min", {bm.min(cells)})
cells = cells - 6551

mesh = HexahedronMesh(nodes, cells)

GD = mesh.geo_dimension()   
NC = mesh.number_of_cells()
print(f"NC: {NC}")
NN = mesh.number_of_nodes()
print(f"NN: {NN}")
node = mesh.entity('node')
cell = mesh.entity('cell')

fixed_node_index = bm.arange(6551, 6813, 1, dtype=bm.int32) - 6551

p = 1
q = p+2
space = LagrangeFESpace(mesh, p=p, ctype='C')
sgdof = space.number_of_global_dofs()
cell2dof = space.cell_to_dof()
tensor_space = TensorFunctionSpace(space, shape=(3, -1)) # dof_priority
print(f"sgdof: {sgdof}")
tgdof = tensor_space.number_of_global_dofs()
print(f"tgdof: {tgdof}")
tldof = tensor_space.number_of_local_dofs()

# 载荷向量
load_nodes = bm.array([9451-6551], dtype=bm.int32)
loads = bm.array([-500, 0, 0], dtype=bm.float64)
F = bm.zeros((NN, GD), dtype=bm.float64)
F = bm.set_at(F, load_nodes, loads)
if tensor_space.dof_priority:
    F = F.T.flatten()
else:
    F = F.flatten() 

# 刚度矩阵
E = 206e3
nu = 0.3
lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
linear_elastic_material = LinearElasticMaterial(name='E_nu', 
                                                elastic_modulus=E, poisson_ratio=nu, 
                                                hypo='3D', device=bm.get_device(mesh))
integrator_K = LinearElasticIntegrator(material=linear_elastic_material, q=q, method='voigt')
bform = BilinearForm(tensor_space)
bform.add_integrator(integrator_K)
K = bform.assembly(format='csr')
# 处理边界条件
scalar_is_bd_dof = bm.zeros(sgdof, dtype=bm.bool)
scalar_is_bd_dof[fixed_node_index] = True
tensor_is_bd_dof = tensor_space.is_boundary_dof(
        threshold=(scalar_is_bd_dof, scalar_is_bd_dof, scalar_is_bd_dof), 
        method='interp')
dbc = DirichletBC(space=tensor_space, 
                    gd=bm.zeros(tgdof), 
                    threshold=tensor_is_bd_dof, 
                    method='interp')
K = dbc.apply_matrix(matrix=K, check=True)
Kdense = K.to_dense()
uh_bd = bm.zeros(tensor_space.number_of_global_dofs(), 
                dtype=bm.float64, device=bm.get_device(mesh))
uh_bd, isDDof = tensor_space.boundary_interpolate(gd=bm.zeros(tgdof), 
                                                uh=uh_bd, 
                                                threshold=tensor_is_bd_dof, 
                                                method='interp')
F = F - K.matmul(uh_bd)
F = bm.set_at(F, isDDof, uh_bd[isDDof])

uh = tensor_space.function()
# from fealpy import logger
# logger.setLevel('INFO')
# uh[:] = cg(K, F, maxiter=10000, atol=1e-8, rtol=1e-8)
uh[:] = spsolve(K, F, solver="mumps")
# uh[:] = spsolve(K, F, solver="scipy")

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
mesh.to_vtk('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/gear_new_fealpy.vtu')
print("-----------")