"""论文中带有线性位移真解的算例（应变和应力为常数）"""
from fealpy.backend import backend_manager as bm

from fealpy.mesh import HexahedronMesh, TetrahedronMesh
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.fem.vector_source_integrator import VectorSourceIntegrator
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem.linear_form import LinearForm
from fealpy.fem.dirichlet_bc import DirichletBC
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.typing import TensorLike
from fealpy.decorator import cartesian
from fealpy.sparse import COOTensor, CSRTensor
from fealpy.solver import cg, spsolve

from app.gearx.utils import *
    
bm.set_backend('numpy')

def read_mtx_file(filename):
    """
    读取 mtx 文件并将数据转换为三维数组
    
    参数:
    filename : str
        mtx 文件的路径
        
    返回:
    numpy.ndarray
        形状为 (7, 24, 24) 的三维数组
    """
    # 初始化一个 7x24x24 的零矩阵
    result = np.zeros((7, 24, 24))
    
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
                matrix_idx = int(parts[0])     # 矩阵索引 (0-6)
                i = int(parts[1]) - 1          # 行索引 (0-23)
                j = int(parts[2]) - 1          # 列索引 (0-23)
                value = float(parts[3])        # 值
                
                # 将值存入对应位置
                if 0 <= matrix_idx < 7 and 0 <= i < 24 and 0 <= j < 24:
                    result[matrix_idx, i, j] = value
    
    return result

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

class BoxDomainLinear3d():
    def __init__(self):
        self.eps = 1e-12

    def domain(self):
        return [0, 1, 0, 1, 0, 1]

    @cartesian
    def source(self, points: TensorLike):
        val = bm.zeros(points.shape, 
                       dtype=points.dtype, device=bm.get_device(points))
        
        return val

    @cartesian
    def solution(self, points: TensorLike):
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        val = bm.zeros(points.shape, 
                       dtype=points.dtype, device=bm.get_device(points))
        
        val[..., 0] = 1e-3 * (2*x + y + z) / 2
        val[..., 1] = 1e-3 * (x + 2*y + z) / 2
        val[..., 2] = 1e-3 * (x + y + 2*z) / 2

        return val

    @cartesian
    def dirichlet(self, points: TensorLike) -> TensorLike:

        result = self.solution(points)

        return result

node = bm.array([[0.249, 0.342, 0.192],
                [0.826, 0.288, 0.288],
                [0.850, 0.649, 0.263],
                [0.273, 0.750, 0.230],
                [0.320, 0.186, 0.643],
                [0.677, 0.305, 0.683],
                [0.788, 0.693, 0.644],
                [0.165, 0.745, 0.702],
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1]],
            dtype=bm.float64)

cell = bm.array([[0, 1, 2, 3, 4, 5, 6, 7],
                [0, 3, 2, 1, 8, 11, 10, 9],
                [4, 5, 6, 7, 12, 13, 14, 15],
                [3, 7, 6, 2, 11, 15, 14, 10],
                [0, 1, 5, 4, 8, 9, 13, 12],
                [1, 2, 6, 5, 9, 10, 14, 13],
                [0, 4, 7, 3, 8, 12, 15, 11]],
                dtype=bm.int32)
mesh = HexahedronMesh(node, cell)

GD = mesh.geo_dimension()
NN = mesh.number_of_nodes()
NC = mesh.number_of_cells()
print(f"NN = {NN} NC = {NC}")
cm = mesh.cell_volume()
node = mesh.entity('node')
cell = mesh.entity('cell')

p = 1
q = p+1
space = LagrangeFESpace(mesh, p=p, ctype='C')
sgdof = space.number_of_global_dofs()
print(f"sgdof: {sgdof}")
# tensor_space = TensorFunctionSpace(space, shape=(3, -1)) # dof_priority
tensor_space = TensorFunctionSpace(space, shape=(-1, 3)) # dof_priority
tldof = tensor_space.number_of_local_dofs()
tgdof = tensor_space.number_of_global_dofs()
print(f"tgdof: {tgdof}")
pde = BoxDomainLinear3d()

filename = "/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/box_linear_exact_abaqus_fealpy_STIF2.mtx"
matrices = read_mtx_file(filename)
KE0_true = matrices[0].round(4)
print(f"KE0_true_max: {bm.max(KE0_true)}")
abs_value0 = bm.abs(KE0_true)
print(f"KE0_abs_min: {bm.min(abs_value0[abs_value0 > 0])}")
print(f"KE0_true_min: {bm.min(KE0_true)}")
KE_true = matrices.round(4)
print(f"KE_true_max: {bm.max(KE_true)}")
abs_value = bm.abs(KE_true)
print(f"KE_abs_min: {bm.min(abs_value[abs_value > 0])}")
print(f"KE_true_min: {bm.min(KE_true)}")

# 刚度矩阵
E = 2.1e5
nu = 0.3
lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
mu = E / (2.0 * (1.0 + nu))
linear_elastic_material = LinearElasticMaterial(name='E_nu', 
                                                elastic_modulus=E, poisson_ratio=nu, 
                                                hypo='3D', device=bm.get_device(mesh))
integrator_K_bbar = LinearElasticIntegrator(material=linear_elastic_material, 
                                            q=q, method='C3D8_BBar')
KE_bbar= integrator_K_bbar.c3d8_bbar_assembly(space=tensor_space)
KE0_bbar = KE_bbar[0].round(4)
bform = BilinearForm(tensor_space)
bform.add_integrator(integrator_K_bbar)
K = bform.assembly(format='csr')
Kdense = K.to_dense()
values = K.values()
K_norm = bm.sqrt(bm.sum(values * values))
print(f"K.shape = {K.shape}")
print(f"Matrix norm before dc: {K_norm:.6f}")

# 载荷向量
integrator_F = VectorSourceIntegrator(source=pde.source, q=q)
lform = LinearForm(tensor_space)    
lform.add_integrator(integrator_F)
F = lform.assembly()
F_norm = bm.sqrt(bm.sum(F * F))   
print(f"F.shape = {F.shape}")
print(f"Load vector norm before dc: {F_norm:.6f}")

# 导入 inp 文件
u = lambda p: (1e-3*(2*p[..., 0]+p[..., 1]+p[..., 2])/2).reshape(-1, 1)
v = lambda p: (1e-3*(p[..., 0]+2*p[..., 1]+p[..., 2])/2).reshape(-1, 1)
w = lambda p: (1e-3*(p[..., 0]+p[..., 1]+2*p[..., 2])/2).reshape(-1, 1)
isBdNode = space.is_boundary_dof(threshold=None)
boundary_nodes_idx = bm.where(isBdNode)[0]
boundary_nodes = node[boundary_nodes_idx]
boundary_nodes_u = bm.concatenate([u(boundary_nodes), v(boundary_nodes), w(boundary_nodes)], axis=1)

export_to_inp_by_u(
            filename='/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/inp/box_linear_exact_ansys.inp', 
            nodes=node, elements=cell, 
            boundary_nodes_idx=boundary_nodes_idx, boundary_nodes_u=boundary_nodes_u,
            young_modulus=E, poisson_ratio=nu, density=7.85e-9, 
            used_app='ansys', mesh_type='hex'
        )

# 边界条件处理
uh_bd = bm.zeros(tensor_space.number_of_global_dofs(), 
                dtype=bm.float64, device=bm.get_device(mesh))
uh_bd, isDDof = tensor_space.boundary_interpolate(gd=pde.dirichlet, 
                                                uh=uh_bd, threshold=None, method='interp')
print(f"isDDof: {isDDof.shape} :\n {isDDof}")
print(f"uh_bd: {uh_bd.shape} :\n {uh_bd}")
F = F - K.matmul(uh_bd)
F = bm.set_at(F, isDDof, uh_bd[isDDof])

dbc = DirichletBC(space=tensor_space, 
                gd=pde.dirichlet, 
                threshold=None, 
                method='interp')
K = dbc.apply_matrix(matrix=K, check=True)
Kdense1 = K.to_dense()
# 求解
uh = tensor_space.function()
# uh[:] = cg(K, F, maxiter=1000, atol=1e-8, rtol=1e-8)
uh[:] = spsolve(K, F, solver="mumps")
print(f"uh: {uh.shape}:\n {uh[:]}")
u_exact = tensor_space.interpolate(pde.solution)
print(f"u_exact: {u_exact[:].shape}:\n {u_exact[:]}")
error = bm.sum(bm.abs(uh - u_exact))
print(f"error: {error:.6f}")

if tensor_space.dof_priority:
    uh_show = uh.reshape(GD, NN).T
else:
    uh_show = uh.reshape(NN, GD)
uh_x = uh_show[:, 0]
uh_y = uh_show[:, 1]
uh_z = uh_show[:, 2]
print(f"uh_x: {uh_x.shape}:\n {uh_x}")
print(f"uh_y: {uh_y.shape}:\n {uh_y}")
print(f"uh_z: {uh_z.shape}:\n {uh_z}")
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

# 计算方式五：使用 B-Bar 修正后的 B 计算
strain5, stress5, nstrain5, nstress5 = compute_strain_stress_at_quadpoints3(tensor_space,
                                                                        uh, B_BBar, D)

# # 单元积分点处的位移梯度
# q = 2
# qf = mesh.quadrature_formula(q)
# bcs_quadrature, ws = qf.get_quadrature_points_and_weights()
# tgphi = tensor_space.grad_basis(bcs_quadrature)             # (NC, NQ, tldof, GD, GD)
# tgrad = bm.einsum('cqimn, ci -> cqmn', tgphi, uh_cell)      # (NC, NQ, GD, GD)

# # 应变张量
# strain = (tgrad + bm.transpose(tgrad, (0, 1, 3, 2))) / 2    # (NC, NQ, GD, GD)
# strain_xx = strain[..., 0, 0] # (NC, NQ)
# strain_yy = strain[..., 1, 1] # (NC, NQ)
# strain_zz = strain[..., 2, 2] # (NC, NQ)
# strain_xy = strain[..., 0, 1] # (NC, NQ)
# strain_yz = strain[..., 1, 2] # (NC, NQ)
# strain_xz = strain[..., 0, 2] # (NC, NQ)

# # 应力张量
# trace_e = bm.einsum("...ii", strain) # (NC, NQ)
# I = bm.eye(GD, dtype=bm.float64)
# stress = 2 * mu * strain + lam * trace_e[..., None, None] * I # (NC, NQ, GD, GD)
# stress_xx = stress[..., 0, 0] # (NC, 8)
# stress_yy = stress[..., 1, 1] # (NC, 8)
# stress_zz = stress[..., 2, 2] # (NC, 8)
# stress_xy = stress[..., 0, 1] # (NC, 8)
# stress_yz = stress[..., 1, 2] # (NC, 8)
# stress_xz = stress[..., 0, 2] # (NC, 8)


# mesh.nodedata['uh'] = uh_show[:]
# mesh.nodedata['uh_magnitude'] = uh_magnitude[:]
# mesh.nodedata['strain_vertices'] = nstrain1
# mesh.nodedata['stress_vertices'] = nstress1
# mesh.nodedata['strian_centers'] = nstrain2
# mesh.nodedata['stress_centers'] = nstress2
# mesh.nodedata['strain_quadpoints'] = nstrain3
# mesh.nodedata['stress_quadpoints'] = nstress3
# mesh.to_vtk('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/box_hex_linear_exact_fealpy.vtu')


# # 验证收敛阶
# maxit = 4
# errorType = ['$|| u_{bd} - uh_{bd}$', '$|| u  - u_h ||_{L2}$', '$|| u -  u_h||_{l2}$']
# errorMatrix = bm.zeros((len(errorType), maxit), dtype=bm.float64)
# NDof = bm.zeros(maxit, dtype=bm.int32)
# for i in range(maxit):
#     space = LagrangeFESpace(mesh, p=p, ctype='C')
#     # tensor_space = TensorFunctionSpace(space, shape=(3, -1)) # dof_priority
#     tensor_space = TensorFunctionSpace(space, shape=(-1, 3)) # gd_priority

#     NDof[i] = tensor_space.number_of_global_dofs()

#     linear_elastic_material = LinearElasticMaterial(name='E_nu', 
#                                                 elastic_modulus=E, poisson_ratio=nu, 
#                                                 hypo='3D', device=bm.get_device(mesh))

#     integrator_K = LinearElasticIntegrator(material=linear_elastic_material, q=q, method='voigt')
#     bform = BilinearForm(tensor_space)
#     bform.add_integrator(integrator_K)
#     K = bform.assembly(format='csr')

#     integrator_F = VectorSourceIntegrator(source=pde.source, q=q)
#     lform = LinearForm(tensor_space)    
#     lform.add_integrator(integrator_F)
#     F = lform.assembly()

#     dbc = DirichletBC(space=tensor_space, 
#                     gd=pde.dirichlet, 
#                     threshold=None, 
#                     method='interp')
#     uh_bd = bm.zeros(tensor_space.number_of_global_dofs(), 
#                     dtype=bm.float64, device=bm.get_device(mesh))
#     uh_bd, isDDof = tensor_space.boundary_interpolate(gd=pde.dirichlet, 
#                                                     uh=uh_bd, threshold=None, method='interp')
#     F = F - K.matmul(uh_bd)
#     F = bm.set_at(F, isDDof, uh_bd[isDDof])

#     K = dbc.apply_matrix(matrix=K, check=True)

#     uh = tensor_space.function()
#     # uh[:] = cg(K, F, maxiter=1000, atol=1e-8, rtol=1e-8)
#     uh[:] = spsolve(K, F, solver="mumps")

#     u_exact = tensor_space.interpolate(pde.solution)
#     errorMatrix[0, i] = bm.sqrt(bm.sum(bm.abs(uh[:] - u_exact)**2 * (1 / NDof[i])))
#     errorMatrix[1, i] = mesh.error(u=uh, v=pde.solution, q=tensor_space.p+3, power=2)
#     errorMatrix[2, i] = bm.sqrt(bm.sum(bm.abs(uh[isDDof] - u_exact[isDDof])**2 * (1 / NDof[i])))

#     if i < maxit-1:
#         mesh.uniform_refine()

# print("errorMatrix:\n", errorType, "\n", errorMatrix)
# print("NDof:", NDof)
# print("order_l2:\n", bm.log2(errorMatrix[0, :-1] / errorMatrix[0, 1:]))
# print("order_L2:\n ", bm.log2(errorMatrix[1, :-1] / errorMatrix[1, 1:]))
# print("----------------------")

