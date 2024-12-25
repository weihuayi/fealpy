"""中建项目线弹性梁的程序"""
from app.gearx.utils import export_to_inp

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TetrahedronMesh, HexahedronMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.material.elastic_material import LinearElasticMaterial
from fealpy.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.fem.vector_source_integrator import VectorSourceIntegrator
from fealpy.fem.bilinear_form import BilinearForm
from fealpy.fem.linear_form import LinearForm
from fealpy.fem.dirichlet_bc import DirichletBC
from fealpy.solver import cg, spsolve

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
    gphix_q1 = space.grad_basis(bcs_q1, variable='x')
    gphix_q1 = gphix_q1.squeeze(axis=1)

    cuh = uh[cell2dof]

    # 计算应变
    strain = bm.zeros((NC, 6), dtype=bm.float64)
    strain[:, 0:3] = bm.einsum('cid, cid -> cd', cuh, gphix_q1)
    strain[:, 3] = bm.sum(
            cuh[:, :, 2]*gphix_q1[:, :, 1] + cuh[:, :, 1]*gphix_q1[:, :, 2], 
            axis=-1)/2.0
    strain[:, 4] = bm.sum(
            cuh[:, :, 2]*gphix_q1[:, :, 0] + cuh[:, :, 0]*gphix_q1[:, :, 2], 
            axis=-1)/2.0
    strain[:, 5] = bm.sum(
            cuh[:, :, 1]*gphix_q1[:, :, 0] + cuh[:, :, 0]*gphix_q1[:, :, 1], 
            axis=-1)/2.0

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

if __name__ == '__main__':
    nset_Set_1 = bm.arange(449, dtype=bm.int32)
    eset_Set_1 = bm.arange(1612, dtype=bm.int32)

    nset_F1 = bm.array([8], dtype=bm.int32) - 1
    nset_F2 = bm.array([5], dtype=bm.int32) - 1

    nset_Set_1 = bm.array([22,  24,  26,  27, 159, 165], dtype=bm.int32) - 1
    eset_Set_2 = bm.array([1395, 1431, 1449, 1470, 1516, 1578, 1589, 1608], dtype=bm.int32) - 1

    nset_Set_12 = bm.array([12,  14,  17,  19,  93, 140], dtype=bm.int32) - 1
    eset_Set_12 = bm.array([471,  473,  573,  590, 1065, 1066, 1260, 1310, 1400, 1438, 1444, 1488, 1504, 1521, 1573, 1588], dtype=bm.int32) - 1

    Density = 2.4e-09
    Youngs_Modulus = 30000.
    poissons_ratio = 0.22

    fixed_nodes = nset_Set_12
    load_nodes = bm.concatenate([nset_F1, nset_F2])
    loads = bm.array([
        [0, -50000, 0],
        [0, -50000, 0]
    ], dtype=bm.float64)

    nx, ny, nz = 10, 10, 10 
    mesh = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], 
                                nx=nx, ny=ny, nz=nz, device=bm.get_device('cpu'))
    NN = mesh.number_of_nodes()
    NC = mesh.number_of_cells()
    GD = mesh.geo_dimension()
    node = mesh.entity('node')
    cell = mesh.entity('cell') # (NC, LDOF)

    mesh.to_vtk('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/zj_beam_hex_fealpy.vtu')

    export_to_inp('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/beam_hex_abaqus.inp', 
                  node, cell, fixed_nodes, load_nodes, loads, 
                  Youngs_Modulus, poissons_ratio, Density, used_app='abaqus', mesh_type='hex')
    export_to_inp('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/beam_hex_ansys.inp', 
                  node, cell, fixed_nodes, load_nodes, loads, 
                  Youngs_Modulus, poissons_ratio, Density, used_app='ansys', mesh_type='hex')

    p = 1
    q = p+1
    space = LagrangeFESpace(mesh, p=p, ctype='C')
    scalar_gdof = space.number_of_global_dofs()
    tensor_space = TensorFunctionSpace(space, shape=(-1, 3)) # gd_priority
    tldof = tensor_space.number_of_local_dofs()
    tgdof = tensor_space.number_of_global_dofs()
    print(f"tgdof: {tgdof}")
    cell2tdof = tensor_space.cell_to_dof()

    # 刚度矩阵
    E = 3e4
    nu = 0.22
    lam = (E * nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    linear_elastic_material = LinearElasticMaterial(name='E_nu', 
                                                    elastic_modulus=E, poisson_ratio=nu, 
                                                    hypo='3D', device=bm.get_device(mesh))
    integrator_K = LinearElasticIntegrator(material=linear_elastic_material, q=q, method='voigt')
    bform = BilinearForm(tensor_space)
    bform.add_integrator(integrator_K)
    K = bform.assembly(format='csr')
    values = K.values()
    K_norm = bm.sqrt(bm.sum(values * values))
    print(f"K.shape = {K.shape}")
    print(f"Matrix norm before dc: {K_norm:.6f}")

    # 载荷向量
    F = bm.zeros((NN, GD), dtype=bm.float64)
    F = bm.set_at(F, load_nodes, loads)
    if tensor_space.dof_priority:
        F = F.T.flatten()
    else:
        F = F.flatten() 

    F_norm = bm.sqrt(bm.sum(F * F))   
    print(f"F.shape = {F.shape}")
    print(f"Load vector norm before dc: {F_norm:.6f}")

    # 处理 Dirichlet 边界条件
    scalar_is_bd_dof = bm.zeros(scalar_gdof, dtype=bm.bool)
    scalar_is_bd_dof[fixed_nodes] = True
    tensor_is_bd_dof = tensor_space.is_boundary_dof(
            threshold=(scalar_is_bd_dof, scalar_is_bd_dof, scalar_is_bd_dof), 
            method='interp')
    dbc = DirichletBC(space=tensor_space, 
                        gd=0, 
                        threshold=tensor_is_bd_dof, 
                        method='interp')
    uh_bd = bm.zeros(tensor_space.number_of_global_dofs(), dtype=bm.float64, device=bm.get_device(mesh))
    isDDof = tensor_is_bd_dof
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

    cell2dof = space.cell_to_dof()
    uh = uh.reshape(NN, GD) # (NN, GD)

    # 计算方式一：在顶点处计算
    strain1, stress1, nstrain1, nstress1 = compute_strain_stress_at_vertices(space, 
                                                                            uh, mu, lam)

    # 计算方式二：在中心点处计算
    strain2, stress2, nstrain2, nstress2 = compute_strain_stress_at_centers(space, 
                                                                            uh, mu, lam)
    
    # 计算方式三：在积分点处计算
    strain3, stress3, nstrain3, nstress3 = compute_strain_stress_at_quadpoints1(space, 
                                                                            uh, mu, lam)

    # 计算方式四：在积分点处计算
    strain4, stress4, nstrain4, nstress4 = compute_strain_stress_at_quadpoints2(space, 
                                                                            uh, mu, lam)
    
    mesh.nodedata['uh'] = uh
    mesh.nodedata['strain_vertices'] = nstrain1
    mesh.nodedata['stress_vertices'] = nstress1
    mesh.nodedata['strian_centers'] = nstrain2
    mesh.nodedata['stress_centers'] = nstress2
    mesh.nodedata['strain_quadpoints'] = nstrain3
    mesh.nodedata['stress_quadpoints'] = nstress3
    mesh.to_vtk('/home/heliang/FEALPy_Development/fealpy/app/soptx/linear_elasticity/zj_beam_hex_fealpy.vtu')
    print("-----------")


    

