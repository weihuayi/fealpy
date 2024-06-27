import numpy as np
from fealpy.mesh import TriangleMesh
from fealpy.pde.bem_model_2d import *
from fealpy.functionspace import LagrangeFESpace


# 构建 PDE 模型与网格
pde = PoissonModelConstantDirichletBC2d()

box = pde.domain()
nx = 5
ny = 5
# 定义网格对象
mesh = TriangleMesh.from_box(box, nx, ny)

# 定义基函数次数
p = 1  # 增加有限元空间的多项式次数

# 定义网格加密次数，与误差矩阵
maxite = 3
errorMatrix = np.zeros(maxite)
for k in range(maxite):
    # 获取网格信息
    NN = mesh.number_of_nodes()
    NE = mesh.number_of_edges()
    NC = mesh.number_of_cells()

    Node = mesh.entity('node')
    Cell = mesh.entity('cell')
    Edge = mesh.entity('edge')

    # 获取边界信息
    bd_Node_idx = mesh.ds.boundary_node_index()
    bd_face_idx = mesh.ds.boundary_face_index()
    bd_face = mesh.ds.boundary_face()
    bd_face_flag = mesh.ds.boundary_face_flag()  # 判断是否为边界边
    bd_face_measure = mesh.entity_measure('face', index=bd_face_idx)
    cell_measure = mesh.entity_measure('cell')

    bd_val = pde.dirichlet(Node)  # 求精确解
    uh = np.zeros_like(bd_val)  # 构造近视解形式
    # 求边界中点和函数值这个函数值
    Pj = mesh.entity_barycenter('edge')[bd_face_flag]  # 求边重（中）心坐标返回二维数组

    # 计算离散边界中点法向导数值（这里给定矩阵大小）
    G = np.zeros((bd_face.shape[0], bd_face.shape[0]), dtype=float)
    H = np.zeros_like(G)
    B = np.zeros(bd_face.shape[0])

    # 构建边界 Gauss 积分子，获取积分点重心坐标及其权重
    face_qf = mesh.integrator(q=2, etype='edge')  # 定义积分对象
    face_bcs, face_ws = face_qf.get_quadrature_points_and_weights()  # 返回积分点在区域内的重心坐标和权重
    # 计算积分点笛卡尔坐标，q 为每个边界上积分点，j 为边界端点，f 为边界，d 为空间维数
    face_ps = np.einsum('qj, fjd->fqd', face_bcs, Node[bd_face])
    # 单元 Gauss 积分子
    cell_qf = mesh.integrator(q=3, etype='cell')
    cell_bcs, cell_ws = cell_qf.get_quadrature_points_and_weights()
    cell_ps = np.einsum('qj, ejd->eqd', cell_bcs, Node[Cell])  # 求高斯积分点

    x1 = Node[bd_face[:, 0]]
    x2 = Node[bd_face[:, 1]]

    for i in range(bd_face.shape[0]):
        xi = Pj[i]  # 边界边中心节点
        c = np.sign((x1[:, 0] - xi[0]) * (x2[:, 1] - xi[1]) - (x2[:, 0] - xi[0]) * (x1[:, 1] - xi[1]))
        # np.sign如果元素为正数，则返回1,如果元素为负数，则返回-1如果元素为0，则返回0
        hij = c * np.abs(
            (xi[0] - Pj[:, 0]) * (x2[:, 1] - x1[:, 1]) - (xi[1] - Pj[:, 1]) * (x2[:, 0] - x1[:, 0])) / bd_face_measure

        rij = np.sqrt(np.sum((face_ps - xi) ** 2, axis=-1))
        H[..., i, :] = np.einsum('f,f,q,fq->f', -bd_face_measure, hij, face_ws, 1 / rij ** 2) / 2 / np.pi
        G[..., i, :] = bd_face_measure / 2 / np.pi * np.einsum('q, fq ->f', face_ws, np.log(1 / rij))

        # 源项数值积分
        cell_rij = np.sqrt(np.sum((cell_ps - xi) ** 2, axis=-1))  # 求解任意两点距离
        b = pde.source(cell_ps)
        B[i] = np.einsum('e,b,eb,eb->', cell_measure, cell_ws, b, np.log(1 / cell_rij)) / np.pi / 2

    np.fill_diagonal(H, 0.5)  # 填充对角线元素
    np.fill_diagonal(G, (bd_face_measure * (np.log(2 / bd_face_measure) + 1) / np.pi / 2))

    bd_u_val = pde.dirichlet(Pj)
    bd_un_val = np.linalg.solve(G, H @ bd_u_val + B)

    # 计算内部节点函数值
    internal_node = Node[~mesh.ds.boundary_node_flag()]
    uh[bd_Node_idx] = bd_val[bd_Node_idx]
    interNode_idx = np.arange(NN)[~mesh.ds.boundary_node_flag()]
    # 计算内部节点相关矩阵元素值
    for i in range(internal_node.shape[0]):
        Hi = 0
        Mi = 0
        xi = internal_node[i]
        c = np.sign((x1[:, 0] - xi[0]) * (x2[:, 1] - xi[1]) - (x2[:, 0] - xi[0]) * (x1[:, 1] - xi[1]))
        hij = c * np.abs(
            (xi[0] - Pj[:, 0]) * (x2[:, 1] - x1[:, 1]) - (xi[1] - Pj[:, 1]) * (x2[:, 0] - x1[:, 0])) / bd_face_measure

        rij = np.sqrt(np.sum((face_ps - xi) ** 2, axis=-1))
        Hi = np.einsum('f...,f,f,q,fq->...', bd_u_val, -bd_face_measure, hij, face_ws, 1 / rij ** 2) / 2 / np.pi
        Mi = np.einsum('f,f...,q,fq->...', bd_un_val, bd_face_measure, face_ws, np.log(1 / rij)) / 2 / np.pi

        cell_rij = np.sqrt(np.sum((cell_ps - xi) ** 2, axis=-1))
        b = -2 * np.pi ** 2 * np.sin(np.pi * cell_ps[..., 0]) * np.sin(np.pi * cell_ps[..., 1])
        Bi = np.einsum('f,q,fq,fq->', cell_measure, cell_ws, b, np.log(1 / cell_rij)) / np.pi / 2
        uh[interNode_idx[i]] = Mi - Hi - Bi  # 近似解在内部节点的函数值

    real_solution = pde.solution(Node)  # 真解值
    h = np.max(mesh.entity_measure('cell'))
    space = LagrangeFESpace(mesh)
    function_u = space.function()
    function_u[:] = uh
    errorMatrix[k] = mesh.error(function_u, pde.solution)
    # 加密网格
    mesh.uniform_refine(1)

print(f'迭代{maxite}次，结果如下：')
print("误差：\n", errorMatrix)
print('误差比：\n', errorMatrix[0:-1] / errorMatrix[1:])