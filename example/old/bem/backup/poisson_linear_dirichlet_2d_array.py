import numpy as np
from fealpy.mesh import TriangleMesh
from fealpy.pde.bem_model_2d import *
from fealpy.functionspace import LagrangeFESpace


pde = PoissonModelConstantDirichletBC2d()
box = pde.domain()
nx = 5
ny = 5
# 定义网格对象
mesh = TriangleMesh.from_box(box, nx, ny)

# 定义网格加密次数，与误差矩阵
maxite = 4
errorMatrix = np.zeros(maxite)

for k in range(maxite):
    # 获取网格实体信息
    NN = mesh.number_of_nodes()
    NE = mesh.number_of_edges()
    NC = mesh.number_of_cells()
    Node = mesh.entity('node')
    Cell = mesh.entity('cell')
    Edge = mesh.entity('edge')
    Face = mesh.entity('face')

    # 获取边界信息
    bd_face = mesh.ds.boundary_face()
    bd_edge = mesh.ds.boundary_edge()
    bd_node_flag = mesh.ds.boundary_node_flag()
    bd_node_index = mesh.ds.boundary_node_index()
    bd_edge_index = mesh.ds.boundary_edge_index()
    bd_face_measure = mesh.entity_measure('face', mesh.ds.boundary_face_index())

    # 构建网格边界面自由度（节点），由面网格到边界边网格的映射
    face2gdof = np.zeros(mesh.number_of_nodes(), dtype=np.int64)
    face2gdof[bd_node_index] = np.arange(bd_node_index.shape[0])

    # 初始化解函数
    bd_val = pde.dirichlet(Node)
    uh = np.zeros_like(bd_val)

    # 构建面的 Gauss 积分点重心坐标及其权重
    qf = mesh.integrator(q=2, etype='face')  # 定义积分对象
    bcs, ws = qf.get_quadrature_points_and_weights()  # 返回积分点在区域内的重心坐标和权重

    # 计算面积分点的笛卡尔坐标
    # NQ * bd_NE * dim
    ps = np.einsum('qj, ejd->qed', bcs, Node[bd_face], optimize=True)

    # 每个面的两个节点
    x1 = Node[bd_face[:, 0]]  # bd_NF * dim
    x2 = Node[bd_face[:, 1]]
    # 获取整体边界网格节点坐标
    # bd_gdof * dim
    xi = Node[bd_node_index]

    # 相关数据计算
    # bd_gdof * bd_NF
    c = np.sign((x1[np.newaxis, :, 0] - xi[..., np.newaxis, 0]) * (
            x2[np.newaxis, :, 1] - xi[..., np.newaxis, 1]) - (
                        x2[np.newaxis, :, 0] - xi[..., np.newaxis, 0]) * (
                        x1[np.newaxis, :, 1] - xi[..., np.newaxis, 1]))
    h = c * np.abs((xi[..., np.newaxis, 0] - x1[np.newaxis, :, 0]) * (
                x2[np.newaxis, :, 1] - x1[np.newaxis, :, 1]) - (
                               xi[..., np.newaxis, 1] - x1[np.newaxis, :, 1]) * (
                               x2[np.newaxis, :, 0] - x1[np.newaxis, :, 0])) / bd_face_measure[np.newaxis, :]
    # bd_gdof * NQ * bd_NF
    r = np.sqrt(np.sum((ps[np.newaxis, ...] - xi[:, np.newaxis, np.newaxis, ...]) ** 2, axis=-1))

    # 获取单元局部基函数
    phi = bcs[np.newaxis, ...]
    # 单元自由度矩阵计算
    Hij = np.einsum('f, nf, q, fqi, nqf -> nfi', bd_face_measure, h, ws, phi, 1 / r ** 2, optimize=True) / (-2 * np.pi)
    Gij = np.einsum('f, q, fqi, nqf -> nfi', bd_face_measure, ws, phi, np.log(1 / r), optimize=True) / (2 * np.pi)
    # 构建单元自由度到全局自由度的映射
    I = np.broadcast_to(np.arange(bd_node_index.shape[0], dtype=np.int64)[:, None, None], shape=Hij.shape)
    J = np.broadcast_to(face2gdof[bd_face][None, ...], shape=Hij.shape)
    # 整体矩阵的初始化与组装
    H = np.zeros((bd_node_index.shape[0], bd_node_index.shape[0]))
    np.add.at(H, (I, J), Hij)
    np.fill_diagonal(H, 0.5)
    G = np.zeros((bd_node_index.shape[0], bd_node_index.shape[0]))
    np.add.at(G, (I, J), Gij)

    # 源项积分
    cell_qf = mesh.integrator(q=3)
    cell_bcs, cell_ws = cell_qf.get_quadrature_points_and_weights()
    cell_ps = np.einsum('qj, ejd->qed', cell_bcs, Node[Cell], optimize=True)
    cell_measure = mesh.entity_measure('cell')
    cell_r = np.sqrt(np.sum((cell_ps[np.newaxis, ...] - xi[:, np.newaxis, np.newaxis, ...]) ** 2, axis=-1))
    f_xi = pde.source(cell_ps)
    f = np.einsum('c,q,nqc,qc->n', cell_measure, cell_ws, np.log(1 / cell_r), f_xi, optimize=True) / (2 * np.pi)

    # 边界值计算，位势与通量
    u = pde.dirichlet(xi)
    q = np.linalg.solve(G, H @ u + f)

    # =========================================
    # 内部节点值计算
    x_inter = Node[~mesh.ds.boundary_node_flag()]

    c_inter = np.sign((x1[np.newaxis, :, 0] - x_inter[..., np.newaxis, 0]) * (
            x2[np.newaxis, :, 1] - x_inter[..., np.newaxis, 1]) - (
                        x2[np.newaxis, :, 0] - x_inter[..., np.newaxis, 0]) * (
                        x1[np.newaxis, :, 1] - x_inter[..., np.newaxis, 1]))
    h_inter = c_inter * np.abs((x_inter[..., np.newaxis, 0] - x1[np.newaxis, :, 0]) * (
                x2[np.newaxis, :, 1] - x1[np.newaxis, :, 1]) - (
                               x_inter[..., np.newaxis, 1] - x1[np.newaxis, :, 1]) * (
                               x2[np.newaxis, :, 0] - x1[np.newaxis, :, 0])) / bd_face_measure[np.newaxis, :]

    r_inter = np.sqrt(np.sum((ps[np.newaxis, ...] - x_inter[:, np.newaxis, np.newaxis, ...]) ** 2, axis=-1))

    Hij_inter = np.einsum('f, nf, q, fqi, nqf -> nfi', bd_face_measure, h_inter, ws, phi, 1 / r_inter ** 2, optimize=True) / (-2 * np.pi)
    Gij_inter = np.einsum('f, q, fqi, nqf -> nfi', bd_face_measure, ws, phi, np.log(1 / r_inter), optimize=True) / (2 * np.pi)
    I_inter = np.broadcast_to(np.arange(x_inter.shape[0], dtype=np.int64)[:, None, None], shape=Hij_inter.shape)
    J_inter = np.broadcast_to(face2gdof[bd_face][None, ...], shape=Hij_inter.shape)

    H_inter = np.zeros((x_inter.shape[0], xi.shape[0]))
    np.add.at(H_inter, (I_inter, J_inter), Hij_inter)
    G_inter = np.zeros((x_inter.shape[0], xi.shape[0]))
    np.add.at(G_inter, (I_inter, J_inter), Gij_inter)

    # 内部节点源项
    cell_r_inter = np.sqrt(np.sum((cell_ps[np.newaxis, ...] - x_inter[:, np.newaxis, np.newaxis, ...]) ** 2, axis=-1))
    f_inter = np.einsum('c,q,nqc,qc->n', cell_measure, cell_ws, np.log(1 / cell_r_inter), f_xi, optimize=True) / (2 * np.pi)

    # 内部节点位势计算
    u_inter = G_inter @ q - H_inter @ u - f_inter

    # 数值解填充
    uh[bd_node_index] = bd_val[bd_node_index]
    uh[~bd_node_flag] = u_inter

    # 计算误差
    space = LagrangeFESpace(mesh)
    function_u = space.function()
    function_u[:] = uh
    errorMatrix[k] = mesh.error(function_u, pde.solution)

    mesh.uniform_refine(1)  # 在这里将网格进行细化成更加小的网格，即类似之前椭圆方程步长减少

print(f'迭代{maxite}次，结果如下：')
print("误差：\n", errorMatrix)
print('误差比：\n', errorMatrix[0:-1] / errorMatrix[1:])
