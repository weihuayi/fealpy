import numpy as np
import matplotlib.pyplot as plt

from fealpy.pde.bem_model_2d import BemModelMixedBC2d, BemModelDirichletBC2d, BemModelNeumannBC2d
from fealpy.mesh import TriangleMesh

# 模型初始化，可尝试三种边界条件，默认是混合边界条件
# 混合边界模型
pde = BemModelMixedBC2d()
# Dirichlet 边界模型
# pde = BemModelDirichletBC2d()
# Neumann 边界模型
# pde = BemModelNeumannBC2d()

# # 网格初始化
box = [0, 1, 0, 1]
nx = 5
ny = 5
mesh = TriangleMesh.from_box(box, nx, ny)

# 定义基函数次数
p = 1

# 定义网格加密次数，与误差矩阵
maxite = 4
errorMatrix = np.zeros(maxite)

for k in range(maxite):
    NN = mesh.number_of_nodes()
    NE = mesh.number_of_edges()
    NC = mesh.number_of_cells()
    node = mesh.entity('node')
    edge = mesh.entity('edge')
    cell = mesh.entity('cell')
    bd_val = pde.dirichlet(node)
    uh = np.zeros_like(bd_val)

    # 边界信息
    bdNode_idx = mesh.ds.boundary_node_index()
    bdEdge = mesh.ds.boundary_edge()
    bdEdge_flag = mesh.ds.boundary_edge_flag()
    bdEdge_idx = mesh.ds.boundary_edge_index()
    bdEdge_measure = mesh.entity_measure('edge', index=bdEdge_idx)

    # 计算离散边界中点坐标与函数值
    Pj = mesh.entity_barycenter('edge')[bdEdge_flag]
    Pj_val = (bd_val[bdEdge[:, 0]] + bd_val[bdEdge[:, 1]]) / 2

    # 计算离散边界中点法向导数值
    M = np.zeros((bdEdge.shape[0], bdEdge.shape[0]), dtype=float)
    H = np.zeros_like(M)
    # Gauss 积分点重心坐标及其权重
    qf = mesh.integrator(q=2, etype='edge')
    bcs, ws = qf.get_quadrature_points_and_weights()
    # 计算相关矩阵元素值
    for i in range(bdEdge.shape[0]):
        xi = (node[bdEdge[i, 1]] + node[bdEdge[i, 0]]) / 2
        x1 = node[bdEdge[:, 0]]
        x2 = node[bdEdge[:, 1]]
        xj = (x1 + x2) / 2
        c = np.sign((x1[:, 0] - xi[0]) * (x2[:, 1] - xi[1]) - (x2[:, 0] - xi[0]) * (x1[:, 1] - xi[1]))
        hij = c * np.abs((xi[0] - xj[:, 0]) * (x2[:, 1] - x1[:, 1]) - (xi[1] - xj[:, 1]) * (x2[:, 0] - x1[:, 0])) / bdEdge_measure
        ps = np.einsum('qj, ejd->eqd', bcs, node[bdEdge])
        rij = np.sqrt(np.sum((ps - xi) ** 2, axis=-1))
        H[..., i, :] = np.einsum('e,e,q,eq->e', -bdEdge_measure, hij, ws, 1 / rij ** 2)/2/np.pi
        M[..., i, :] = bdEdge_measure / 2 / np.pi * np.einsum('q, eq ->e', ws, np.log(1 / rij))

    np.fill_diagonal(H, 0.5)
    np.fill_diagonal(M, (bdEdge_measure * (np.log(2 / bdEdge_measure) + 1) / np.pi / 2))


    is_dirichlet_node = pde.is_dirichlet_boundary(node)
    is_neumann_node = pde.is_neumann_boundary(node)
    # Dirichlet 边和 Neumann 边在全部边界边上的局部索引
    dirichlet_dbEdge_idx = np.arange(bdEdge.shape[0])[(is_dirichlet_node[bdEdge[:, 0]] & is_dirichlet_node[bdEdge[:, 1]])]
    neumann_dbEdge_idx = np.arange(bdEdge.shape[0])[(is_neumann_node[bdEdge[:, 0]] & is_neumann_node[bdEdge[:, 1]])]

    bd_u_val = np.zeros(bdEdge.shape[0])
    bd_un_val = np.zeros(bdEdge.shape[0])
    dirichlet_node = (node[bdEdge[dirichlet_dbEdge_idx][:, 0]]+node[bdEdge[dirichlet_dbEdge_idx][:, 1]])/2
    neumann_node = (node[bdEdge[neumann_dbEdge_idx][:, 0]]+node[bdEdge[neumann_dbEdge_idx][:, 1]])/2
    n = mesh.face_unit_normal(bdEdge_idx)[neumann_dbEdge_idx]

    temp_A = np.zeros_like(M)
    temp_X = np.zeros_like(M)
    temp_F = np.zeros(Pj.shape[0])

    temp_A[:, dirichlet_dbEdge_idx] = -M[:, dirichlet_dbEdge_idx]
    temp_A[:, neumann_dbEdge_idx] = H[:, neumann_dbEdge_idx]
    temp_X[:, dirichlet_dbEdge_idx] = -H[:, dirichlet_dbEdge_idx]
    temp_X[:, neumann_dbEdge_idx] = M[:, neumann_dbEdge_idx]
    temp_F[dirichlet_dbEdge_idx] = pde.dirichlet(dirichlet_node)
    temp_F[neumann_dbEdge_idx] = pde.neumann(neumann_node, n)

    if neumann_dbEdge_idx.shape[0] == bdEdge.shape[0]:
        CC = bdEdge_measure[np.newaxis, :]
        temp_Matrix = np.block([[temp_A, CC.T], [CC, 0]])
        temp_result = np.linalg.solve(temp_Matrix, np.block([temp_X @ temp_F, 0]))
    else:
        temp_result = np.linalg.solve(temp_A, temp_X @ temp_F)


    bd_u_val[dirichlet_dbEdge_idx] = temp_F[dirichlet_dbEdge_idx]
    bd_u_val[neumann_dbEdge_idx] = temp_result[neumann_dbEdge_idx]
    bd_un_val[dirichlet_dbEdge_idx] = temp_result[dirichlet_dbEdge_idx]
    bd_un_val[neumann_dbEdge_idx] = temp_F[neumann_dbEdge_idx]

    # 计算内部节点函数值
    internal_node = node[~mesh.ds.boundary_node_flag()]
    uh[bdNode_idx] = bd_val[bdNode_idx]
    interNode_idx = np.arange(NN)[~mesh.ds.boundary_node_flag()]

    # 计算内部节点相关矩阵元素值
    for i in range(internal_node.shape[0]):
        Hi = 0
        Mi = 0
        xi = internal_node[i]
        x1 = node[bdEdge[:, 0]]
        x2 = node[bdEdge[:, 1]]
        xj = (x1 + x2) / 2
        c = np.sign((x1[:, 0] - xi[0]) * (x2[:, 1] - xi[1]) - (x2[:, 0] - xi[0]) * (x1[:, 1] - xi[1]))
        hij = c * np.abs(
            (xi[0] - xj[:, 0]) * (x2[:, 1] - x1[:, 1]) - (xi[1] - xj[:, 1]) * (x2[:, 0] - x1[:, 0])) / bdEdge_measure
        ps = np.einsum('qj, ejd->eqd', bcs, node[bdEdge])
        rij = np.sqrt(np.sum((ps - xi) ** 2, axis=-1))
        Hi = np.einsum('e...,e,e,q,eq->...', bd_u_val, -bdEdge_measure, hij, ws, 1 / rij ** 2) / 2 / np.pi
        Mi = np.einsum('e,e...,q,eq->...', bd_un_val, bdEdge_measure, ws, np.log(1 / rij)) / 2 / np.pi
        uh[interNode_idx[i]] = Mi - Hi

    real_solution = pde.solution(node)
    h = np.max(mesh.entity_measure('cell'))
    errorMatrix[k] = np.sqrt(np.sum((uh - real_solution) ** 2) * h)

    mesh.uniform_refine(1)

print(f'迭代{maxite}次，结果如下：')
print('误差：\n', errorMatrix)
print('误差比：\n', errorMatrix[0:-1]/errorMatrix[1:])

# 绘图
# fig, axes = plt.subplots()
#
# mesh.add_plot(axes)
# plt.axis('off')
# plt.axis('equal')
# plt.show()
