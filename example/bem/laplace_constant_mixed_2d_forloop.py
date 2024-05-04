import numpy as np
import matplotlib.pyplot as plt

from fealpy.pde.bem_model_2d import LaplaceBemModelMixedBC2d, LaplaceBemModelDirichletBC2d, LaplaceBemModelNeumannBC2d
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace

# 模型初始化，可尝试三种边界条件，默认是混合边界条件
# 混合边界模型
# pde = LaplaceBemModelMixedBC2d()
# Dirichlet 边界模型
# pde = LaplaceBemModelDirichletBC2d()
# Neumann 边界模型
# pde = LaplaceBemModelNeumannBC2d()

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
    bd_face = mesh.ds.boundary_face()
    bd_face_flag = mesh.ds.boundary_face_flag()
    bd_face_idx = mesh.ds.boundary_face_index()
    bd_face_measure = mesh.entity_measure('face', index=bd_face_idx)

    # 计算离散边界中点坐标与函数值
    Pj = mesh.entity_barycenter('edge')[bd_face_flag]
    Pj_val = (bd_val[bd_face[:, 0]] + bd_val[bd_face[:, 1]]) / 2

    # 计算离散边界中点法向导数值
    G = np.zeros((bd_face.shape[0], bd_face.shape[0]), dtype=float)
    H = np.zeros_like(G)
    # Gauss 积分点重心坐标及其权重
    qf = mesh.integrator(q=2, etype='edge')
    bcs, ws = qf.get_quadrature_points_and_weights()
    # 计算相关矩阵元素值
    x1 = node[bd_face[:, 0]]
    x2 = node[bd_face[:, 1]]
    xj = (x1 + x2) / 2
    ps = np.einsum('qj, ejd->eqd', bcs, node[bd_face])
    for i in range(bd_face.shape[0]):
        xi = (node[bd_face[i, 1]] + node[bd_face[i, 0]]) / 2
        c = np.sign((x1[:, 0] - xi[0]) * (x2[:, 1] - xi[1]) - (x2[:, 0] - xi[0]) * (x1[:, 1] - xi[1]))
        hij = c * np.abs((xi[0] - xj[:, 0]) * (x2[:, 1] - x1[:, 1]) - (xi[1] - xj[:, 1]) * (x2[:, 0] - x1[:, 0])) / bd_face_measure
        rij = np.sqrt(np.sum((ps - xi) ** 2, axis=-1))
        H[..., i, :] = np.einsum('e,e,q,eq->e', -bd_face_measure, hij, ws, 1 / rij ** 2) / 2 / np.pi
        G[..., i, :] = bd_face_measure / 2 / np.pi * np.einsum('q, eq ->e', ws, np.log(1 / rij))

    np.fill_diagonal(H, 0.5)
    np.fill_diagonal(G, (bd_face_measure * (np.log(2 / bd_face_measure) + 1) / np.pi / 2))


    is_dirichlet_node = pde.is_dirichlet_boundary(node)
    is_neumann_node = pde.is_neumann_boundary(node)
    # Dirichlet 边和 Neumann 边在全部边界边上的局部索引
    dirichlet_dbEdge_idx = np.arange(bd_face.shape[0])[(is_dirichlet_node[bd_face[:, 0]] & is_dirichlet_node[bd_face[:, 1]])]
    neumann_dbEdge_idx = np.arange(bd_face.shape[0])[(is_neumann_node[bd_face[:, 0]] & is_neumann_node[bd_face[:, 1]])]

    bd_u_val = np.zeros(bd_face.shape[0])
    bd_un_val = np.zeros(bd_face.shape[0])
    dirichlet_node = (node[bd_face[dirichlet_dbEdge_idx][:, 0]] + node[bd_face[dirichlet_dbEdge_idx][:, 1]]) / 2
    neumann_node = (node[bd_face[neumann_dbEdge_idx][:, 0]] + node[bd_face[neumann_dbEdge_idx][:, 1]]) / 2
    n = mesh.face_unit_normal(bd_face_idx)[neumann_dbEdge_idx]

    temp_A = np.zeros_like(G)
    temp_X = np.zeros_like(G)
    temp_F = np.zeros(Pj.shape[0])

    temp_A[:, dirichlet_dbEdge_idx] = -G[:, dirichlet_dbEdge_idx]
    temp_A[:, neumann_dbEdge_idx] = H[:, neumann_dbEdge_idx]
    temp_X[:, dirichlet_dbEdge_idx] = -H[:, dirichlet_dbEdge_idx]
    temp_X[:, neumann_dbEdge_idx] = G[:, neumann_dbEdge_idx]
    temp_F[dirichlet_dbEdge_idx] = pde.dirichlet(dirichlet_node)
    temp_F[neumann_dbEdge_idx] = pde.neumann(neumann_node, n)

    if neumann_dbEdge_idx.shape[0] == bd_face.shape[0]:
        CC = bd_face_measure[np.newaxis, :]
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
        Gi = 0
        xi = internal_node[i]

        c = np.sign((x1[:, 0] - xi[0]) * (x2[:, 1] - xi[1]) - (x2[:, 0] - xi[0]) * (x1[:, 1] - xi[1]))
        hij = c * np.abs(
            (xi[0] - xj[:, 0]) * (x2[:, 1] - x1[:, 1]) - (xi[1] - xj[:, 1]) * (x2[:, 0] - x1[:, 0])) / bd_face_measure

        rij = np.sqrt(np.sum((ps - xi) ** 2, axis=-1))
        Hi = np.einsum('e...,e,e,q,eq->...', bd_u_val, -bd_face_measure, hij, ws, 1 / rij ** 2) / 2 / np.pi
        Gi = np.einsum('e,e...,q,eq->...', bd_un_val, bd_face_measure, ws, np.log(1 / rij)) / 2 / np.pi
        uh[interNode_idx[i]] = Gi - Hi

    space = LagrangeFESpace(mesh)
    function_u = space.function()
    function_u[:] = uh
    errorMatrix[k] = mesh.error(function_u, pde.solution)

    mesh.uniform_refine(1)

print(f'迭代{maxite}次，结果如下：')
print('误差：\n', errorMatrix)
print('误差比：\n', errorMatrix[0:-1]/errorMatrix[1:])
