import numpy as np
from fealpy.mesh.uniform_mesh_3d import UniformMesh3d
from fealpy.pde.bem_model_3d import *


def error_calculator(mesh, u, v, q=3, power=2):
    qf = mesh.integrator(q, etype='cell')
    bcs, ws = qf.get_quadrature_points_and_weights()
    ps = mesh.bc_to_point(bcs)

    cell = mesh.entity('cell')
    cell_node_val = u[cell]

    bc0 = bcs[0].reshape(-1, 2)  # (NQ0, 2)
    bc1 = bcs[1].reshape(-1, 2)  # (NQ1, 2)
    bc2 = bcs[2].reshape(-1, 2)  # (NQ2, 2)
    bc = np.einsum('im, jn, kl->ijkmnl', bc0, bc1, bc2).reshape(-1, 8)  # (NQ0, NQ1, NQ2, 2, 2, 2)  (NQ0*NQ1*NQ2, 8)

    u = np.einsum('...j, cj->...c', bc, cell_node_val)


    if callable(v):
        if not hasattr(v, 'coordtype'):
            v = v(ps)
        else:
            if v.coordtype == 'cartesian':
                v = v(ps)
            elif v.coordtype == 'barycentric':
                v = v(bcs)

    if u.shape[-1] == 1:
        u = u[..., 0]

    if v.shape[-1] == 1:
        v = v[..., 0]

    cm = mesh.entity_measure('cell')

    f = np.power(np.abs(u - v), power)

    e = np.einsum('q, qc..., c->c...', ws, f, cm)
    e = np.power(np.sum(e), 1 / power)

    return e


pde = PoissonModelConstantDirichletBC3d()
nx = 5
ny = 5
nz = 5

hx = (1 - 0) / nx
hy = (1 - 0) / ny
hz = (1 - 0) / nz

maxite = 3

errorMatrix = np.zeros(maxite)
mesh = UniformMesh3d((0, nx, 0, ny, 0, nz), h=(hx, hy, hz), origin=(0, 0, 0))  #

for k in range(maxite):
    # 获取网格信息
    NN = mesh.number_of_nodes()
    NC = mesh.number_of_cells()
    NE = mesh.number_of_edges()
    NF = mesh.number_of_faces()

    Node = mesh.entity('node')
    Cell = mesh.entity('cell')
    Edge = mesh.entity('edge')
    Face = mesh.entity('face')

    # 获取边界信息
    bd_node_idx = mesh.ds.boundary_node_index()
    bd_face = mesh.ds.boundary_face()
    bd_face_index = mesh.ds.boundary_face_index()
    bd_face_flag = mesh.ds.boundary_face_flag()

    # 求近视解与真解
    bd_val = pde.dirichlet(Node)
    uh = np.zeros_like(bd_val)

    # 边界节点坐标
    Pj = mesh.entity_barycenter('face')[bd_face_flag]

    G = np.zeros((bd_face.shape[0], bd_face.shape[0]), dtype=float)
    H = np.zeros_like(G)
    B = np.zeros(bd_face.shape[0])

    # Gauss 积分点重心坐标及其权重
    face_qs = mesh.integrator(q=2, etype=2)
    face_bcs, face_ws = face_qs.get_quadrature_points_and_weights()
    face_ps = mesh.bc_to_point(face_bcs, bd_face_index)

    cell_qs = mesh.integrator(q=3)
    cell_bcs, cell_ws = cell_qs.get_quadrature_points_and_weights()
    cell_ps = mesh.bc_to_point(cell_bcs)

    # 计算边界面外法向量，面积
    n = mesh.face_normal(index=bd_face_index)
    s = mesh.entity_measure('face', bd_face_index)
    # 计算体单元体积
    v = mesh.entity_measure('cell')

    x1 = Node[bd_face[:, 1]]
    b = pde.source(cell_ps)
    for i in range(bd_face.shape[0]):
        xi = Pj[i, :]
        hij = np.einsum('fi, fi -> f', n, x1-xi) / np.linalg.norm(n, axis=-1)

        rij = np.sqrt(np.sum((face_ps - xi) ** 2, axis=-1))
        rij1 = np.sqrt(np.sum((cell_ps - xi) ** 2, axis=-1))

        H[..., i, :] = -np.einsum('f,q,qf,f->f', hij, face_ws, 1 / rij ** 3, s) / np.pi / 4
        G[..., i, :] = np.einsum('q,qf,f->f', face_ws, 1 / rij, s) / np.pi / 4
        B[i] = np.einsum("f,if,i,if->", v, 1 / rij1, cell_ws, b) / 4 / np.pi

    # 计算给定矩阵主对角线元素
    np.fill_diagonal(H, 0.5)

    bd_u_val = pde.dirichlet(Pj)
    bd_un_val = np.linalg.solve(G, H @ bd_u_val + B)

    # 内部节点处理
    internal_node = Node[~mesh.ds.boundary_node_flag()]
    uh[bd_node_idx] = bd_val[bd_node_idx]
    interNode_idx = np.arange(NN)[~mesh.ds.boundary_node_flag()]

    # 计算内部节点相关矩阵元素值
    for i in range(internal_node.shape[0]):
        Hi = 0
        Mi = 0
        xi = internal_node[i]
        hij = np.einsum('fi, fi -> f', n, x1-xi) / np.linalg.norm(n, axis=-1)
        rij = np.sqrt(np.sum((face_ps - xi) ** 2, axis=-1))
        rij1 = np.sqrt(np.sum((cell_ps - xi) ** 2, axis=-1))
        Hi = -np.einsum('f,f,q,qf,f->', bd_u_val, hij, face_ws, 1 / rij ** 3, s) / np.pi / 4
        Mi = np.einsum('f,q,qf,f->', bd_un_val, face_ws, 1 / rij, s) / np.pi / 4
        Bi = np.einsum("f,if,i,if->", v, 1 / rij1, cell_ws, b) / 4 / np.pi
        uh[interNode_idx[i]] = Mi - Hi - Bi

    # 计算误差
    errorMatrix[k] = error_calculator(mesh, uh, pde.solution)

    mesh.uniform_refine(1)

print(f'迭代{maxite}次，结果如下：')
print("误差：\n", errorMatrix)
print('误差比：\n', errorMatrix[0:-1] / errorMatrix[1:])