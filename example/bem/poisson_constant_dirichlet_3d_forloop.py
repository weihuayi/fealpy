import numpy as np
from fealpy.mesh.uniform_mesh_3d import UniformMesh3d
from fealpy.pde.bem_model_3d import *



pde = PoissonModelConstantDirichletBC3d()
nx = 5
ny = 5
nz = 5
# 第一个问题：边界节点和边界面不同数，在用线性方程求解时会出现零矩阵
# 第三个问题：J矩阵以及G的对角元素待求，J=1/4面面积
# 根据网格剖分数计算网格节点间距
hx = (1 - 0) / nx
hy = (1 - 0) / ny
hz = (1 - 0) / nz

maxite = 2

errorMatrix = np.zeros(maxite)
# mesh = UniformMesh3d((0, 10, 0, 10, 0, 10))
mesh = UniformMesh3d((0, nx, 0, ny, 0, nz), h=(hx, hy, hz), origin=(0, 0, 0))  #

for k in range(maxite):
    # 给出各元素数量
    NN = mesh.number_of_nodes()
    NC = mesh.number_of_cells()
    NE = mesh.number_of_edges()
    NF = mesh.number_of_faces()

    # 获取坐标
    Node = mesh.entity('node')
    Cell = mesh.entity('cell')
    Edge = mesh.entity('edge')
    Face = mesh.entity('face')

    # 获取边界信息
    bdCell = mesh.ds.boundary_cell()  # 给出组成单元的节点全局编号
    bdNode = mesh.ds.boundary_node_index()

    bdCell_idx = mesh.ds.boundary_cell_index()  # 边界单元个数
    bdEdge = mesh.ds.boundary_edge()
    bdface = mesh.ds.boundary_face()
    bdEdge_idx = mesh.ds.boundary_edge_index()
    # print("1",bdEdge_idx[0])
    # print("2",bdNode.shape[0])
    bdface_flag = mesh.ds.boundary_face_flag()
    # 计算函数值
    bd_val = pde.dirichlet(Node)  # 求精确解
    uh = np.zeros_like(bd_val)

    # 计算每个体单元的面积
    index = np.s_[:]  # 表示获取所有的元素索引
    v1 = Node[Cell[index, 1], :] - Node[Cell[index, 0], :]
    v2 = Node[Cell[index, 2], :] - Node[Cell[index, 1], :]
    v3 = Node[Cell[index, 3], :] - Node[Cell[index, 2], :]
    v4 = Node[Cell[index, 3], :] - Node[Cell[index, 0], :]
    h1 = Node[Cell[index, 4], :] - Node[Cell[index, 0], :]
    h2 = Node[Cell[index, 5], :] - Node[Cell[index, 1], :]
    h3 = Node[Cell[index, 6], :] - Node[Cell[index, 2], :]
    h4 = Node[Cell[index, 7], :] - Node[Cell[index, 3], :]
    tsa1 = Node[Cell[index, 5], :] - Node[Cell[index, 4], :]
    tsa2 = Node[Cell[index, 6], :] - Node[Cell[index, 5], :]
    ba = np.sqrt(np.square(np.cross(v2, -v1)).sum(axis=1))  # 这个np.square是计算向量模长
    sa = np.sqrt(np.square(np.cross(h1, -v1)).sum(axis=1))
    sa2 = np.sqrt(np.square(np.cross(h2, -v2)).sum(axis=1))
    sa3 = np.sqrt(np.square(np.cross(h3, -v3)).sum(axis=1))
    sa4 = np.sqrt(np.square(np.cross(h4, -v4)).sum(axis=1))
    ta = np.sqrt(np.square(np.cross(tsa2, -tsa1)).sum(axis=1))
    v = np.sqrt(np.square(v1).sum(axis=1)) ** 3  # 正六面体体单元体积(125.3)

    bdCell_measure = ba + ta + sa + sa2 + sa3 + sa4  # 体单元面积

    # 求单元边长
    bdEdge_measure = mesh.entity_measure('edge')
    bdFace_measure = mesh.entity_measure('face')
    # 求面单元面积
    s1 = Node[bdface[index, 1]] - Node[bdface[index, 0]]
    s2 = Node[bdface[index, 3]] - Node[bdface[index, 0]]
    s = np.sqrt(np.square(np.cross(s2, -s1)).sum(axis=1))  # 面单元面积
    s3 = Node[Face[index, 1]] - Node[Face[index, 0]]
    s4 = Node[Face[index, 3]] - Node[Face[index, 0]]
    s2 = np.sqrt(np.square(np.cross(s4, -s3)).sum(axis=1))
    # 求近视解与真解
    bd_val = pde.dirichlet(Node)
    uh = np.zeros_like(bd_val)


    Pj = mesh.entity_barycenter('face')[bdface_flag]  # 计算边界面单元重心坐标
    # (150,3)
    # Pj1=mesh.entity_barycenter('edge')
    # 这里给定矩阵大小
    G = np.zeros((bdface.shape[0], bdface.shape[0]), dtype=float)
    H = np.zeros_like(G)
    B = np.zeros(bdface.shape[0])
    # Gauss 积分点重心坐标及其权重
    qs = mesh.integrator(q=2, etype=2)
    bcs, ws = qs.get_quadrature_points_and_weights()  # ws(8,),bcs(3,2,2)
    # ps = mesh.bc_to_point(bcs)#bcs是0-1区间内的高斯坐标即重心坐标，此时通过重心坐标和网格节点求解边界点坐标
    # ps维数（NQ,NC，3），NQ为积分点个数，即每个积分点所对应的笛卡尔坐标
    gs = mesh.integrator(q=3)
    bcs1, ws1 = gs.get_quadrature_points_and_weights()  # ws1(27,)
    ps1 = mesh.bc_to_point(bcs1)

    bc0 = bcs[0].reshape(-1, 2)  # (NQ0, 2)
    bc1 = bcs[1].reshape(-1, 2)  # (NQ1, 2)
    bc = np.einsum('im, jn->ijmn', bc0, bc1).reshape(-1, 4)  # (NQ0, NQ1, 2, 2)

    # node[cell].shape == (NC, 4, 2)
    # bc.shape == (NQ, 4)
    ps = np.einsum('...j, cjk->...ck', bc, Node[bdface[:]])  # (4,150,3)(NQ, NC, 2),面单元情况下每个积分点对应的笛卡尔坐标
    J = s
    for i in range(bdface.shape[0]):
        # 取每个平面的重心点
        xi = Pj[i, :]  # (3,)
        e1 = Node[bdface[:, 1]] - Node[bdface[:, 0]]
        e2 = Node[bdface[:, 3]] - Node[bdface[:, 0]]
        x1 = Node[bdface[:, 1]]  # （600，3）
        n = np.cross(e2, e1)  # （150，3)
        hij = np.einsum('ei, ei -> e', n, x1-xi) / np.linalg.norm(n, axis=-1)
        # ps1=np.einsum('qji,edq->ejq',bcs,Node[bdface])#(150,2,3)

        rij = np.sqrt(np.sum((ps - xi) ** 2, axis=-1))  # (4,150)
        rij1 = np.sqrt(np.sum((ps1 - xi) ** 2, axis=-1))  # (27,125)
        b = pde.source(ps1)  # (27,125)
        # B计算有误
        B[i] = np.einsum("k,mk,m,mk->", v, 1 / rij1, ws1, b) / 4 / np.pi
        # H[..., i, :] = -np.einsum('ij,q,qi,i->i', hij, ws, 1 / rij, J) / np.pi
        # G[..., i, :] = np.einsum('q,qi,i->i', ws, 1 / rij, J) / np.pi
        H[..., i, :] = -np.einsum('i,q,qi,i->i', hij, ws, 1 / rij**3, J) / np.pi / 4
        G[..., i, :] = np.einsum('q,qi,i->i', ws, 1 / rij, J) / np.pi / 4

    # 计算给定矩阵主对角线元素
    np.fill_diagonal(H, 0.5)
    # np.fill_diagonal(G, J)

    bd_u_val = pde.dirichlet(Pj)
    bd_un_val = np.linalg.solve(G, H @ bd_u_val + B)

    # 内部节点处理
    internal_node = Node[~mesh.ds.boundary_node_flag()]  # 内部节点坐标这里意思是除去边界节点
    uh[bdNode] = bd_val[bdNode]  # 将每条边的全局编号处的精确解赋值给近视解
    interNode_idx = np.arange(NN)[~mesh.ds.boundary_node_flag()]

    # 计算内部节点相关矩阵元素值
    for i in range(internal_node.shape[0]):
        Hi = 0
        Mi = 0
        xi = internal_node[i]
        e1 = Node[bdface[:, 1]] - Node[bdface[:, 0]]
        e2 = Node[bdface[:, 3]] - Node[bdface[:, 0]]
        x1 = Node[bdface[:, 1]]  # （150，3）
        n = np.cross(e2, e1)  # （150，3）#边界面的法向量
        hij = np.einsum('ei, ei -> e', n, x1-xi) / np.linalg.norm(n, axis=-1)
        rij = np.sqrt(np.sum((ps - xi) ** 2, axis=-1))
        rij1 = np.sqrt(np.sum((ps1 - xi) ** 2, axis=-1))
        Hi = -np.einsum('i,i,q,qi,i->...', bd_u_val, hij, ws, 1 / rij ** 3, J) / np.pi / 4
        Mi = np.einsum('i,q,qi,i->...', bd_un_val, ws, 1 / rij, J) / np.pi / 4
        Bi = np.einsum("k,mk,m,mk->", v, 1 / rij1, ws1, b) / 4 / np.pi
        uh[interNode_idx[i]] = Mi - Hi - Bi

    real_solution = pde.solution(Node)  # 真解值
    h = np.max(v)  # 返回的是单元测度的最大值一般是体积或者大小
    errorMatrix[k] = np.sqrt(np.sum((uh - real_solution) ** 2) * h)
    mesh.uniform_refine(1)  # 在这里将网格进行细化成更加小的网格，即类似之前椭圆方程步长减少
print(f'迭代{maxite}次，结果如下：')
print("误差：\n", errorMatrix)
print('误差比：\n', errorMatrix[0:-1] / errorMatrix[1:])