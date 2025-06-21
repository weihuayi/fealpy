
"""

Notes
-----
"""
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye, tril, triu
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.linalg import spsolve

from ..mesh import TriangleMesh, HalfEdgeMesh2d 


class ComputationalConformalGeometryAlg():
    def __init__(self):
        """

        Notes
        -----

        传入一个三角化曲面网格，
        """
        pass

    def tri_cut_graph(self, mesh):
        """
        Notes
        -----
            给定一个封闭曲面三角形网格， 得到这个网格的割图基本元。

            1. 考虑把边长当做权重，看结果如何？
            2. 单元树中应该加入什么样的权重？
            3. 什么的生成树是最优的生成树？应该设定什么样的标准？

        Authors
        -----
            Huayi Wei
            Chunyu Chen
        """

        # HalfEdgeMesh2d 可以存储表示非常丰富的二维网格类型
        # 
        # NV == 3， 说明网格是三角形网格
        assert mesh.ds.NV == 3

        # 检查网格的封闭性 
        # 该程序只处理定向封闭曲面
        # HalfEdgeMesh2d 存储的网格单元编号从 cellstart 开始， 0:cellstart 编号
        # 的意义是网格中的洞或者外界区域
        # cellstart == 0, 表示网格是封闭的
        assert mesh.ds.cellstart == 0 

        # 获取单元实体的个数
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()

        cell2cell = mesh.ds.cell_to_cell(return_sparse=True)
        celltree = minimum_spanning_tree(cell2cell)
        celltree = celltree + celltree.T # 对称化

        flag = np.asarray(celltree[edge2cell[:, 0], edge2cell[:, 1]]).reshape(-1)
        index, = np.nonzero(flag == 0) # 不在 celltree 中的边编号，形成初始的割图 
    
        cutEdge = edge[index]
        nc = len(cutEdge)

        val = np.ones(nc, dtype=np.bool_)
        n2n = csr_matrix((val, (cutEdge[:, 0], cutEdge[:, 1])), shape=(NN, NN))
        n2n += csr_matrix((val, (cutEdge[:, 1], cutEdge[:, 0])), shape=(NN, NN))
        ctree = minimum_spanning_tree(n2n) # 初始割图的最小生成树
        ctree = ctree + ctree.T # 对称化

        flag = np.asarray(ctree[cutEdge[:, 0], cutEdge[:, 1]]).reshape(-1)
        flag = (flag == 0)
        index0 = index[flag] # 没有在生成树中的边的编号
        index1 = index[~flag]# 在生成树中的边的编号

        # 这里可以多线程并行处理
        gamma = []
        count = np.zeros(NN, dtype=np.int8)
        for i in index0:
            isKeepEdge = np.ones(len(index1), dtype=np.bool_)
            while True:
                np.add.at(count, edge[i], 1)
                np.add.at(count, edge[index1[isKeepEdge]], 1)
                isDelEdge = (count[edge[index1, 0]] == 1) | (count[edge[index1, 1]] == 1)
                count[:] = 0
                if np.any(isDelEdge):
                    isKeepEdge = isKeepEdge & (~isDelEdge)
                else:
                    break
            loop = np.r_['0', i, index1[isKeepEdge]]
            gamma.append(loop)

        return gamma

    def harmonic_map(self, mesh):

        """

        Notes
        -----

        读入一个与圆盘拓扑同胚的人脸曲面，计算计算它到单位圆的 Harmonic 映射
        """

        NN = mesh.number_of_nodes()
        NC = mesh.number_of_cells()
        cell = mesh.entity('cell')
        area = mesh.entity_measure('cell')
        Dlambda = mesh.grad_lambda()


        # 组装 Laplace 算子离散矩阵 
        A = np.einsum('jkl, jml, j->jkm', Dlambda, Dlambda, area)
        I = np.broadcast_to(cell[:, :, None], shape=A.shape)
        J = np.broadcast_to(cell[:, None, :], shape=A.shape)
        A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(NN, NN))

        isBdNode = mesh.ds.boundary_node_flag()
        NBN = isBdNode.sum()

        halfedge = mesh.entity('halfedge')
        bhedge = np.zeros(NBN, dtype=mesh.itype) # 逆时针存储边界的主半边
        start = mesh.ds.hcell[0] # 无界区域起始半边
        end = start 
        i = 0
        bhedge[i] = halfedge[start, 4]
        while True: 
            i += 1
            start = halfedge[start, 3] # 前一条半边
            if start == end:
                break
            else:
                bhedge[i] = halfedge[start, 4]

        node = mesh.entity('node')

        bc = np.sum(node, axis=0)/NN
        h = np.max(node, axis=0) - np.min(node, axis=0)
        bc[2] += h[2]

        v = node[halfedge[bhedge, 0]] - node[halfedge[halfedge[bhedge, 4], 0]]
        theta = np.zeros(NBN + 1, dtype=mesh.ftype)
        theta[1:] = np.sum(v**2, axis=-1)
        np.cumsum(theta, out=theta)
        theta /= theta[-1]
        theta *= 2*np.pi

        node = np.zeros((NN, 3), dtype=mesh.ftype)
        idx = halfedge[halfedge[bhedge, 4], 0]
        node[idx, 0] = np.cos(theta[:-1])
        node[idx, 1] = np.sin(theta[:-1])

        x = node[:, 0] 
        F = np.zeros(NN, dtype=mesh.ftype)
        F -= A@x
        bdIdx = np.zeros(A.shape[0], dtype=np.int)
        bdIdx[isBdNode] = 1
        Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
        T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
        AD = T@A@T + Tbd
        F[isBdNode] = x[isBdNode]
        node[:, 0] = spsolve(AD, F)

        x = node[:, 1] 
        F = np.zeros(NN, dtype=mesh.ftype)
        F -= A@x
        bdIdx = np.zeros(A.shape[0], dtype=np.int_)
        bdIdx[isBdNode] = 1
        Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
        T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
        AD = T@A@T + Tbd
        F[isBdNode] = x[isBdNode]
        node[:, 1] = spsolve(AD, F)

        cmesh = HalfEdgeMesh2d(bc + h.min()/2*node, halfedge.copy(), mesh.ds.subdomain.copy(),
                NV=3)
        return cmesh



           
            
