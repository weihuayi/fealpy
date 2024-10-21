import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.sparse import spdiags, eye, tril, triu, bmat
from scipy.spatial import KDTree
from .mesh_base import Mesh, Plotable
from .mesh_data_structure import Mesh3dDataStructure
from .mphtxt_file_reader import MPHTxtFileReader

class TetrahedronMeshDataStructure(Mesh3dDataStructure):
    OFace = np.array([(1, 2, 3),  (0, 3, 2), (0, 1, 3), (0, 2, 1)])
    SFace = np.array([(1, 2, 3),  (0, 2, 3), (0, 1, 3), (0, 1, 2)])
    ccw = np.array([0, 1, 2])

    localFace = np.array([(1, 2, 3),  (0, 3, 2), (0, 1, 3), (0, 2, 1)])
    localEdge = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    localFace2edge = np.array([(5, 4, 3), (5, 1, 2), (4, 2, 0), (3, 0, 1)])
    localEdge2face = np.array([[2, 3], [3, 1], [1, 2], [0, 3], [2, 0], [0, 1]])
    localCell = np.array([
       (0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2),
       (1, 2, 0, 3), (1, 0, 3, 2), (1, 3, 2, 0),
       (2, 0, 1, 3), (2, 1, 3, 0), (2, 3, 0, 1),
       (3, 0, 2, 1), (3, 2, 1, 0), (3, 1, 0, 2)])


    def face_to_edge_sign(self):
        face2edge = self.face_to_edge()
        edge = self.edge
        face = self.face
        NF = len(face2edge)
        NEF = 3
        face2edgeSign = np.zeros((NF, NEF), dtype=np.bool_)
        n = [1, 2, 0]
        for i in range(3):
            face2edgeSign[:, i] = (face[:, n[i]] == edge[face2edge[:, i], 0])
        return face2edgeSign

    def cell_to_face_permutation(self, locFace = None):
        """
        局部面到全局面的映射
        """
        if locFace is None:
            locFace = self.localFace

        c2f  = self.cell_to_face()
        cell = self.cell
        face = self.face
        face_g_idx = np.argsort(face)

        c2f_glo = face[c2f.reshape(-1)]
        c2f_loc = cell[:, locFace].reshape(-1, 3)

        c2f_glo = np.argsort(c2f_glo, axis=1)
        c2f_glo = np.argsort(c2f_glo, axis=1)
        c2f_loc = np.argsort(c2f_loc, axis=1)

        NC = len(cell)
        c2f_order = c2f_loc[np.arange(NC*4)[:, None], c2f_glo]
        return c2f_order.reshape(NC, 4, 3)

## @defgroup MeshGenerators TetrhedronMesh Common Region Mesh Generators
## @defgroup MeshQuality
class TetrahedronMesh(Mesh, Plotable):
    def __init__(self, node, cell, showmemory=False):
        """
        @brief Initializes a TetrahedronMesh object.

        @param[in] node The node array representing the vertices of the tetrahedral mesh.
        @param[in] cell The cell array representing the connectivity of the tetrahedral mesh.
        @param[in] showmemory Optional boolean to show memory usage (default is False).

        This method initializes a TetrahedronMesh object by setting up its attributes
        and data structures based on the provided node and cell arrays.
        """
        self.node = node
        NN = node.shape[0]
        self.ds = TetrahedronMeshDataStructure(NN, cell)

        self.meshtype = 'tet'
        self.type = 'TET'
        self.p = 1

        self.itype = cell.dtype
        self.ftype = node.dtype

        self.celldata = {}
        self.edgedata = {}
        self.facedata = {}
        self.nodedata = {}
        self.meshdata = {}

        self.edge_bc_to_point = self.bc_to_point
        self.face_bc_to_point = self.bc_to_point
        self.cell_bc_to_point = self.bc_to_point
        self.shape_function = self._shape_function
        self.cell_shape_function = self._shape_function
        self.face_shape_function = self._shape_function
        self.edge_shape_function = self._shape_function

        if showmemory:
            nsize = self.node.size*self.node.itemsize/2**30
            csize = self.ds.cell.size*self.ds.cell.itemsize/2**30
            fsize = self.ds.face.size*self.ds.face.itemsize/2**30
            esize = self.ds.edge.size*self.ds.edge.itemsize/2**30
            f2csize = self.ds.face2cell.size*self.ds.face2cell.itemsize/2**30
            c2esize = self.ds.cell2edge.size*self.ds.cell2edge.itemsize/2**30
            total = nsize + csize + fsize + esize + f2csize + c2esize
            print("memory size of node array (GB): ", nsize)
            print("memory size of cell array (GB): ", csize)
            print("memory size of face array (GB): ", fsize)
            print("memory size of edge array (GB): ", esize)
            print("memory size of face2cell array (GB): ", f2csize)
            print("memory size of cell2edge array (GB): ", c2esize)
            print("Total memory size (GB): ",  total)

    def ref_cell_measure(self):
        return 1.0/6.0

    def ref_face_measure(self):
        return 1.0/2.0

    def integrator(self, q, etype=3):
        """
        @brief 获取不同维度网格实体上的积分公式
        """
        if etype in {'cell', 3}:
            from ..quadrature import TetrahedronQuadrature
            return TetrahedronQuadrature(q)
        elif etype in {'face', 2}:
            from ..quadrature import TriangleQuadrature
            return TriangleQuadrature(q)
        elif etype in {'edge', 1}:
            from ..quadrature import GaussLegendreQuadrature
            return GaussLegendreQuadrature(q)

    def entity_measure(self, etype=3, index=np.s_[:]):
        if etype in {'cell', 3}:
            return self.cell_volume(index=index)
        elif etype in {'face', 2}:
            return self.face_area(index=index)
        elif etype in {'edge', 1}:
            return self.edge_length(index=index)
        elif etype in {'node', 0}:
            return np.zeros(1, dtype=self.ftype)
        else:
            raise ValueError(f"entity type: {etype} is wrong!")

    def cell_volume(self, index=np.s_[:]):
        """
        @brief 计算网格单元的体积
        """
        cell = self.ds.cell
        node = self.node
        v01 = node[cell[index, 1]] - node[cell[index, 0]]
        v02 = node[cell[index, 2]] - node[cell[index, 0]]
        v03 = node[cell[index, 3]] - node[cell[index, 0]]
        volume = np.sum(v03*np.cross(v01, v02), axis=1)/6.0
        return volume

    def face_area(self, index=np.s_[:]):
        """
        @brief 计算所有网格面的面积
        """
        face = self.ds.face
        node = self.node
        v01 = node[face[index, 1], :] - node[face[index, 0], :]
        v02 = node[face[index, 2], :] - node[face[index, 0], :]
        nv = np.cross(v01, v02)
        area = np.sqrt(np.square(nv).sum(axis=1))/2.0
        return area

    def grad_lambda(self, index=np.s_[:]):
        localFace = self.ds.localFace
        node = self.node
        cell = self.ds.cell
        NC = self.number_of_cells() if index == np.s_[:] else len(index)
        Dlambda = np.zeros((NC, 4, 3), dtype=self.ftype)
        volume = self.entity_measure('cell', index=index)
        for i in range(4):
            j,k,m = localFace[i]
            vjk = node[cell[index, k],:] - node[cell[index, j],:]
            vjm = node[cell[index, m],:] - node[cell[index, j],:]
            Dlambda[:, i, :] = np.cross(vjm, vjk)/(6*volume.reshape(-1, 1))
        return Dlambda
    
    def grad_face_lambda(self, index=np.s_[:]):

        node = self.entity('node')
        face = self.entity('face', index=index)
        NF = face.shape[0]
        v0 = node[face[..., 2]] - node[face[..., 1]]
        v1 = node[face[..., 0]] - node[face[..., 2]]
        v2 = node[face[..., 1]] - node[face[..., 0]]
        GD = self.geo_dimension()
        nv = np.cross(v1, v2)
        Dlambda = np.zeros((NF, 3, GD), dtype=self.ftype)

        length = np.linalg.norm(nv, axis=-1, keepdims=True)
        n = nv / length
        Dlambda[:, 0] = np.cross(n, v0) / length
        Dlambda[:, 1] = np.cross(n, v1) / length
        Dlambda[:, 2] = np.cross(n, v2) / length
        return Dlambda

    def grad_shape_function(self, bc, p=1, index=np.s_[:], variables='x'):
        R = self._grad_shape_function(bc, p=p)
        if variables == 'x':
            Dlambda = self.grad_lambda(index=index)
            gphi = np.einsum('...ij, kjm->...kim', R, Dlambda, optimize=True)
            return gphi #(..., NC, ldof, GD)
        elif variables == 'u':
            return R

    cell_grad_shape_function = grad_shape_function

    def grad_shape_function_on_face(self, bc, cindex, lidx, p=1, direction=True):
        pass

    def grad_shape_function_on_edge(self, bc, cindex, lidx, p=1, direction=True):
        pass

    def prolongation_matrix(self, p0:int, p1:int):
        """
        @brief 生成从 p0 元到 p1 元的延拓矩阵，假定 0 < p0 < p1

        @todo 测试程序正确性
        """

        assert 0 < p0 < p1

        TD = self.top_dimension()
        gdof0 = self.number_of_global_ipoints(p0)
        gdof1 = self.number_of_global_ipoints(p1)

        # 1. 网格节点上的插值点 
        NN = self.number_of_nodes()
        I = range(NN)
        J = range(NN)
        V = np.ones(NN, dtype=self.ftype)
        P = coo_matrix((V, (I, J)), shape=(gdof1, gdof0))

        # 2. 网格边内部的插值点 
        NE = self.number_of_edges()
        # p1 元在边上插值点对应的重心坐标
        bcs = self.multi_index_matrix(p1, 1)/p1 
        # p0 元基函数在 p1 元对应的边内部插值点处的函数值
        phi = self.edge_shape_function(bcs[1:-1], p=p0) # (ldof1 - 2, ldof0)  
       
        e2p1 = self.edge_to_ipoint(p1)[:, 1:-1]
        e2p0 = self.edge_to_ipoint(p0)
        shape = (NE, ) + phi.shape

        I = np.broadcast_to(e2p1[:, :, None], shape=shape).flat
        J = np.broadcast_to(e2p0[:, None, :], shape=shape).flat
        V = np.broadcast_to( phi[None, :, :], shape=shape).flat

        P += coo_matrix((V, (I, J)), shape=(gdof1, gdof0))

        # 3. 网格面内部的插值点
        if p1 > 2:
            NF = self.number_of_faces()
            # p1 元在单元上对应插值点的重心坐标
            bcs = self.multi_index_matrix(p1, 2)/p1
            flag = np.sum(bcs>0, axis=1) == 3
            # p0 元基函数在 p1 元对应的单元内部插值点处的函数值
            phi = self.face_shape_function(bcs[flag, :], p=p0)
            f2p1 = self.face_to_ipoint(p1)[:, flag]
            f2p0 = self.face_to_ipoint(p0)

            shape = (NF, ) + phi.shape

            I = np.broadcast_to(f2p1[:, :, None], shape=shape).flat
            J = np.broadcast_to(f2p0[:, None, :], shape=shape).flat
            V = np.broadcast_to( phi[None, :, :], shape=shape).flat

            P += coo_matrix((V, (I, J)), shape=(gdof1, gdof0))

        # 3. 单元内部的插值点
        if p1 > 3:
            NC = self.number_of_cells()
            # p1 元在单元上对应插值点的重心坐标
            bcs = self.multi_index_matrix(p1, 3)/p1
            flag = np.sum(bcs>0, axis=1) == 4
            # p0 元基函数在 p1 元对应的单元内部插值点处的函数值
            phi = self.cell_shape_function(bcs[flag, :], p=p0)
            c2p1 = self.cell_to_ipoint(p1)[:, flag]
            c2p0 = self.cell_to_ipoint(p0)

            shape = (NC, ) + phi.shape

            I = np.broadcast_to(c2p1[:, :, None], shape=shape).flat
            J = np.broadcast_to(c2p0[:, None, :], shape=shape).flat
            V = np.broadcast_to( phi[None, :, :], shape=shape).flat

            P += coo_matrix((V, (I, J)), shape=(gdof1, gdof0))

        return P.tocsr()

    def interpolation_points(self, p, index=np.s_[:]):
        """
        @brief 获取整个四面体网格上的全部插值点
        """

        node = self.entity('node')
        cell = self.entity('cell')

        if p == 1:
            return node

        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        GD = self.geo_dimension()
        TD = self.top_dimension()

        ldof = self.number_of_local_ipoints(p)
        gdof = self.number_of_global_ipoints(p)
        ipoints = np.zeros((gdof, GD), dtype=self.ftype)
        ipoints[:NN, :] = node

        if p > 1:
            NE = self.number_of_edges()
            edge = self.entity('edge')
            w = np.zeros((p-1,2), dtype=self.ftype) #TODO: fix it
            w[:, 0] = np.arange(p-1, 0, -1)/p
            w[:, 1] = w[-1::-1, 0]
            ipoints[NN:NN+(p-1)*NE, :] = np.einsum('ij, kj...->ki...', w, node[edge,:]).reshape(-1, GD)

        if p > 2:
            mi = self.multi_index_matrix(p, TD-1)
            NF = self.number_of_faces()
            fidof = (p+1)*(p+2)//2 - 3*p
            face = self.entity('face')
            isInFaceIPoints = np.sum(mi > 0, axis=-1) == 3
            w = mi[isInFaceIPoints, :]/p
            ipoints[NN+(p-1)*NE:NN+(p-1)*NE+fidof*NF, :] = np.einsum('ij, kj...->ki...', w, node[face, :]).reshape(-1, GD)

        if p > 3:
            mi = self.multi_index_matrix(p, TD)
            isInCellIPoints = np.sum(mi > 0, axis=-1) == 4
            w = mi[isInCellIPoints, :]/p
            ipoints[NN+(p-1)*NE+fidof*NF:, :] = np.einsum('ij, kj...->ki...', w,
                    node[cell,:]).reshape(-1, GD)
        return ipoints[index]

    def number_of_local_ipoints(self, p, iptype='cell'):
        """
        @brief 每个四面体单元上插值点的个数
        """
        if iptype in {'cell', 3}:
            return (p+1)*(p+2)*(p+3)//6
        elif iptype in {'face', 2}:
            return (p+1)*(p+2)//2
        elif iptype in {'edge', 1}:
            return self.p + 1
        elif iptype in {'node', 0}:
            return 1

    def number_of_global_ipoints(self, p):
        """
        @brief 四面体网格上插值点的总数
        """
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NF = self.number_of_faces()
        NC = self.number_of_cells()
        return NN + NE*(p-1) + NF*(p-2)*(p-1)//2 + NC*(p-3)*(p-2)*(p-1)//6

    def face_to_ipoint(self, p, index=np.s_[:]):
        """
        @brief 获取网格中每个三角形面与插值点的对应关系
        """
        TD = self.top_dimension()
        fdof = (p+1)*(p+2)//2

        edgeIdx = np.zeros((2, p+1), dtype=np.int_)
        edgeIdx[0, :] = range(p+1)
        edgeIdx[1, :] = edgeIdx[0, -1::-1]


        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NF = self.number_of_faces()

        face = self.entity('face')
        edge = self.entity('edge')
        face2edge = self.ds.face_to_edge()
        edge2ipoint = self.edge_to_ipoint(p)
        face2ipoint = np.zeros((NF, fdof), dtype=np.int_)

        faceIdx = self.multi_index_matrix(p, TD-1)
        isEdgeIPoint = (faceIdx == 0)

        fe = np.array([1, 0, 0])
        for i in range(3):
            I = np.ones(NF, dtype=np.int_)
            sign = (face[:, fe[i]] == edge[face2edge[:, i], 0])
            I[sign] = 0
            face2ipoint[:, isEdgeIPoint[:, i]] = edge2ipoint[face2edge[:, [i]], edgeIdx[I]]

        if p > 2:
            base = NN + (p-1)*NE
            isInFaceIPoint = ~(isEdgeIPoint[:, 0] | isEdgeIPoint[:, 1] | isEdgeIPoint[:, 2])
            fidof = fdof - 3*p
            face2ipoint[:, isInFaceIPoint] = base + np.arange(NF*fidof).reshape(NF, fidof)

        return face2ipoint[index]

    def cell_to_ipoint(self, p, index=np.s_[:]):
        """
        @brief 获取单元与插值点的对应关系

        @param[in] p 正整数

        @return  cell2ipoints 数组， 形状为 (NC, ldof)
        """

        TD = self.top_dimension()
        edof = p+1
        fdof = (p+1)*(p+2)//2
        ldof = (p+1)*(p+2)*(p+3)//6

        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NF = self.number_of_faces()
        NC = self.number_of_cells()

        face = self.entity('face')
        cell = self.entity('cell')
        cell2face = self.ds.cell_to_face()

        cell2ipoint = np.zeros((NC, ldof), dtype=np.int_)

        face2ipoint = self.face_to_ipoint(p)
        m2 = self.multi_index_matrix(p, TD-1).T
        m3 = self.multi_index_matrix(p, TD).T
        isFaceIPoint = (m3 == 0)

        fidx = np.argsort(face, axis=1) # 第 i 个全局面顶点做一个排序
        fidx = np.argsort(fidx, axis=1)
        for i in range(4):
            idx = list(range(4))
            idx.remove(i)
            idxj = np.argsort(cell[:, idx], axis=1) #  (NC, 3)

            idxi = fidx[cell2face[:, i]]

            order = idxj[np.arange(NC).reshape(-1, 1), idxi] # (NC, 3)
            # order 满足条件: fi - fj[np.arange(NC)[:, None], idx] = 0

            mi = m2[order]  # (NC, 3, fdof)
            k = mi[:, 1] + mi[:, 2] # (NC, fdof)
            a = k*(k+1)//2 + mi[:, 2] # (NC, fdof)
            cell2ipoint[:, isFaceIPoint[i]] = face2ipoint[cell2face[:, [i]], a]

        if p > 3:
            base = NN + (p-1)*NE + (fdof - 3*p)*NF
            idof = ldof - 4 - 6*(p - 1) - 4*(fdof - 3*p)
            isInCellIPoint = ~(isFaceIPoint[0] | isFaceIPoint[1] | isFaceIPoint[2] | isFaceIPoint[3])
            cell2ipoint[:, isInCellIPoint] = base + np.arange(NC*idof).reshape(NC, idof)

        return cell2ipoint


    def vtk_cell_type(self, etype='cell'):
        if etype in {'cell', 3}:
            VTK_TETRA = 10
            return VTK_TETRA
        elif etype in {'face', 2}:
            VTK_TRIANGLE = 5
            return VTK_TRIANGLE
        elif etype in {'edge', 1}:
            VTK_LINE = 3
            return VTK_LINE

    def to_vtk(self, etype='cell', index=np.s_[:], fname=None):
        from .vtk_extent import vtk_cell_index, write_to_vtu

        node = self.entity('node')
        GD = self.geo_dimension()

        cell = self.entity(etype)[index]
        NC = len(cell)

        cell = np.r_['1', np.zeros((NC, 1), dtype=cell.dtype), cell]
        cell[:, 0] = cell.shape[1]-1

        if etype == 'cell':
            cellType = 10  # 四面体
            celldata = self.celldata
        elif etype == 'face':
            cellType = 5  # 三角形
            celldata = self.facedata
        elif etype == 'edge':
            cellType = 3  # segment
            celldata = self.edgedata

        if fname is None:
            return node, cell.flatten(), cellType, NC
        else:
            print("Writting to vtk...")
            write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                    nodedata=self.nodedata,
                    celldata=celldata)


    def location(self, points):

        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        NP = points.shape[0]

        self.celldata['cell'] = np.zeros(NC)

        node = self.entity('node')
        cell = self.entity('cell')
        cell2cell = self.ds.cell_to_cell()

        start = np.zeros(NN, dtype=self.itype)
        start[cell[:, 0]] = range(NC)
        start[cell[:, 1]] = range(NC)
        start[cell[:, 2]] = range(NC)
        start[cell[:, 3]] = range(NC)

        tree = KDTree(node)
        _, loc = tree.query(points)
        start = start[loc] # 设置一个初始单元位置

        print("start:", start)

        self.celldata['cell'][start] = 1

        localFace = self.ds.localFace
        isNotOK = np.ones(NP, dtype=np.bool_)
        while np.any(isNotOK):
            idx = start[isNotOK] # 试探的单元编号
            pp = points[isNotOK] # 还没有找到所在单元的点的坐标

            v = node[cell[idx, :]] - pp[:, None, :] # (NP, 4, 3) - (NP, 1, 3)
            # 计算点和当前四面体四个面形成四面体的体积
            a = np.zeros((len(idx), 4), dtype=self.ftype)
            for i in range(4):
                vv = np.cross(v[:, localFace[i, 0]], v[:, localFace[i, 1]])
                a[:, i] = np.sum(vv*v[:, localFace[i, 2]], axis=-1)
            lidx = np.argmin(a, axis=-1)

            # 最小体积小于 0, 说明点在单元外
            isOutCell = a[range(a.shape[0]), lidx] < 0.0

            idx0, = np.nonzero(isNotOK)
            flag = (idx[isOutCell] == cell2cell[idx[isOutCell],
                lidx[isOutCell]])

            start[idx0[isOutCell][~flag]] = cell2cell[idx[isOutCell][~flag],
                    lidx[isOutCell][~flag]]
            start[idx0[isOutCell][flag]] = -1

            self.celldata['cell'][start[start > -1]] = 1

            isNotOK[idx0[isOutCell][flag]] = False
            isNotOK[idx0[~isOutCell]] = False

        return start

    def direction(self, i):
        """ Compute the direction on every node of

        0 <= i < 4
        """
        node = self.node
        cell = self.ds.cell
        index = self.ds.localCell
        v10 = node[cell[:, index[3*i, 0]]] - node[cell[:, index[3*i, 1]]]
        v20 = node[cell[:, index[3*i, 0]]] - node[cell[:, index[3*i, 2]]]
        v30 = node[cell[:, index[3*i, 0]]] - node[cell[:, index[3*i, 3]]]
        l1 = np.sum(v10**2, axis=1, keepdims=True)
        l2 = np.sum(v20**2, axis=1, keepdims=True)
        l3 = np.sum(v30**2, axis=1, keepdims=True)

        return l1*np.cross(v20, v30) + l2*np.cross(v30, v10) + l3*np.cross(v10, v20)

    def face_normal(self, index=np.s_[:]):
        face = self.ds.face
        node = self.node
        v01 = node[face[index, 1], :] - node[face[index, 0], :]
        v02 = node[face[index, 2], :] - node[face[index, 0], :]
        nv = np.cross(v01, v02)
        return nv/2.0 # 长度为三角形面的面积

    def face_unit_normal(self, index=np.s_[:]):
        face = self.ds.face
        node = self.node

        v01 = node[face[index, 1], :] - node[face[index, 0], :]
        v02 = node[face[index, 2], :] - node[face[index, 0], :]
        nv = np.cross(v01, v02)
        length = np.sqrt(np.square(nv).sum(axis=1))
        return nv/length.reshape(-1, 1)


    def dihedral_angle(self):
        """
        @brief 计算所有单元的四个二面角
        """
        node = self.entity('node')
        cell = self.entity('cell')
        localFace = self.ds.localFace

        n = [np.cross(node[cell[:, j],:] - node[cell[:, i],:],
            node[cell[:, k],:] - node[cell[:, i],:]) for i, j, k in localFace]
        l =[np.sqrt(np.sum(ni**2, axis=1)) for ni in n]
        n = [ ni/li.reshape(-1, 1) for ni, li in zip(n, l)]
        localEdge = self.ds.localEdge
        angle = [(np.pi - np.arccos((n[i]*n[j]).sum(axis=1)))/np.pi*180 for i,j in localEdge[-1::-1]]
        return np.array(angle).T


    def circumcenter(self, index=np.s_[:], returnradius=False):
        """
        @brief 计算外接圆圆心和半径
        """
        node = self.node
        cell = self.ds.cell
        v = [ node[cell[index, 0]] - node[cell[index, i]] for i in range(1,4)]
        l = [ np.sum(vi**2, axis=1, keepdims=True) for vi in v]
        d = l[2]*np.cross(v[0], v[1]) + l[0]*np.cross(v[1], v[2]) + l[1]*np.cross(v[2],v[0])
        volume = self.cell_volume(index)
        d /=12*volume[:, None]
        c = node[cell[index,0]] + d
        R = np.sqrt(np.sum(d**2, axis=1))
        if returnradius:
            return c, R
        else:
            return c

    ## @ingroup MeshQuality
    def cell_quality(self):
        """
        @brief  计算单元的质量，这里的质量定义单元外接球的半径比上 3 倍的内接球的半径
        """
        s = self.face_area()
        cell2face = self.ds.cell_to_face()
        ss = np.sum(s[cell2face], axis=1)
        d = self.direction(0)
        ld = np.sqrt(np.sum(d**2, axis=1))
        vol = self.cell_volume()
        R = ld/vol/12.0
        r = 3.0*vol/ss
        return R/r/3.0

    ## @ingroup MeshQuality
    def grad_quality(self):
        """
        @brief 计算单元质量关于节点坐标的导数

        """

        NN = self.number_of_nodes()
        NC = self.number_of_cells()

        cell = self.entity('cell')
        node = self.entity('node')

        s = self.face_area()
        cell2face = self.ds.cell_to_face()
        s = s[cell2face]

        ss = np.sum(s, axis=1)
        d = [self.direction(i) for i in range(4)]
        dd = np.sum(d[0]**2, axis=1)

        ld = np.sqrt(np.sum(d[0]**2, axis=1))
        vol = self.cell_volume()
        R = ld/vol/12.0
        r = 3.0*vol/ss
        q = R/r/3.0
        index = self.ds.localCell
        g = np.zeros((NC, 4, 3), dtype=self.ftype)
        w = np.zeros((NC, 4), dtype=self.ftype)
        for idx in range(12):
            i = index[idx, 0]
            j = index[idx, 1]
            k = index[idx, 2]
            m = index[idx, 3]
            vji = node[cell[:, i]] - node[cell[:, j]]
            w0 = 2.0*np.sum(np.cross(node[cell[:, i]] - node[cell[:, k]],
                node[cell[:, i]] - node[cell[:, m]])*d[i], axis=1)/dd
            w1 = 0.25*(np.sum((node[cell[:, i]] - node[cell[:,
                m]])*(node[cell[:, j]] - node[cell[:, m]]), axis=1)/s[:, k]
                    + np.sum((node[cell[:, i]] - node[cell[:,
                        k]])*(node[cell[:, j]] - node[cell[:, k]]), axis=1)/s[:, m])/ss

            g[:, i, :] += (w0 + w1).reshape(-1, 1)*vji
            w[:, i] += (w0 + w1)

            w2 = (np.sum((node[cell[:, i]] - node[cell[:, m]])**2, axis=1) -
                    np.sum((node[cell[:, i]]-node[cell[:, k]])**2, axis=1))/dd
            g[:, i, :] += w2.reshape(-1, 1)*np.cross(d[i], vji)
            g[:, i, :] += np.cross(node[cell[:, k]] + node[cell[:, j]] - 2*node[cell[:, m]], vji)/vol.reshape(-1, 1)/9.0

        g *= q.reshape(-1, 1, 1)
        w *= q.reshape(-1, 1)
        grad = np.zeros((NN, 3), dtype=self.ftype)
        np.add.at(grad, cell.flatten(), g.reshape(-1, 3))
        wgt = np.zeros(NN, dtype=self.ftype)
        np.add.at(wgt, cell.flat, w.flat)

        return grad/wgt.reshape(-1, 1)


    def label(self, node=None, cell=None, cellidx=None):
        """
        @brief 单元顶点的重新排列，使得cell[:, :2] 存储了单元的最长边

        """

        rflag = False
        if node is None:
            node = self.entity('node')

        if cell is None:
            cell = self.entity('cell')
            rflag = True

        if cellidx is None:
            cellidx = np.arange(len(cell))

        NC = cellidx.shape[0]
        localEdge = self.ds.localEdge
        totalEdge = cell[cellidx][:, localEdge].reshape(
                -1, localEdge.shape[1])
        NE = totalEdge.shape[0]
        length = np.sum(
                (node[totalEdge[:, 1]] - node[totalEdge[:, 0]])**2,
                axis = -1)
        #length += 0.1*np.random.rand(NE)*length
        cellEdgeLength = length.reshape(NC, 6)
        lidx = np.argmax(cellEdgeLength, axis=-1)

        flag = (lidx == 1)
        if  sum(flag) > 0:
            cell[cellidx[flag], :] = cell[cellidx[flag]][:, [2, 0, 1, 3]]

        flag = (lidx == 2)
        if sum(flag) > 0:
            cell[cellidx[flag], :] = cell[cellidx[flag]][:, [0, 3, 1, 2]]

        flag = (lidx == 3)
        if sum(flag) > 0:
            cell[cellidx[flag], :] = cell[cellidx[flag]][:, [1, 2, 0, 3]]

        flag = (lidx == 4)
        if sum(flag) > 0:
            cell[cellidx[flag], :] = cell[cellidx[flag]][:, [1, 3, 2, 0]]

        flag = (lidx == 5)
        if sum(flag) > 0:
            cell[cellidx[flag], :] = cell[cellidx[flag]][:, [3, 2, 1, 0]]

        if rflag == True:
            self.ds.construct()

    def uniform_bisect(self, n=1):
        for i in range(n):
            self.bisect()

    def bisect_options(self, HB = None, data=None, disp=None):
        options = {'HB' : HB, 'data': data, 'disp': disp}
        return options

    def bisect(self, isMarkedCell=None, data=None, returnim=False, options={'disp': True}):

        if options['disp']:
            print('Bisection begining.......')

        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        NE = self.number_of_edges()

        if options['disp']:
            print('Current number of nodes:', NN)
            print('Current number of edges:', NE)
            print('Current number of cells:', NC)

        if ('data' in options) and (options['data'] is not None):
            oldnode = self.entity('node')
            oldcell = self.entity('cell')
        
        if("HB" in options) & (options["HB"] is not None):
            HB = np.tile(np.arange(NC*4)[:, None], (1, 2))
            options["HB"] = HB
   
        if isMarkedCell is None: # 加密所有的单元
            markedCell = np.arange(NC, dtype=self.itype)
        else:
            markedCell, = np.nonzero(isMarkedCell)

        # allocate new memory for node and cell
        node = np.zeros((9*NN, 3), dtype=self.ftype)
        cell = np.zeros((4*NC, 4), dtype=self.itype)

        node[:NN] = self.entity('node')
        cell[:NC] = self.entity('cell')

        for key in self.celldata:
            data = np.zeros(4*NC, dtype=self.ftype)
            data[:NC] = self.celldata[key]
            self.celldata[key] = data.copy()

        # 用于存储网格节点的代数，初始所有节点都为第 0 代
        generation = np.zeros(NN + 6*NC, dtype=np.uint8)

        # 用于记录被二分的边及其中点编号
        cutEdge = np.zeros((8*NN, 3), dtype=self.itype)

        # 当前的二分边的数目
        nCut = 0

        # 非协调边的标记数组
        nonConforming = np.ones(8*NN, dtype=np.bool_)
        IM = eye(NN)
        while len(markedCell) != 0:
            # 标记最长边
            self.label(node, cell, markedCell)

            # 获取标记单元的四个顶点编号
            p0 = cell[markedCell, 0]
            p1 = cell[markedCell, 1]
            p2 = cell[markedCell, 2]
            p3 = cell[markedCell, 3]

            # 找到新的二分边和新的中点
            nMarked = len(markedCell)
            p4 = np.zeros(nMarked, dtype=self.itype)

            if nCut == 0: # 如果是第一次循环
                idx = np.arange(nMarked) # cells introduce new cut edges
            else:
                # all non-conforming edges
                ncEdge = np.nonzero(nonConforming[:nCut])
                NE = len(ncEdge)
                I = cutEdge[ncEdge][:, [2, 2]].reshape(-1)
                J = cutEdge[ncEdge][:, [0, 1]].reshape(-1)
                val = np.ones(len(I), dtype=np.bool_)
                nv2v = csr_matrix(
                        (val, (I, J)),
                        shape=(NN, NN))
                i, j =  np.nonzero(nv2v[:, p0].multiply(nv2v[:, p1]))
                p4[j] = i
                idx, = np.nonzero(p4 == 0)

            if len(idx) != 0:
                # 把需要二分的边唯一化
                NE = len(idx)
                cellCutEdge = np.array([p0[idx], p1[idx]])
                cellCutEdge.sort(axis=0)
                s = csr_matrix(
                    (
                        np.ones(NE, dtype=np.bool_),
                        (
                            cellCutEdge[0, ...],
                            cellCutEdge[1, ...]
                        )
                    ), shape=(NN, NN), dtype=np.bool_)
                # 获得唯一的边
                i, j = s.nonzero()
                nNew = len(i)
                newCutEdge = np.arange(nCut, nCut+nNew)
                cutEdge[newCutEdge, 0] = i
                cutEdge[newCutEdge, 1] = j
                cutEdge[newCutEdge, 2] = range(NN, NN+nNew)
                node[NN:NN+nNew, :] = (node[i, :] + node[j, :])/2.0
                if returnim is True:
                    val = np.full(nNew, 0.5)
                    I = coo_matrix(
                            (val, (range(nNew), i)), shape=(nNew, NN),
                            dtype=self.ftype)
                    I += coo_matrix(
                            (val, (range(nNew), j)), shape=(nNew, NN),
                            dtype=self.ftype)
                    I = bmat([[eye(NN)], [I]], format='csr')
                    IM = I@IM

                nCut += nNew
                NN += nNew

                # 新点和旧点的邻接矩阵
                I = cutEdge[newCutEdge][:, [2, 2]].reshape(-1)
                J = cutEdge[newCutEdge][:, [0, 1]].reshape(-1)
                val = np.ones(len(I), dtype=np.bool_)
                nv2v = csr_matrix(
                        (val, (I, J)),
                        shape=(NN, NN))
                i, j =  np.nonzero(nv2v[:, p0].multiply(nv2v[:, p1]))
                p4[j] = i

            # 如果新点的代数仍然为 0
            idx = (generation[p4] == 0)
            cellGeneration = np.max(
                    generation[cell[markedCell[idx]]],
                    axis=-1)
            # 第几代点
            generation[p4[idx]] = cellGeneration + 1
            cell[markedCell, 0] = p3
            cell[markedCell, 1] = p0
            cell[markedCell, 2] = p2
            cell[markedCell, 3] = p4
            cell[NC:NC+nMarked, 0] = p2
            cell[NC:NC+nMarked, 1] = p1
            cell[NC:NC+nMarked, 2] = p3
            cell[NC:NC+nMarked, 3] = p4

            for key in self.celldata:
                data = self.celldata[key]
                data[NC:NC+nMarked] = data[markedCell]

            if("HB" in options) and (options["HB"] is not None):
                HB = options['HB']
                HB[NC:NC+nMarked, 1] = HB[markedCell, 1]
            
            NC = NC + nMarked
            del cellGeneration, p0, p1, p2, p3, p4

            # 找到非协调的单元
            checkEdge, = np.nonzero(nonConforming[:nCut])
            isCheckNode = np.zeros(NN, dtype=np.bool_)
            isCheckNode[cutEdge[checkEdge]] = True
            isCheckCell = np.sum(
                    isCheckNode[cell[:NC]],
                    axis= -1) > 0
            # 找到所有包含检查节点的单元编号
            checkCell, = np.nonzero(isCheckCell)
            I = np.repeat(checkCell, 4)
            J = cell[checkCell].reshape(-1)
            val = np.ones(len(I), dtype=np.bool_)
            cell2node = csr_matrix((val, (I, J)), shape=(NC, NN))
            i, j = np.nonzero(
                    cell2node[:, cutEdge[checkEdge, 0]].multiply(
                        cell2node[:, cutEdge[checkEdge, 1]]
                        ))
            markedCell = np.unique(i)
            nonConforming[checkEdge] = False
            nonConforming[checkEdge[j]] = True;


        self.node = node[:NN]
        cell = cell[:NC]
        self.ds.reinit(NN, cell)

        for key in self.celldata:
            self.celldata[key] = self.celldata[key][:NC]
            

        if("HB" in options) & (options["HB"] is not None):
            options['HB'] = options['HB'][:NC]

        if ('data' in options) and (options['data'] is not None):
            options['data'] = self.interpolation_with_HB(oldnode, oldcell, options['HB'], options['data'])
            

        if returnim is True:
            return IM

    def interpolation_with_HB(self, oldnode, oldcell, HB, data={}):

        node = self.entity('node')
        cell = self.entity('cell')
        NN = self.number_of_nodes()
        NC = self.number_of_cells()

        v01 = oldnode[oldcell[..., 1]] - oldnode[oldcell[..., 0]]
        v02 = oldnode[oldcell[..., 2]] - oldnode[oldcell[..., 0]]
        v03 = oldnode[oldcell[..., 3]] - oldnode[oldcell[..., 0]]
        volume = np.sum(v03*np.cross(v01, v02), axis=1)/6.0
        
        idx = HB[..., 1]
        ret = {"nodedata": [], "celldata": []}

        for u0 in data['celldata']: 
            fval = u0[idx]
            ret["celldata"].append(fval)

        if 'nodedata' in data:
            lambdai = np.zeros((NC, 4, 4), dtype=np.float64)

            # 计算所有单元的第 j 个点的第 i 个重心坐标分量
            for j in range(4):
                flag = node[cell[:, j]]
                localface = np.array([[2, 1, 3], [2, 3, 0], [1, 0, 3], [0, 1, 2]])
                for i in range(4):
                    v1 = oldnode[oldcell[idx, localface[i, 0]]] - flag
                    v2 = oldnode[oldcell[idx, localface[i, 1]]] - flag
                    v3 = oldnode[oldcell[idx, localface[i, 2]]] - flag
                    volume1 = np.sum(np.cross(v2, v1)*v3, axis=-1)/6
                    lambdai[:, j, i] = volume1/volume[idx]

            fval0 = np.zeros((NC, 4), dtype=np.float64)
            for u0 in data['nodedata']: 
                fval = np.zeros((NN), dtype=np.float64)
                for i in range(4):
                    w = lambdai[:, i, :]
                    fval0[:, i] = np.sum(w*u0[oldcell[idx]], axis=1) 
                fval[cell] = fval0 
                ret["nodedata"].append(fval)
        return ret

    def uniform_refine(self, n=1, returnim=False):
        """
        Perform uniform refinement on the tetrahedral mesh.

        @param n Number of refinement iterations (default: 1)
        """
        if returnim:
            nodeIMatrix = []
            cellIMatrix = []

        for i in range(n):
            NN = self.number_of_nodes()
            NC = self.number_of_cells()
            NE = self.number_of_edges()

            node = self.entity('node')
            edge = self.entity('edge')
            cell = self.entity('cell')
            cell2edge = self.ds.cell_to_edge()

            edge2newNode = np.arange(NN, NN+NE)
            newNode = (node[edge[:, 0], :]+node[edge[:, 1], :])/2.0

            self.node = np.concatenate((node, newNode), axis=0)

            if returnim:
                A = coo_matrix((np.ones(NN), (range(NN), range(NN))), shape=(NN+NE, NN), dtype=self.ftype)
                A += coo_matrix((0.5*np.ones(NE), (range(NN, NN+NE), edge[:, 0])), shape=(NN+NE, NN), dtype=self.ftype)
                A += coo_matrix((0.5*np.ones(NE), (range(NN, NN+NE), edge[:, 1])), shape=(NN+NE, NN), dtype=self.ftype)
                nodeIMatrix.append(A.tocsr())

                B = eye(NC, dtype=self.ftype)
                B = bmat([[B], [B], [B], [B], [B], [B], [B], [B]])
                cellIMatrix.append(B.tocsr())

            p = edge2newNode[cell2edge]
            newCell = np.zeros((8*NC, 4), dtype=self.itype)

            newCell[0:4*NC, 3] = cell.flatten('F')
            newCell[0:NC, 0:3] = p[:, [0, 2, 1]]
            newCell[NC:2*NC, 0:3] = p[:, [0, 3, 4]]
            newCell[2*NC:3*NC, 0:3] = p[:, [1, 5, 3]]
            newCell[3*NC:4*NC, 0:3] = p[:, [2, 4, 5]]

            l = np.zeros((NC, 3), dtype=self.ftype)
            node = self.node
            l[:, 0] = np.sum((node[p[:, 0]] - node[p[:, 5]])**2, axis=1)
            l[:, 1] = np.sum((node[p[:, 1]] - node[p[:, 4]])**2, axis=1)
            l[:, 2] = np.sum((node[p[:, 2]] - node[p[:, 3]])**2, axis=1)

            # Here one should connect the shortest edge
            # idx = np.argmax(l, axis=1)
            idx = np.argmin(l, axis=1)
            T = np.array([
                (1, 3, 4, 2, 5, 0),
                (0, 2, 5, 3, 4, 1),
                (0, 4, 5, 1, 3, 2)
                ])[idx]
            newCell[4*NC:5*NC, 0] = p[range(NC), T[:, 0]]
            newCell[4*NC:5*NC, 1] = p[range(NC), T[:, 1]]
            newCell[4*NC:5*NC, 2] = p[range(NC), T[:, 4]]
            newCell[4*NC:5*NC, 3] = p[range(NC), T[:, 5]]

            newCell[5*NC:6*NC, 0] = p[range(NC), T[:, 1]]
            newCell[5*NC:6*NC, 1] = p[range(NC), T[:, 2]]
            newCell[5*NC:6*NC, 2] = p[range(NC), T[:, 4]]
            newCell[5*NC:6*NC, 3] = p[range(NC), T[:, 5]]

            newCell[6*NC:7*NC, 0] = p[range(NC), T[:, 2]]
            newCell[6*NC:7*NC, 1] = p[range(NC), T[:, 3]]
            newCell[6*NC:7*NC, 2] = p[range(NC), T[:, 4]]
            newCell[6*NC:7*NC, 3] = p[range(NC), T[:, 5]]

            newCell[7*NC:, 0] = p[range(NC), T[:, 3]]
            newCell[7*NC:, 1] = p[range(NC), T[:, 0]]
            newCell[7*NC:, 2] = p[range(NC), T[:, 4]]
            newCell[7*NC:, 3] = p[range(NC), T[:, 5]]
            import ipdb
            ipdb.set_trace()
            print(newCell)

            self.ds.reinit(NN+NE, newCell)

        if returnim:
            return nodeIMatrix, cellIMatrix

    def is_valid(self, threshold=1e-15):
        """
        Check if the tetrahedral mesh is valid.

        @param threshold

        @return True if all tetrahedra have positive volume, False otherwise
        """
        vol = self.cell_volume()
        return np.all(vol > threshold)

    ## @ingroup MeshGenerators
    @classmethod
    def from_meshpy(cls, points, facets, h,
        hole_points=None,
        facet_markers=None,
        point_markers=None):

        from meshpy.tet import MeshInfo, build

        mesh_info = MeshInfo()
        mesh_info.set_points(points)
        mesh_info.set_facets(facets)
        mesh = build(mesh_info, max_volume=h**3/6.0)

        node = np.array(mesh.points, dtype=np.float64)
        cell = np.array(mesh.elements, dtype=np.int_)

        return cls(node, cell)
    @classmethod
    def from_mphtxt(cls,filename):
        reader = MPHTxtFileReader(filename)
        reader.parse()
        node = reader.mesh['vertices']
        cell = reader.mesh['element']['tet']['Element']
        return cls(node,cell)

    @classmethod
    def from_step(cls,filename):
        import os
        import gmsh
        gmsh.initialize()
        gmsh.open(filename)
        gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 10)
        
        #获取所有几何实体
        ent = gmsh.model.getEntities()

        physicals = {}
        for e in ent:
            n = gmsh.model.getEntityName(e[0], e[1])
            # 获取从step读取的实体标签，并为在 / 分隔的标签路径中具有相同第三个
            # 标签的所有实体创建一个物理组
            if n:
                print('Entity ' + str(e) + ' has label ' + n + ' (and mass ' +
                      str(gmsh.model.occ.getMass(e[0], e[1])) + ')')
                path = n.split('/')
                if e[0] == 3 and len(path) > 3:
                    if (path[2] not in physicals):
                        physicals[path[2]] = []
                    physicals[path[2]].append(e[1])
        #创建物理组
        for name, tags in physicals.items():
            p = gmsh.model.addPhysicalGroup(3, tags)
            gmsh.model.setPhysicalName(3, p, name)
        gmsh.fltk.run()
        gmsh.model.mesh.generate(3)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node = np.array(node_coords, dtype=np.float64).reshape(-1, 3) 
        
        #节点的编号映射 
        nodetags_map = dict({j:i for i,j in enumerate(node_tags)})
        
        # 获取四面体单元信息
        tetrahedron_type = 4  # 四面体单元的类型编号为 4
        tetrahedron_tags, tetrahedron_connectivity = gmsh.model.mesh.getElementsByType(tetrahedron_type)
        evid = np.array([nodetags_map[j] for j in tetrahedron_connectivity])
        cell = evid.reshape((tetrahedron_tags.shape[-1],-1))
        return cls(node,cell)

    ## @ingroup MeshGenerators
    @classmethod
    def from_domain_distmesh(cls, domain, hmin, maxit=100, output=False):
        from .DistMesher3d import DistMesher3d
        mesher = DistMesher3d(domain, hmin, output=output)
        mesh = mesher.meshing(maxit)
        return mesh


    ## @ingroup MeshGenerators
    @classmethod
    def from_one_tetrahedron(cls, meshtype='equ'):
        """
        """
        if meshtype == 'equ':
            node = np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, np.sqrt(3)/2, 0.0],
                [0.5, np.sqrt(3)/6, np.sqrt(2/3)]], dtype=np.float64)
        elif meshtype == 'iso':
            node = np.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]], dtype=np.float64)
        cell = np.array([[0, 1, 2, 3]], dtype=np.int_)
        return cls(node, cell)

    ## @ingroup MeshGenerators
    @classmethod
    def from_cylinder_gmsh(cls, radius, height, lc):
        """
        @brief Generate a tetrahedral mesh for a cylinder domain
        """
        import gmsh
        gmsh.initialize()
        gmsh.model.add("Cylinder")

        # 几何定义
        gmsh.model.occ.addCylinder(0.0,0.0,0.0,0,0,height,radius)
        gmsh.model.occ.synchronize()

        # 设置网格尺寸
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0),lc)

        # 网格生成
        gmsh.model.mesh.generate(3)

        # 获取节点信息
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node = np.array(node_coords, dtype=np.float64).reshape(-1, 3)

        #节点的编号映射
        nodetags_map = dict({j:i for i,j in enumerate(node_tags)})

        # 获取四面体单元信息
        tetrahedron_type = 4  # 四面体单元的类型编号为 4
        tetrahedron_tags, tetrahedron_connectivity = gmsh.model.mesh.getElementsByType(tetrahedron_type)
        evid = np.array([nodetags_map[j] for j in tetrahedron_connectivity])
        cell = evid.reshape((tetrahedron_tags.shape[-1],-1))

        # 输出节点和单元数量
        print(f"Number of nodes: {node.shape[0]}")
        print(f"Number of tetrahedra: {cell.shape[0]}")

        gmsh.finalize()
        return cls(node, cell)


    ## @ingroup MeshGenerators
    @classmethod
    def from_unit_sphere_gmsh(cls, h):
        """
        Generate a tetrahedral mesh for a unit sphere by gmsh.

        @param h Parameter controlling mesh density
        @return TetrhedronMesh instance
        """
        import gmsh
        gmsh.initialize()
        gmsh.model.add("UnitSphere")

        # 创建球体
        gmsh.model.occ.addSphere(0.0,0.0,0.0,1,1)

        # 同步几何模型
        gmsh.model.occ.synchronize()

        # 设置网格尺寸
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0),h)

        # 生成网格
        gmsh.model.mesh.generate(3)

        # 获取节点信息
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node = np.array(node_coords, dtype=np.float64).reshape(-1, 3)

        #节点的编号映射
        nodetags_map = dict({j:i for i,j in enumerate(node_tags)})

        # 获取四面体单元信息
        tetrahedron_type = 4  # 四面体单元的类型编号为 4
        tetrahedron_tags, tetrahedron_connectivity = gmsh.model.mesh.getElementsByType(tetrahedron_type)
        evid = np.array([nodetags_map[j] for j in tetrahedron_connectivity])
        cell = evid.reshape((tetrahedron_tags.shape[-1],-1))

        # 输出节点和单元数量
        print(f"Number of nodes: {node.shape[0]}")
        print(f"Number of tetrahedra: {cell.shape[0]}")

        gmsh.finalize()
        return cls(node, cell)
    
    ## @ingroup MeshGenerators
    @classmethod
    def from_fuel_rod_gmsh(cls,R1,R2,L,w,h,l,p,meshtype='segmented'):
        """
        Generate a tetrahedron mesh for a fuel-rod region by gmsh

        @param R1 The radius of semicircles
        @param R2 The radius of quarter circles
        @param L The length of straight segments
        @param w The thickness of caldding
        @param h Parameter controlling mesh density
        @param l The length of the fuel-rod
        @param p The pitch of the fuel-rod
        @return TetrahedronMesh instance
        """
        import gmsh
        import math
        gmsh.initialize()
        gmsh.model.add("fuel_rod_3D")

        # 内部单元大小
        Lc1 = h
        # 包壳单元大小
        Lc2 = h/2.5

        factory = gmsh.model.geo
        # 外圈点
        factory.addPoint( -R1 -R2 -L, 0 , 0 , Lc2 , 1 )#圆心1
        factory.addPoint( -R1 -R2 -L, -R1 , 0 , Lc2 , 2)
        factory.addPoint( -R1 -R2 , -R1 , 0 , Lc2 , 3)
        factory.addPoint( -R1 -R2 , -R1 -R2 , 0 , Lc2 , 4)#圆心2
        factory.addPoint( -R1 , -R1 -R2 , 0 , Lc2 , 5)
        factory.addPoint( -R1 , -R1 -R2 -L , 0 , Lc2 , 6)
        factory.addPoint( 0 , -R1 -R2 -L , 0 , Lc2 , 7)#圆心3
        factory.addPoint( R1 , -R1 -R2 -L , 0 , Lc2 , 8)
        factory.addPoint( R1 , -R1 -R2 , 0 , Lc2 , 9)
        factory.addPoint( R1 +R2 , -R1 -R2 , 0, Lc2 , 10)#圆心4
        factory.addPoint( R1 +R2 , -R1 , 0 , Lc2 , 11) 
        factory.addPoint( R1 +R2 +L , -R1 , 0 , Lc2 , 12)
        factory.addPoint( R1 +R2 +L , 0 , 0 , Lc2 , 13)#圆心5
        factory.addPoint( R1 +R2 +L , R1 , 0 , Lc2 , 14)
        factory.addPoint( R1 +R2 , R1 , 0 , Lc2 , 15)
        factory.addPoint( R1 +R2 , R1 +R2 , 0 , Lc2 , 16)#圆心6
        factory.addPoint( R1 , R1 +R2 , 0 , Lc2 , 17)
        factory.addPoint( R1 , R1 +R2 +L , 0 , Lc2 , 18)
        factory.addPoint( 0 , R1 +R2 +L , 0 , Lc2 , 19)#圆心7
        factory.addPoint( -R1 , R1 +R2 +L , 0 , Lc2 , 20)
        factory.addPoint( -R1 , R1 +R2 , 0 , Lc2 , 21)
        factory.addPoint( -R1 -R2 , R1 +R2 , 0 , Lc2 , 22)#圆心8
        factory.addPoint( -R1 -R2 , R1 , 0 , Lc2 , 23)
        factory.addPoint( -R1 -R2 -L , R1 , 0 , Lc2 , 24)

        # 外圈线
        line_list_out = []
        for i in range(8):
            if i == 0:
                factory.addCircleArc(24 , 3*i+1 , 3*i+2, 2*i+1)
                factory.addLine( 3*i+2 , 3*i+3 , 2*(i+1) )
            else:
                factory.addCircleArc(3*i , 3*i+1 , 3*i+2 , 2*i+1)
                factory.addLine( 3*i+2 , 3*i+3 , 2*(i+1) )
            # 填充线环中的线
            line_list_out.append(2*i+1)
            line_list_out.append(2*(i+1))
        # 生成外圈线环
        factory.addCurveLoop(line_list_out,17)

        # 内圈点
        factory.addPoint( -R1 -R2 -L, -R1 +w , 0 , Lc1 , 25)
        factory.addPoint( -R1 -R2 , -R1 +w , 0 , Lc1 , 26)
        factory.addPoint( -R1 +w , -R1 -R2 , 0 , Lc1 , 27)
        factory.addPoint( -R1 +w , -R1 -R2 -L , 0 , Lc1 , 28)
        factory.addPoint( R1 -w , -R1 -R2 -L , 0 , Lc1 , 29)
        factory.addPoint( R1 -w , -R1 -R2 , 0 , Lc1 , 30)
        factory.addPoint( R1 +R2 , -R1 +w , 0 , Lc1 , 31) 
        factory.addPoint( R1 +R2 +L , -R1 +w , 0 , Lc1 , 32)
        factory.addPoint( R1 +R2 +L , R1 -w , 0 , Lc1 , 33)
        factory.addPoint( R1 +R2 , R1 -w , 0 , Lc1 , 34)
        factory.addPoint( R1 -w , R1 +R2 , 0 , Lc1 , 35)
        factory.addPoint( R1 -w , R1 +R2 +L , 0 , Lc1 , 36)
        factory.addPoint( -R1 +w , R1 +R2 +L , 0 , Lc1 , 37)
        factory.addPoint( -R1 +w , R1 +R2 , 0 , Lc1 , 38)
        factory.addPoint( -R1 -R2 , R1 -w, 0 , Lc1 , 39)
        factory.addPoint( -R1 -R2 -L , R1 -w, 0 , Lc1 , 40)

        # 内圈线
        line_list_in = []
        for j in range(8):
            if j == 0:
                factory.addCircleArc(40 , 3*j+1 , 25+2*j , 18+2*j)
                factory.addLine(25+2*j , 26+2*j , 19+2*j)
            else:
                factory.addCircleArc(24+2*j , 3*j+1 , 25+2*j, 18+2*j)
                factory.addLine(25+2*j , 26+2*j , 19+2*j)
            line_list_in.append(18+2*j)
            line_list_in.append(19+2*j)
        # 生成内圈线环  
        factory.addCurveLoop(line_list_in,34)

        # 内圈面
        factory.addPlaneSurface([34],35)
        # 包壳截面
        factory.addPlaneSurface([17, 34],36)

        factory.synchronize()

        N = math.ceil((2*l)/p)
        angle = ((2*l)/p* math.pi) / N
        nsection = math.ceil(l/(N*h))
        if meshtype == 'segmented':
            for i in range(N):
                if i == 0:
                    ov1 = factory.twist([(2,35)],0,0,0,0,0,l/N,0,0,1,angle,[nsection],[],False)
                    ov2 = factory.twist([(2,36)],0,0,0,0,0,l/N,0,0,1,angle,[nsection],[],False)
                else:
                    ov1 = factory.twist([(2,ov1[0][1])],0,0,0,0,0,l/N,0,0,1,angle,[nsection],[],False)
                    ov2 = factory.twist([(2,ov2[0][1])],0,0,0,0,0,l/N,0,0,1,angle,[nsection],[],False)
        elif meshtype == 'unsegmented':
            for i in range(N):
                if i == 0:
                    ov1 = factory.twist([(2,35)],0,0,0,0,0,l/N,0,0,1,angle)
                    ov2 = factory.twist([(2,36)],0,0,0,0,0,l/N,0,0,1,angle)
                else:
                    ov1 = factory.twist([(2,ov1[0][1])],0,0,0,0,0,l/N,0,0,1,angle)
                    ov2 = factory.twist([(2,ov2[0][1])],0,0,0,0,0,l/N,0,0,1,angle)

        factory.synchronize()
        # 生成网格
        gmsh.model.mesh.generate(3)
        # 获取节点信息
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node = np.array(node_coords, dtype=np.float64).reshape(-1, 3)

        #节点的编号映射
        nodetags_map = dict({j:i for i,j in enumerate(node_tags)})

        # 获取四面体单元信息
        tetrahedron_type = 4  
        tetrahedron_tags, tetrahedron_connectivity = gmsh.model.mesh.getElementsByType(tetrahedron_type)
        evid = np.array([nodetags_map[j] for j in tetrahedron_connectivity])
        cell = evid.reshape((tetrahedron_tags.shape[-1],-1))

        gmsh.finalize()
        print(f"Number of nodes: {node.shape[0]}")
        print(f"Number of cells: {cell.shape[0]}")

        return cls(node,cell)
    
    ## @ingroup MeshGenerators
    @classmethod
    def from_unit_cube(cls, nx=10, ny=10, nz=10, threshold=None):
        """
        Generate a tetrahedral mesh for a unit cube.

        @param nx Number of divisions along the x-axis (default: 10)
        @param ny Number of divisions along the y-axis (default: 10)
        @param nz Number of divisions along the z-axis (default: 10)
        @param threshold Optional function to filter cells based on their barycenter coordinates (default: None)
        @return TetrahedronMesh instance
        """
        return cls.from_box(box=[0, 1, 0, 1, 0, 1], nx=nx, ny=ny, nz=nz, threshold=threshold)

    ## @ingroup MeshGenerators
    @classmethod
    def from_box(cls, box=[0, 1, 0, 1, 0, 1], nx=10, ny=10, nz=10, threshold=None):
        """
        Generate a tetrahedral mesh for a box domain.

        @param nx Number of divisions along the x-axis (default: 10)
        @param ny Number of divisions along the y-axis (default: 10)
        @param nz Number of divisions along the z-axis (default: 10)
        @param threshold Optional function to filter cells based on their barycenter coordinates (default: None)
        @return TetrahedronMesh instance
        """
        NN = (nx+1)*(ny+1)*(nz+1)
        NC = nx*ny*nz
        node = np.zeros((NN, 3), dtype=np.float64)
        X, Y, Z = np.mgrid[
                box[0]:box[1]:(nx+1)*1j,
                box[2]:box[3]:(ny+1)*1j,
                box[4]:box[5]:(nz+1)*1j
                ]
        node[:, 0] = X.flat
        node[:, 1] = Y.flat
        node[:, 2] = Z.flat

        idx = np.arange(NN).reshape(nx+1, ny+1, nz+1)
        c = idx[:-1, :-1, :-1]

        cell = np.zeros((NC, 8), dtype=np.int_)
        nyz = (ny + 1)*(nz + 1)
        cell[:, 0] = c.flatten()
        cell[:, 1] = cell[:, 0] + nyz
        cell[:, 2] = cell[:, 1] + nz + 1
        cell[:, 3] = cell[:, 0] + nz + 1
        cell[:, 4] = cell[:, 0] + 1
        cell[:, 5] = cell[:, 4] + nyz
        cell[:, 6] = cell[:, 5] + nz + 1
        cell[:, 7] = cell[:, 4] + nz + 1

        localCell = np.array([
            [0, 1, 2, 6],
            [0, 5, 1, 6],
            [0, 4, 5, 6],
            [0, 7, 4, 6],
            [0, 3, 7, 6],
            [0, 2, 3, 6]], dtype=np.int_)
        cell = cell[:, localCell].reshape(-1, 4)

        if threshold is not None:
            NN = len(node)
            bc = np.sum(node[cell, :], axis=1)/cell.shape[1]
            isDelCell = threshold(bc)
            cell = cell[~isDelCell]
            isValidNode = np.zeros(NN, dtype=np.bool_)
            isValidNode[cell] = True
            node = node[isValidNode]
            idxMap = np.zeros(NN, dtype=cell.dtype)
            idxMap[isValidNode] = range(isValidNode.sum())
            cell = idxMap[cell]
        mesh = TetrahedronMesh(node, cell)

        bdface = mesh.ds.boundary_face_index()
        f2n = mesh.face_unit_normal()[bdface]
        isLeftBd   = np.abs(f2n[:, 0]+1)<1e-14
        isRightBd  = np.abs(f2n[:, 0]-1)<1e-14
        isFrontBd  = np.abs(f2n[:, 1]+1)<1e-14
        isBackBd   = np.abs(f2n[:, 1]-1)<1e-14
        isBottomBd = np.abs(f2n[:, 2]+1)<1e-14
        isUpBd     = np.abs(f2n[:, 2]-1)<1e-14
        mesh.meshdata["leftface"]   = bdface[isLeftBd]
        mesh.meshdata["rightface"]  = bdface[isRightBd]
        mesh.meshdata["frontface"]  = bdface[isFrontBd]
        mesh.meshdata["backface"]   = bdface[isBackBd]
        mesh.meshdata["upface"]     = bdface[isUpBd]
        mesh.meshdata["bottomface"] = bdface[isBottomBd]
        return mesh

    ## @ingroup MeshGenerators
    @classmethod
    def from_crack_box(cls, box=[0, 2, 0, 5, 0, 10], nx=2, ny=5, nz=10,
            threshold=None):
        """
        Generate a tetrahedral mesh for a box domain.

        @param nx Number of divisions along the x-axis (default: 10)
        @param ny Number of divisions along the y-axis (default: 10)
        @param nz Number of divisions along the z-axis (default: 10)
        @param threshold Optional function to filter cells based on their barycenter coordinates (default: None)
        @return TetrahedronMesh instance
        """
        NN = (nx+1)*(ny+1)*(nz+1)
        NC = nx*ny*nz
        node = np.zeros((NN, 3), dtype=np.float64)
        X, Y, Z = np.mgrid[
                box[0]:box[1]:(nx+1)*1j,
                box[2]:box[3]:(ny+1)*1j,
                box[4]:box[5]:(nz+1)*1j
                ]
        node[:, 0] = X.flat
        node[:, 1] = Y.flat
        node[:, 2] = Z.flat

        idx = np.arange(NN).reshape(nx+1, ny+1, nz+1)
        c = idx[:-1, :-1, :-1]

        cell = np.zeros((NC, 8), dtype=np.int_)
        nyz = (ny + 1)*(nz + 1)
        cell[:, 0] = c.flatten()
        cell[:, 1] = cell[:, 0] + nyz
        cell[:, 2] = cell[:, 1] + nz + 1
        cell[:, 3] = cell[:, 0] + nz + 1
        cell[:, 4] = cell[:, 0] + 1
        cell[:, 5] = cell[:, 4] + nyz
        cell[:, 6] = cell[:, 5] + nz + 1
        cell[:, 7] = cell[:, 4] + nz + 1

        localCell = np.array([
            [0, 1, 2, 6],
            [0, 5, 1, 6],
            [0, 4, 5, 6],
            [0, 7, 4, 6],
            [0, 3, 7, 6],
            [0, 2, 3, 6]], dtype=np.int_)
        cell = cell[:, localCell].reshape(-1, 4)

        if threshold is not None:
            NN = len(node)
            bc = np.sum(node[cell, :], axis=1)/cell.shape[1]
            isDelCell = threshold(bc)
            cell = cell[~isDelCell]
            isValidNode = np.zeros(NN, dtype=np.bool_)
            isValidNode[cell] = True
            node = node[isValidNode]
            idxMap = np.zeros(NN, dtype=cell.dtype)
            idxMap[isValidNode] = range(isValidNode.sum())
            cell = idxMap[cell]

        # 切口节点重复
        NN = node.shape[0]
        # 找到切口处 node
        nidx = np.where((np.abs(node[:, 2] - 5)<1e-5) & (node[:, 1] > 3.01))[0]
        # 找到切口处节点所在单元
        nidxmap = np.arange(NN, dtype=np.int_)

        nidxmap[nidx] = NN+np.arange(len(nidx))
        # 计算 z 坐标平均值
        flag = np.mean(node[:, 2][cell], axis=1)>5
        cell[flag] = nidxmap[cell[flag]]

        node = np.r_[node, node[nidx]]
        mesh = TetrahedronMesh(node, cell)
        return mesh


    def print_cformat(self):
        def print_cpp_array(arr):
            print("int arr[{}][{}] = {{".format(arr.shape[0], arr.shape[1]))
            for i in range(arr.shape[0]):
                if(i%4==3):
                    print("{" + ", ".join(str(x) for x in arr[i]) + "},", end='\n')
                elif(i%4==0):
                    print("    {" + ", ".join(str(x) for x in arr[i]) + "},", end='')
                else:
                    print("{" + ", ".join(str(x) for x in arr[i]) + "},", end='')
            print("};")

        print("Node:")
        print_cpp_array(self.node)
        print("Cell:")
        print_cpp_array(self.ds.cell)
        print("Edge:")
        print_cpp_array(self.ds.edge)
        print("Face:")
        print_cpp_array(self.ds.face)
        print("Face2cell:")
        print_cpp_array(self.ds.face2cell)
        print("Cell2face:")
        print_cpp_array(self.ds.cell_to_face())

    def to_mfem_file(self, filename, isBdFace=None):
        """!
        @brief 将网格保存为 MFEM 网格文件格式
               格式类型见 : https://mfem.org/mesh-format-v1.0/
        """
        # Open file for writing
        with open(filename, 'w') as f:
            # Write header
            f.write("MFEM mesh v1.0\n\n")
            
            # Write dimension
            dim = 3
            f.write("dimension\n")
            f.write(str(dim) + "\n\n")
            
            # Write elements
            NC = self.number_of_cells()
            f.write("elements\n")
            f.write(str(NC) + "\n")

            cellattr = np.ones(NC, dtype=np.int_)
            if 'attributes' in self.celldata:
                cellattr = self.celldata['attributes']
            cell = self.entity('cell')
            for i in range(NC):
                attr = cellattr[i]
                geom_type = '4'
                s = " ".join([str(j) for j in cell[i]])
                f.write(str(attr) + " " + geom_type + " " + s + "\n")
            f.write("\n")
            
            # Write boundary
            if isBdFace is None:
                isBdFace = self.ds.boundary_face_flag()
            boundary = self.entity('face')[isBdFace]

            NB = np.sum(isBdFace)
            faceattr = np.ones(NB, dtype=np.int_)
            if 'attributes' in self.facedata:
                faceattr = self.facedata['attributes'][isBdFace]

            f.write("boundary\n")
            f.write(str(NB) + "\n")
            for i in range(NB):
                attr = faceattr[i]
                geom_type = '2'
                s = " ".join([str(j) for j in boundary[i]])
                f.write(str(attr) + " " + geom_type + " " + s + "\n")
            f.write("\n")
            
            # Write vertices
            NN = self.number_of_nodes()
            node = self.entity('node')
            vdim = node.shape[1]

            f.write("vertices\n")
            f.write(str(NN) + "\n")
            f.write(str(vdim) + "\n")
            for i in range(NN):
                x = node[i]
                s = " ".join([str(x[j]) for j in range(vdim)])
                f.write(s + "\n")

TetrahedronMesh.set_ploter('3d')
