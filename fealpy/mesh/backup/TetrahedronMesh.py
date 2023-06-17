import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix
from scipy.sparse import spdiags, eye, tril, triu, bmat
from scipy.spatial import KDTree
from .Mesh3d import Mesh3d, Mesh3dDataStructure
from ..quadrature import TetrahedronQuadrature, TriangleQuadrature, GaussLegendreQuadrature
from ..decorator import timer


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
       (3, 0, 2, 1), (3, 2, 1, 0), (3, 1, 0, 2)]);
    NVC = 4
    NEC = 6
    NFC = 4
    NVF = 3
    NEF = 3

    def __init__(self, NN, cell):
        super().__init__(NN, cell)

    def number_of_vertices_of_cells(self):
        return self.NVC

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


## @defgroup MeshGenerators TetrhedronMesh Common Region Mesh Generators
## @defgroup MeshQuality
class TetrahedronMesh(Mesh3d):
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
        self.p = 1  

        self.itype = cell.dtype
        self.ftype = node.dtype

        self.celldata = {}
        self.edgedata = {}
        self.facedata = {}
        self.nodedata = {}
        self.meshdata = {}

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

    def integrator(self, q, etype=3):
        """
        @brief 获取不同维度网格实体上的积分公式 
        """
        if etype in {'cell', 3}:
            return TetrahedronQuadrature(q)
        elif etype in {'face', 2}:
            return TriangleQuadrature(q)
        elif etype in {'edge', 1}:
            return GaussLegendreQuadrature(q)

    def bc_to_point(self, bc, index=np.s_[:]):
        """
        @brief 把重心坐标积分点变换到实际网格实体上的笛卡尔坐标点
        """
        TD = bc.shape[-1] - 1 #
        node = self.node
        entity = self.entity(etype=TD)[index]
        p = np.einsum('...j, ijk->...ik', bc, node[entity])
        return p

    def edge_bc_to_point(self, bc, index=np.s_[:]):
        node = self.node
        edge = self.entity('edge')[index]
        p = np.einsum('...j, ijk->...ik', bc, node[edge])
        return p

    def face_bc_to_point(self, bc, index=np.s_[:]):
        node = self.node
        face = self.entity('face')[index]
        p = np.einsum('...j, ijk->...ik', bc, node[face])
        return p

    def cell_bc_to_point(self, bc, index=np.s_[:]):
        node = self.node
        cell = self.entity('cell')[index]
        p = np.einsum('...j, ijk->...ik', bc, node[cell])
        return p

    def multi_index_matrix(self, p, etype='cell'):
        """
        @brief 获取四面体上的 p 次的多重指标矩阵

        @param[in] p 正整数 

        @return multiIndex  ndarray with shape (ldof, 4)
        """
        if etype in {'cell', 3}:
            ldof = (p+1)*(p+2)*(p+3)//6
            idx = np.arange(1, ldof)
            idx0 = (3*idx + np.sqrt(81*idx*idx - 1/3)/3)**(1/3)
            idx0 = np.floor(idx0 + 1/idx0/3 - 1 + 1e-4) # a+b+c
            idx1 = idx - idx0*(idx0 + 1)*(idx0 + 2)/6
            idx2 = np.floor((-1 + np.sqrt(1 + 8*idx1))/2) # b+c
            multiIndex = np.zeros((ldof, 4), dtype=np.int_)
            multiIndex[1:, 3] = idx1 - idx2*(idx2 + 1)/2
            multiIndex[1:, 2] = idx2 - multiIndex[1:, 3]
            multiIndex[1:, 1] = idx0 - idx2
            multiIndex[:, 0] = p - np.sum(multiIndex[:, 1:], axis=1)
            return multiIndex
        elif etype in {'face', 2}:
            ldof = (p+1)*(p+2)//2
            idx = np.arange(0, ldof)
            idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
            multiIndex = np.zeros((ldof, 3), dtype=np.int_)
            multiIndex[:,2] = idx - idx0*(idx0 + 1)/2
            multiIndex[:,1] = idx0 - multiIndex[:,2]
            multiIndex[:,0] = p - multiIndex[:, 1] - multiIndex[:, 2]
            return multiIndex
        elif etype in {'edge', 1}:
            ldof = p+1
            multiIndex = np.zeros((ldof, 2), dtype=np.int_)
            multiIndex[:, 0] = np.arange(p, -1, -1)
            multiIndex[:, 1] = p - multiIndex[:, 0]
            return multiIndex

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

    def shape_function(self, bc, p=1, etype='cell'):
        """
        @brief 四面体单元上的形函数 
        """
        TD = bc.shape[-1] - 1 
        multiIndex = self.multi_index_matrix(p, etype=etype)
        c = np.arange(1, p+1, dtype=np.int_)
        P = 1.0/np.multiply.accumulate(c)
        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = np.arange(TD+1)
        phi = np.prod(A[..., multiIndex, idx], axis=-1)
        return phi

    def grad_shape_function(self, bc, p=1, index=np.s_[:], variables='x'):

        TD = self.top_dimension()

        multiIndex = self.multi_index_matrix(p)

        c = np.arange(1, p+1, dtype=self.itype)
        P = 1.0/np.multiply.accumulate(c)

        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)

        FF = np.einsum('...jk, m->...kjm', A[..., 1:, :], np.ones(p))
        FF[..., range(p), range(p)] = p
        np.cumprod(FF, axis=-2, out=FF)
        F = np.zeros(shape, dtype=self.ftype)
        F[..., 1:, :] = np.sum(np.tril(FF), axis=-1).swapaxes(-1, -2)
        F[..., 1:, :] *= P.reshape(-1, 1)

        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)

        Q = A[..., multiIndex, range(TD+1)]
        M = F[..., multiIndex, range(TD+1)]
        ldof = self.number_of_local_ipoints(p)
        shape = bc.shape[:-1]+(ldof, TD+1)
        R = np.zeros(shape, dtype=self.ftype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            R[..., i] = M[..., i]*np.prod(Q[..., idx], axis=-1)

        if variables == 'x':
            Dlambda = self.grad_lambda(index=index)
            gphi = np.einsum('...ij, kjm->...kim', R, Dlambda)
            return gphi #(..., NC, ldof, GD)
        elif variables == 'u':
            return R

    def grad_shape_function_on_face(self, bc, cindex, lidx, p=1, direction=True):
        pass

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

        ldof = self.number_of_local_ipoints(p)
        gdof = self.number_of_global_ipoints(p)
        ipoints = np.zeros((gdof, GD), dtype=self.ftype)
        ipoints[:NN, :] = node

        if p > 1:
            NE = self.number_of_edges()
            edge = self.entity('edge') 
            w = np.zeros((p-1,2), dtype=self.ftype)
            w[:, 0] = np.arange(p-1, 0, -1)/p
            w[:, 1] = w[-1::-1, 0]
            ipoints[NN:NN+(p-1)*NE, :] = np.einsum('ij, kj...->ki...', w, node[edge,:]).reshape(-1, GD) 

        if p > 2:
            mi = self.multi_index_matrix(p, 'face') 
            NF = self.number_of_faces()
            fidof = (p+1)*(p+2)//2 - 3*p
            face = self.entity('face') 
            isInFaceIPoints = np.sum(mi > 0, axis=-1) == 3
            w = mi[isInFaceIPoints, :]/p
            ipoints[NN+(p-1)*NE:NN+(p-1)*NE+fidof*NF, :] = np.einsum('ij, kj...->ki...', w, node[face, :]).reshape(-1, GD)

        if p > 3:
            mi = self.multi_index_matrix(p, 'cell') 
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
    
    def edge_to_ipoint(self, p, index=np.s_[:]):
        """
        @brief 获取网格中每条边与插值点的对应关系
        """
        NN = self.number_of_nodes()
        NE = self.number_of_edges()

        base = NN
        edge = self.entity('edge')
        edge2ipoint = np.zeros((NE, p+1), dtype=np.int_)
        edge2ipoint[:, [0, -1]] = edge
        if p > 1:
            edge2ipoint[:,1:-1] = base + np.arange(NE*(p-1)).reshape(NE, p-1)
        return edge2ipoint

    def face_to_ipoint(self, p, index=np.s_[:]):
        """
        @brief 获取网格中每个三角形面与插值点的对应关系
        """
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

        faceIdx = self.multi_index_matrix(p, etype='face') 
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
        m2 = self.multi_index_matrix(p, 'face').T
        m3 = self.multi_index_matrix(p, 'cell').T
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
        NVC = self.ds.NVC 
        NC = len(cell)

        cell = np.r_['1', np.zeros((NC, 1), dtype=cell.dtype), cell]
        cell[:, 0] = NVC

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

    def edge_length(self, index=np.s_[:]):
        """
        @brief 计算网格边的长度
        """
        edge = self.entity('edge')
        node = self.entity('node')
        v = node[edge[index, 1]] - node[edge[index, 0]]
        length = np.sqrt(np.sum(v**2, axis=-1))
        return length


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

    def bisect(self, isMarkedCell=None, data=None, returnim=False):

        NN = self.number_of_nodes()
        NC = self.number_of_cells()

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
            data = np.zeros(4*NC, dtype=self.itype)
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

        if returnim is True:
            return IM

    @timer
    def uniform_refine(self, n=1):
        """
        Perform uniform refinement on the tetrahedral mesh.

        @param n Number of refinement iterations (default: 1)
        """
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
 
            self.ds.reinit(NN+NE, newCell)

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
