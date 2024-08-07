from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S
from .mesh_base import SimplexMesh

class TetrahedronMesh(SimplexMesh): 
    def __init__(self, node, cell):
        super().__init__(TD=3,itype=cell.dtype,ftype=node.dtype)
        self.node = node
        self.cell = cell

        self.meshtype = 'tet'
        self.p = 1 # linear mesh

        #kwargs = {"dtype": self.cell.dtype, } # TODO: 增加 device 参数
        kwargs = bm.context(cell)
        self.localEdge = bm.tensor([
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], **kwargs)
        self.localFace = bm.tensor([
            (1, 2, 3),  (0, 3, 2), (0, 1, 3), (0, 2, 1)], **kwargs)
        self.localCell = bm.tensor([
            (0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2),
            (1, 2, 0, 3), (1, 0, 3, 2), (1, 3, 2, 0),
            (2, 0, 1, 3), (2, 1, 3, 0), (2, 3, 0, 1),
            (3, 0, 2, 1), (3, 2, 1, 0), (3, 1, 0, 2)], **kwargs)

        self.ccw = bm.tensor([0, 1, 2, 4], **kwargs)
        self.construct()
        self.OFace = bm.tensor([
            (1, 2, 3),  (0, 3, 2), (0, 1, 3), (0, 2, 1)], **kwargs)
        self.SFace = bm.tensor([
            (1, 2, 3),  (0, 2, 3), (0, 1, 3), (0, 1, 2)], **kwargs)
        self.localFace2edge = bm.tensor([
            (5, 4, 3), (5, 1, 2), (4, 2, 0), (3, 0, 1)], **kwargs)
        self.localEdge2face = bm.tensor(
                [[2, 3], [3, 1], [1, 2], [0, 3], [2, 0], [0, 1]], **kwargs)

        self.nodedata = {}
        self.edgedata = {}
        self.facedata = {} 
        self.celldata = {}
        self.meshdata = {}

    ## @ingroup MeshGenerators
    @classmethod
    def from_one_tetrahedron(cls, meshtype='equ'):
        """
        """
        if meshtype == 'equ':
            node = bm.tensor([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, bm.sqrt(bm.tensor(3))/2, 0.0],
                [0.5, bm.sqrt(bm.tensor(3))/6, bm.sqrt(bm.tensor(2/3))]], dtype=bm.float64)
        elif meshtype == 'iso':
            node = bm.tensor([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]], dtype=bm.float64)
        cell = bm.tensor([[0, 1, 2, 3]], dtype=bm.int32)
        return cls(node, cell)

    def face_to_edge_sign(self):
        face2edge = self.face_to_edge()
        edge = self.edge
        face = self.face
        NF = len(face2edge)
        NEF = 3
        face2edgeSign = bm.zeros((NF, NEF), dtype=bm.bool)
        n = [1, 2, 0]
        for i in range(3):
            face2edgeSign[:, i] = (face[:, n[i]] == edge[face2edge[:, i], 0])
        return face2edgeSign

    def face_unit_normal(self, index=_S):
        face = self.face
        node = self.node

        v01 = node[face[index, 1], :] - node[face[index, 0], :]
        v02 = node[face[index, 2], :] - node[face[index, 0], :]
        nv = bm.cross(v01, v02)
        length = bm.sqrt(bm.square(nv).sum(axis=1))
        return nv/length.reshape(-1, 1)

    def integrator(self, q, etype=3):
        """
        @brief 获取不同维度网格实体上的积分公式
        """
        if etype in {'cell', 3}:
            a = self.ftype
            print(a)
            from ..quadrature import TetrahedronQuadrature
            return TetrahedronQuadrature(q, dtype=self.ftype)
        elif etype in {'face', 2}:
            from ..quadrature import TriangleQuadrature
            return TriangleQuadrature(q, dtype=self.dtype)
        elif etype in {'edge', 1}:
            from ..quadrature import GaussLegendreQuadrature
            return GaussLegendreQuadrature(q, dtype=self.dtype)

    def cell_volume(self, index=_S):
        """
        @brief 计算网格单元的体积
        """
        cell = self.cell
        node = self.node
        v01 = node[cell[index, 1]] - node[cell[index, 0]]
        v02 = node[cell[index, 2]] - node[cell[index, 0]]
        v03 = node[cell[index, 3]] - node[cell[index, 0]]
        volume = bm.sum(v03*bm.cross(v01, v02), axis=1)/6.0
        return volume


    def face_area(self, index=_S):
        """
        @brief 计算所有网格面的面积
        """
        face = self.face
        node = self.node
        v01 = node[face[index, 1], :] - node[face[index, 0], :]
        v02 = node[face[index, 2], :] - node[face[index, 0], :]
        nv = bm.cross(v01, v02)
        area = bm.sqrt(bm.square(nv).sum(axis=1))/2.0
        return area


    def entity_measure(self, etype=3, index=_S):
        if etype in {'cell', 3}:
            return self.cell_volume(index=index)
        elif etype in {'face', 2}:
            return self.face_area(index=index)
        elif etype in {'edge', 1}:
            return self.edge_length(index=index)
        elif etype in {'node', 0}:
            return bm.zeros(1, dtype=self.ftype)
        else:
            raise ValueError(f"entity type: {etype} is wrong!")

    def grad_lambda(self, index=_S):
        localFace = self.localFace
        node = self.node
        cell = self.cell
        NC = self.number_of_cells() if index == _S else len(index)
        Dlambda = bm.zeros((NC, 4, 3), dtype=self.ftype)
        volume = self.entity_measure('cell', index=index)
        for i in range(4):
            j,k,m = localFace[i]
            vjk = node[cell[index, k],:] - node[cell[index, j],:]
            vjm = node[cell[index, m],:] - node[cell[index, j],:]
            Dlambda[:, i, :] = bm.cross(vjm, vjk)/(6*volume.reshape(-1, 1))
        return Dlambda

    def grad_shape_function(self, bc, p=1, index=_S, variables='x'):
        R = bm.simplex_grad_shape_function(bc, p=p)
        if variables == 'x':
            Dlambda = self.grad_lambda(index=index)
            gphi = bm.einsum('...ij, kjm->...kim', R, Dlambda)
            return gphi #(..., NC, ldof, GD)
        elif variables == 'u':
            return R

    cell_grad_shape_function = grad_shape_function

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

    def interpolation_points(self, p, index=_S):
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
        ipoints = bm.zeros((gdof, GD), dtype=self.ftype)
        ipoints[:NN, :] = node

        if p > 1:
            NE = self.number_of_edges()
            edge = self.entity('edge')
            w = bm.zeros((p-1,2), dtype=self.ftype) #TODO: fix it
            w[:, 0] = bm.arange(p-1, 0, -1)/p
            w[:, 1] = bm.flip(w,axis=0)[:,0]
            ipoints[NN:NN+(p-1)*NE, :] = bm.einsum('ij, kj...->ki...', w, node[edge,:]).reshape(-1, GD)

        if p > 2:
            mi = self.multi_index_matrix(p, TD-1, dtype=self.ftype)
            NF = self.number_of_faces()
            fidof = (p+1)*(p+2)//2 - 3*p
            face = self.entity('face')
            isInFaceIPoints = bm.sum(mi > 0, axis=-1) == 3
            w = mi[isInFaceIPoints, :]/p
            ipoints[NN+(p-1)*NE:NN+(p-1)*NE+fidof*NF, :] = bm.einsum('ij, kj...->ki...', w, node[face, :]).reshape(-1, GD)

        if p > 3:
            mi = self.multi_index_matrix(p, TD, dtype=self.ftype)
            isInCellIPoints = bm.sum(mi > 0, axis=-1) == 4
            w = mi[isInCellIPoints, :]/p
            ipoints[NN+(p-1)*NE+fidof*NF:, :] = bm.einsum('ij, kj...->ki...', w,
                    node[cell,:]).reshape(-1, GD)
        return ipoints[index]

    def face_to_ipoint(self, p, index=_S):
        """
        @brief 获取网格中每个三角形面与插值点的对应关系
        """
        TD = self.top_dimension()
        fdof = (p+1)*(p+2)//2

        edgeIdx = bm.zeros((2, p+1), dtype=bm.int64)
        edgeIdx[0, :] = bm.arange(p+1)
        edgeIdx[1, :] = bm.flip(edgeIdx[0])

        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NF = self.number_of_faces()

        face = self.entity('face')
        edge = self.entity('edge')
        face2edge = self.face_to_edge()
        edge2ipoint = self.edge_to_ipoint(p)
        face2ipoint = bm.zeros((NF, fdof), dtype=bm.int32)

        faceIdx = self.multi_index_matrix(p, TD-1, dtype=bm.float64)
        isEdgeIPoint = (faceIdx == 0)

        fe = bm.array([1, 0, 0])
        for i in range(3):
            I = bm.ones(NF, dtype=bm.int64)
            sign = (face[:, fe[i]] == edge[face2edge[:, i], 0])
            I[sign] = 0
            face2ipoint[:, isEdgeIPoint[:, i]] = edge2ipoint[face2edge[:, [i]], edgeIdx[I]]

        if p > 2:
            base = NN + (p-1)*NE
            isInFaceIPoint = ~(isEdgeIPoint[:, 0] | isEdgeIPoint[:, 1] | isEdgeIPoint[:, 2])
            fidof = fdof - 3*p
            face2ipoint[:, isInFaceIPoint] = base + bm.arange(NF*fidof,dtype=self.itype).reshape(NF, fidof)

        return face2ipoint[index]

    def cell_to_ipoint(self, p, index=_S):
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
        cell2face = self.cell_to_face()

        cell2ipoint = bm.zeros((NC, ldof), dtype=self.itype)

        face2ipoint = self.face_to_ipoint(p)
        m2 = self.multi_index_matrix(p, TD-1).T
        m3 = self.multi_index_matrix(p, TD).T
        isFaceIPoint = (m3 == 0)

        fidx = bm.argsort(face, axis=1) # 第 i 个全局面顶点做一个排序
        fidx = bm.argsort(fidx, axis=1)
        for i in range(4):
            idx = list(bm.arange(4))
            idx.remove(i)
            idxj = bm.argsort(cell[:, idx], axis=1) #  (NC, 3)

            idxi = fidx[cell2face[:, i]]

            order = idxj[bm.arange(NC).reshape(-1, 1), idxi] # (NC, 3)
            # order 满足条件: fi - fj[bm.arange(NC)[:, None], idx] = 0

            mi = m2[order]  # (NC, 3, fdof)
            k = mi[:, 1] + mi[:, 2] # (NC, fdof)
            a = k*(k+1)//2 + mi[:, 2] # (NC, fdof)
            cell2ipoint[:, isFaceIPoint[i]] = face2ipoint[cell2face[:, [i]], a]

        if p > 3:
            base = NN + (p-1)*NE + (fdof - 3*p)*NF
            idof = ldof - 4 - 6*(p - 1) - 4*(fdof - 3*p)
            isInCellIPoint = ~(isFaceIPoint[0] | isFaceIPoint[1] | isFaceIPoint[2] | isFaceIPoint[3])
            cell2ipoint[:, isInCellIPoint] = base + bm.arange(NC*idof,dtype=self.itype).reshape(NC, idof)

        return cell2ipoint

    def face_normal(self, index=_S):
        face = self.face
        node = self.node
        v01 = node[face[index, 1], :] - node[face[index, 0], :]
        v02 = node[face[index, 2], :] - node[face[index, 0], :]
        nv = bm.cross(v01, v02)
        return nv/2.0 # 长度为三角形面的面积

    def face_unit_normal(self, index=_S):
        face = self.face
        node = self.node

        v01 = node[face[index, 1], :] - node[face[index, 0], :]
        v02 = node[face[index, 2], :] - node[face[index, 0], :]
        nv = bm.cross(v01, v02)
        length = bm.sqrt(bm.square(nv).sum(axis=1))
        return nv/length.reshape(-1, 1)





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
            cell2edge = self.cell_to_edge()

            edge2newNode = bm.arange(NN, NN+NE)
            newNode = (node[edge[:, 0], :]+node[edge[:, 1], :])/2.0

            self.node = bm.concatenate((node, newNode), axis=0)

            if returnim:
                A = coo_matrix((bm.ones(NN), (range(NN), range(NN))), shape=(NN+NE, NN), dtype=self.ftype)
                A += coo_matrix((0.5*bm.ones(NE), (range(NN, NN+NE), edge[:, 0])), shape=(NN+NE, NN), dtype=self.ftype)
                A += coo_matrix((0.5*bm.ones(NE), (range(NN, NN+NE), edge[:, 1])), shape=(NN+NE, NN), dtype=self.ftype)
                nodeIMatrix.append(A.tocsr())

                B = eye(NC, dtype=self.ftype)
                B = bmat([[B], [B], [B], [B], [B], [B], [B], [B]])
                cellIMatrix.append(B.tocsr())

            p = edge2newNode[cell2edge]
            newCell = bm.zeros((8*NC, 4), dtype=self.itype)

            newCell[0:4*NC, 3] = cell.T.flatten()
            newCell[0:NC, 0:3] = p[:, [0, 2, 1]]
            newCell[NC:2*NC, 0:3] = p[:, [0, 3, 4]]
            newCell[2*NC:3*NC, 0:3] = p[:, [1, 5, 3]]
            newCell[3*NC:4*NC, 0:3] = p[:, [2, 4, 5]]

            l = bm.zeros((NC, 3), dtype=self.ftype)
            node = self.node
            l[:, 0] = bm.sum((node[p[:, 0]] - node[p[:, 5]])**2, axis=1)
            l[:, 1] = bm.sum((node[p[:, 1]] - node[p[:, 4]])**2, axis=1)
            l[:, 2] = bm.sum((node[p[:, 2]] - node[p[:, 3]])**2, axis=1)

            # Here one should connect the shortest edge
            # idx = bm.argmax(l, axis=1)
            idx = bm.argmin(l, axis=1)
            T = bm.array([
                (1, 3, 4, 2, 5, 0),
                (0, 2, 5, 3, 4, 1),
                (0, 4, 5, 1, 3, 2)
                ])[idx]
            newCell[4*NC:5*NC, 0] = p[bm.arange(NC), T[:, 0]]
            newCell[4*NC:5*NC, 1] = p[bm.arange(NC), T[:, 1]]
            newCell[4*NC:5*NC, 2] = p[bm.arange(NC), T[:, 4]]
            newCell[4*NC:5*NC, 3] = p[bm.arange(NC), T[:, 5]]

            newCell[5*NC:6*NC, 0] = p[bm.arange(NC), T[:, 1]]
            newCell[5*NC:6*NC, 1] = p[bm.arange(NC), T[:, 2]]
            newCell[5*NC:6*NC, 2] = p[bm.arange(NC), T[:, 4]]
            newCell[5*NC:6*NC, 3] = p[bm.arange(NC), T[:, 5]]

            newCell[6*NC:7*NC, 0] = p[bm.arange(NC), T[:, 2]]
            newCell[6*NC:7*NC, 1] = p[bm.arange(NC), T[:, 3]]
            newCell[6*NC:7*NC, 2] = p[bm.arange(NC), T[:, 4]]
            newCell[6*NC:7*NC, 3] = p[bm.arange(NC), T[:, 5]]

            newCell[7*NC:, 0] = p[bm.arange(NC), T[:, 3]]
            newCell[7*NC:, 1] = p[bm.arange(NC), T[:, 0]]
            newCell[7*NC:, 2] = p[bm.arange(NC), T[:, 4]]
            newCell[7*NC:, 3] = p[bm.arange(NC), T[:, 5]]
            self.cell = newCell
            self.construct()

            #self.ds.reinit(NN+NE, newCell)
    def circumcenter(self, index=_S, returnradius=False):
            """
            @brief 计算外接圆圆心和半径
            """
            node = self.node
            cell = self.cell
            v = [ node[cell[index, 0]] - node[cell[index, i]] for i in range(1,4)]
            l = [ bm.sum(vi**2, axis=1, keepdims=True) for vi in v]
            d = l[2]*bm.cross(v[0], v[1]) + l[0]*bm.cross(v[1], v[2]) + l[1]*bm.cross(v[2],v[0])
            volume = self.cell_volume(index)
            d /=12*volume[:, None]
            c = node[cell[index,0]] + d
            R = bm.sqrt(bm.sum(d**2, axis=1))
            if returnradius:
                return c, R
            else:
                return c






   
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
        node = bm.zeros((NN, 3), dtype=bm.float64)
        x = bm.linspace(box[0], box[1], nx+1, dtype=bm.float64)
        y = bm.linspace(box[2], box[3], ny+1, dtype=bm.float64)
        z = bm.linspace(box[4], box[5], nz+1, dtype=bm.float64)
        X, Y, Z = bm.meshgrid(x, y, z, indexing='ij')
 
        node = bm.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)), axis=1)

        idx = bm.arange(NN, dtype=bm.int32).reshape(nx+1, ny+1, nz+1)
        c = idx[:-1, :-1, :-1]

        nyz = (ny + 1)*(nz + 1)
        cell0 = idx[:-1, :-1, :-1] 
        cell1 = cell0 + nyz
        cell2 = cell1 + nz + 1
        cell3 = cell0 + nz + 1
        cell4 = cell0 + 1
        cell5 = cell4 + nyz
        cell6 = cell5 + nz + 1
        cell7 = cell4 + nz + 1
        cell = bm.concatenate((cell0.reshape(-1, 1), cell1.reshape(-1, 1),
            cell2.reshape(-1, 1), cell3.reshape(-1, 1), cell4.reshape(-1, 1),
            cell5.reshape(-1, 1), cell6.reshape(-1, 1), cell7.reshape(-1, 1)),
            axis = 1)

        localCell = bm.tensor([
            [0, 1, 2, 6],
            [0, 5, 1, 6],
            [0, 4, 5, 6],
            [0, 7, 4, 6],
            [0, 3, 7, 6],
            [0, 2, 3, 6]], dtype=bm.int32)
        cell = cell[:, localCell].reshape(-1, 4)

        if threshold is not None:
            NN = len(node)
            bc = bm.sum(node[cell, :], axis=1)/cell.shape[1]
            isDelCell = threshold(bc)
            cell = cell[~isDelCell]
            isValidNode = bm.zeros(NN, dtype=bm.bool)
            isValidNode[cell] = True
            node = node[isValidNode]
            idxMap = bm.zeros(NN, dtype=cell.dtype)
            idxMap[isValidNode] = bm.arange(isValidNode.sum(), dtype=cell.dtype)
            cell = idxMap[cell]
        mesh = TetrahedronMesh(node, cell)

        bdface = mesh.boundary_face_index()
        f2n = mesh.face_unit_normal()[bdface]
        isLeftBd   = bm.abs(f2n[:, 0]+1)<1e-14
        isRightBd  = bm.abs(f2n[:, 0]-1)<1e-14
        isFrontBd  = bm.abs(f2n[:, 1]+1)<1e-14
        isBackBd   = bm.abs(f2n[:, 1]-1)<1e-14
        isBottomBd = bm.abs(f2n[:, 2]+1)<1e-14
        isUpBd     = bm.abs(f2n[:, 2]-1)<1e-14
        mesh.meshdata["leftface"]   = bdface[isLeftBd]
        mesh.meshdata["rightface"]  = bdface[isRightBd]
        mesh.meshdata["frontface"]  = bdface[isFrontBd]
        mesh.meshdata["backface"]   = bdface[isBackBd]
        mesh.meshdata["upface"]     = bdface[isUpBd]
        mesh.meshdata["bottomface"] = bdface[isBottomBd]
        return mesh

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
        node = bm.array(node_coords, dtype=bm.float64).reshape(-1, 3)

        #节点的编号映射
        nodetags_map = dict({j:i for i,j in enumerate(node_tags)})

        # 获取四面体单元信息
        tetrahedron_type = 4  # 四面体单元的类型编号为 4
        tetrahedron_tags, tetrahedron_connectivity = gmsh.model.mesh.getElementsByType(tetrahedron_type)
        evid = bm.array([nodetags_map[j] for j in tetrahedron_connectivity])
        cell = evid.reshape((tetrahedron_tags.shape[-1],-1))

        # 输出节点和单元数量
        print(f"Number of nodes: {node.shape[0]}")
        print(f"Number of tetrahedra: {cell.shape[0]}")

        gmsh.finalize()
        return cls(node, cell)
 
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
        node = bm.array(node_coords, dtype=bm.float64).reshape(-1, 3)

        #节点的编号映射
        nodetags_map = dict({j:i for i,j in enumerate(node_tags)})

        # 获取四面体单元信息
        tetrahedron_type = 4  # 四面体单元的类型编号为 4
        tetrahedron_tags, tetrahedron_connectivity = gmsh.model.mesh.getElementsByType(tetrahedron_type)
        evid = bm.array([nodetags_map[j] for j in tetrahedron_connectivity])
        cell = evid.reshape((tetrahedron_tags.shape[-1],-1))

        # 输出节点和单元数量
        print(f"Number of nodes: {node.shape[0]}")
        print(f"Number of tetrahedra: {cell.shape[0]}")

        gmsh.finalize()
        return cls(node, cell)


