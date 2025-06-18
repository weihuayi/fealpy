from typing import Union, Optional
from math import sqrt
from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S
from .mesh_base import SimplexMesh
from .plot import Plotable
from fealpy.sparse import coo_matrix,csr_matrix

class TetrahedronMesh(SimplexMesh, Plotable): 
    def __init__(self, node, cell):
        super().__init__(TD=3, itype=cell.dtype, ftype=node.dtype)
        self.node = node
        self.cell = cell

        self.meshtype = 'tet'
        self.p = 1 # linear mesh

        #kwargs = {"dtype": self.cell.dtype, } # TODO: 增加 device 参数
        self.ikwargs = bm.context(cell)
        self.fkwargs = bm.context(node)
        self.localEdge = bm.tensor([
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], **self.ikwargs)
        self.localFace = bm.tensor([
            (1, 2, 3),  (0, 3, 2), (0, 1, 3), (0, 2, 1)], **self.ikwargs)
        self.localCell = bm.tensor([
            (0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2),
            (1, 2, 0, 3), (1, 0, 3, 2), (1, 3, 2, 0),
            (2, 0, 1, 3), (2, 1, 3, 0), (2, 3, 0, 1),
            (3, 0, 2, 1), (3, 2, 1, 0), (3, 1, 0, 2)], **self.ikwargs)

        self.ccw = bm.tensor([0, 1, 2], **self.ikwargs)
        self.construct()
        self.OFace = bm.tensor([
            (1, 2, 3),  (0, 3, 2), (0, 1, 3), (0, 2, 1)], **self.ikwargs)
        self.SFace = bm.tensor([
            (1, 2, 3),  (0, 2, 3), (0, 1, 3), (0, 1, 2)], **self.ikwargs)
        self.localFace2edge = bm.tensor([
            (5, 4, 3), (5, 1, 2), (4, 2, 0), (3, 0, 1)], **self.ikwargs)
        self.localEdge2face = bm.tensor(
                [[2, 3], [3, 1], [1, 2], [0, 3], [2, 0], [0, 1]], **self.ikwargs)

        self.nodedata = {}
        self.edgedata = {}
        self.facedata = {} 
        self.celldata = {}
        self.meshdata = {}

    def cell_to_face_permutation(self, locFace = None):
        """
        局部面到全局面的映射
        c2f_loc[c2f_order]=c2f_glo
        """
        if locFace is None:
            locFace = self.localFace

        c2f  = self.cell_to_face()
        cell = self.cell
        face = self.face
        face_g_idx = bm.argsort(face)

        c2f_glo = face[c2f.reshape(-1)]
        c2f_loc = cell[:, locFace].reshape(-1, 3)

        c2f_glo = bm.argsort(c2f_glo, axis=1)
        c2f_glo = bm.argsort(c2f_glo, axis=1)
        c2f_loc = bm.argsort(c2f_loc, axis=1)

        NC = len(cell)
        c2f_order = c2f_loc[bm.arange(NC*4)[:, None], c2f_glo]
        return c2f_order.reshape(NC, 4, 3)
    
    def cell_to_face_sign(self):
        
        NC = self.number_of_cells()
        NFC = self.number_of_faces_of_cells()
        cell2faceSign = bm.zeros((NC, NFC), dtype=bm.bool, device=self.device)
        f2c = self.face_to_cell()
        cell2faceSign[f2c[:, 0], f2c[:, 2]] = True
        return cell2faceSign


    ## @ingroup MeshGenerators
    @classmethod
    def from_one_tetrahedron(cls, meshtype='equ', device=None):
        """
        """
        if meshtype == 'equ':
            node = bm.tensor([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, sqrt(3)/2, 0.0],
                [0.5, sqrt(3)/6, sqrt(2/3)]], dtype=bm.float64, device=device)
        elif meshtype == 'iso':
            node = bm.tensor([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]], dtype=bm.float64)
        cell = bm.tensor([[0, 1, 2, 3]], dtype=bm.int32, device=device)
        return cls(node, cell)

    def face_to_edge_sign(self):
        face2edge = self.face_to_edge()
        edge = self.edge
        face = self.face
        NF = len(face2edge)
        NEF = 3
        face2edgeSign = bm.zeros((NF, NEF),dtype=bm.bool, device=self.device)
        n = [1, 2, 0]
        for i in range(3):
            face2edgeSign[:, i] = (face[:, n[i]] == edge[face2edge[:, i], 0])
        return face2edgeSign
    
    def cell_to_edge_sign(self, cell=None):
        """
        TODO: true 代表相同方向
        """
        if cell==None:
            cell = self.cell
        NC = self.number_of_cells()
        NEC = self.number_of_edges_of_cells()
        cell2edgeSign = bm.zeros((NC, NEC), dtype=bm.bool, device=self.device)
        localEdge = self.localEdge
        E = localEdge.shape[0]
        #for i, (j, k) in zip(range(E), localEdge):
        #    cell2edgeSign[:, i] = cell[:, j] < cell[:, k]
        edge = self.edge
        c2e = self.cell_to_edge()
        cell2edgeSign = edge[c2e, 0]==cell[:, localEdge[:, 0]]
        return cell2edgeSign

    def face_unit_normal(self, index=_S):
        face = self.face
        node = self.node

        v01 = node[face[index, 1], :] - node[face[index, 0], :]
        v02 = node[face[index, 2], :] - node[face[index, 0], :]
        nv = bm.cross(v01, v02, axis=1)
        length = bm.sqrt(bm.square(nv).sum(axis=1))
        return nv/length.reshape(-1, 1)

    def quadrature_formula(self, q: int, etype: Union[int, str] = 'cell',
                           qtype: str = 'legendre'):
        """
        @brief 获取不同维度网格实体上的积分公式
        """
        kwargs = {'dtype': self.ftype, 'device': self.device}

        if etype in {'cell', 3}:
            if q > 7:
                from ..quadrature.stroud_quadrature import StroudQuadrature
                return StroudQuadrature(3, q)
            else:
                from ..quadrature import TetrahedronQuadrature
                return TetrahedronQuadrature(q, **kwargs)
        elif etype in {'face', 2}:
            from ..quadrature import TriangleQuadrature
            return TriangleQuadrature(q, **kwargs)
        elif etype in {'edge', 1}:
            from ..quadrature import GaussLegendreQuadrature
            return GaussLegendreQuadrature(q, **kwargs)

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
            return bm.zeros(1, **self.fkwargs)
        else:
            raise ValueError(f"entity type: {etype} is wrong!")
    
    def grad_lambda(self, index: Index=_S, TD:int=3) -> TensorLike:
        node = self.entity('node')
        entity = self.entity(TD, index=index) 
        if TD == 1:
            return bm.interval_grad_lambda(entity, node)
        elif TD == 2:
            return bm.triangle_grad_lambda_3d(entity, node)
        elif TD == 3:
            localFace = self.localFace
            return bm.tetrahedron_grad_lambda_3d(entity, node, localFace)
        else:
            raise ValueError("Unsupported topological dimension: {TD}")

    '''
    def grad_lambda(self, index=_S):
        localFace = self.localFace
        node = self.node
        cell = self.cell
        NC = self.number_of_cells() if bm.all(index == _S) else len(index)
        Dlambda = bm.zeros((NC, 4, 3), device=self.device, **self.fkwargs)
        volume = self.entity_measure('cell', index=index)
        for i in range(4):
            j,k,m = localFace[i]
            vjk = node[cell[index, k],:] - node[cell[index, j],:]
            vjm = node[cell[index, m],:] - node[cell[index, j],:]
            Dlambda[:, i, :] = bm.cross(vjm, vjk)/(6*volume.reshape(-1, 1))
        return Dlambda
    
    def grad_face_lambda(self, index=_S):

        node = self.entity('node')
        face = self.entity('face', index=index)
        NF = face.shape[0]
        v0 = node[face[..., 2]] - node[face[..., 1]]
        v1 = node[face[..., 0]] - node[face[..., 2]]
        v2 = node[face[..., 1]] - node[face[..., 0]]
        GD = self.geo_dimension()
        nv = bm.cross(v1, v2)
        Dlambda = bm.zeros((NF, 3, GD),device=bm.get_device(face), **self.fkwargs)

        length = bm.linalg.norm(nv, axis=-1, keepdims=True)
        n = nv / length
        Dlambda[:, 0] = bm.cross(n, v0) / length
        Dlambda[:, 1] = bm.cross(n, v1) / length
        Dlambda[:, 2] = bm.cross(n, v2) / length
        return Dlambda
    '''

    def boundary_edge_flag(self):
        """
        @brief 判断边界边 
        """
        NE = self.number_of_edges()
        face2edge = self.face_to_edge()
        isBdFace = self.boundary_face_flag()
        isBdEdge = bm.zeros(NE, dtype=bm.bool, device=self.device)
        isBdEdge[face2edge[isBdFace, :]] = True
        return isBdEdge 
        

    """
    def grad_shape_function(self, bc, p=1, index=_S, variables='x'):
        R = bm.simplex_grad_shape_function(bc, p=p)
        if variables == 'x':
            Dlambda = self.grad_lambda(index=index)
            gphi = bm.einsum('...ij, kjm->...kim', R, Dlambda)
            return gphi #(..., NC, ldof, GD)
        elif variables == 'u':
            return R
    """

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
        ipoints = bm.zeros((gdof, GD), **self.fkwargs)
        ipoints[:NN, :] = node

        if p > 1:
            NE = self.number_of_edges()
            edge = self.entity('edge')
            w = bm.zeros((p-1,2), **self.fkwargs) #TODO: fix it
            w[:, 0] = bm.arange(p-1, 0, -1)/p
            w[:, 1] = bm.flip(w,axis=0)[:,0]
            ipoints[NN:NN+(p-1)*NE, :] = bm.einsum('ij, kj...->ki...', w, node[edge,:]).reshape(-1, GD)

        if p > 2:
            mi = self.multi_index_matrix(p, TD-1, **self.fkwargs)
            NF = self.number_of_faces()
            fidof = (p+1)*(p+2)//2 - 3*p
            face = self.entity('face')
            isInFaceIPoints = bm.sum(mi > 0, axis=-1) == 3
            w = mi[isInFaceIPoints, :]/p
            ipoints[NN+(p-1)*NE:NN+(p-1)*NE+fidof*NF, :] = bm.einsum('ij, kj...->ki...', w, node[face, :]).reshape(-1, GD)

        if p > 3:
            mi = self.multi_index_matrix(p, TD, **self.fkwargs)
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
            face2ipoint[:, isInFaceIPoint] = base + bm.arange(NF*fidof,**self.ikwargs).reshape(NF, fidof)

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

        cell2ipoint = bm.zeros((NC, ldof), **self.ikwargs)

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
            cell2ipoint[:, isInCellIPoint] = base + bm.arange(NC*idof,**self.ikwargs).reshape(NC, idof)

        return cell2ipoint

    def direction(self,i):
        """
        Compute the direction on every node of 0 <= i < 4
        """
        node = self.node
        cell = self.cell
        index = self.localCell
        v10 = node[cell[:, index[3*i, 0]]] - node[cell[:, index[3*i, 1]]]
        v20 = node[cell[:, index[3*i, 0]]] - node[cell[:, index[3*i, 2]]]
        v30 = node[cell[:, index[3*i, 0]]] - node[cell[:, index[3*i, 3]]]
        l1 = bm.sum(v10**2, axis=1, keepdims=True)
        l2 = bm.sum(v20**2, axis=1, keepdims=True)
        l3 = bm.sum(v30**2, axis=1, keepdims=True)

        return l1*bm.cross(v20, v30) + l2*bm.cross(v30, v10) + l3*bm.cross(v10, v20)

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


    def update_bcs(self, bcs, toetype: Union[int, str]='cell'):
        TD = bcs.shape[-1] - 1
        if toetype == 'cell' or toetype == 3: 
            if TD == 3:
                return bcs
            elif TD == 2: # edge up to cell
                result = bm.stack([bm.insert(bcs, i, 0.0, axis=-1) for i in range(4)], axis=0)
                return result
            else:
                raise ValueError("Unsupported topological dimension: {TD}")
                    
        else:
            raise ValueError("The etype only support face, other etype is not implemented.")


    def prolongation_matrix(self, p0: int, p1: int):
        """
        Return the prolongation_matrix from p0 to p1: 0 < p0 < p1

        Parameters:
            p0(int): The degree of the lowest-order space.
            p1(int): The degree of the highest-order space.

        Returns:
            CSRTensor: the prolongation_matrix from p0 to p1
        """
        assert 0 < p0 < p1

        TD = self.top_dimension()#Geometric Dimension
        gdof0 = self.number_of_global_ipoints(p0)
        gdof1 = self.number_of_global_ipoints(p1)
        matrix_shape = (gdof1,gdof0)

        # 1. Interpolation points on the mesh nodes: Inherit the original interpolation points
        NN = self.number_of_nodes()
        V_1 = bm.ones(NN)
        I_1 = bm.arange(NN)
        J_1 = bm.arange(NN)

        # 2. Interpolation points within the mesh edges
        NE = self.number_of_edges()
        bcs = self.multi_index_matrix(p1, 1) / p1  
        phi = self.edge_shape_function(bcs[1:-1], p=p0)  # (ldof1 - 2, ldof0)

        e2p1 = self.edge_to_ipoint(p1)[:, 1:-1]
        e2p0 = self.edge_to_ipoint(p0)
        shape = (NE,) + phi.shape

        I_2 = bm.broadcast_to(e2p1[:, :, None], shape=shape).flat
        J_2 = bm.broadcast_to(e2p0[:, None, :], shape=shape).flat
        V_2 = bm.broadcast_to(phi[None, :, :], shape=shape).flat

        # 3. Interpolation points within the mesh faces
        if p1 > 2:
            NF = self.number_of_faces()
            bcs = self.multi_index_matrix(p1, 2) / p1
            flag = bm.sum(bcs > 0, axis=1) == 3
            phi = self.face_shape_function(bcs[flag, :], p=p0)
            f2p1 = self.face_to_ipoint(p1)[:, flag]
            f2p0 = self.face_to_ipoint(p0)

            shape = (NF,) + phi.shape

            I_3 = bm.broadcast_to(f2p1[:, :, None], shape=shape).flat
            J_3 = bm.broadcast_to(f2p0[:, None, :], shape=shape).flat
            V_3 = bm.broadcast_to(phi[None, :, :], shape=shape).flat

        # 4. Interpolation points within the mesh cells
        if p1 > 3:
            NC = self.number_of_cells()
            bcs = self.multi_index_matrix(p1, 3)/p1
            flag = bm.sum(bcs>0, axis=1) == 4
            phi = self.shape_function(bcs[flag, :], p=p0)
            c2p1 = self.cell_to_ipoint(p1)[:, flag]
            c2p0 = self.cell_to_ipoint(p0)

            shape = (NC, ) + phi.shape

            I_4 = bm.broadcast_to(c2p1[:, :, None], shape=shape).flat
            J_4 = bm.broadcast_to(c2p0[:, None, :], shape=shape).flat
            V_4 = bm.broadcast_to( phi[None, :, :], shape=shape).flat
        
        # 5.concatenate
        if p1 <=2:
            V = bm.concatenate((V_1, V_2), axis=0) 
            I = bm.concatenate((I_1, I_2), axis=0) 
            J = bm.concatenate((J_1, J_2), axis=0) 
            P = csr_matrix((V, (I, J)), matrix_shape)
        elif p1 == 3:
            V = bm.concatenate((V_1, V_2, V_3), axis=0) 
            I = bm.concatenate((I_1, I_2, I_3), axis=0) 
            J = bm.concatenate((J_1, J_2, J_3), axis=0) 
            P = csr_matrix((V, (I, J)), matrix_shape)
        else:
            V = bm.concatenate((V_1, V_2, V_3,V_4), axis=0) 
            I = bm.concatenate((I_1, I_2, I_3,I_4), axis=0) 
            J = bm.concatenate((J_1, J_2, J_3,J_4), axis=0) 
            P = csr_matrix((V, (I, J)), matrix_shape)

        return P

    def uniform_refine(self, n=1, returnim=False):
        """
        Uniform refine the tetrahedral mesh n times.

        Parameters:
            n (int): Times refine the triangle mesh.
            returnirm (bool): Return the prolongation matrix list or not,from the finest to the the coarsest
        
        Returns:
            mesh: The mesh obtained after uniformly refining n times.
            List(CSRTensor): The prolongation matrix from the finest to the the coarsest
        """
        if returnim:
            IM = []

        for i in range(n):
            NN = self.number_of_nodes()
            NC = self.number_of_cells()
            NE = self.number_of_edges()

            node = self.entity('node')
            edge = self.entity('edge')
            cell = self.entity('cell')
            cell2edge = self.cell_to_edge()

            kargs = bm.context(cell)
            edge2newNode = bm.arange(NN, NN + NE, **kargs)
            newNode = (node[edge[:, 0], :]+node[edge[:, 1], :])/2.0

            self.node = bm.concatenate((node, newNode), axis=0)

            if returnim is True:
                shape = (NN + NE, NN)
                kargs = bm.context(node)
                values = bm.ones(NN+2*NE, **kargs) 
                values = bm.set_at(values, bm.arange(NN, NN+2*NE), 0.5)

                kargs = bm.context(cell)
                i0 = bm.arange(NN, **kargs) 
                I = bm.concatenate((i0, edge2newNode, edge2newNode))
                J = bm.concatenate((i0, edge[:, 0], edge[:, 1]))   

                P = csr_matrix((values, (I, J)), shape)

                IM.append(P)

            p = edge2newNode[cell2edge]
            newCell = bm.zeros((8*NC, 4), **self.ikwargs)

            newCell = bm.set_at(newCell , (slice(4*NC),3) , cell.T.flatten())
            newCell = bm.set_at(newCell , (slice(NC),slice(3)) , p[:,[0,2,1]])
            newCell = bm.set_at(newCell , (slice(NC,2*NC),slice(3)) , p[:, [0, 3, 4]])
            newCell = bm.set_at(newCell , (slice(2*NC , 3*NC),slice(3)) , p[:, [1, 5, 3]])
            newCell = bm.set_at(newCell , (slice(3*NC , 4*NC),slice(3)) , p[:, [2, 4, 5]])

            l = bm.zeros((NC, 3), **self.fkwargs)
            node = self.node
            l = bm.set_at(l , (slice(None) , 0) , bm.sum((node[p[:, 0]] - node[p[:, 5]])**2, axis=1))
            l = bm.set_at(l , (slice(None) , 1) , bm.sum((node[p[:, 1]] - node[p[:, 4]])**2, axis=1))
            l = bm.set_at(l , (slice(None) , 2) , bm.sum((node[p[:, 2]] - node[p[:, 3]])**2, axis=1))

            # Here one should connect the shortest edge
            # idx = bm.argmax(l, axis=1)
            idx = bm.argmin(l, axis=1)
            T = bm.array([
                (1, 3, 4, 2, 5, 0),
                (0, 2, 5, 3, 4, 1),
                (0, 4, 5, 1, 3, 2)
                ])[idx]
            newCell = bm.set_at(newCell , (slice(4*NC , 5*NC),0) , p[bm.arange(NC), T[:, 0]])
            newCell = bm.set_at(newCell , (slice(4*NC , 5*NC),1) , p[bm.arange(NC), T[:, 1]])
            newCell = bm.set_at(newCell , (slice(4*NC , 5*NC),2) , p[bm.arange(NC), T[:, 4]])
            newCell = bm.set_at(newCell , (slice(4*NC , 5*NC),3) , p[bm.arange(NC), T[:, 5]])

            newCell = bm.set_at(newCell , (slice(5*NC , 6*NC),0) , p[bm.arange(NC), T[:, 1]])
            newCell = bm.set_at(newCell , (slice(5*NC , 6*NC),1) , p[bm.arange(NC), T[:, 2]])
            newCell = bm.set_at(newCell , (slice(5*NC , 6*NC),2) , p[bm.arange(NC), T[:, 4]])
            newCell = bm.set_at(newCell , (slice(5*NC , 6*NC),3) , p[bm.arange(NC), T[:, 5]])
            
            newCell = bm.set_at(newCell , (slice(6*NC , 7*NC),0) , p[bm.arange(NC), T[:, 2]])
            newCell = bm.set_at(newCell , (slice(6*NC , 7*NC),1) , p[bm.arange(NC), T[:, 3]])
            newCell = bm.set_at(newCell , (slice(6*NC , 7*NC),2) , p[bm.arange(NC), T[:, 4]])
            newCell = bm.set_at(newCell , (slice(6*NC , 7*NC),3) , p[bm.arange(NC), T[:, 5]])

            newCell = bm.set_at(newCell , (slice(7*NC , 8*NC),0) , p[bm.arange(NC), T[:, 3]])
            newCell = bm.set_at(newCell , (slice(7*NC , 8*NC),1) , p[bm.arange(NC), T[:, 0]])
            newCell = bm.set_at(newCell , (slice(7*NC , 8*NC),2) , p[bm.arange(NC), T[:, 4]])
            newCell = bm.set_at(newCell , (slice(7*NC , 8*NC),3) , p[bm.arange(NC), T[:, 5]])
            self.cell = newCell
            self.construct()

        if returnim is True:
            IM.reverse()
            return IM

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
            cellidx = bm.arange(len(cell))

        NC = cellidx.shape[0]
        localEdge = self.localEdge
        totalEdge = cell[cellidx][:, localEdge].reshape(
                -1, localEdge.shape[1])
        NE = totalEdge.shape[0]
        length = bm.sum(
                (node[totalEdge[:, 1]] - node[totalEdge[:, 0]])**2,
                axis = -1)
        #length += 0.1*bm.random.rand(NE)*length
        cellEdgeLength = length.reshape(NC, 6)
        lidx = bm.argmax(cellEdgeLength, axis=-1)

        flag = (lidx == 1)
        if  sum(flag) > 0:
            cell = bm.set_at(cell, cellidx[flag], cell[cellidx[flag]][:, [2, 0, 1, 3]])

        flag = (lidx == 2)
        if sum(flag) > 0:
            cell = bm.set_at(cell, cellidx[flag], cell[cellidx[flag]][:, [0, 3, 1, 2]])

        flag = (lidx == 3)
        if sum(flag) > 0:
            cell = bm.set_at(cell, cellidx[flag], cell[cellidx[flag]][:, [1, 2, 0, 3]])

        flag = (lidx == 4)
        if sum(flag) > 0:
            cell = bm.set_at(cell, cellidx[flag], cell[cellidx[flag]][:, [1, 3, 2, 0]])

        flag = (lidx == 5)
        if sum(flag) > 0:
            cell = bm.set_at(cell, cellidx[flag], cell[cellidx[flag]][:, [3, 2, 1, 0]])

        if rflag == True:
            self.construct()

    def uniform_bisect(self, n=1):
        for i in range(n):
            self.bisect()

    def bisect_options(self, HB=None, data=None, disp=None):
        options = {'HB': HB, 'data': data, 'disp': disp}
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
        
        if ('HB' in options) and (options['HB'] is not None):
            HB = bm.tile(bm.arange(NC*4)[:, None], (1, 2))
            options["HB"] = HB
   
        if isMarkedCell is None: # 加密所有的单元
            markedCell = bm.arange(NC, **self.ikwargs)
        else:
            markedCell, = bm.nonzero(isMarkedCell)

        # allocate new memory for node and cell
        node = bm.zeros((9*NN, 3), **self.fkwargs)
        cell = bm.zeros((4*NC, 4), **self.ikwargs)

        node = bm.set_at(node, slice(NN), self.entity('node'))
        cell = bm.set_at(cell, slice(NC), self.entity('cell'))

        for key in self.celldata:
            data = bm.zeros(4*NC, **self.fkwargs)
            data = bm.set_at(data, slice(NC), self.celldata[key])
            data = bm.set_at(self.celldata , key, data.copy())

        # 用于存储网格节点的代数，初始所有节点都为第 0 代
        generation = bm.zeros(NN + 6*NC, dtype=bm.uint8)

        # 用于记录被二分的边及其中点编号
        cutEdge = bm.zeros((8*NN, 3), **self.ikwargs)

        # 当前的二分边的数目
        nCut = 0

        # 非协调边的标记数组
        nonConforming = bm.ones(8*NN, dtype=bm.bool, device=self.device)
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
            p4 = bm.zeros(nMarked, **self.ikwargs)

            if nCut == 0: # 如果是第一次循环
                idx = bm.arange(nMarked) # cells introduce new cut edges
            else:
                # all non-conforming edges
                ncEdge = bm.nonzero(nonConforming[:nCut])
                NE = len(ncEdge)
                I = cutEdge[ncEdge][:, [2, 2]].reshape(-1)
                J = cutEdge[ncEdge][:, [0, 1]].reshape(-1)
                val = bm.ones(len(I), dtype=bm.bool, device=self.device)
                nv2v = csr_matrix(
                        (val, (I, J)),
                        shape=(NN, NN))
                i, j =  (nv2v[:, p0].multiply(nv2v[:, p1])).nonzero()
                p4 = bm.set_at(p4, bm.array(j,**self.ikwargs), bm.array(i,**self.ikwargs))
                idx, = bm.nonzero(p4 == 0)

            if len(idx) != 0:
                # 把需要二分的边唯一化
                NE = len(idx)
                cellCutEdge = bm.stack([p0[idx], p1[idx]])
                cellCutEdge = bm.sort(cellCutEdge,axis=0)
                s = csr_matrix(
                    (
                        bm.ones(NE, dtype=bm.bool, device=self.device),
                        (
                            cellCutEdge[0, ...],
                            cellCutEdge[1, ...]
                        )
                    ), shape=(NN, NN))
                # 获得唯一的边
                i, j = s.nonzero()
                i = bm.tensor(i,**self.ikwargs)
                j = bm.tensor(j,**self.ikwargs)
                nNew = len(i)
                newCutEdge = bm.arange(nCut, nCut+nNew)
                cutEdge = bm.set_at(cutEdge, (newCutEdge,0), i)
                cutEdge = bm.set_at(cutEdge, (newCutEdge,1), j)
                cutEdge = bm.set_at(cutEdge, (newCutEdge,2), bm.arange(NN, NN+nNew,**self.ikwargs))
                node = bm.set_at(node, slice(NN, NN+nNew), (node[i, :] + node[j, :])/2.0)

                if returnim is True:
                    val = bm.full(nNew, 0.5)
                    I = coo_matrix(
                            (val, (range(nNew), i)), shape=(nNew, NN),
                            **self.fkwargs)
                    I += coo_matrix(
                            (val, (range(nNew), j)), shape=(nNew, NN),
                            **self.fkwargs)
                    I = bmat([[eye(NN)], [I]], format='csr')
                    IM = I@IM

                nCut += nNew
                NN += nNew

                # 新点和旧点的邻接矩阵
                I = cutEdge[newCutEdge][:, [2, 2]].reshape(-1)
                J = cutEdge[newCutEdge][:, [0, 1]].reshape(-1)
                val = bm.ones(len(I), dtype=bm.bool, device=self.device)
                nv2v = csr_matrix(
                        (val, (I, J)),
                        shape=(NN, NN))
                i, j =  (nv2v[:, p0].multiply(nv2v[:, p1])).nonzero()
                p4 = bm.set_at(p4, bm.array(j,**self.ikwargs), bm.array(i,**self.ikwargs))

            # 如果新点的代数仍然为 0
            idx = (generation[p4] == 0)
            cellGeneration = bm.max(
                    generation[cell[markedCell[idx]]],
                    axis=-1)
            # 第几代点
            generation = bm.set_at(generation, p4[idx], cellGeneration + 1)
            cell = bm.set_at(cell, (markedCell,0), p3)
            cell = bm.set_at(cell, (markedCell,1), p0)
            cell = bm.set_at(cell, (markedCell,2), p2)
            cell = bm.set_at(cell, (markedCell,3), p4)
            cell = bm.set_at(cell, (slice(NC, NC+nMarked),0), p2)
            cell = bm.set_at(cell, (slice(NC, NC+nMarked),1), p1)
            cell = bm.set_at(cell, (slice(NC, NC+nMarked),2), p3)
            cell = bm.set_at(cell, (slice(NC, NC+nMarked),3), p4)

            for key in self.celldata:
                data = self.celldata[key]
                data = bm.set_at(data, slice(NC, NC+nMarked), data[markedCell])

            if("HB" in options) and (options["HB"] is not None):
                HB = options['HB']
                HB = bm.set_at(HB, (slice(NC, NC+nMarked),1), HB[markedCell,1])
            
            NC = NC + nMarked
            del cellGeneration, p0, p1, p2, p3, p4

            # 找到非协调的单元
            checkEdge, = bm.nonzero(nonConforming[:nCut])
            isCheckNode = bm.zeros(NN, dtype=bm.bool, device=self.device)
            isCheckNode = bm.set_at(isCheckNode, cutEdge[checkEdge], True)
            isCheckCell = bm.sum(
                    isCheckNode[cell[:NC]],
                    axis= -1) > 0
            # 找到所有包含检查节点的单元编号
            checkCell, = bm.nonzero(isCheckCell)
            I = bm.repeat(checkCell, 4)
            J = cell[checkCell].reshape(-1)
            val = bm.ones(len(I), dtype=bm.bool, device=self.device)
            cell2node = csr_matrix((val, (I, J)), shape=(NC, NN))
            i, j =  (cell2node[:, cutEdge[checkEdge, 0]].multiply(
                        cell2node[:, cutEdge[checkEdge, 1]]
                        )).nonzero()
            markedCell = bm.unique(bm.array(i))
            nonConforming = bm.set_at(nonConforming, checkEdge, False)
            nonConforming = bm.set_at(nonConforming, checkEdge[j], True)


        self.node = node[:NN]
        self.cell = cell[:NC]
        self.construct()
        

        for key in self.celldata:
            self.celldata = bm.set_at(self.celldata, key, self.celldata[key][:NC])
            

        if("HB" in options) and (options["HB"] is not None):
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
        volume = bm.sum(v03*bm.cross(v01, v02), axis=1,dtype=bm.float64)/6.0
        
        idx = HB[..., 1]
        ret = {"nodedata": [], "celldata": []}

        for u0 in data['celldata']: 
            fval = u0[idx]
            ret["celldata"].append(fval)

        if 'nodedata' in data:
            lambdai = bm.zeros((NC, 4, 4), dtype=bm.float64)

            # 计算所有单元的第 j 个点的第 i 个重心坐标分量
            for j in range(4):
                flag = node[cell[:, j]]
                localface = bm.array([[2, 1, 3], [2, 3, 0], [1, 0, 3], [0, 1, 2]])
                for i in range(4):
                    v1 = oldnode[oldcell[idx, localface[i, 0]]] - flag
                    v2 = oldnode[oldcell[idx, localface[i, 1]]] - flag
                    v3 = oldnode[oldcell[idx, localface[i, 2]]] - flag
                    volume1 = bm.sum(bm.cross(v2, v1)*v3, axis=-1,dtype=bm.float64)/6
                    lambdai = bm.set_at(lambdai , (slice(None),j ,i), volume1/volume[idx])

            fval0 = bm.zeros((NC, 4), dtype=bm.float64)
            for u0 in data['nodedata']: 
                fval = bm.zeros((NN), dtype=bm.float64)
                for i in range(4):
                    w = lambdai[:, i, :]
                    fval0 = bm.set_at(fval0 , (slice(None) , i) , bm.sum(w*u0[oldcell[idx]], axis=1))
                fval = bm.set_at(fval , cell , fval0)
                ret["nodedata"].append(fval)
        return ret


   
    ## @ingroup MeshGenerators
    @classmethod
    def from_box(cls, box=[0, 1, 0, 1, 0, 1], nx=10, ny=10, nz=10, 
                threshold=None, device: str = None):
        """
        Generate a tetrahedral mesh for a box domain.
        """
        NN = (nx+1)*(ny+1)*(nz+1)
        NC = nx*ny*nz
        node = bm.zeros((NN, 3), dtype=bm.float64, device=device)
        x = bm.linspace(box[0], box[1], nx+1, dtype=bm.float64, device=device)
        y = bm.linspace(box[2], box[3], ny+1, dtype=bm.float64, device=device)
        z = bm.linspace(box[4], box[5], nz+1, dtype=bm.float64, device=device)
        X, Y, Z = bm.meshgrid(x, y, z, indexing='ij')
 
        node = bm.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)), axis=1)

        idx = bm.arange(NN, dtype=bm.int32, device=device).reshape(nx+1, ny+1, nz+1)
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
            [0, 2, 3, 6]], dtype=bm.int32, device=device)
        cell = cell[:, localCell].reshape(-1, 4)

        if threshold is not None:
            NN = len(node)
            bc = bm.sum(node[cell, :], axis=1)/cell.shape[1]
            isDelCell = threshold(bc)
            cell = cell[~isDelCell]
            isValidNode = bm.zeros(NN, dtype=bm.bool, device=device)
            isValidNode[cell] = True
            node = node[isValidNode]
            idxMap = bm.zeros(NN, dtype=cell.dtype, device=device)
            idxMap[isValidNode] = bm.arange(isValidNode.sum(), dtype=cell.dtype)
            cell = idxMap[cell]
        mesh = cls(node, cell)

        mesh.box = box
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

    @classmethod
    def from_crack_box(cls, box=[0, 2, 0, 5, 0, 10], nx=2, ny=5, nz=10,
                       threshold=None, itype=None, ftype=None, device=None):
        """
        Generate a tetrahedral mesh for a box domain.

        @param nx Number of divisions along the x-axis (default: 2)
        @param ny Number of divisions along the y-axis (default: 5)
        @param nz Number of divisions along the z-axis (default: 10)
        @param threshold Optional function to filter cells based on their barycenter coordinates (default: None)
        @return TetrahedronMesh instance
        """
        if itype is None:
            itype = bm.int32
        if ftype is None:
            ftype = bm.float64

        NN = (nx + 1) * (ny + 1) * (nz + 1)
        NC = nx * ny * nz
        node = bm.zeros((NN, 3), dtype=ftype)
        x = bm.linspace(box[0], box[1], nx + 1, dtype=ftype, device=device)
        y = bm.linspace(box[2], box[3], ny + 1, dtype=ftype, device=device)
        z = bm.linspace(box[4], box[5], nz + 1, dtype=ftype, device=device)
        X, Y, Z = bm.meshgrid(x, y, z, indexing='ij')
        node = bm.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)), axis=1)

        idx = bm.arange(NN, dtype=itype, device=device).reshape(nx + 1, ny + 1, nz + 1)
        c = idx[:-1, :-1, :-1]
        nyz = (ny + 1) * (nz + 1)

        cell0 = c.flatten().reshape((-1, 1))
        cell1 = cell0 + nyz
        cell2 = cell1 + nz + 1
        cell3 = cell0 + nz + 1
        cell4 = cell0 + 1
        cell5 = cell4 + nyz
        cell6 = cell5 + nz + 1
        cell7 = cell4 + nz + 1
        cell = bm.concatenate((cell0, cell1, cell2, cell3, cell4, cell5, cell6, cell7), axis=1)

        localCell = bm.tensor([
            [0, 1, 2, 6],
            [0, 5, 1, 6],
            [0, 4, 5, 6],
            [0, 7, 4, 6],
            [0, 3, 7, 6],
            [0, 2, 3, 6]], dtype=itype, device=device)
        cell = cell[:, localCell].reshape(-1, 4)

        if threshold is not None:
            bc = bm.sum(node[cell, :], axis=1) / cell.shape[1]
            isDelCell = threshold(bc)
            cell = cell[~isDelCell]
            isValidNode = bm.zeros(NN, dtype=bm.bool, device=device)
            isValidNode = bm.set_at(isValidNode, cell, True)
            node = node[isValidNode]
            idxMap = bm.zeros(NN, dtype=itype, device=device)
            # idxMap[isValidNode] = bm.arange(isValidNode.sum(), dtype=cell.dtype)
            idxMap = bm.set_at(
                idxMap, isValidNode, bm.arange(isValidNode.sum(), dtype=itype, device=device)
            )
            cell = idxMap[cell]

        # 切口节点重复
        NN = node.shape[0]
        # 找到切口处 node
        nidx = bm.nonzero((bm.abs(node[:, 2] - 5) < 1e-5) & (node[:, 1] > 3.01))[0]
        # 找到切口处节点所在单元
        nidxmap = bm.arange(NN, dtype=itype, device=device)

        nidxmap = bm.set_at(nidxmap, nidx, NN + bm.arange(len(nidx), dtype=itype, device=device))
        # 计算 z 坐标平均值
        flag = bm.mean(node[:, 2][cell], axis=1) > 5
        cell = bm.set_at(cell, flag, nidxmap[cell[flag]])

        node = bm.concatenate((node, node[nidx]), axis=0)
        mesh = cls(node, cell)
        return mesh

    @classmethod
    def from_spherical_shell(cls, r1=0.05, r2=0.5, h=0.04,
                             itype=None, ftype=None, device=None) -> 'TetrahedronMesh':
        """
        Generate a tetrahedral mesh for a spherical shell.
        Parameters
        ----------
        r1: float
            Inner radius of the spherical shell.
        r2: float
            Outer radius of the spherical shell.
        h: float
            Mesh size parameter.
        itype: int type for indices, default is bm.int32
        ftype: float type for coordinates, default is bm.float64
        device: str, optional

        Returns
        -------
        TetrahedronMesh
            An instance of TetrahedronMesh containing the mesh data.
        """
        try:
            import gmsh
        except ImportError:
            raise ImportError("Please install gmsh to use this function.")

        if itype is None:
            itype = bm.int32
        if ftype is None:
            ftype = bm.float64

        # 1. Initialize GMSH and create spherical shell geometry
        gmsh.initialize()
        gmsh.model.add("spherical_shell")
        outer = gmsh.model.occ.addSphere(0, 0, 0, r2)
        inner = gmsh.model.occ.addSphere(0, 0, 0, r1)
        shell, _ = gmsh.model.occ.cut([(3, outer)], [(3, inner)],
                                      removeObject=True, removeTool=False)
        gmsh.model.occ.synchronize()

        # 2. Get the boundary faces of the shell
        faces = gmsh.model.getBoundary(shell, oriented=False, recursive=False)
        # Get inner sphere's face tags
        inner_faces = gmsh.model.getBoundary([(3, inner)], oriented=False, recursive=False)
        inner_face_tags = [f[1] for f in inner_faces]
        # Match shell faces that belong to inner sphere
        inner_face_tags_in_shell = [f[1] for f in faces if f[0] == 2 and f[1] in inner_face_tags]

        gmsh.model.occ.remove([(3, inner)])
        gmsh.model.occ.synchronize()

        # 3.  Create mesh fields for size control  Distance + Threshold
        h_min, h_max = h, 10.0 * h
        # （a）Distance field
        gmsh.model.mesh.field.add("Distance", 1)
        gmsh.model.mesh.field.setNumbers(1, "FacesList", inner_face_tags_in_shell)
        gmsh.model.mesh.field.setNumber(1, "Sampling", 100)  # the number of sampling points
        # （b）Threshold field
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", h_min)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", h_max)
        gmsh.model.mesh.field.setNumber(2, "DistMin", r1/1000)
        gmsh.model.mesh.field.setNumber(2, "DistMax", r2 - r1)
        gmsh.model.mesh.field.setAsBackgroundMesh(2)

        # 4. generate the mesh
        gmsh.model.mesh.generate(3)

        # Extract mesh data: nodes and elements
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node = bm.array(node_coords, dtype=ftype, device=device).reshape(-1, 3)

        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=3)

        cell = bm.array(elem_node_tags[0], dtype=itype, device=device).reshape(-1, 4) - 1  # Convert to 0-based index

        gmsh.finalize()

        return cls(node, cell)

    @classmethod
    def from_vtu(cls,file):
        import meshio
        data = meshio.read(file)
        node = data.points
        cell = data.cells_dict['tetra']
        mesh = cls(node, cell)
        return mesh
    
    @classmethod
    def from_medit(cls,file):
        '''
        Read medit format file (.mesh) to create tetrahedral mesh.
        Parameters:
            file (str): Path to the medit file.
        Returns:
            TetrahedronMesh: An instance of TetrahedronMesh containing the mesh
            data.
        '''
        import meshio
        data = meshio.read(file)
        node = bm.from_numpy(data.points)
        cell = bm.from_numpy(data.cells_dict['tetra'])
        mesh = cls(node, cell)
        return mesh

    def to_vtk(self, fname=None, etype='cell', index:Index=_S):
        from .vtk_extent import  write_to_vtu

        node = self.entity('node')
        GD = self.geo_dimension()

        cell = self.entity(etype)[index]
        NC = len(cell)
        NV = cell.shape[-1]

        cell = bm.concatenate((bm.zeros((len(cell), 1), dtype=cell.dtype), cell), axis=1)
        cell = bm.set_at(cell, (slice(NC), 0), NV)

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

TetrahedronMesh.set_ploter('3d')
