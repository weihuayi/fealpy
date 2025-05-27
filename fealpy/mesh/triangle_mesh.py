from typing import Union, Optional, Callable
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .utils import simplex_gdof, simplex_ldof
from .mesh_base import SimplexMesh, estr2dim
from .plot import Plotable
from fealpy.sparse import csr_matrix
from fealpy.sparse import CSRTensor,COOTensor
class TriangleMesh(SimplexMesh, Plotable):
    def __init__(self, node: TensorLike, cell: TensorLike) -> None:
        """
        """
        super().__init__(TD=2, itype=cell.dtype, ftype=node.dtype)
        kwargs = bm.context(cell)
        
        self.node = node
        self.cell = cell


        self.localEdge = bm.tensor([(1, 2), (2, 0), (0, 1)], **kwargs)
        self.localFace = bm.tensor([(1, 2), (2, 0), (0, 1)], **kwargs)
        self.ccw = bm.tensor([0, 1, 2], **kwargs)

        self.localCell = bm.tensor([
            (0, 1, 2),
            (1, 2, 0),
            (2, 0, 1)], **kwargs)

        self.construct()
        self.meshtype = 'tri'

        self.nodedata = {}
        self.edgedata = {}
        self.facedata = self.edgedata
        self.celldata = {}
        self.meshdata = {}

    face_unit_normal = SimplexMesh.edge_unit_normal

    # entity
    def entity_measure(self, etype: Union[int, str], index: Optional[Index]=None) -> TensorLike:
        """
        """
        node = self.node
        kwargs = bm.context(node)

        if isinstance(etype, str):
            etype = estr2dim(self, etype)

        if etype == 0:
            return bm.tensor([0,], **kwargs)
        elif etype == 1:
            edge = self.entity(1, index)
            return bm.edge_length(edge, node)
        elif etype == 2:
            cell = self.entity(2, index)
            if self.geo_dimension()==2:
                return bm.simplex_measure(cell, node)
            else: 
                v0 = node[cell[:, 1], :] - node[cell[:, 0], :]
                v1 = node[cell[:, 2], :] - node[cell[:, 0], :]

                nv = bm.cross(v0, v1)
                return bm.sqrt(bm.sum(nv ** 2, axis=1)) / 2.0
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")
  
    # quadrature
    def quadrature_formula(self, q: int, etype: Union[int, str]='cell',
                           qtype: str='legendre'): # TODO: other qtype
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        kwargs = {'dtype': self.ftype, 'device': self.device}

        if etype == 2:
            from ..quadrature.stroud_quadrature import StroudQuadrature
            from ..quadrature import TriangleQuadrature
            if q > 9:
                quad = StroudQuadrature(2, q)
            else:
                quad = TriangleQuadrature(q, **kwargs)
        elif etype == 1:
            from ..quadrature import GaussLegendreQuadrature
            quad = GaussLegendreQuadrature(q, **kwargs)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")
        return quad

    def update_bcs(self, bcs, toetype: Union[int, str]='cell'):
        TD = bcs.shape[-1] - 1
        if toetype == 'cell' or toetype == 2: 
            if TD == 2:
                return bcs
            elif TD == 1: # edge up to cell
                result = bm.stack([bm.insert(bcs, i, 0.0, axis=-1) for i in range(3)], axis=0)
                return result
            else:
                raise ValueError("Unsupported topological dimension: {TD}")
                    
        else:
            raise ValueError("The etype only support face, other etype is not implemented.")
    
    # shape function
    def grad_lambda(self, index: Index=_S, TD:int=2) -> TensorLike:
        """
        """
        node = self.entity('node')
        entity = self.entity(TD, index=index)
        GD = self.GD
        if TD == 1:
            return bm.interval_grad_lambda(entity, node)
        elif TD == 2:
            if GD == 2:
                return bm.triangle_grad_lambda_2d(entity, node)
            elif GD == 3:
                return bm.triangle_grad_lambda_3d(entity, node)
        else:
            raise ValueError("Unsupported topological dimension: {TD}")
        '''
        node = self.node
        cell = self.cell[index]
        GD = self.GD
        if GD == 2:
            return bm.triangle_grad_lambda_2d(cell, node)
        elif GD == 3:
            return bm.triangle_grad_lambda_3d(cell, node)
        '''
    def rot_lambda(self, index: Index=_S): # TODO
        pass
    
    def grad_shape_function(self, bc, p=1, index: Index=_S, variables='x'):
        """
        @berif 这里调用的是网格空间基函数的梯度
        """
        TD = bc.shape[-1] - 1
        R = bm.simplex_grad_shape_function(bc, p)
        if variables == 'x':
            Dlambda = self.grad_lambda(index=index, TD=TD)
            gphi = bm.einsum('...ij, kjm -> k...im', R, Dlambda)
            return gphi  # (NC, NQ, ldof, GD)
        elif variables == 'u':
            return R  # (NQ, ldof, TD+1)

    cell_grad_shape_function = grad_shape_function

    def grad_shape_function_on_edge(self, bc, cindex, lidx, p=1, direction=True):
        """
        @brief 计算单元上所有形函数在边上的积分点处的导函数值

        @param bc 边上的一组积分点
        @param cindex 边所在的单元编号
        @param lidx 边在该单元的局部编号
        @param direction  True 表示边的方向和单元的逆时针方向一致，False 表示不一致
        """
        pass

    # ipoint
    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell'):
        if isinstance(iptype, str):
            iptype = estr2dim(self, iptype)
        return simplex_ldof(p, iptype)

    def number_of_global_ipoints(self, p: int):
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()
        num = (NN, NE, NC)
        return simplex_gdof(p, num)
    
    def interpolation_points(self, p: int, index: Index=_S):
        """Fetch all p-order interpolation points on the triangle mesh."""
        node = self.entity('node')
        if p == 1:
            return node[index]
        if p <= 0:
            raise ValueError("p must be a integer larger than 0.")

        ipoint_list = []
        kwargs = {'dtype': self.ftype}

        GD = self.geo_dimension()
        ipoint_list.append(node) # ipoints[:NN, :]

        edge = self.entity('edge')
        w = bm.multi_index_matrix(p, 1, dtype=self.ftype)
        w = w[1:-1]/p
        ipoints_from_edge = bm.einsum('ij, ...jm->...im', w,
                                         node[edge, :]).reshape(-1, GD) # ipoints[NN:NN + (p - 1) * NE, :]
        ipoint_list.append(ipoints_from_edge)

        if p >= 3:
            TD = self.top_dimension()
            cell = self.entity('cell')
            multiIndex = bm.multi_index_matrix(p, TD, dtype=self.ftype)
            isEdgeIPoints = (multiIndex == 0)
            isInCellIPoints = ~(isEdgeIPoints[:, 0] | isEdgeIPoints[:, 1] |
                                isEdgeIPoints[:, 2])
            multiIndex = multiIndex[isInCellIPoints, :]
            w = multiIndex / p
            
            ipoints_from_cell = bm.einsum('ij, kj...->ki...', w,
                                          node[cell, :]).reshape(-1, GD) # ipoints[NN + (p - 1) * NE:, :]
            ipoint_list.append(ipoints_from_cell)

        return bm.concatenate(ipoint_list, axis=0)[index]  # (gdof, GD)

    def cell_to_ipoint(self, p: int, index: Index=_S):
        """
        Get the map from local index to global index for interpolation points.
        """
        cell = self.cell

        if p == 1:
            return cell[index]

        TD = self.top_dimension()
        mi = self.multi_index_matrix(p, TD)
        idx0, = bm.nonzero(mi[:, 0] == 0)
        idx1, = bm.nonzero(mi[:, 1] == 0)
        idx2, = bm.nonzero(mi[:, 2] == 0)

        face2cell = self.face_to_cell()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()

        e2p = self.edge_to_ipoint(p)
        ldof = self.number_of_local_ipoints(p, 'cell')

        kwargs = bm.context(cell)
        c2p = bm.zeros((NC, ldof), **kwargs)

        flag = face2cell[:, 2] == 0
        c2p = bm.set_at(c2p, (face2cell[flag, 0][:, None], idx0), e2p[flag])

        flag = face2cell[:, 2] == 1
        idx1_ = bm.flip(idx1, axis=0)
        c2p = bm.set_at(c2p, (face2cell[flag, 0][:, None], idx1_), e2p[flag])

        flag = face2cell[:, 2] == 2
        c2p = bm.set_at(c2p, (face2cell[flag, 0][:, None], idx2), e2p[flag])

        iflag = face2cell[:, 0] != face2cell[:, 1]
        flag = iflag & (face2cell[:, 3] == 0)
        idx0_ = bm.flip(idx0, axis=0)
        c2p = bm.set_at(c2p, (face2cell[flag, 1][:, None], idx0_), e2p[flag])

        flag = iflag & (face2cell[:, 3] == 1)
        c2p = bm.set_at(c2p, (face2cell[flag, 1][:, None], idx1), e2p[flag])

        flag = iflag & (face2cell[:, 3] == 2)
        idx2_ = bm.flip(idx2, axis=0)
        c2p = bm.set_at(c2p, (face2cell[flag, 1][:, None], idx2_),  e2p[flag])

        cdof = (p-1)*(p-2)//2
        flag = bm.sum(mi > 0, axis=1) == 3
        val = NN + NE*(p-1) + bm.arange(NC*cdof, **kwargs).reshape(NC, cdof)
        c2p = bm.set_at(c2p, (..., flag), val)
        return c2p[index]

    def face_to_ipoint(self, p: int, index: Index=_S):
        return self.edge_to_ipoint(p, index)

    def boundary_edge_flag(self):
        return self.boundary_face_flag()

    def cell_to_face_sign(self):
        """
        """
        NC = self.number_of_cells()
        NEC = self.number_of_edges_of_cells()
        face2cell = self.face_to_cell() 
        cell2faceSign = bm.zeros((NC, NEC), dtype=bm.bool, device=self.device)
        cell2faceSign = bm.set_at(cell2faceSign, (face2cell[:, 0], face2cell[:, 2]), True)
        return cell2faceSign
    
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

        kargs_node = bm.context(self.entity('node'))
        kargs_cell = bm.context(self.entity('cell'))

        # 1. Interpolation points on the mesh nodes: Inherit the original interpolation points
        NN = self.number_of_nodes()
        V_1 = bm.ones(NN,**kargs_node)
        I_1 = bm.arange(NN,**kargs_cell)
        J_1 = bm.arange(NN,**kargs_cell)

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

        # 3. Interpolation points within the mesh cells
        if p1 > 2:
            NC = self.number_of_cells()
            bcs = self.multi_index_matrix(p1, TD) / p1
            flag = bm.sum(bcs > 0, axis=1) == 3
            phi = self.shape_function(bcs[flag, :], p=p0)
            c2p1 = self.cell_to_ipoint(p1)[:, flag]
            c2p0 = self.cell_to_ipoint(p0)

            shape = (NC,) + phi.shape

            I_3 = bm.broadcast_to(c2p1[:, :, None], shape=shape).flat
            J_3 = bm.broadcast_to(c2p0[:, None, :], shape=shape).flat
            V_3 = bm.broadcast_to(phi[None, :, :], shape=shape).flat

        # 4.concatenate
        if p1 <=2:
            V = bm.concatenate((V_1, V_2), axis=0) 
            I = bm.concatenate((I_1, I_2), axis=0) 
            J = bm.concatenate((J_1, J_2), axis=0) 
            P = csr_matrix((V, (I, J)), matrix_shape)
        else:
            V = bm.concatenate((V_1, V_2, V_3), axis=0) 
            I = bm.concatenate((I_1, I_2, I_3), axis=0) 
            J = bm.concatenate((J_1, J_2, J_3), axis=0) 
            P = csr_matrix((V, (I, J)), matrix_shape)

        return P

    def edge_frame(self, index: Index=_S):
        """
        @brief 计算二维网格中每条边上的局部标架
        """
        pass
    
    def edge_unit_tangent(self, index=_S):
        """
        @brief Calculate the tangent vector with unit length of each edge.See `Mesh.edge_tangent`.
        """
        node = self.entity('node') 
        edge = self.entity('edge', index=index)
        v = node[edge[:, 1], :] - node[edge[:, 0], :]
        length = bm.sqrt(bm.square(v).sum(axis=1))
        return v/length.reshape(-1, 1)

    def uniform_refine(self, n=1, surface=None, interface=None, returnim=False):
        """
        Uniform refine the triangle mesh n times.

        Parameters:
            n (int): Times refine the triangle mesh.
            surface (function): The surface function.
            returnirm (bool): Return the prolongation matrix list or not,from the finest to the the coarsest
        
        Returns:
            mesh: The mesh obtained after uniformly refining n times.
            List(CSRTensor): The prolongation matrix from the finest to the the coarsest
        """
        if returnim is True:
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
            newNode = (node[edge[:, 0], :] + node[edge[:, 1], :]) / 2.0
            
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
            
            self.node = bm.concatenate((node, newNode), axis=0)
            p = bm.concatenate((cell, edge2newNode[cell2edge]), axis=1)
            self.cell = bm.concatenate(
                    (p[:,[0,5,4]], p[:,[5,1,3]], p[:,[4,3,2]], p[:,[3,4,5]]),
                    axis=0)
            self.construct()

        if returnim is True:
            IM.reverse()
            return IM

    def is_crossed_cell(self, point, segment):
        """
        @berif 给定一组线段，找到这些线段的一个邻域单元集合, 且这些单元要满足一定的连通
        性
        """
        pass
    
    def location(self, points):
        """
        @breif  给定一组点 p , 找到这些点所在的单元

        这里假设：

        1. 所有点在网格内部，
        2. 网格中没有洞
        3. 区域还要是凸的
        """
        pass

    def circumcenter(self, index: Index=_S, returnradius=False):
        """
        @brief 计算三角形外接圆的圆心和半径
        """
        node = self.node
        cell = self.cell
        GD = self.geo_dimension()

        v0 = node[cell[index, 2], :] - node[cell[index, 1], :]
        v1 = node[cell[index, 0], :] - node[cell[index, 2], :]
        v2 = node[cell[index, 1], :] - node[cell[index, 0], :]
        nv = bm.cross(v2, -v1)
        if GD == 2:
            area = nv / 2.0
            x2 = bm.sum(node ** 2, axis=1, keepdims=True)
            w0 = x2[cell[index, 2]] + x2[cell[index, 1]]
            w1 = x2[cell[index, 0]] + x2[cell[index, 2]]
            w2 = x2[cell[index, 1]] + x2[cell[index, 0]]
            W = bm.array([[0, -1], [1, 0]], dtype=self.ftype)
            fe0 = w0 * v0 @ W
            fe1 = w1 * v1 @ W
            fe2 = w2 * v2 @ W
            c = 0.25 * (fe0 + fe1 + fe2) / area.reshape(-1, 1)
            R = bm.sqrt(bm.sum((c - node[cell[index, 0], :]) ** 2, axis=1))
        elif GD == 3:
            length = bm.sqrt(bm.sum(nv ** 2, axis=1))
            n = nv / length.reshape((-1, 1))
            l02 = bm.sum(v1 ** 2, axis=1, keepdims=True)
            l01 = bm.sum(v2 ** 2, axis=1, keepdims=True)
            d = 0.5 * (l02 * bm.cross(n, v2) + l01 * bm.cross(-v1, n)) / length.reshape(-1, 1)
            c = node[cell[index, 0]] + d
            R = bm.sqrt(bm.sum(d ** 2, axis=1))

        if returnradius:
            return c, R
        else:
            return c

    def angle(self):
        NC = self.number_of_cells()
        cell = self.entity('cell')
        node = self.entity('node')
        localEdge = self.localEdge
        angle = bm.zeros((NC, 3), dtype=self.ftype, device=self.device)
        for i,(j,k) in zip(range(3),localEdge):
            v0 = node[cell[:, j]] - node[cell[:, i]]
            v1 = node[cell[:, k]] - node[cell[:, i]]
            angle = bm.set_at(angle,(...,i), bm.arccos(bm.sum(v0*v1,axis=1)/bm.sqrt(bm.sum(v0**2,axis=1)*bm.sum(v1**2,axis=1))))
        return angle

    def cell_quality(self, measure='radius_ratio'):
        if measure == 'radius_ratio':
            return radius_ratio(self)

    def show_quality(self, axes, qtype=None, quality=None):
        """
        @brief 显示网格质量分布的分布直方图
        """
        pass

    def edge_swap(self):
        pass

    def odt_iterate(self):
        pass

    def uniform_bisect(self, n=1):
        for i in range(n):
            self.bisect()

    def bisect_options(
            self,
            HB=None,
            IM=None,
            data=None,
            disp=True,
    ):

        options = {
            'HB': HB,
            'IM': IM,
            'data': data,
            'disp': disp
        }
        return options

    def bisect(self, isMarkedCell=None, options={'disp': True}): #TODO
        if options['disp']:
            print('Bisection begining......')

        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        NE = self.number_of_edges()

        if options['disp']:
            print('Current number of nodes:', NN)
            print('Current number of edges:', NE)
            print('Current number of cells:', NC)

        if isMarkedCell is None:
            isMarkedCell = bm.ones(NC, dtype=bm.bool)

        cell = self.entity('cell')
        edge = self.entity('edge')

        cell2edge = self.cell_to_edge()
        cell2cell = self.cell_to_cell()
        #cell2ipoint = self.cell_to_ipoint(self.p)
        isCutEdge = bm.zeros((NE,), dtype=bm.bool, device=self.device)

        if options['disp']:
            print('The initial number of marked elements:', isMarkedCell.sum())

        markedCell, = bm.nonzero(isMarkedCell)
        while len(markedCell) > 0:
            isCutEdge = bm.set_at(isCutEdge, cell2edge[markedCell, 0], True)
            refineNeighbor = cell2cell[markedCell, 0]
            markedCell = refineNeighbor[~isCutEdge[cell2edge[refineNeighbor, 0]]]

        if options['disp']:
            print('The number of markedg edges: ', isCutEdge.sum())

        edge2newNode = bm.zeros((NE,), dtype=self.itype, device=self.device)
        edge2newNode = bm.set_at(edge2newNode, isCutEdge, bm.arange(NN, NN + isCutEdge.sum(), dtype=self.itype, device=self.device))

        node = self.node
        newNode = 0.5 * (node[edge[isCutEdge, 0], :] + node[edge[isCutEdge, 1], :])
        self.node = bm.concatenate((node, newNode), axis=0)
        cell2edge0 = cell2edge[:, 0]

        if 'data' in options:
            pass

        if 'IM' in options:
            nn = len(newNode)
            IM = COOTensor( indices=bm.stack((bm.arange(NN), bm.arange(NN)), axis=0),
                            values=bm.ones(NN), 
                            spshape=(NN + nn, NN))
            # IM = coo_matrix((bm.ones(NN), (bm.arange(NN), bm.arange(NN))),
            #                 shape=(NN + nn, NN))
            val = bm.full((nn,), 0.5)
            IM += COOTensor(indices=bm.stack((NN + bm.arange(nn), edge[isCutEdge, 0]), axis=0),
                            values=val,
                            spshape=(NN + nn, NN))
            # IM += coo_matrix(
            #     (
            #         val,
            #         (
            #             NN + bm.arange(nn),
            #             edge[isCutEdge, 0]
            #         )
            #     ), shape=(NN + nn, NN))
            IM += COOTensor(indices=bm.stack((NN + bm.arange(nn), edge[isCutEdge, 1]), axis=0),
                            values=val,
                            spshape=(NN + nn, NN))
            # IM += coo_matrix(
            #     (
            #         val,
            #         (
            #             NN + bm.arange(nn),
            #             edge[isCutEdge, 1]
            #         )
            #     ), shape=(NN + nn, NN))
            options['IM'] = IM.tocsr()

        if 'HB' in options:
            options['HB'] = bm.arange(NC)

        for k in range(2):
            idx, = bm.nonzero(edge2newNode[cell2edge0] > 0)
            nc = len(idx)
            if nc == 0:
                break

            if 'HB' in options:
                HB = options['HB']
                options['HB'] = bm.concatenate((HB, HB[idx]), axis=0)

            L = idx
            R = bm.arange(NC, NC + nc)
            if ('data' in options) and (options['data'] is not None):
                for key, value in options['data'].items():
                    if value.shape == (NC,):  # 分片常数
                        value = bm.concatenate((value[:], value[idx]))
                        options['data'][key] = value
                    #elif value.ndim == 2 and value.shape[0] == NC:  # 处理(NC, NQ)的情况
                    #    value = bm.concatenate((value, value[idx, :])) 
                    #    options['data'][key] = value
                    elif value.shape == (NN + k * nn,):
                        if k == 0:
                            value = bm.concatenate((value, bm.zeros((nn,),  dtype=self.ftype, device=self.device)))
                            value = bm.set_at(value , slice(NN, None), 0.5 * (value[edge[isCutEdge, 0]] + value[edge[isCutEdge, 1]]))
                            options['data'][key] = value
                    else:
                        ldof = value.shape[-1]
                        p = int((bm.sqrt(1 + 8 * bm.array(ldof)) - 3) // 2)
                        bc = self.multi_index_matrix(p, etype=2) / p

                        bcl = bm.zeros_like(bc, dtype=self.ftype, device=self.device)
                        bcl = bm.set_at(bcl , (slice(None), 0), bc[:, 1])
                        bcl = bm.set_at(bcl , (slice(None), 1), 0.5 * bc[:, 0] + bc[:, 2])
                        bcl = bm.set_at(bcl , (slice(None), 2), 0.5 * bc[:, 0])

                        bcr = bm.zeros_like(bc,dtype=self.ftype, device=self.device)
                        bcr = bm.set_at(bcr , (slice(None), 0), bc[:, 2])
                        bcr = bm.set_at(bcr , (slice(None), 1), 0.5 * bc[:, 0])
                        bcr = bm.set_at(bcr , (slice(None), 2), 0.5 * bc[:, 0] + bc[:, 1])

                        value = bm.concatenate((value, bm.zeros((nc, ldof), dtype=self.ftype, device=self.device)))

                        phi = self.shape_function(bcr, p=p)
                        value = bm.set_at(value , slice(NC , None), bm.einsum('cj,kj->ck', value[idx], phi))

                        phi = self.shape_function(bcl, p=p)
                        value = bm.set_at(value , (idx, slice(None)), bm.einsum('cj,kj->ck', value[idx], phi))

                        options['data'][key] = value

            p0 = cell[idx, 0]
            p1 = cell[idx, 1]
            p2 = cell[idx, 2]
            p3 = edge2newNode[cell2edge0[idx]]
            cell = bm.concatenate((cell, bm.zeros((nc, 3), dtype=self.itype, device=self.device)), axis=0)
            
            cell = bm.set_at(cell , (L, 0), p3)
            cell = bm.set_at(cell , (L, 1), p0)
            cell = bm.set_at(cell , (L, 2), p1)
            cell = bm.set_at(cell , (R, 0), p3)
            cell = bm.set_at(cell , (R, 1), p2)
            cell = bm.set_at(cell , (R, 2), p0)
            if k == 0:
                cell2edge0 = bm.zeros((NC + nc,), dtype=self.itype, device=self.device)
                cell2edge0 = bm.set_at(cell2edge0 , slice(NC) , cell2edge[:, 0])
                cell2edge0 = bm.set_at(cell2edge0 , L , cell2edge[idx, 2])
                cell2edge0 = bm.set_at(cell2edge0 , R , cell2edge[idx, 1])
            NC = NC + nc

        self.NN = self.node.shape[0]
        self.cell = cell
        self.construct()

    def coarsen(self, isMarkedCell=None, options={}):
        """
        @brief

        https://lyc102.github.io/ifem/afem/coarsen/
        """
        from .utils import inverse_relation

        if isMarkedCell is None:
            return

        NN = self.number_of_nodes()
        NC = self.number_of_cells()

        cell = self.entity('cell')
        node = self.entity('node')

        valence = bm.zeros(NN, dtype=self.itype, device=self.device)
        valence = bm.index_add(valence, cell, 1)

        valenceNew = bm.zeros(NN, dtype=self.itype, device=self.device)
        valenceNew = bm.index_add(valenceNew, cell[isMarkedCell][:, 0], 1)

        isIGoodNode = (valence == valenceNew) & (valence == 4)
        isBGoodNode = (valence == valenceNew) & (valence == 2)

        # node2cell = self.node_to_cell()

        # I, J = bm.nonzero(node2cell[isIGoodNode, :])
        _, J, _ = inverse_relation(cell, NN, isIGoodNode)
        nodeStar = J.reshape(-1, 4)

        ix = (cell[nodeStar[:, 0], 2] == cell[nodeStar[:, 3], 1])
        iy = (cell[nodeStar[:, 1], 1] == cell[nodeStar[:, 2], 2])
        nodeStar = bm.set_at(nodeStar , ix & (~iy) ,nodeStar[ix & (~iy), :][:, [0, 2, 1, 3]])
        nodeStar = bm.set_at(nodeStar , (~ix) & iy ,nodeStar[(~ix) & iy, :][:, [0, 3, 1, 2]])

        t0 = nodeStar[:, 0]
        t1 = nodeStar[:, 1]
        t2 = nodeStar[:, 2]
        t3 = nodeStar[:, 3]

        p1 = cell[t0, 2]
        p2 = cell[t1, 1]
        p3 = cell[t0, 1]
        p4 = cell[t2, 1]

        cell = bm.set_at(cell , (t0, 0) , p3)
        cell = bm.set_at(cell , (t0, 1) , p1)
        cell = bm.set_at(cell , (t0, 2) , p2)
        cell = bm.set_at(cell , (t1, 0) , -1)

        cell = bm.set_at(cell , (t2, 0) , p4)
        cell = bm.set_at(cell , (t2, 1) , p2)
        cell = bm.set_at(cell , (t2, 2) , p1)
        cell = bm.set_at(cell , (t3, 0) , -1)

        # I, J = bm.nonzero(node2cell[isBGoodNode, :])
        _, J, _ = inverse_relation(cell, NN, isBGoodNode)
        nodeStar = J.reshape(-1, 2)
        idx = (cell[nodeStar[:, 0], 2] == cell[nodeStar[:, 1], 1])
        nodeStar = bm.set_at(nodeStar , idx , nodeStar[idx, :][:, [0, 1]])

        t4 = nodeStar[:, 0]
        t5 = nodeStar[:, 1]
        p0 = cell[t4, 0]
        p1 = cell[t4, 2]
        p2 = cell[t5, 1]
        p3 = cell[t4, 1]
        cell = bm.set_at(cell , (t4, 0) , p3)
        cell = bm.set_at(cell , (t4, 1) , p1)
        cell = bm.set_at(cell , (t4, 2) , p2)
        cell = bm.set_at(cell , (t5, 0) , -1)

        isKeepCell = cell[:, 0] > -1
        if ('data' in options) and (options['data'] is not None):
            # value.shape == (NC, (p+1)*(p+2)//2)
            lidx = bm.concatenate((t0, t2, t4))
            ridx = bm.concatenate((t1, t3, t5))
            for key, value in options['data'].items():
                ldof = value.shape[1]
                p = int((bm.sqrt(bm.tensor(8 * ldof + 1)) - 3) / 2)
                bc = self.multi_index_matrix(p=p, etype=2) / p
                bcl = bm.zeros_like(bc, dtype=self.ftype, device=self.device)
                bcl = bm.set_at(bcl , (slice(None), 0) , 2 * bc[:, 2])
                bcl = bm.set_at(bcl , (slice(None), 1) , bc[:, 0])
                bcl = bm.set_at(bcl , (slice(None), 2) , bc[:, 1] - bc[:, 2])
 
                bcr = bm.zeros_like(bc,dtype=self.ftype, device=self.device)
                bar = bm.set_at(bcr , (slice(None), 0) , 2 * bc[:, 1])
                bar = bm.set_at(bcr , (slice(None), 1) , bc[:, 2] - bc[:, 1])
                bar = bm.set_at(bcr , (slice(None), 2) , bc[:, 0])

                phi = self.shape_function(bcl, p=p)  # (NQ, ldof)
                value = bm.set_at(value , lidx , bm.einsum('ci, qi->cq', value[lidx, :], phi))

                phi = self.shape_function(bcr, p=p)  # (NQ, ldof)

                value = bm.index_add(value , lidx , bm.einsum('ci, qi->cq', value[ridx, :], phi))
                value = bm.set_at(value , lidx , 0.5 * value[lidx])
                options['data'] = bm.set_at(options['data'],key , value[isKeepCell])

        cell = cell[isKeepCell]
        isGoodNode = (isIGoodNode | isBGoodNode)

        idxMap = bm.zeros(NN, dtype=self.itype, device=self.device)
        self.node = node[~isGoodNode]

        NN = self.node.shape[0]
        arange = bm.arange(NN, dtype=self.itype, device=self.device)
        idxMap = bm.set_at(idxMap , ~isGoodNode , arange)
        cell = idxMap[cell]

        self.cell = cell
        self.construct()

    def label(self, node=None, cell=None, cellidx=None):
        """
        单元顶点的重新排列，使得cell[:, [1, 2]] 存储了单元的最长边
        Parameter
        -------
        Return 
        -------
        cell ： in-place modify
        """
        """
        单元顶点的重新排列，使得cell[:, [1, 2]] 存储了单元的最长边
        Parameter
        -------
        Return 
        -------
        cell ： in-place modify
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
            (node[totalEdge[:, 1]] - node[totalEdge[:, 0]]) ** 2,
            axis=-1)
        length += 0.1 * bm.random.rand(NE) * length
        cellEdgeLength = length.reshape(NC, 3)
        lidx = bm.argmax(cellEdgeLength, axis=-1)

        flag = (lidx == 1)
        if sum(flag) > 0:
            cell = bm.set_at(cell , cellidx[flag] , cell[cellidx[flag]][:, [0, 1, 2]])

        flag = (lidx == 2)
        if sum(flag) > 0:
            cell = bm.set_at(cell , cellidx[flag] , cell[cellidx[flag]][:, [2, 0, 1]])

        if rflag == True:
            self.construct()

    def delete_degree_4(self):
        pass

    @staticmethod
    def adaptive_options(
            method='mean',
            maxrefine=5,
            maxcoarsen=0,
            theta=1.0,
            tol=1e-6,  # 目标误差
            HB=None,
            imatrix=False,
            data=None,
            disp=True,
        ):

        options = {
            'method': method,
            'maxrefine': maxrefine,
            'maxcoarsen': maxcoarsen,
            'theta': theta,
            'tol': tol,
            'data': data,
            'HB': HB,
            'imatrix': imatrix,
            'disp': disp
        }
        return options

    def adaptive(self, eta, options):
        theta = options['theta']
        if options['method'] == 'mean':
            options['numrefine'] = bm.round(
                bm.log2(eta / (theta * bm.mean(eta)))
            )
        elif options['method'] == 'max':
            options['numrefine'] = bm.round(
                bm.log2(eta / (theta * bm.max(eta)))
            )
        elif options['method'] == 'median':
            options['numrefine'] = bm.round(
                bm.log2(eta / (theta * bm.mean(eta)))
            )
        elif options['method'] == 'min':
            options['numrefine'] = bm.round(
                bm.log2(eta / (theta * bm.min(eta)))
            )
        elif options['method'] == 'target':
            NT = self.number_of_cells()
            e = options['tol'] / bm.sqrt(NT)
            options['numrefine'] = bm.round(
                bm.log2(eta / (theta * e)
                        ))
        else:
            raise ValueError(
                "I don't know anyting about method %s!".format(options['method']))

        flag = options['numrefine'] > options['maxrefine']
        options['numrefine'][flag] = options['maxrefine']
        flag = options['numrefine'] < -options['maxcoarsen']
        options['numrefine'][flag] = -options['maxcoarsen']

        # refine
        NC = self.number_of_cells()
        print("Number of cells before:", NC)
        isMarkedCell = (options['numrefine'] > 0)
        while sum(isMarkedCell) > 0:
            self.bisect_1(isMarkedCell, options)
            print("Number of cells after refine:", self.number_of_cells())
            isMarkedCell = (options['numrefine'] > 0)

        # coarsen
        if options['maxcoarsen'] > 0:
            isMarkedCell = (options['numrefine'] < 0)
            while sum(isMarkedCell) > 0:
                NN0 = self.number_of_cells()
                self.coarsen(isMarkedCell, options)
                NN = self.number_of_cells()
                if NN == NN0:
                    break
                print("Number of cells after coarsen:", self.number_of_cells())
                isMarkedCell = (options['numrefine'] < 0)

    def bisect_1(self, isMarkedCell=None, options={'disp': True}):
        GD = self.geo_dimension()
        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        NN0 = NN  # 记录下二分加密之前的节点数目

        if isMarkedCell is None:
            # 默认加密所有的单元
            markedCell = bm.arange(NC, dtype=self.itype)
        else:
            markedCell, = bm.nonzero(isMarkedCell)

        # allocate new memory for node and cell
        node = bm.zeros((5 * NN, GD), dtype=self.ftype, device=self.device)
        cell = bm.zeros((3 * NC, 3), dtype=self.itype, device=self.device)

        if ('numrefine' in options) and (options['numrefine'] is not None):
            options['numrefine'] = bm.concatenate((options['numrefine'], bm.zeros(2 * NC)))

        node = bm.set_at(node , slice(NN), self.entity('node'))
        cell = bm.set_at(cell , slice(NC), self.entity('cell'))

        # 用于存储网格节点的代数，初始所有节点都为第 0 代
        generation = bm.zeros(NN + 2 * NC, dtype=bm.uint8, device=self.device)

        # 用于记录被二分的边及其中点编号
        cutEdge = bm.zeros((4 * NN, 3), dtype=self.itype, device=self.device)

        # 当前的二分边的数目
        nCut = 0
        # 非协调边的标记数组
        nonConforming = bm.ones(4 * NN, dtype=bm.bool, device=self.device)
        while len(markedCell) != 0:
            # 标记最长边
            self.label(node, cell, markedCell)

            # 获取标记单元的四个顶点编号
            p0 = cell[markedCell, 0]
            p1 = cell[markedCell, 1]
            p2 = cell[markedCell, 2]

            # 找到新的二分边和新的中点
            nMarked = len(markedCell)
            p3 = bm.zeros(nMarked, dtype=self.itype, device=self.device)

            if nCut == 0:  # 如果是第一次循环
                idx = bm.arange(nMarked)  # cells introduce new cut edges
            else:
                # all non-conforming edges
                ncEdge = bm.nonzero(nonConforming[:nCut])
                NE = len(ncEdge)
                I = cutEdge[ncEdge][:, [2, 2]].reshape(-1)
                J = cutEdge[ncEdge][:, [0, 1]].reshape(-1)
                val = bm.ones(len(I), dtype=bm.bool)
                nv2v = csr_matrix(
                    (val, (I, J)),
                    shape=(NN, NN))
                i, j = (nv2v[:, p1].multiply(nv2v[:, p2])).nonzero()
                p3 = bm.set_at(p3, bm.array(j,dtype=self.itype), bm.array(i,dtype=self.itype))
                idx, = bm.nonzero(p3 == 0)

            if len(idx) != 0:
                # 把需要二分的边唯一化
                NE = len(idx)
                cellCutEdge = bm.stack([p1[idx], p2[idx]])
                cellCutEdge = bm.sort(cellCutEdge,axis=0)
                s = csr_matrix(
                    (
                        bm.ones(NE, dtype=bm.bool),
                        (
                            cellCutEdge[0, :],
                            cellCutEdge[1, :]
                        )
                    ), shape=(NN, NN))
                # 获得唯一的边
                i, j = s.nonzero()
                i = bm.tensor(i,dtype=self.itype, device=self.device)
                j = bm.tensor(j,dtype=self.itype, device=self.device)
                nNew = len(i)
                newCutEdge = bm.arange(nCut, nCut + nNew, device=self.device)
                cutEdge = bm.set_at(cutEdge , (newCutEdge, 0) , i)
                cutEdge = bm.set_at(cutEdge , (newCutEdge, 1) , j)
                cutEdge = bm.set_at(cutEdge , (newCutEdge, 2) , bm.arange(NN, NN + nNew, device=self.device))
                node = bm.set_at(node, slice(NN, NN + nNew), 0.5 * (node[i, :] + node[j, :]))
                nCut += nNew
                NN += nNew

                # 新点和旧点的邻接矩阵
                I = cutEdge[newCutEdge][:, [2, 2]].reshape(-1)
                J = cutEdge[newCutEdge][:, [0, 1]].reshape(-1)
                val = bm.ones(len(I), dtype=bm.bool, device=self.device)
                nv2v = csr_matrix(
                    (val, (I, J)),
                    shape=(NN, NN))
                i, j = (nv2v[:, p1].multiply(nv2v[:, p2])).nonzero()
                p3 = bm.set_at(p3, bm.array(j,dtype=self.itype, device=self.device), bm.array(i,dtype=self.itype, device=self.device))

            # 如果新点的代数仍然为 0
            idx = (generation[p3] == 0)
            cellGeneration = bm.max(
                generation[cell[markedCell[idx]]],
                axis=-1)
            # 第几代点
            generation = bm.set_at(generation , p3[idx] , cellGeneration + 1)
            cell = bm.set_at(cell ,(markedCell,0) , p3)
            cell = bm.set_at(cell ,(markedCell,1) , p0)
            cell = bm.set_at(cell ,(markedCell,2) , p1)
            cell = bm.set_at(cell ,(slice(NC,NC+nMarked),0) , p3)
            cell = bm.set_at(cell ,(slice(NC,NC+nMarked),1) , p2)
            cell = bm.set_at(cell ,(slice(NC,NC+nMarked),2) , p0)

            if ('numrefine' in options) and (options['numrefine'] is not None):
                bm.add_at(options['numrefine'], markedCell, -1)
                options['numrefine'] = bm.set_at(options['numrefine'], slice(NC, NC + nMarked), 
                                                 options['numrefine'][markedCell])

            NC = NC + nMarked
            del cellGeneration, p0, p1, p2, p3

            # 找到非协调的单元
            checkEdge, = bm.nonzero(nonConforming[:nCut])
            isCheckNode = bm.zeros(NN, dtype=bm.bool, device=self.device)
            isCheckNode = bm.set_at(isCheckNode, cutEdge[checkEdge], True)
            isCheckCell = bm.sum(
                isCheckNode[cell[:NC]],
                axis=-1) > 0
            # 找到所有包含检查节点的单元编号
            checkCell, = bm.nonzero(isCheckCell)
            I = bm.repeat(checkCell, 3)
            J = cell[checkCell].reshape(-1)
            val = bm.ones(len(I), dtype=bm.bool, device=self.device)
            cell2node = csr_matrix((val, (I, J)), shape=(NC, NN))
            i, j = (cell2node[:, cutEdge[checkEdge, 0]].multiply(
                    cell2node[:, cutEdge[checkEdge, 1]]
                )).nonzero()
              
            markedCell = bm.unique(bm.array(i))
            nonConforming = bm.set_at(nonConforming , checkEdge , False)
            nonConforming = bm.set_at(nonConforming , checkEdge[j] , True)

        if ('imatrix' in options) and (options['imatrix'] is True):
            nn = NN - NN0
            IM = coo_matrix(
                (
                    bm.ones(NN0),
                    (
                        bm.arange(NN0),
                        bm.arange(NN0)
                    )
                ), shape=(NN, NN), dtype=self.ftype)
            cutEdge = cutEdge[:nn]
            val = bm.full((nn, 2), 0.5, dtype=self.ftype)

            g = 2
            markedNode, = bm.nonzero(generation == g)

            N = len(markedNode)
            while N != 0:
                nidx = markedNode - NN0
                i = cutEdge[nidx, 0]
                j = cutEdge[nidx, 1]
                ic = bm.zeros((N, 2), dtype=self.ftype)
                jc = bm.zeros((N, 2), dtype=self.ftype)
                ic = bm.set_at(ic, (i < NN0,0), 1.0)
                jc = bm.set_at(jc, (j < NN0,1), 1.0)
                ic = bm.set_at(ic, i >= NN0, val[i[i >= NN0] - NN0])
                jc = bm.set_at(jc, j >= NN0, val[j[j >= NN0] - NN0])

                val = bm.set_at(val , markedNode - NN0 , 0.5 * (ic + jc))
                cutEdge = bm.set_at(cutEdge , (nidx[i >= NN0],0) , cutEdge[i[i >= NN0] - NN0,0])
                cutEdge = bm.set_at(cutEdge , (nidx[j >= NN0],1) , cutEdge[j[j >= NN0] - NN0,1])
                g += 1
                markedNode, = bm.nonzero(generation == g)
                N = len(markedNode)

            IM += coo_matrix(
                (
                    val.flat,
                    (
                        cutEdge[:, [2, 2]].flat,
                        cutEdge[:, [0, 1]].flat
                    )
                ), shape=(NN, NN0), dtype=self.ftype)
            options['imatrix'] = IM.tocsr()

        self.node = node[:NN]
        self.cell = cell[:NC]
        self.construct()

    def jacobian_matrix(self, index: Index=_S):
        """
        @brief 获得三角形单元对应的 Jacobian 矩阵
        """
        NC = self.number_of_cells()
        GD = self.geo_dimension()

        node = self.entity('node')
        cell = self.entity('cell')

        J = bm.zeros((NC, GD, 2), dtype=self.ftype, device=self.device)

        J[..., 0] = node[cell[:, 1]] - node[cell[:, 0]]
        J[..., 1] = node[cell[:, 2]] - node[cell[:, 0]]

        return J

    def point_to_bc(self, point):
        """
        @brief 找到定点 point 所在的单元，并计算其重心坐标 
        """
        pass

    def mark_interface_cell(self, phi):
        """
        @brief 标记穿过界面的单元
        """
        pass

    def mark_interface_cell_with_curvature(self, phi, hmax=None):
        """
        @brief 标记曲率大的单元
        """
        pass

    def mark_interface_cell_with_type(self, phi, interface):
        """
        @brief 等腰直角三角形，可以分为两类
            - Type A：两条直角边和坐标轴平行
            - Type B: 最长边和坐标轴平行
        """
        pass

    def bisect_interface_cell_with_curvature(self, interface, hmax):
        pass

    def show_function(self, plot, uh, cmap=None):
        pass

    @classmethod
    def show_lattice(cls, p=1, shownltiindex=False):
        """
        @berif 展示三角形上的单纯形格点
        """
        pass

    @classmethod
    def show_shape_function(cls, p=1, funtype='L'):
        """
        @brief 可视化展示三角形单元上的 p 次基函数
        """
        pass

    @classmethod
    def show_global_basis_function(cls, p=3):
        """
        @brief 展示通过单元基函数的拼接+零扩展的方法获取整体基函数的过程
        """
        pass

    @classmethod
    def from_one_triangle(cls, meshtype='iso'):
        if meshtype == 'equ':
            node = bm.tensor([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, bm.sqrt(bm.tensor(3)) / 2]], dtype=bm.float64)
        elif meshtype == 'iso':
            node = bm.tensor([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0]], dtype=bm.float64)
        cell = bm.tensor([[0, 1, 2]], dtype=bm.int32)
        return cls(node, cell)

    ## @ingroup MeshGenerators
    @classmethod
    def from_square_domain_with_fracture(cls, device=None):
        node = bm.tensor([
            [0.0, 0.0],
            [0.0, 0.5],
            [0.0, 0.5],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.5, 1.0],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0]], dtype=bm.float64, device=device)

        cell = bm.tensor([
            [1, 0, 5],
            [4, 5, 0],
            [2, 5, 3],
            [6, 3, 5],
            [4, 7, 5],
            [8, 5, 7],
            [6, 5, 9],
            [8, 9, 5]], dtype=bm.int32, device=device)

        return cls(node, cell)

    ## @ingroup MeshGenerators
    @classmethod
    def from_unit_square(cls, nx=10, ny=10, threshold=None):
        """
        Generate a triangle mesh for a unit square.

        @param nx Number of divisions along the x-axis (default: 10)
        @param ny Number of divisions along the y-axis (default: 10)
        @param threshold Optional function to filter cells based on their barycenter coordinates (default: None)
        @return TriangleMesh instance
        """
        return cls.from_box(box=[0, 1, 0, 1], nx=nx, ny=ny, 
                threshold=threshold, ftype=bm.float64, itype=bm.int32)

    ## @ingroup MeshGenerators
    @classmethod
    def from_box(cls, box=[0, 1, 0, 1], nx=10, ny=10, *, threshold=None,
                 itype=None, ftype=None, device=None):
        """Generate a triangle mesh for a box domain.

        @param box
        @param nx Number of divisions along the x-axis (default: 10)
        @param ny Number of divisions along the y-axis (default: 10)
        @param threshold Optional function to filter cells based on their barycenter coordinates (default: None)
        @return TriangleMesh instance
        """
        if itype is None:
            itype = bm.int32
        if ftype is None:
            ftype = bm.float64
        
        NN = (nx + 1) * (ny + 1)
        x = bm.linspace(box[0], box[1], nx+1, dtype=ftype, device=device)
        y = bm.linspace(box[2], box[3], ny+1, dtype=ftype, device=device)
        X, Y = bm.meshgrid(x, y, indexing='ij')

        node = bm.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)

        idx = bm.arange(NN, dtype=itype, device=device).reshape(nx + 1, ny + 1)
        cell0 = bm.concatenate((
            idx[1:, 0:-1].T.reshape(-1, 1),
            idx[1:, 1:].T.reshape(-1, 1),
            idx[0:-1, 0:-1].T.reshape(-1, 1),
            ), axis=1)
        cell1 = bm.concatenate((
            idx[0:-1, 1:].T.reshape(-1, 1),
            idx[0:-1, 0:-1].T.reshape(-1, 1),
            idx[1:, 1:].T.reshape(-1, 1)
            ), axis=1)
        cell = bm.concatenate((cell0, cell1), axis=0)

        if threshold is not None:
            bc = bm.sum(node[cell, :], axis=1) / cell.shape[1]
            isDelCell = threshold(bc)
            cell = cell[~isDelCell]
            isValidNode = bm.zeros(NN, dtype=bm.bool, device=device)
            isValidNode = bm.set_at(isValidNode, cell, True)
            node = node[isValidNode]
            idxMap = bm.zeros(NN, dtype=itype, device=device)
            idxMap = bm.set_at(
                idxMap, isValidNode, bm.arange(isValidNode.sum(), dtype=itype, device=device)
            )
            cell = idxMap[cell]

        return cls(node, cell)

    ## @ingroup MeshGenerators
    @classmethod
    def from_unit_sphere_surface(cls, refine=0, *, itype=None, ftype=None, device=None):
        """Generate a triangular mesh on a unit sphere surface."""
        if itype is None:
            itype = bm.int32
        if ftype is None:
            ftype = bm.float64

        t = (bm.sqrt(bm.tensor(5)) - 1) / 2
        node = bm.array([
            [0, 1, t], [0, 1, -t], [1, t, 0], [1, -t, 0],
            [0, -1, -t], [0, -1, t], [t, 0, 1], [-t, 0, 1],
            [t, 0, -1], [-t, 0, -1], [-1, t, 0], [-1, -t, 0]], dtype=ftype, device=device)
        cell = bm.array([
            [6, 2, 0], [3, 2, 6], [5, 3, 6], [5, 6, 7],
            [6, 0, 7], [3, 8, 2], [2, 8, 1], [2, 1, 0],
            [0, 1, 10], [1, 9, 10], [8, 9, 1], [4, 8, 3],
            [4, 3, 5], [4, 5, 11], [7, 10, 11], [0, 10, 7],
            [4, 11, 9], [8, 4, 9], [5, 7, 11], [10, 9, 11]], dtype=itype, device=device)
        mesh = cls(node, cell)
        mesh.uniform_refine(refine)
        node = mesh.node
        cell = mesh.entity('cell')
        d = bm.sqrt(node[:, 0] ** 2 + node[:, 1] ** 2 + node[:, 2] ** 2) - 1
        l = bm.sqrt(bm.sum(node ** 2, axis=1))
        n = node / l[..., None]
        node = node - d[..., None] * n
        return cls(node, cell)

    @classmethod
    def from_unit_circle_gmsh(cls, h):
        """
        Generate a triangular mesh for a unit circle by gmsh.

        @param h Parameter controlling mesh density
        @return TriangleMesh instance
        """
        import gmsh
        gmsh.initialize()
        gmsh.model.add("UnitCircle")

        # 创建单位圆
        gmsh.model.occ.addDisk(0.0, 0.0, 0.0, 1, 1, 1)

        # 同步几何模型
        gmsh.model.occ.synchronize()

        # 设置网格尺寸
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)

        # 生成网格
        gmsh.model.mesh.generate(2)

        # 获取节点信息
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_tags = bm.from_numpy(node_tags)
        node_coords = bm.from_numpy(node_coords)
        node = node_coords.reshape((-1, 3))[:, :2]

        # 节点编号映射
        nodetags_map = dict({int(j): i for i, j in enumerate(node_tags)})

        # 获取单元信息
        cell_type = 2  # 三角形单元的类型编号为 2
        cell_tags, cell_connectivity = gmsh.model.mesh.getElementsByType(cell_type)

        # 节点编号映射到单元
        evid = bm.array([nodetags_map[int(j)] for j in cell_connectivity])
        cell = evid.reshape((cell_tags.shape[-1], -1))

        gmsh.finalize()

        # 输出节点和单元数量
        print(f"Number of nodes: {node.shape[0]}")
        print(f"Number of cells: {cell.shape[0]}")

        return cls(node, cell)

    ## @ingroup MeshGenerators
    @classmethod
    def from_polygon_gmsh(cls, vertices, h):
        """
        Generate a triangle mesh for a polygonal region by gmsh.

        @param vertices List of tuples representing vertices of the polygon
        @param h Parameter controlling mesh density
        @return TriangleMesh instance
        """

        import gmsh
        gmsh.initialize()
        gmsh.model.add("Polygon")

        # 创建多边形
        lc = h  # 设置网格大小
        polygon_points = []
        for i, vertex in enumerate(vertices):
            point = gmsh.model.geo.addPoint(vertex[0], vertex[1], 0, lc)
            polygon_points.append(point)

        # 添加线段和循环
        lines = []
        for i in range(len(polygon_points)):
            line = gmsh.model.geo.addLine(polygon_points[i], polygon_points[(i + 1) % len(polygon_points)])
            lines.append(line)
        curve_loop = gmsh.model.geo.addCurveLoop(lines)

        # 创建平面表面
        surface = gmsh.model.geo.addPlaneSurface([curve_loop])

        # 同步几何模型
        gmsh.model.geo.synchronize()

        # 生成网格
        gmsh.model.mesh.generate(2)

        # 获取节点信息
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        node_tags = bm.from_numpy(node_tags)
        node = bm.array(node_coords, dtype=bm.float64).reshape(-1, 3)[:, 0:2]

        # 获取三角形单元信息
        cell_type = 2  # 三角形单元的类型编号为 2
        cell_tags, cell_connectivity = gmsh.model.mesh.getElementsByType(cell_type)
        cell = bm.array(cell_connectivity, dtype=bm.int64).reshape(-1, 3) - 1

        # 输出节点和单元数量
        print(f"Number of nodes: {node.shape[0]}")
        print(f"Number of cells: {cell.shape[0]}")

        gmsh.finalize()

        NN = len(node)
        isValidNode = bm.zeros(NN, dtype=bm.bool)
        isValidNode = bm.set_at(isValidNode, cell, True)
        node = node[isValidNode]
        idxMap = bm.zeros(NN, dtype=cell.dtype)
        idxMap = bm.set_at(idxMap, isValidNode, bm.arange(isValidNode.sum(), dtype=bm.int64))
        cell = idxMap[cell]

        return cls(node, cell)
    
    ## @ingroup MeshGenerators
    @classmethod
    def from_ellipsoid(cls, radius=[9, 3, 1], refine=0, *, itype=None, ftype=None, device=None):
        """
        a: 椭球的长半轴
        b: 椭球的中半轴
        c: 椭球的短半轴
        """
        a, b, c = radius
        mesh = TriangleMesh.from_unit_sphere_surface(itype=itype, ftype=ftype, device=device)
        mesh.uniform_refine(refine)
        node = mesh.node
        cell = mesh.entity('cell')
        node[:, 0]*=a 
        node[:, 1]*=b 
        node[:, 2]*=c
        return cls(node, cell)

    ## @ingroup MeshGenerators
    @classmethod
    def from_ellipsoid_surface(
        cls, ntheta=10, nphi=10, radius=(1, 1, 1), theta=None, phi=None,
        returnuv=False, *, itype=None, ftype=None, device=None):
        """
        @brief 给定椭球面的三个轴半径 radius=(a, b, c)，以及天顶角 theta 的范围,
        生成相应带状区域的三角形网格

        x = a \\sin\\theta \\cos\\phi
        y = b \\sin\\theta \\sin\\phi
        z = c \\cos\\theta

        @param[in] ntheta \\theta 方向的剖分段数
        @param[in] nphi \\phi 方向的剖分段数 
        """
        if theta is None:
            theta = (bm.pi / 4, 3 * bm.pi / 4)

        a, b, c = radius
        if phi is None:  # 默认为一封闭的带状区域
            NN = (ntheta + 1) * nphi
        else:  # 否则为四边形区域
            NN = (ntheta + 1) * (nphi + 1)

        NC = ntheta * nphi

        if phi is None:
            theta = bm.linspace(theta[0], theta[1], ntheta+1, dtype=bm.float64)
            l = bm.linspace(0, 2*bm.pi, nphi+1, dtype=bm.float64)
            U, V = bm.meshgrid(theta, l, indexing='ij')
            U = U[:, 0:-1]  # 去掉最后一列
            V = V[:, 0:-1]  # 去年最后一列
        else:
            theta = bm.linspace(theta[0], theta[1], ntheta+1, dtype=bm.float64)
            phi = bm.linspace(phi[0], phi[1], nphi+1, dtype=bm.float64)
            U, V = bm.meshgrid(theta, phi, indexing='ij')

        node = bm.zeros((NN, 3), dtype=bm.float64)
        X = a * bm.sin(U) * bm.cos(V)
        Y = b * bm.sin(U) * bm.sin(V)
        Z = c * bm.cos(U)
        node = bm.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)), axis=1)
        
        idx = bm.zeros((ntheta + 1, nphi + 1), dtype=bm.int32)
        if phi is None:
            idx[:, 0:-1] = bm.arange(NN).reshape(ntheta + 1, nphi)
            idx[:, -1] = idx[:, 0]
        else:
            idx = bm.arange(NN).reshape(ntheta + 1, nphi + 1)
        cell = bm.zeros((2 * NC, 3), dtype=bm.int32)
        cell0 = bm.concatenate((
            idx[1:, 0:-1].T.reshape(-1, 1),
            idx[1:, 1:].T.reshape(-1, 1),
            idx[0:-1, 0:-1].T.reshape(-1, 1)), axis=1)
        cell1 = bm.concatenate((
            idx[0:-1, 1:].T.reshape(-1, 1),
            idx[0:-1, 0:-1].T.reshape(-1, 1),
            idx[1:, 1:].T.reshape(-1, 1)), axis=1)
        cell = bm.concatenate((cell0, cell1), axis=1).reshape(-1, 3)

        if returnuv:
            return cls(node, cell), U.flatten(), V.flatten()
        else:
            return cls(node, cell)

    ### 界面网格 ###
    # NOTE: 均匀网格改成作为一个参数传入，避免循环内调用本函数时反复实例化。
    # 利用网格实体的缓存机制，节约生成实体时的性能消耗。
    @classmethod
    def interfacemesh_generator(cls, uniform_mesh_2d, /, phi: Callable[[TensorLike], TensorLike]):
        """Generate a triangle mesh fitting the interface.

        Parameters:
            uniform_mesh_2d (UniformMesh2d): A 2d uniform mesh as the background, constant.
            phi (Callable): A level-set function of the interface.

        Returns:
            TriangleMesh: The triangle mesh fitting the interface.
        """
        from scipy.spatial import Delaunay
        from .uniform_mesh_2d import UniformMesh2d

        if not isinstance(uniform_mesh_2d, UniformMesh2d):
            raise TypeError("Only UniformMesh2d is supported.")

        concat = bm.concat
        mesh = uniform_mesh_2d
        device = mesh.device

        iCellNodeIndex, cutNode, auxNode, isInterfaceCell = mesh.find_interface_node(phi)
        nonInterfaceCellIndex = bm.nonzero(~isInterfaceCell)[0]

        NN = mesh.number_of_nodes()
        nonInterfaceCell = mesh.entity('cell')[nonInterfaceCellIndex, :]
        node = mesh.entity('node')

        interfaceNode = concat(
            (node[iCellNodeIndex, :], cutNode, auxNode),
            axis = 0
        )
        dt = Delaunay(bm.to_numpy(interfaceNode))
        tri = bm.from_numpy(dt.simplices)
        tri = bm.device_put(tri, device)
        del dt, interfaceNode # 释放内存
        # 如果 3 个顶点至少有一个是切点（不都在前 NI 个里），则纳入考虑
        NI = iCellNodeIndex.shape[0]
        isNecessaryCell = bm.sum(tri < NI, axis=1) != 3
        tri = tri[isNecessaryCell, :]
        # 把顶点在 Delaunay 内的编号，转换为整个三角形内的编号
        interfaceNodeIdx = concat(
            [bm.astype(iCellNodeIndex, mesh.itype),
             NN + bm.arange(cutNode.shape[0] + auxNode.shape[0], dtype=mesh.itype, device=device)],
            axis = 0
        )
        tri = interfaceNodeIdx[tri]

        pnode = concat((node, cutNode, auxNode), axis=0)
        pcell = concat(
            [nonInterfaceCell[:, [2, 3, 0]], nonInterfaceCell[:, [1, 0, 3]], tri],
            axis = 0
        )
        return cls(pnode, pcell)

    def vtk_cell_type(self, etype='cell'):
        if etype in {'cell', 2}:
            VTK_TRIANGLE = 5
            return VTK_TRIANGLE
        elif etype in {'face', 'edge', 1}:
            VTK_LINE = 3
            return VTK_LINE

    def to_vtk(self, fname=None, etype='cell', index: Index=_S):
        """
        @brief 把网格转化为 vtk 的数据格式
        """
        from .vtk_extent import  write_to_vtu

        node = self.entity('node')
        GD = self.geo_dimension()
        if GD == 2:
            node = bm.concatenate((node, bm.zeros((node.shape[0], 1), dtype=bm.float64)), axis=1)

        cell = self.entity(etype)[index]
        cellType = self.vtk_cell_type(etype)
        NV = cell.shape[-1]

        cell = bm.concatenate((bm.zeros((len(cell), 1), dtype=cell.dtype), cell), axis=1)
        NC = len(cell)
        cell = bm.set_at(cell, (slice(NC), 0), NV)
        
        if fname is None:
            return node, cell.flatten(), cellType, NC
        else:
            print("Writting to vtk...")
            write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                         nodedata=self.nodedata,
                         celldata=self.celldata)
    @classmethod        
    def from_vtu(cls, file, show=False):
        import meshio
        data = meshio.read(file)
        node = bm.from_numpy(data.points)
        cell = bm.from_numpy(data.cells_dict['triangle'])

        mesh = cls(node, cell)
        if show:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            mesh.add_plot(ax)
            plt.show()
        return mesh
    
    @classmethod
    def from_medit(cls,file):
        '''
        Read medit format file (.mesh) to create triangle mesh.
        Parameters:
            file (str): Path to the medit format file.
        Returns:
            TriangleMesh: An instance of TriangleMesh created from the medit
            file.
        '''
        import meshio
        data = meshio.read(file)
        node = bm.from_numpy(data.points)
        cell = bm.from_numpy(data.cells_dict['triangle'])

        mesh = cls(node, cell)
        return mesh

    @classmethod
    def from_domain_distmesh(cls, domain, maxit=100, output=False, itype=None, ftype=None, device=None):
        from fealpy.old.mesh import DistMesher2d
        if itype is None:
            itype = bm.int32
        if ftype is None:
            ftype = bm.float64

        mesher = DistMesher2d(domain, domain.hmin, output=output)
        mesh = mesher.meshing(maxit=maxit)
        node = bm.array(mesh.entity('node'), dtype=ftype, device=device)
        cell = bm.array(mesh.entity('cell'), dtype=itype, device=device)

        return cls(node, cell)

TriangleMesh.set_ploter('2d')


