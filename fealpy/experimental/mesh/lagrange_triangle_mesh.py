from typing import Union, Optional, Sequence, Tuple, Any

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .utils import simplex_gdof, simplex_ldof 
from .mesh_base import HomogeneousMesh, estr2dim
from .triangle_mesh import TriangleMesh

class LagrangeTriangleMesh(HomogeneousMesh):
    def __init__(self, node: TensorLike, cell: TensorLike, p=1, surface=None,
            construct=False):
        super().__init__(TD=2, itype=cell.dtype, ftype=node.dtype)

        kwargs = bm.context(cell)
        self.p = p
        self.node = node
        self.cell = cell
        self.surface = surface

        self.localEdge = self.generate_local_lagrange_edges(p)
        self.localFace = self.localEdge
        self.ccw  = bm.array([0, 1, 2], **kwargs)

        self.localCell = bm.array([
            (0, 1, 2),
            (1, 2, 0),
            (2, 0, 1)], **kwargs)

        if construct:
            self.construct()

        self.meshtype = 'ltri'

        self.nodedata = {}
        self.edgedata = {}
        self.celldata = {}
        self.meshdata = {}

    def reference_cell_measure(self):
        return 0.5

    def generate_local_lagrange_edges(self, p: int) -> TensorLike:
        """
        Generate the local edges for Lagrange elements of order p.
        """
        TD = self.top_dimension()
        multiIndex = bm.multi_index_matrix(p, TD)

        localEdge = bm.zeros((3, p+1), dtype=bm.int32)
        localEdge[2, :], = bm.where(multiIndex[:, 2] == 0)
        localEdge[1,:] = bm.flip(bm.where(multiIndex[:, 1] == 0)[0])
        #localEdge[1, -1::-1], = bm.where(multiIndex[:, 1] == 0)
        localEdge[0, :],  = bm.where(multiIndex[:, 0] == 0)

        return localEdge

    @classmethod
    def from_triangle_mesh(cls, mesh, p: int, surface=None):
        node = mesh.interpolation_points(p)
        cell = mesh.cell_to_ipoint(p)
        if surface is not None:
            node, _ = surface.project(node)

        lmesh = cls(node, cell, p=p, construct=True)

        lmesh.edge2cell = mesh.edge2cell # (NF, 4)
        lmesh.cell2edge = mesh.cell_to_edge()
        #lmesh.edge  = mesh.edge_to_ipoint(p)
        return lmesh 

    # quadrature
    def quadrature_formula(self, q: int, etype: Union[int, str]='cell'):
        from ..quadrature import TriangleQuadrature
        from ..quadrature import GaussLegendreQuadrature

        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        if etype == 2:
            quad = TriangleQuadrature(q)
        elif etype == 1:
            quad = GaussLegendreQuadrature(q)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")

        return quad

    def bc_to_point(self, bc: TensorLike, index: Index=_S, etype='cell'):
        """
        """
        node = self.node
        TD = bc.shape[-1] - 1
        entity = self.entity(TD, index=index) # 
        phi = self.shape_function(bc) # (NC, NQ, ldof)
        p = bm.einsum('cqn, cni -> cqi', phi, node[entity])
        return p
    
    # shape function
    def shape_function(self, bc: TensorLike, p=None):
        p = self.p if p is None else p
        phi = bm.simplex_shape_function(bc, p=p)
        return phi[None, :, :]

    def grad_shape_function(self, bc: TensorLike, p=None, index: Index=_S, variables='x'):
        """
        @berif 计算单元形函数关于参考单元变量 u=(xi, eta) 或者实际变量 x 梯度。

        lambda_0 = 1 - xi - eta
        lambda_1 = xi
        lambda_2 = eta

        """
        p = self.p if p is None else p 
        TD = bc.shape[-1] - 1
        if TD == 2:
            Dlambda = bm.array([[-1, -1], [1, 0], [0, 1]], dtype=bm.float64)
        else:
            Dlambda = bm.array([[-1], [1]], dtype=bm.float64)
        R = bm.simplex_grad_shape_function(bc, p=p) # (..., ldof, TD+1)
        gphi = bm.einsum('...ij, jn -> ...in', R, Dlambda) # (..., ldof, TD)
        
        if variables == 'u':
            return gphi[..., None, :, :] #(..., 1, ldof, TD)
        elif variables == 'x':
            G, J = self.first_fundamental_form(bc, index=index, return_jacobi=True)
            G = bm.linalg.inv(G)
            gphi = bm.einsum('q...km, q...mn, ...ln -> q...lk', J, G, gphi) 
            return gphi

    # ipoint --> copy TriangleMesh
    def number_of_local_ipoints(self, p:int, iptype:Union[int, str]='cell'):
        """
        @berif 每个ltri单元上插值点的个数
        """
        if isinstance(iptype, str):
            iptype = estr2dim(self, iptype)
        return simplex_ldof(p, iptype)

    def number_of_global_ipoints(self, p:int):
        """
        @berif ltri网格上插值点总数
        """
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()
        num = (NN, NE, NC)
        return simplex_gdof(p, num)

    def interpolation_points(self, p:int, index:Index=_S):
        """
        @berif 获取ltri网格上全部插值点
        """
        node = self.entity('node')
        if p == 1:
            return node
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

        return bm.concatenate(ipoint_list, axis=0)  # (gdof, GD)

    def cell_to_ipoint(self, p:int, index:Index=_S):
        """
        @berif 获取单元与插值点的对应关系
        """
        sp = self.p
        cell = self.cell[:, [0, -sp-1, 1]]  # 取角点

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
 
    def entity_measure(self, etype=2, index:Index=_S):
        if etype in {'cell', 2}:
            return self.cell_area(index=index)
        elif etype in {'edge', 1}:
            return self.edge_length(index=index)
        elif etype in {'node', 0}:
            return bm.zeros(1, dtype=bm.float64)
        else:
            raise ValueError(f"entity type:{etype} is erong!")
        
    def cell_area(self, q=None, index: Index=_S):
        """
        Calculate the area of a cell.
        """
        p = self.p
        q = p if q is None else q
        GD = self.geo_dimension()

        qf = self.quadrature_formula(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        J = self.jacobi_matrix(bcs, index=index)
        n = bm.cross(J[..., 0], J[..., 1], axis=-1)
        if GD == 3:
            n = bm.sqrt(bm.sum(n**2, axis=-1)) # (NC, NQ)
        a = bm.einsum('i, ji -> j', ws, n)/2.0
        return a

    def edge_length(self, q=None, index: Index=_S):
        """
        Calculate the length of the side.
        """
        p = self.p
        q = p if q is None else q
        qf = self.quadrature_formula(q, etype='edge')
        bcs, ws = qf.get_quadrature_points_and_weights()
        
        J = self.jacobi_matrix(bcs, index=index)
        a = bm.sqrt(bm.sum(J**2, axis=(-1, -2)))
        l = bm.einsum('i, ij -> j', ws, a)
        return l

    def cell_unit_normal(self, bc: TensorLike, index: Index=_S):
        """
        When calculating the surface,the direction of the unit normal at the integration point. 
        """
        J = self.jacobi_matrix(bc, index=index)
        n = bm.cross(J[..., 0], J[..., 1], axis=-1)
        if self.GD == 3:
            l = bm.sqrt(bm.sum(n**2, axis=-1, keepdims=True))
            n /= l
        return n

    def jacobi_matrix(self, bc: TensorLike, p=None, index: Index=_S, return_grad=False):
        """
        @berif 计算参考单元 （xi, eta) 到实际 Lagrange 三角形(x) 之间映射的 Jacobi 矩阵。

        x(xi, eta) = phi_0 x_0 + phi_1 x_1 + ... + phi_{ldof-1} x_{ldof-1}
        """

        TD = bc.shape[-1] - 1
        entity = self.entity(TD, index)
        gphi = self.grad_shape_function(bc, p=p, variables='u')
        J = bm.einsum(
                'cin, ...cim -> c...nm',
                self.node[entity[index], :], gphi) #(NC,ldof,GD),(NC,NQ,ldof,TD)
        if return_grad is False:
            return J #(NC,NQ,GD,TD)
        else:
            return J, gphi

    def uniform_refine(self, n=1):
        pass
    
    # fundamental form
    def first_fundamental_form(self, bc: Union[TensorLike, Tuple[TensorLike]], 
            index: Index=_S, return_jacobi=False, return_grad=False):
        """
        Compute the first fundamental form of a mesh surface at integration points.
        """
        TD = bc.shape[-1] - 1

        J = self.jacobi_matrix(bc, index=index,
                return_grad=return_grad)
        
        if return_grad:
            J, gphi = J

        shape = J.shape[0:-2] + (TD, TD)
        G = bm.zeros(shape, dtype=self.ftype)
        for i in range(TD):
            G[..., i, i] = bm.sum(J[..., i]**2, axis=-1)
            for j in range(i+1, TD):
                G[..., i, j] = bm.sum(J[..., i]*J[..., j], axis=-1)
                G[..., j, i] = G[..., i, j]
        if (return_jacobi is False) & (return_grad is False):
            return G
        elif (return_jacobi is True) & (return_grad is False): 
            return G, J
        elif (return_jacobi is False) & (return_grad is True): 
            return G, gphi 
        else:
            return G, J, gphi

    def second_fundamental_form(self, bc: Union[TensorLike, Tuple[TensorLike]], 
            index: Index=_S, return_jacobi=False, return_grad=False):
        """
        Compute the second fundamental form of a mesh surface at integration points.
        """
        TD = bc.shape[-1] - 1
        pass

    def vtk_cell_type(self, etype='cell'):
        """
        @berif  返回网格单元对应的 vtk类型。
        """
        if etype in {'cell', 2}:
            VTK_LAGRANGE_TRIANGLE = 69
            return VTK_LAGRANGE_TRIANGLE 
        elif etype in {'face', 'edge', 1}:
            VTK_LAGRANGE_CURVE = 68
            return VTK_LAGRANGE_CURVE

    def to_vtk(self, etype='cell', index: Index=_S, fname=None):
        """
        Parameters
        ----------

        @berif 把网格转化为 VTK 的格式
        """
        from fealpy.mesh.vtk_extent import vtk_cell_index, write_to_vtu

        node = self.entity('node')
        GD = self.geo_dimension()
        if GD == 2:
            node = np.concatenate((node, bm.zeros((node.shape[0], 1), dtype=bm.float64)), axis=1)

        #cell = self.entity(etype)[index]
        cell = self.entity(etype, index)
        cellType = self.vtk_cell_type(etype)
        idx = vtk_cell_index(self.p, cellType)
        NV = cell.shape[-1]

        cell = bm.concatenate((bm.zeros((len(cell), 1), dtype=cell.dtype), cell[:, idx]), axis=1)
        cell[:, 0] = NV

        NC = len(cell)
        if fname is None:
            return node, cell.flatten(), cellType, NC 
        else:
            print("Writting to vtk...")
            write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                    nodedata=self.nodedata,
                    celldata=self.celldata)
