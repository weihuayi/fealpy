from typing import Union, Optional, Sequence, Tuple, Any
from math import sqrt,cos,sin,pi
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
from .utils import estr2dim
from ..sparse import COOTensor

from .mesh_base import Mesh
from .plot import Plotable


class PolygonMesh(Mesh, Plotable):
    def __init__(self, node: TensorLike, 
                 cell: Tuple[TensorLike, Optional[TensorLike]]) -> None:
        """
        """
        super().__init__(TD=2, itype=cell[0].dtype, ftype=node.dtype)
        kwargs = bm.context(cell[0]) 
        self.node = node
        if cell[1] is None: 
            assert cell[0].ndim == 2
            NC = cell[0].shape[0]
            NV = cell[0].shape[1]
            self.cell = (cell[0].reshape(-1), bm.arange(0, (NC+1)*NV, NV))
        else:
            self.cell = cell

        self.meshtype = 'polygon'

        self.construct()

        self.nodedata = {}
        self.edgedata = {}
        self.facedata = self.edgedata
        self.celldata = {}
        self.meshdata = {}

    def total_edge(self) -> TensorLike:
        cell, cellLocation = self.cell
        kwargs = bm.context(cell)
        totalEdge = bm.zeros((len(cell), 2), **kwargs)
        totalEdge = bm.set_at(totalEdge, (...,0), cell)
        totalEdge = bm.set_at(totalEdge, (slice(None,-1),1), cell[1:])
        totalEdge = bm.set_at(totalEdge, (cellLocation[1:]-1,1), cell[cellLocation[:-1]])
        return totalEdge

    total_face = total_edge

    def construct(self):
        """
        """
        cell, cellLocation = self.cell
        kwargs = bm.context(cell)
        totalEdge = self.total_edge()
        j0, i0, i1, j,j1 = bm.unique_all_(bm.sort(totalEdge, axis=1), axis=0)

        NE = i0.shape[0]
        self.edge2cell = bm.zeros((NE, 4), **kwargs)

        self.edge = totalEdge[i0]

        NC = self.number_of_cells()
        NV = self.number_of_vertices_of_cells() # (NC, )
         
        cellIdx = bm.repeat(bm.arange(NC), NV)
        shifts = bm.cumsum(NV,axis=0)
        id_arr = bm.ones(shifts[-1], **kwargs)
        id_arr = bm.set_at(id_arr, (shifts[:-1]), -bm.asarray(NV[:-1])+1)
        id_arr = bm.set_at(id_arr, 0, 0)
        localIdx = bm.cumsum(id_arr,axis=0)

        self.edge2cell = bm.set_at(self.edge2cell, (...,0), cellIdx[i0])
        self.edge2cell = bm.set_at(self.edge2cell, (...,1), cellIdx[i1])
        self.edge2cell = bm.set_at(self.edge2cell, (...,2), localIdx[i0])
        self.edge2cell = bm.set_at(self.edge2cell, (...,3), localIdx[i1])
        self.face2cell = self.edge2cell
        self.cell2edge = j

    def entity_barycenter(self, etype: Union[int, str]='cell', index=_S):
        node = self.entity('node')
        GD = self.geo_dimension()
        if isinstance(etype,str):
            etype = estr2dim(self,etype)
        if etype == 2:
            cell2node = self.cell_to_node(return_sparse=True)
            cell2node._values = bm.astype(cell2node._values,self.ftype)
            NV = self.number_of_vertices_of_cells().reshape(-1, 1)
            bc = cell2node@node/NV
            #bc = bm.dot(cell2node,node)/NV
            #bc = bm.einsum('ij,jk->ik',cell2node,node)/NV
        elif etype == 1:
            edge = self.entity('edge')
            bc = bm.mean(node[edge, :], axis=1).reshape(-1, GD)
        elif etype == 0:
            bc = node
        return bc

    def entity_measure(self,etype:Union[int,str],index:Index=_S) ->TensorLike:
        node = self.node
        if isinstance(etype, str):
            etype = estr2dim(self,etype)
        if etype == 0:
            return bm.tensor([0,],dtype = self.ftype)
        elif etype == 1:
            edge = self.entity(1,index)
            return bm.edge_length(edge,node)
        elif etype == 2:
            return self.cell_area(index=index)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")

    def cell_area(self,index=_S):
        NC = self.number_of_cells()
        node = self.entity('node')
        edge = self.entity('edge')
        edge2cell = self.edge2cell

        t = self.edge_tangent()
        val = t[:, 1]*node[edge[:,0],0]-t[:,0]*node[edge[:,0],1]
        a = bm.zeros((NC,),dtype = self.ftype)
         
        a = bm.index_add(a,edge2cell[:,0],val)
        
        isInEdge = (edge2cell[:,0] != edge2cell[:,1])
        a = bm.index_add(a,edge2cell[isInEdge,1],-val[isInEdge])
        
        a /= 2.0
        return a[index]

    def quadrature_formula(self, q, etype: Union[int, str] = 'cell', qtype='legendre'):
        if isinstance(etype,str):
            etype = estr2dim(self, etype)
        kwargs = {'dtype': self.ftype, 'device': self.device}
        if etype == 2:
            from ..quadrature import TriangleQuadrature
            return TriangleQuadrature(q, **kwargs)
        elif etype == 1:
            if qtype in {'legendre'}:
                from ..quadrature import GaussLegendreQuadrature
                return GaussLegendreQuadrature(q, **kwargs)
            elif qtype in {'lobatto'}:
                from ..quadrature import GaussLobattoQuadrature
                return GaussLobattoQuadrature(q, **kwargs)

    def number_of_global_ipoints(self, p: int) -> int:
        """
        @brief 插值点总数
        """
        gdof = self.number_of_nodes()
        if p > 1:
            NE = self.number_of_edges()
            NC = self.number_of_cells()
            gdof += NE*(p-1) + NC*(p-1)*p//2
        return gdof

    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='all') -> Union[TensorLike, int]:
        """
        @brief 获取局部插值点的个数
        """
        if iptype in {'all'}:
            NV = self.number_of_vertices_of_cells()
            ldof = NV + (p-1)*NV + (p-1)*p//2
            return ldof
        if isinstance(iptype, str):
            iptype = estr2dim(self,iptype)
        if iptype== 2:
            return (p-1)*p//2
        elif iptype==1:
            return (p+1)
        elif iptype == 0:
            return 1

    def edge_normal(self, index=_S):
        """
        @brief 计算二维网格中每条边上单位法线
        """
        assert self.geo_dimension() == 2
        v = self.edge_tangent(index=index)
        w = bm.tensor([(0,-1),(1,0)],dtype=self.ftype)
        return v@w

    def edge_unit_normal(self, index=_S):
        """
        @brief 计算二维网格中每条边上单位法线
        """
        assert self.geo_dimension() == 2
        v = self.edge_tangent(index=index,unit=True)
        w = bm.tensor([(0,-1),(1,0)],dtype=self.ftype)
        return v@w

    def interpolation_points(self, p: int, index=_S, scale: float=0.3):
        """
        @brief 获取多边形网格上的插值点
        
        @TODO: 边上可以取不同的插值点，lagrange, lobatto, legendre
        """
        node = self.entity('node')

        if p == 1:
            return node

        gdof = self.number_of_global_ipoints(p)

        GD = self.geo_dimension()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()
        start = 0
        ipoint = bm.zeros((gdof, GD), dtype=self.ftype)
        if bm.backend_name in ["numpy","pytorch"]:
            ipoint[start:NN, :] = node

            start += NN

            edge = self.entity('edge')
            qf = self.quadrature_formula(p+1, etype='edge', qtype='lobatto')
            quadpts,ws = qf.get_quadrature_points_and_weights()
            bcs = quadpts[1:-1, :]
            ipoint[start:NN+(p-1)*NE, :] = bm.einsum('ij, ...jm->...im', bcs, node[edge, :]).reshape(-1, GD)
            start += (p-1)*NE

            if p == 2:
                ipoint[start:] = self.entity_barycenter('cell')
                return ipoint

            h = bm.sqrt(self.cell_area())[:, None]*scale
            bc = self.entity_barycenter('cell')
            t = bm.tensor([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, sqrt(3)/2]], dtype=self.ftype)
            t -= bm.tensor([0.5, sqrt(3)/6.0], dtype=self.ftype)

            tri = bm.zeros((NC, 3, GD), dtype=self.ftype)
            tri[:, 0, :] = bc + t[0]*h
            tri[:, 1, :] = bc + t[1]*h
            tri[:, 2, :] = bc + t[2]*h

            bcs = self.multi_index_matrix(p-2, 2)/(p-2)
            bcs = bm.astype(bcs,self.ftype)
            ipoint[start:] = bm.einsum('ij, ...jm->...im', bcs, tri).reshape(-1, GD)
            return ipoint

        elif bm.backend_name == "jax":
            ipoint = ipoint.at[start:NN,:].set(node)
            start += NN

            edge = self.entity('edge')
            qf = self.quadrature_formula(p+1, etype='edge', qtype='lobatto')
            quadpts,ws = qf.get_quadrature_points_and_weights()
            bcs = quadpts[1:-1, :]
            ipoint = ipoint.at[start:NN+(p-1)*NE,:].set(bm.einsum('ij, ...jm->...im',bcs,node[edge,:]).reshape(-1, GD))
            start += (p-1)*NE

            if p == 2:
                ipoint = ipoint.at[start:].set(self.entity_barycenter('cell'))
                return ipoint

            h = bm.sqrt(self.cell_area())[:, None]*scale
            bc = self.entity_barycenter('cell')
            t = bm.tensor([
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, sqrt(3)/2]], dtype=self.ftype)
            t -= bm.tensor([0.5, sqrt(3)/6.0], dtype=self.ftype)

            tri = bm.zeros((NC, 3, GD), dtype=self.ftype)
            tri = tri.at[...,0,:].set(bc + t[0]*h)
            tri = tri.at[...,1,:].set(bc + t[1]*h)
            tri = tri.at[...,2,:].set(bc + t[2]*h)

            bcs = self.multi_index_matrix(p-2, 2)/(p-2)
            ipoint = ipoint.at[start:].set(bm.einsum('ij, ...jm->...im', bcs,tri).reshape(-1, GD))
            return ipoint
     
    #需要功能 hsplit
    def cell_to_ipoint(self, p: int, index=_S):
        """
        @brief
        """
        cell,cellLocation = self.entity('cell')
        if p == 1:
            return cell[index]
        else:
            if bm.backend_name in ["numpy","pytorch"]:
                NC = self.number_of_cells()
                ldof = self.number_of_local_ipoints(p, iptype='all')

                location = bm.zeros(NC+1, dtype=self.itype)
                location[1:] = bm.cumsum(ldof,axis=0)

                cell2ipoint = bm.zeros(location[-1], dtype=self.itype)

                edge2ipoint = self.edge_to_ipoint(p)
                edge2cell = self.edge2cell

                idx = location[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + bm.arange(p)
                cell2ipoint[idx] = edge2ipoint[:, 0:p]

                isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
                idx = (location[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*p).reshape(-1,1) + bm.arange(p)
                selected_elements = edge2ipoint[isInEdge, 1:p+1]
                reversed_elements = bm.flip(selected_elements, axis=1)
                cell2ipoint[idx] = reversed_elements

                NN = self.number_of_nodes()
                NV = self.number_of_vertices_of_cells()
                NE = self.number_of_edges()
                cdof = self.number_of_local_ipoints(p, iptype='cell')
                idx = (location[:-1] + NV*p).reshape(-1, 1) + bm.arange(cdof)
                cell2ipoint[idx] = NN + NE*(p-1) + bm.arange(NC*cdof).reshape(NC, cdof)
                return bm.split(cell2ipoint, location[1:-1])[index]
            elif bm.backend_name == "jax": 
                NC = self.number_of_cells()
                ldof = self.number_of_local_ipoints(p, iptype='all')

                location = bm.zeros(NC+1, dtype=self.itype)
                location = location.at[1:].set(bm.cumsum(ldof,axis=0))

                cell2ipoint = bm.zeros(location[-1], dtype=self.itype)

                edge2ipoint = self.edge_to_ipoint(p)
                edge2cell = self.edge2cell

                idx = location[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + bm.arange(p)
                cell2ipoint = cell2ipoint.at[idx].set(edge2ipoint[:, 0:p])

                isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
                idx = (location[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*p).reshape(-1,1) + bm.arange(p)
                cell2ipoint = cell2ipoint.at[idx].set(edge2ipoint[isInEdge, p:0:-1])

                NN = self.number_of_nodes()
                NV = self.number_of_vertices_of_cells()
                NE = self.number_of_edges()
                cdof = self.number_of_local_ipoints(p, iptype='cell')
                idx = (location[:-1] + NV*p).reshape(-1, 1) + bm.arange(cdof)
                cell2ipoint = cell2ipoint.at[idx].set(NN + NE*(p-1) + bm.arange(NC*cdof).reshape(NC,cdof))
                return bm.split(cell2ipoint, location[1:-1])[index]

    def shape_function(self, bcs: TensorLike, p: int,index: Index = _S) -> TensorLike:
        raise NotImplementedError

    def grad_shape_function(self, bcs: TensorLike, p: int, index:Index = _S) -> TensorLike:
        raise NotImplementedError

    def uniform_refine(self, n: int=1) -> None:
        raise NotImplementedError

    def integral(self, u, q=3, celltype=False):
        """
        @brief 多边形网格上的数值积分

        @param[in] u 被积函数, 需要两个参数 (x, index)
        @param[in] q 积分公式编号
        """
        node = self.entity('node')
        edge = self.entity('edge')
        edge2cell = self.edge2cell
        NC = self.number_of_cells()
        NE = self.number_of_edges()

        bcs, ws = self.quadrature_formula(q).get_quadrature_points_and_weights()
        bcs = bm.astype(bcs,self.ftype) 
        ws = bm.astype(ws,self.ftype)
        bc = self.entity_barycenter('cell')
        tri = bm.zeros((3,NE,2),dtype=self.ftype)
        
        tri = bm.set_at(tri,(0),bc[edge2cell[:,0]])
        tri = bm.set_at(tri,(1),node[edge[:,0]])
        tri = bm.set_at(tri,(2),node[edge[:,1]])
        
        v1 = node[edge[:, 0]] - bc[edge2cell[:, 0]]
        v2 = node[edge[:, 1]] - bc[edge2cell[:, 0]]
        a = (v1[:,0]*v2[:,1] - v1[:,1]*v2[:,0])/2.0
    
        pp = bm.einsum('ij, jkm->ikm', bcs, tri)
        val = u(pp, edge2cell[:, 0])

        shape = (NC, ) + val.shape[2:]
        e = bm.zeros(shape, dtype=self.ftype)

        ee = bm.einsum('i, ij..., j->j...', ws, val, a)
        e = bm.index_add(e, edge2cell[:, 0], ee)

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        if bm.sum(isInEdge) > 0:
            tri = bm.zeros((3,bm.sum(isInEdge),2),dtype=self.ftype)
            tri = bm.set_at(tri,(0),bc[edge2cell[isInEdge,1]])
            tri = bm.set_at(tri,(1),node[edge[isInEdge,1]])
            tri = bm.set_at(tri,(2),node[edge[isInEdge,0]])

            v1 = node[edge[isInEdge, 1]] - bc[edge2cell[isInEdge, 1]]
            v2 = node[edge[isInEdge, 0]] - bc[edge2cell[isInEdge, 1]]
            
            a = (v1[:,0]*v2[:,1] - v1[:,1]*v2[:,0])/2.0

            pp = bm.einsum('ij, jkm->ikm', bcs, tri)
            val = u(pp, edge2cell[isInEdge, 1])
            ee = bm.einsum('i, ij..., j->j...', ws, val, a)
            e = bm.index_add(e, edge2cell[isInEdge, 1], ee)
            
        if celltype is True:
            return e
        else:
            return e.sum(axis=0)
    def edge_to_cell(self):
        return self.edge2cell

    def cell_to_node(self,return_sparse=False):
        if return_sparse:
            NN = self.number_of_nodes()
            NC = self.number_of_cells()

            NV = self.number_of_vertices_of_cells()
            cell = self.cell
            cellLocation = cell[1]
            cell = cell[0]
            kwargs = bm.context(cell)

            row = bm.repeat(bm.arange(NC),NV)
            col = cell
            data = bm.ones_like(row,**kwargs)
            
            indice = bm.stack([row,col],axis=0)
             
            cell2node = COOTensor(indice,data,spshape=(NC,NN))
            return cell2node
        else:
            return self.cell

    @classmethod
    def from_triangle_mesh_by_dual(cls, mesh, bc=True):
        """
        @brief 生成三角形网格的对偶网格，目前默认用三角形的重心做为对偶网格的顶点

        @param mesh
        @param bc bool 如果为真，则对偶网格点为三角形单元重心; 否则为三角形单元外心
        """
        raise NotImplementedError


    @classmethod
    def from_box(cls, box=[0, 1, 0, 1], nx=10, ny=10, threshold=None):
        """
        @brief Generate a polygon mesh for a box domain
        """
        raise NotImplementedError

    @classmethod
    def from_one_triangle(cls,meshtype='iso',*,device=None): 
        if meshtype == 'equ':
            node = bm.tensor([
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.5, sqrt(3)/2]], dtype=bm.float64,device=device)
        elif meshtype =='iso':
            node = bm.tensor([
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0]], dtype=bm.float64)
        cell = (bm.tensor([[0, 1, 2]],dtype=bm.int64,device=device),None)
        return cls(node, cell)

    @classmethod
    def from_one_square(cls,*,device=None):
        node = bm.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]],dtype=bm.float64,device=device)
        cell = (bm.tensor([[0, 1, 2, 3]], dtype=bm.int64,device=device),None)
        return cls(node, cell)

    @classmethod
    def from_one_pentagon(cls,*,device=None):
        node = bm.tensor([
            (0.0, 0.0),
            (cos(2/5*pi), -sin(2/5*pi)),
            (cos(2/5*pi)+1, -sin(2/5*pi)),
            ( 2*cos(1/5*pi), 0.0),
            (cos(1/5*pi), sin(1/5*pi))],dtype=bm.float64,device=device)
        cell = (bm.tensor([0, 1, 2, 3, 4], dtype=bm.int64,device=device),bm.tensor([0,5],dtype=bm.int64))
        return cls(node, cell)

    @classmethod
    def from_one_hexagon(cls,*,device=None):
        node = bm.tensor([
            [0.0, 0.0],
            [1/2, -sqrt(3)/2],
            [3/2, -sqrt(3)/2],
            [2.0, 0.0],
            [3/2, sqrt(3)/2],
            [1/2, sqrt(3)/2]], dtype=bm.float64,device=device)
        cell = (bm.tensor([0, 1, 2, 3, 4, 5], dtype=bm.int64,device=device),bm.tensor([0, 6], dtype=bm.int64))
        return cls(node, cell)

    @classmethod
    def from_mesh(cls, mesh: Mesh):
        """
        @brief 把一个由同一类型单元组成网格转化为多边形网格的格式
        """
        node = mesh.entity('node')
        cell = (mesh.entity('cell'),None)
        return cls(node, cell)

    @classmethod
    def distorted_concave_rhombic_quadrilaterals_mesh(cls, box=[0, 1, 0, 1], nx=10, ny=10, ratio=0.618):
        """
        @brief 虚单元网格，矩形内部包含一个菱形，两者共用左下和右上的节点

        @param box 网格所占区域
        @param nx 沿 x 轴方向剖分段数
        @param ny 沿 y 轴方向剖分段数
        @param ratio 矩形内部菱形的大小比例
        """
        from .quadrangle_mesh import QuadrangleMesh
        from .uniform_mesh_2d import UniformMesh2d

        hx = (box[1] - box[0]) / nx
        hy = (box[3] - box[2]) / ny

        mesh0 = UniformMesh2d([0, nx, 0, ny], h=(hx, hy), origin=(box[0], box[2]))
        node0 = mesh0.entity("node")
        cell0 = mesh0.entity("cell")[:, [0, 2, 3, 1]]
        mesh = QuadrangleMesh(node0, cell0)

        edge = mesh.entity("edge")
        node = mesh.entity("node")
        cell = mesh.entity("cell")
        ftype = node.dtype
        itype = cell.dtype
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()

        node_append1 = node[cell[:, 3]] * (1-ratio) + node[cell[:, 1]] * ratio
        node_append2 = node[cell[:, 3]] * ratio + node[cell[:, 1]] * (1-ratio)
        new_node = bm.concatenate((node, node_append1, node_append2),axis=0,
                                  dtype=ftype)

        cell = bm.tile(cell, (3, 1))
        idx1 = bm.arange(NN, NN + NC, dtype=itype)
        idx2 = bm.arange(NN + NC, NN + 2 * NC, dtype=itype)
        cell[0:NC, 3] = idx1
        cell[NC:2 * NC, 1] = idx1
        cell[NC:2 * NC, 3] = idx2
        cell[2 * NC:3 * NC, 1] = idx2
        cellLocation = bm.arange(0, 4*(NC*3+1),4, dtype=itype)
        cell = (cell.reshape(-1), cellLocation)

        return cls(new_node, cell)

    @classmethod
    def nonconvex_octagonal_mesh(cls, box=[0, 1, 0, 1], nx=10, ny=10):
        """
        @brief 虚单元网格，矩形网格的每条内部边上加一个点后形成的八边形网格

        @param box 网格所占区域
        @param nx 沿 x 轴方向剖分段数
        @param ny 沿 y 轴方向剖分段数
        """
        from .quadrangle_mesh import QuadrangleMesh
        from .uniform_mesh_2d import UniformMesh2d

        hx = (box[1] - box[0]) / nx
        hy = (box[3] - box[2]) / ny
        NN = (nx + 1) * (ny + 1)

        mesh0 = UniformMesh2d([0, nx, 0, ny], h=(hx, hy), origin=(box[0], box[2]))
        node0 = mesh0.entity("node")
        cell0 = mesh0.entity("cell")[:, [0, 2, 3, 1]]
        mesh = QuadrangleMesh(node0, cell0)

        edge = mesh.entity("edge")
        node = mesh.entity("node")
        cell = mesh.entity("cell")
        ftype = node.dtype
        itype = cell.dtype
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        cell2edge = mesh.cell_to_edge()
        isbdedge = mesh.boundary_face_flag()
        isbdcell = mesh.boundary_cell_flag()

        nie = bm.sum(~isbdedge)
        hx = 1 / nx
        hy = 1 / ny
        newnode = bm.zeros((NN + nie, 2), dtype=ftype)
        newnode[:NN] = node
        newnode[NN:] = 0.5 * node[edge[~isbdedge, 0]] + 0.5 * node[edge[~isbdedge, 1]]
        newnode[NN: NN + (nx - 1) * ny] = newnode[NN: NN + (nx - 1) * ny] + bm.array([[0.2 * hx, 0.1 * hy]])
        newnode[NN + (nx - 1) * ny:] = newnode[NN + (nx - 1) * ny:] + bm.array([[0.1 * hx, 0.2 * hy]])

        edge2newnode = -bm.ones(NE, dtype=itype)
        edge2newnode[~isbdedge] = bm.arange(NN, newnode.shape[0], dtype=itype)
        newcell = bm.zeros((NC, 8), dtype=itype)
        newcell[:, ::2] = cell
        newcell[:, 1::2] = edge2newnode[cell2edge]

        flag = newcell > -1
        num = bm.zeros(NC + 1, dtype=itype)
        num[1:] = bm.sum(flag, axis=-1)
        newcell = newcell[flag]
        cellLocation = bm.cumsum(num, axis=0)
        cell = (newcell, cellLocation)
        return cls(newnode, cell)




PolygonMesh.set_ploter('polygon2d')
