from typing import Union

from .. import logger
from ..backend import backend_manager as bm
from ..typing import TensorLike , Index, _S
from ..sparse import coo_matrix, csr_matrix
from .mesh_data_structure import MeshDS
from .utils import estr2dim
from .plot import Plotable
from .mesh_base import SimplexMesh


class IntervalMeshDataStructure(MeshDS):
    def total_face(self):
        return self.cell.reshape(-1, 1)

class IntervalMesh(SimplexMesh,Plotable):
    def __init__(self, node: TensorLike ,cell:TensorLike):
        super().__init__(TD=1, itype=cell.dtype, ftype=node.dtype)
        
        if node.ndim == 1:
            self.node = node.reshape(-1, 1)
        else:
            self.node = node
        self.cell = cell
        self.edge = self.cell
        #self.face = bm.arange(self.node.shape[0]).reshape(-1,1)

        self.TD = 1

        self.meshtype = 'interval'
        self.meshtype = 'INT'

        self.nodedata = {}
        self.celldata = {}
        self.edgedata = self.celldata # celldata and edgedata are the same thing
        self.facedata = self.nodedata # facedata and nodedata are the same thing

        self.cell_length = self.edge_length
        self.cell_tangent = self.edge_tangent

        self.cell_to_ipoint = self.edge_to_ipoint
        self.localEdge = bm.tensor([0,1],dtype=bm.int32)
        self.localFace = bm.tensor([[0],[1]],dtype=bm.int32)

        self.itype = self.cell.dtype
        self.ftype = self.node.dtype

        self.construct()
        self.face2cell = self.face_to_cell()

    def ref_cell_measure(self):
        return 1.0

    def ref_face_measure(self):
        return 0.0
    
    def integrator(self, q: int, etype: Union[str, int]='cell'):
        """
        @brief 返回第 k 个高斯积分公式。
        """
        from ..quadrature import GaussLegendreQuadrature
        return GaussLegendreQuadrature(q)
    
    def entity_measure(self, etype: Union[int, str]='cell', index:Index=_S, node=None):
        """
        """
        if etype in {1, 'cell', 'edge'}:
            return self.cell_length(index=index)
        elif etype in {0, 'face', 'node'}:
            return bm.tensor([0.0], dtype=self.ftype)
        else:
            raise ValueError(f"entity type: {etype} is wrong!")

    def quadrature_formula(self, q: int, etype: Union[int, str]='cell',
                           qtype: str='legendre'):
        from ..quadrature import GaussLegendreQuadrature

        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        kwargs = {'dtype': self.ftype, 'device': self.device}
        if etype == 1:
            quad = GaussLegendreQuadrature(q, **kwargs)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")

        return quad

    def grad_lambda(self, index:Index=_S, TD=1):
        """
        @brief 计算所有单元上重心坐标函数的导数
        """
        assert TD == 1
        node = self.entity('node')
        cell = self.entity('cell', index=index)
        Dlambda = bm.interval_grad_lambda(cell, node)
        return Dlambda
    
    def prolongation_matrix(self, p0:int, p1:int):
        """
        @brief 生成从 p0 元到 p1 元的延拓矩阵，假定 0 < p0 < p1
        """
        assert 0 < p0 < p1

        TD = self.top_dimension()
        gdof0 = self.number_of_global_ipoints(p0)
        gdof1 = self.number_of_global_ipoints(p1)

        # 1. 网格节点上的插值点
        NN = self.number_of_nodes()
        I = range(NN)
        J = range(NN)
        V = bm.ones(NN, dtype=self.ftype)
        P = coo_matrix((V, (I, J)), shape=(gdof1, gdof0))

        # 2. 网格边内部的插值点
        NE = self.number_of_edges()
        # p1 元在边上插值点对应的重心坐标
        bcs = self.multi_index_matrix(p1, TD)/p1
        # p0 元基函数在 p1 元对应的边内部插值点处的函数值
        phi = self.shape_function(bcs[1:-1], p=p0) # (ldof1 - 2, ldof0)

        e2p1 = self.cell_to_ipoint(p1)[:, 1:-1]
        e2p0 = self.cell_to_ipoint(p0)
        shape = (NE, ) + phi.shape

        I = bm.broadcast_to(e2p1[:, :, None], shape).flatten()
        J = bm.broadcast_to(e2p0[:, None, :], shape).flatten()
        V = bm.broadcast_to( phi[None, :, :], shape).flatten()

        P += coo_matrix((V, (I, J)), shape=(gdof1, gdof0))

        return P.tocsr()
    
    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell') -> int:
        return p + 1

    def number_of_global_ipoints(self, p: int) -> int:
        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        return NN + (p-1)*NC

    def interpolation_points(self, p: int, index:Index = _S) -> TensorLike:
        GD = self.geo_dimension()
        node = self.entity('node')

        if p == 1:
            return node
        else:
            NN = self.number_of_nodes()
            NC = self.number_of_cells()
            gdof = NN + NC*(p-1)
            ipoint = bm.zeros((gdof, GD), dtype=self.ftype)
            ipoint[:NN] = node
            cell = self.entity('cell')
            w = bm.zeros((p-1,2), dtype=bm.float64)
            w[:,0] = bm.arange(p-1, 0, -1)/p
            w[:,1] = w[-1::-1, 0]
            GD = self.geo_dimension()
            ipoint[NN:NN+(p-1)*NC] = bm.einsum('ij, kj...->ki...', w,
                    node[cell]).reshape(-1, GD)

            return ipoint
    
    def cell_to_ipoint(self, p: int, index: Index=_S) -> TensorLike:
        return self.edge_to_ipoint(p, index)

    def face_to_ipoint(self, p: int, index: Index=_S) -> TensorLike:
        NN = self.number_of_nodes()
        return bm.arange(NN, dtype=self.itype)
    
    def face_unit_normal(self, index: Index = _S, node=None):
        """
        @brief
        """
        raise NotImplementedError
    
    def cell_normal(self, index: Index = _S, node=None):
        """
        @brief 单元的法线方向
        """
        assert self.geo_dimension() == 2
        v = self.cell_tangent(index=index)
        w = bm.tensor([(0, -1),(1, 0)],dtype=bm.float64)
        return v@w
    
    def uniform_refine(self, n=1, options={},returnim = False):
        """
        Uniform refine the interval mesh n times.

        Parameters:
            n (int): Times refine the triangle mesh.
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
            node = self.entity('node')
            cell = self.entity('cell')
            newNode = (node[cell[:, 0]] + self.node[cell[:, 1]])/2
            self.node = bm.concatenate((node, newNode),axis=0)
            
            if returnim is True:
                shape = (NN + NC, NN)
                kargs = bm.context(node)
                values = bm.ones(NN+2*NC, **kargs) 
                values = bm.set_at(values, bm.arange(NN, NN+2*NC), 0.5)

                kargs = bm.context(cell)
                i0 = bm.arange(NN, **kargs) 
                i1 = bm.arange(NN, NN + NC, **kargs)
                I = bm.concatenate((i0, i1, i1))
                J = bm.concatenate((i0, cell[:, 0], cell[:, 1]))   

                P = csr_matrix((values, (I, J)), shape)

                IM.append(P)            
            
            part1 = bm.concatenate((cell[:,0],bm.arange(NN,NN+NC)),axis = 0)
            part2 = bm.concatenate((bm.arange(NN, NN+NC),cell[:, 1]),axis = 0)
            self.cell = bm.stack((part1,part2),axis=1)
            self.construct()
        if returnim is True:
            IM.reverse()
            return IM



    def refine(self, isMarkedCell, options={}):
        """
        @brief 自适应加密网格
        """
        node = self.entity('node')
        cell = self.entity('cell')
        NC = self.number_of_cells()
        NN = self.number_of_nodes()

        N = isMarkedCell.sum()
        if N > 0:
            bc = self.entity_barycenter('cell', index=isMarkedCell)
            self.node = bm.concatenate((node,bc),axis=0) #将新的节点添加到总的节点中去，得到的node

            newCell = bm.zeros((NC+N, 2), dtype=self.itype)
            newCell = bm.set_at(newCell, slice(NC), cell)
            newCell = bm.set_at(newCell[:NC], (isMarkedCell,1), bm.arange(NN, NN+N))
            newCell = bm.set_at(newCell, (slice(NC, None),0), bm.arange(NN, NN+N))
            newCell = bm.set_at(newCell, (slice(NC, None),1), cell[isMarkedCell, 1])
            self.cell = newCell
            self.construct()
                        
    @classmethod
    def from_interval_domain(cls, interval=[0, 1], nx=10):
        node = bm.linspace(interval[0], interval[1], nx+1, dtype=bm.float64)
        c0 = bm.arange(0,nx)
        c1 = bm.arange(1,nx+1)
        cell = bm.stack([c0 , c1] , axis=1)
        return cls(node, cell)

    @classmethod
    def from_mesh_boundary(cls, mesh, /):
        assert mesh.top_dimension() == 2
        itype = mesh.itype
        device = mesh.device
        is_bd_node = mesh.boundary_node_flag()
        is_bd_face = mesh.boundary_face_flag()
        node = mesh.entity('node', index=is_bd_node)
        face = mesh.entity('face', index=is_bd_face)
        NN = mesh.number_of_nodes()
        NN_bd = node.shape[0]

        I = bm.zeros((NN, ), dtype=itype, device=device)
        bm.set_at(I, is_bd_node, bm.arange(NN_bd, dtype=itype, device=device))
        face2bdnode = I[face]
        return cls(node=node, cell=face2bdnode)

    @classmethod
    def from_circle_boundary(cls, center=(0, 0), radius=1.0, n=10):
        dt = 2*bm.pi/n
        theta  = bm.arange(0, 2*bm.pi, dt , dtype=bm.float64)

        n0 = radius*bm.cos(theta) + center[0]
        n1 = radius*bm.sin(theta) + center[1]
        node = bm.stack([n0,n1] , axis=1)
        c0 = bm.arange(n)
        c1 = bm.concatenate((bm.arange(1,n),bm.array([0])))
        cell = bm.stack([c0,c1] , axis = 1)

        return cls(node, cell)
    
    def vtk_cell_type(self):
        VTK_LINE = 3
        return VTK_LINE
    

    def to_vtk(self, fname=None, etype='edge', index:Index=_S):
        """

        Parameters
        ----------

        Notes
        -----
        把网格转化为 VTK 的格式
        """
        from fealpy.mesh.vtk_extent import  write_to_vtu

        node = self.entity('node')
        GD = self.geo_dimension()
        if GD < 3:
            node = bm.concatenate((node, bm.zeros((node.shape[0], 3-GD), dtype=bm.float64)), axis=1)

        cell = self.entity(etype)[index]
        NV = cell.shape[-1]
        NC = len(cell)

        cell = bm.concatenate((bm.zeros((len(cell), 1), dtype=cell.dtype), cell), axis=1)
        cell[:, 0] = NV

        cellType = self.vtk_cell_type()  # segment
        if fname is None:
            return node, cell.flatten(), cellType, NC
        else:
            print("Writting to vtk...")
            write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                    nodedata=self.nodedata,
                    celldata=self.celldata)
    


'''''
    def entity(self, etype: Union[int, str], index:Index = _S) -> TensorLike:
        """
        @brief Get entities.

        @param etype: Type of entities. Accept dimension or name.
        @param index: Index for entities.

        @return: A tensor representing the entities in this mesh.
        """
        TD = 1
        GD = self.geo_dimension()
        if etype in {'cell', TD}:
            return self.cell[index]
        elif etype in {'edge', 1}:
            return self.edge[index]
        elif etype in {'node', 0}:
            return self.node[index]
        elif etype in {'face', TD-1}: # Try 'face' in the last
            return self.face[index]
        raise ValueError(f"Invalid etype '{etype}'.")


    def geo_dimension(self) -> int:
        node = self.node
        if node is None:
            raise RuntimeError('Can not get the geometrical dimension as the node '
                               'has not been assigned.')
        return node.shape[-1]
    
    def multi_index_matrix(self, p: int, etype: int) -> TensorLike:
        return bm.multi_index_matrix(p, etype, dtype=self.itype)
    
    def edge_length(self, index: Index=_S, out=None) -> TensorLike:
        edge = self.entity(1, index=index)
        return bm.edge_length(edge, self.node, out=out)
    
    def edge_normal(self, index: Index=_S, normalize: bool=False, out=None) -> TensorLike:
        edge = self.entity(1, index=index)
        return bm.edge_normal(edge, self.node, normalize=normalize, out=out)
    
    def edge_unit_normal(self, index: Index=_S, out=None) -> TensorLike:
        return self.edge_normal(index=index, normalize=True, out=out)
    
    def edge_tangent(self, index: Index=_S,unit=False, *, out=None) -> TensorLike:
        edge = self.entity(1, index=index)
        return bm.edge_tangent(edge, self.node, unit=unit, out=out)
    
    def edge_to_ipoint(self, p: int, index: Index=_S) -> TensorLike:
        """Get the relationship between edges and integration points."""
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        edges = self.edge[index]
        kwargs = {'dtype': edges.dtype}
        indices = bm.arange(NE, **kwargs)[index]
        return bm.concatenate([
            edges[:, 0].reshape(-1, 1),
            (p-1) * indices.reshape(-1, 1) + bm.arange(0, p-1, **kwargs) + NN,
            edges[:, 1].reshape(-1, 1),
        ], axis=-1)
    
    #counters
    def count(self, etype: Union[int, str]) -> int:
        """Return the number of entities of the given type."""
        entity = self.entity(etype)

        if entity is None:
            logger.info(f'count: entity {etype} is not found and 0 is returned.')
            return 0

        if hasattr(entity, 'location'):
            return entity.location.shape[0] - 1
        else:
            return entity.shape[0]

    def number_of_nodes(self): return self.count('node')
    def number_of_edges(self): return self.count('edge')
    def number_of_faces(self): return self.count('face')
    def number_of_cells(self): return self.count('cell')
    
    def _nv_entity(self, etype: Union[int, str]) -> TensorLike:
        entity = self.entity(etype)
        if hasattr(entity, 'location'):
            loc = entity.location
            return loc[1:] - loc[:-1]
        else:
            return bm.tensor((entity.shape[-1],), dtype=self.itype)

    def number_of_vertices_of_cells(self): return self._nv_entity('cell')
    def number_of_vertices_of_faces(self): return self._nv_entity('face')
    def number_of_vertices_of_edges(self): return self._nv_entity('edge')
    number_of_nodes_of_cells = number_of_vertices_of_cells
    number_of_edges_of_cells: _int_func = lambda self: self.localEdge.shape[0]
    number_of_faces_of_cells: _int_func = lambda self: self.localFace.shape[0]

    def top_dimension(self):
        return self.TD

    # SimpleXMesh
    # ipoints
    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell'):
        if isinstance(iptype, str):
            iptype = estr2dim(self, iptype)
        return bm.simplex_ldof(p, iptype)

    def number_of_global_ipoints(self, p: int):
        nums = [self.entity(i).shape[0] for i in range(self.TD+1)]
        return bm.simplex_gdof(p, nums)

    # shape function
    def shape_function(self, bcs: TensorLike, p: int=1, *, index: Index=_S,
                       variables: str='u', mi: Optional[TensorLike]=None) -> TensorLike:
        TD = bcs.shape[-1] - 1
        if mi is None:
            mi = bm.multi_index_matrix(p, TD, dtype=self.itype)
        phi = bm.simplex_shape_function(bcs, p, mi)
        if variables == 'u':
            return phi
        elif variables == 'x':
            return phi[None, ...]
        else:
            raise ValueError("Variables type is expected to be 'u' or 'x', "
                             f"but got '{variables}'.")

    def grad_shape_function(self, bcs: TensorLike, p: int=1, *, index: Index=_S,
                            variables: str='u', mi: Optional[TensorLike]=None) -> TensorLike:
        TD = bcs.shape[-1] - 1
        if mi is None:
            mi = bm.multi_index_matrix(p, TD, dtype=self.itype)
        R = bm.simplex_grad_shape_function(bcs, p, mi) # (NQ, ldof, bc)
        if variables == 'u':
            return R
        elif variables == 'x':
            Dlambda = self.grad_lambda(index=index)
            gphi = bm.einsum('...bm, qjb -> ...qjm', Dlambda, R) # (NC, NQ, ldof, dim)
            # NOTE: the subscript 'q': NQ, 'm': dim, 'j': ldof, 'b': bc, '...': cell
            return gphi
        else:
            raise ValueError("Variables type is expected to be 'u' or 'x', "
                             f"but got '{variables}'.")

    # entity
    def entity_barycenter(self, etype: Union[int, str], index: Index = _S) -> TensorLike:
        node = self.entity('node')
        if etype in ('node', 0):
            return node if index is None else node[index]
        entity = self.entity(etype, index)
        return bm.barycenter(entity, node)

    def bc_to_point(self, bcs: Union[TensorLike, Sequence[TensorLike]],
                    etype: Union[int, str]='cell', index: Index=_S) -> TensorLike:
        """Convert barycenter coordinate points to cartesian coordinate points
        on mesh entities.
        """
        node = self.entity('node')
        entity = self.entity(etype, index)
        order = getattr(entity, 'bc_order', None)
        return bm.bc_to_points(bcs, node, entity)
    

    # 1D data_structure

    def cell_to_node(self, index: Index = _S) -> TensorLike:
        return self.entity('cell', index)

    def face_to_node(self, index: Index = _S) -> TensorLike:
        return self.entity('face', index)

    def edge_to_node(self, index: Index = _S) -> TensorLike:
        return self.entity('edge', index)

    def cell_to_edge(self, index: Index = _S) -> TensorLike:
        NC = self.number_of_cells()
        return bm.arange(NC,dtype=self.itype)[index].reshape(-1,1)
    
    def edge_to_cell(self, index: Index = _S) -> TensorLike:
        return self.cell_to_edge(index)
    
    def cell_to_face(self, index: Index = _S) -> TensorLike:
        return self.entity('cell', index)
    
    def face_to_cell(self, index: Index = _S) -> TensorLike:
        cell = self.cell
        NF = self.number_of_faces()
        NC = self.number_of_cells()
        NFC = self.number_of_faces_of_cells()

        totalface = cell.reshape(-1,1)

        _,i0,j = bm.unique(totalface , return_index=True 
                           ,return_inverse= True , axis = 0)
        i1 = bm.zeros(NF, dtype=bm.int_)
        b = bm.arange(NFC*NC , dtype= bm.int_)

        if bm.backend_name == 'jax':
            i1 = i1.at(j).set(b)
        else:
            i1[j] = b

        face2cell = bm.stack([i0//NFC , i1//NFC , i0%NFC , i1%NFC] , axis= -1)

        return face2cell[index]
    
    def face_to_edge(self, index: Index = _S) -> TensorLike:
        return self.face_to_cell(index)
    
    # boundary flag
    def boundary_face_flag(self) -> TensorLike:
        """Return a boolean tensor indicating the boundary faces.

        Returns:
            Tensor: boundary face flag.
        """
        return self.face_to_cell[:, 0] == self.face_to_cell[:, 1]
    
    def boundary_node_flag(self) -> TensorLike:
        """Return a boolean tensor indicating the boundary nodes.

        Returns:
            Tensor: boundary face flag.
        """
        return self.boundary_face_flag()

    def boundary_cell_flag(self) -> TensorLike:
        """Return a boolean tensor indicating the boundary cells.

        Returns:
            Tensor: boundary cell flag.
        """
        NC = self.number_of_cells()
        face2cell = self.face_to_cell()
        is_bd_face = self.boundary_face_flag()
        is_bd_cell = bm.zeros((NC, ), dtype=bm.bool_)
        is_bd_cell[face2cell[is_bd_face, 0]] = True
        return is_bd_cell
    
    def boundary_edge_flag(self)-> TensorLike:
        """Return a boolean tensor indicating the boundary edges.

        Returns:
            Tensor: boundary cell flag.
        """
        return self.boundary_cell_flag()
    
    def total_face(self) -> TensorLike:
        cell = self.entity(self.TD)
        local_face = self.localFace
        NVF = local_face.shape[-1]
        total_face = cell[..., local_face].reshape(-1, NVF)
        return total_face
    
    def is_homogeneous(self, etype: Union[int, str]='cell') -> bool:
        """Return True if the mesh entity is homogeneous.

        Returns:
            bool: Homogeneous indicator.
        """
        entity = self.entity(etype)
        if entity is None:
            raise RuntimeError(f'{etype} is not found.')
        return entity.ndim == 2
    
    def construct(self):
        if not self.is_homogeneous():
            raise RuntimeError('Can not construct for a non-homogeneous mesh.')

        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        NFC = self.number_of_faces_of_cells()

        totalFace = self.total_face()
        _, i0, j = bm.unique(
            bm.sort(totalFace, axis=1),
            return_index=True,
            return_inverse=True,
            axis=0
        )
        self.face = totalFace[i0, :] # this also adds the edge in 2-d meshes
        self.edge = self.cell
        NF = i0.shape[0]

        i1 = bm.zeros(NF, dtype=i0.dtype)
        b = bm.arange(0, NFC*NC, dtype=i0.dtype)
        if bm.backend_name == 'jax':
            i1 = i1.at[j].set(b)
        else:
            i1[j] = b 

        self.cell2face = j.reshape(NC, NFC)

        self.face2cell = bm.stack([i0//NFC, i1//NFC, i0%NFC, i1%NFC], axis=-1)


        logger.info(f"Mesh toplogy relation constructed, with {NC} cells, {NF} "
                    f"faces, {NN} nodes "
                    f"on device ?")
'''

IntervalMesh.set_ploter('1d')
