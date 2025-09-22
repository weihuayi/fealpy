from typing import Union, Optional, Sequence, Tuple, Any, List
import numpy as np

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .plot import Plotable
from .mesh_base import Mesh, HomogeneousMesh
from . import TriangleMesh, IntervalMesh
from .utils import simplex_gdof, simplex_ldof, tensor_ldof


class TensorPrismMesh(HomogeneousMesh, Plotable):
    
    def __init__(self, tmesh: TriangleMesh, imesh: IntervalMesh):
        """
        Initialize the 3D tensor-based prism mesh.
        
        Parameters:
            tmesh: TriangleMesh
            imesh: IntervalMesh
        """
        assert isinstance(tmesh, TriangleMesh), "tmesh must be a TriangleMesh"
        assert isinstance(imesh, IntervalMesh), "imesh must be an IntervalMesh"
        
        super().__init__(TD=3, itype=tmesh.itype, ftype=tmesh.ftype)
        
        self.tmesh = tmesh
        self.imesh = imesh

        self.meshtype = 'tensorprism'
        self.p = 1

        kwargs = bm.context(tmesh.cell)

        self.ccw = bm.array([0, 1, 2, 3], **kwargs)
        self.construct()

    def construct(self):
        tmesh = self.tmesh
        imesh = self.imesh

        tnode = tmesh.entity('node') # (NN_t, 2)
        inode = imesh.entity('node') # (NN_i, 1)
        tedge = tmesh.entity('edge')
        tcell = tmesh.entity('cell')

        iNN = imesh.number_of_nodes()
        tNE =tmesh.number_of_edges()
        tNC =tmesh.number_of_cells()
        
        node = bm.concat([bm.repeat(tnode, inode.shape[0], axis=0), 
                          bm.tile(inode.T, tnode.shape[0]).T], axis=1)
        
        all_cell = iNN * tcell[None, :, :] + bm.arange(iNN)[:, None, None]
        all_cell = all_cell.reshape(-1, tcell.shape[1])
        cell = bm.concat([all_cell[:-tNC], all_cell[tNC:]], axis=1)

        all_edge = iNN * tedge[None, :, :] + bm.arange(iNN)[:, None, None]
        all_edge = all_edge.reshape(-1, tedge.shape[1])
        qface = bm.concat([all_edge[:-tNE], all_edge[tNE:][:,::-1]], axis=1)
        self.node = node
        self.cell = cell

        self.tface = all_cell
        self.qface = qface
        self.edge = all_edge
    
    def total_face(self) -> TensorLike:
        cell = self.entity(self.TD)
        local_face = self.localFace
        NVF = local_face.shape[-1]
        cell2face = bm.set_at(cell[..., local_face], (slice(None), bm.arange(2), -1), -1)
        total_face = cell2face.reshape(-1, NVF)
        return total_face

    # entity
    def entity_barycenter(self, etype: Union[int, str], index: Optional[Index]=None) -> TensorLike:
        """Calculate barycenters of mesh entities.
        
        Parameters
            etype : int | str
                Entity type (dimension or name like 'cell', 'face')
            index : Index, optional
                Indices of specific entities to compute
                
        Returns
            TensorLike: Barycenter coordinates (N, GD)
        """
        node = self.entity('node')
        if etype in ('node', 0):
            return node if index is None else node[index]
        entity = self.entity(etype, index)
        
        if etype in ('face', 2):
            tflag = self.tface_flag(index=index)
            qflag = self.qface_flag(index=index)
            return bm.concat([bm.barycenter(entity[tflag, :-1], node), bm.barycenter(entity[qflag], node)], axis=0)

        return bm.barycenter(entity, node)
    
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

    def cell_volume(self, index=_S):
        """Compute the volume of an element.

        The volume is calculated using the formula:
            ∫_c dx = ∫_τ |J| dξ
        where c is the physical element, τ is the reference element, and J is the Jacobian matrix.
        """
        s0 = self.tmesh.entity_measure('cell')
        s1 = self.imesh.entity_measure('cell')
        s = bm.einsum('i,j->ij', s0, s1).ravel()
        return s
    
    def face_area(self, index=_S):
        """Compute the area of all mesh faces.
        """

        pass
    
    # counters
    def number_of_tri_faces(self)->int:
        tri_NF = self.tmesh.number_of_cells() * self.imesh.number_of_nodes()
        return tri_NF
    
    def number_of_quad_faces(self)->int:
        quad_NF = self.tmesh.number_of_edges() * self.imesh.number_of_cells()
        return quad_NF

    # shape function
    def shape_function(self, bcs: Tuple[TensorLike], p: int=1, *, index: Index=_S,
                       variables: str='u', mi: Optional[TensorLike]=None) -> TensorLike:
        """Compute the shape function values of the reference element at integration points.

        Parameters:
            bcs (Tensor): Tuple[(NQ0, 3), (NQ1, 2)], the integration points.
            p (int, optional): The order of the shape function.
            index (int | slice | Tensor, optional): The index of the cell.
            variables : str, default='u'
                Variable space ('u' or 'x').
            mi (Tensor, optional): The multi-index matrix. Defaults to None.

        Returns:
            Tensor: (NQ0*NQ1, ldof). 
        """           
        raw_phi = [bm.simplex_shape_function(bc, p) for bc in bcs] # ((NQ0, ldof0), (NQ1, ldof1))
        phi = bm.tensorprod(*raw_phi)
        if variables == 'u':
            return phi
        elif variables == 'x':
            return phi[None, ...]
        else:
            raise ValueError("Variables type is expected to be 'u' or 'x', "
                             f"but got '{variables}'.")

    # ipoint
    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell')->int:
        if iptype == 'cell':
            return simplex_ldof(p, 2) * simplex_ldof(p, 1)
        elif iptype == 'tface':
            return simplex_ldof(p, 2)
        elif iptype == 'qface':
            return tensor_ldof(p, 2)
        elif iptype == 'edge':
            return simplex_ldof(p, 1)
        elif iptype == 'node':
            return 1
        else:
            raise KeyError(f'{iptype} is not a valid entity name in FEALPy.')

    def number_of_global_ipoints(self, p: int, q=None)->int:
        q = p if q is None else q
        return (self.tmesh.number_of_global_ipoints(p) *
                self.imesh.number_of_global_ipoints(q))

    def tri_to_ipoint(self, p: int, index: Index=_S):
        igdof = self.imesh.number_of_global_ipoints(p)
        iNN = self.imesh.number_of_nodes()
        tc2i = self.tmesh.cell_to_ipoint(p)
        t2i = igdof * tc2i[None, :, :] + bm.arange(iNN)[:, None, None]
        t2i = t2i.reshape(-1, tc2i.shape[1])
        return t2i[index]
    
    def quad_to_ipoint(self, p: int, index: Index=_S):
        te2i = self.tmesh.edge_to_ipoint(p)
        ic2i = self.imesh.cell_to_ipoint(p)
        iNC = self.imesh.number_of_cells()
        tNE = self.tmesh.number_of_edges()
        igdof = self.imesh.number_of_global_ipoints(p)
        q2i = bm.zeros((iNC * tNE, te2i.shape[1] * ic2i.shape[1]), dtype=bm.int32)

        for i in range(iNC):
            q2i[i*tNE:(i+1)*tNE, :] = (igdof* te2i[None, :, :] + ic2i[i][:, None, None]).transpose(1, 0, 2).reshape(-1, q2i.shape[1])

        return  q2i[index]

    def cell_to_ipoint(self, p: int, index: Index=_S):
        cell = self.cell
        if p == 1:
            return cell[:, [0, 3, 1, 4, 2, 5]][index]
        tc2i = self.tmesh.cell_to_ipoint(p)
        ic2i = self.imesh.cell_to_ipoint(p)
        iNC = self.imesh.number_of_cells()
        tNC = self.tmesh.number_of_cells()
        igdof = self.imesh.number_of_global_ipoints(p)
        c2i = bm.zeros((iNC * tNC, tc2i.shape[1] * ic2i.shape[1]), dtype=bm.int32)
        idx = bm.arange(tc2i.shape[1] * ic2i.shape[1]).reshape(ic2i.shape[1], tc2i.shape[1]).T.flatten()
        for i in range(iNC):
            c2i[i*tNC:(i+1)*tNC, :] = (igdof* tc2i[None, :, :] + ic2i[i][:, None, None]).transpose(1, 0, 2).reshape(tNC, -1)[:, idx]
        return  c2i[index]

    def interpolation_points(self, p, index=_S):
        """
        @brief Generate interpolation points for the entire mesh
        """
        cell = self.entity('cell')

        c2ip = self.cell_to_ipoint(p)
        gp = self.number_of_global_ipoints(p)
        ipoint = bm.zeros([gp, 3], dtype=self.ftype, device=bm.get_device(cell))
        mi = self.multi_index_matrix(p, 2)
        line = (bm.linspace(0, 1, p+1, endpoint=True,
                        dtype=self.ftype, device=bm.get_device(cell))).reshape(-1, 1)
        line = bm.concatenate([1-line, line], axis=1)
        bcs = (mi/p, line)
        cip = self.bc_to_point(bcs)
        ipoint = bm.set_at(ipoint, (c2ip, slice(None)), cip)
        # import ipdb;ipdb.set_trace()

        return ipoint
    
    def bc_to_point(self, bcs: Union[TensorLike, Sequence[TensorLike]], index: Index=_S) -> TensorLike:
        """Convert barycentric coordinates to Cartesian coordinates.
        
        Parameters:
            bcs : Union[TensorLike, Sequence[TensorLike]], Tuple[(NQ0, 3), (NQ1, 2)]
                Barycentric coordinates (sequence of tensors)
            index : Index, optional
                Entity indices to compute points for
                
        Returns:
            TensorLike: Cartesian coordinates of points
        """
        node = self.entity('node')

        if isinstance(bcs, tuple) and len(bcs) == 2 and (bcs[0].shape[1] == 3):
            cell = self.entity('cell', index)
            phi = self.shape_function(bcs)
            # points = bm.einsum('cim,qi->cqm', node[cell[:, [0,3,1,4,2,5]]], phi)
            points = bm.einsum('cim,qi->cqm', node[cell[:, [0,3,1,4,2,5]]], phi)
        
        elif isinstance(bcs, tuple) and len(bcs) == 2 and len(bcs[0] == 3):
            pass
        return points
      
    # boundary
    def boundary_face_index(self, index: Index=_S):
        tNC = self.tmesh.number_of_cells()
        iNC = self.imesh.number_of_cells()
        tNE = self.tmesh.number_of_edges()

        bottom_index = bm.arange(tNC)
        top_index = bottom_index + iNC * tNC
        tface_index = bm.concat([bottom_index, top_index], axis=0)
        
        tbdeflag = self.tmesh.boundary_face_index()
        qface_index = (tbdeflag[None, :] + tNE * bm.arange(iNC)[:, None]).ravel() 
        
        return tface_index, qface_index

    # map
    def tface_flag(self, type=None, index: Index=_S):
        flag = (self.entity('face')[index, -1] == self.entity('face')[index, -2])
        if type == 'bool':
            return flag

        return bm.where(flag)[0]

    def qface_flag(self, type=None, index: Index=_S):
        flag = ~(self.entity('face')[index, -1] == self.entity('face')[index, -2])
        if type == 'bool':
            return flag
        
        return bm.where(flag)[0]
    
    def face_to_tface(self, index: Index=_S):
        """Given the global face index of a triangular face, return its local face index.
        """
        a = self.tface_flag()
        sort_idx = bm.argsort(a)
        sorted_a = a[sort_idx]

        pos = bm.searchsorted(sorted_a, index)
        return sort_idx[pos]
    
    def face_to_qface(self, index: Index=_S):
        """Given the global face index of a quadrilateral face, return its local face index.
        """
        a = self.qface_flag()
        sort_idx = bm.argsort(a)
        sorted_a = a[sort_idx]
        pos = bm.searchsorted(sorted_a, index)
        return sort_idx[pos]
  
    # quadrature
    def quadrature_formula(self, q: int, etype: Union[int, str]='cell',
                        qtype: str='legendre'): # TODO: other qtype
        from ..quadrature import GaussLegendreQuadrature, TensorProductQuadrature, TriangleQuadrature
        qf0 = TriangleQuadrature(q)
        qf1 = GaussLegendreQuadrature(q)

        if etype in {'cell', 3}:
            return TensorProductQuadrature((qf0, qf1)) 
        elif etype in {'face', 2}:
            return qf0, TensorProductQuadrature((qf1, qf1))
        elif etype in {'tface'}:
            return qf0
        elif etype in {'qface'}:
            return TensorProductQuadrature((qf1, qf1))
        elif etype in {'edge', 1}:
            return qf1 
    



