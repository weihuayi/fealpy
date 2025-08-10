from typing import Union, Optional, Sequence, Tuple, Any

from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger

from .plot import Plotable
from .mesh_base import Mesh, HomogeneousMesh
from . import TriangleMesh
from .utils import simplex_gdof, simplex_ldof, tensor_ldof


class PrismMesh(HomogeneousMesh, Plotable):
    def __init__(self, node: TensorLike, cell: TensorLike):
        super().__init__(TD=3, itype=cell.dtype, ftype=node.dtype)
        self.node = node
        self.cell = cell

        self.meshtype = 'prism'
        self.p = 1

        kwargs = bm.context(cell)

        self.localEdge = bm.array([
            (0, 1), (1, 2), (0, 2),
            (0, 3), (1, 4), (2, 5),
            (3, 4), (4, 5), (3, 5)], **kwargs)
        self.localFace = bm.array([
            (0, 2, 1, 1), (3, 4, 5, 5), # bottom and top faces
            (0, 1, 4, 3), (1, 2, 5, 4), (0, 3, 5, 2)], **kwargs)
        self.localFace2edge = bm.array([
            (1, 0, 2, 2), (7, 8, 6, 6), 
            (0, 4, 6,  3), (1, 5, 7,  4), (3, 8, 5, 2)], **kwargs)
        self.localEdge2face = bm.array([
            [2, 0], [3, 0], [0, 4],
            [4, 0], [0, 3], [3, 4], 
            [1, 0], [1, 3], [4, 1]], **kwargs)
        self.ccw = bm.array([0, 1, 2, 3], **kwargs)
        self.construct()       

        self.nodedata = {}
        self.edgedata = {}
        self.facedata = {}
        self.celldata = {}
        self.meshdata = {}

    def construct(self):
        super().construct()
        row = bm.concat([self.cell2face[:, 0], self.cell2face[:, 1]], axis=0)
        self.face = bm.set_at(self.face, (row, -1), self.face[row, -2])

    def total_face(self) -> TensorLike:
        """Return all cell faces sorted by localFace.

        Parameters
            None

        Returns
            total_face : TensorLike
                Array of shape (NC * NFC, NVF), where each row is a face represented
                by vertex indices in localFace order.
        """
        cell = self.entity(self.TD)
        local_face = self.localFace
        NVF = local_face.shape[-1]
        # ipdb.set_trace()
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
        qf = self.quadrature_formula(2, etype=3)
        bcs, ws = qf.get_quadrature_points_and_weights()
        G = self.first_fundamental_form(bcs, index=index)
        l = bm.sqrt(bm.linalg.det(G))
        val = 0.5 * bm.einsum('q, cq -> c', ws, l)

        return val
    
    def face_area(self, index=_S):
        """Compute the area of all mesh faces.
        """

        pass
    
    # counters
    def number_of_tri_faces(self)->int:
        flag = self.tface_flag(type='bool')
        return flag.sum()
    
    def number_of_quad_faces(self)->int:
        flag = self.qface_flag(type='bool')
        return flag.sum()
    
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
        
    # shape function
    def grad_lambda(self, index: Index=_S, TD:int=2) -> TensorLike:
        pass

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
        
    def grad_shape_function(self, bcs: Tuple[TensorLike], p: int=1, *, index: Index=_S,
                            variables: str='u', mi: Optional[TensorLike]=None) -> TensorLike:
        """Compute the gradient of shape functions of an element with respect to reference variables u = (eta, zeta, xi) or physical variables x.
            lambda_0 = 1 - eta - zeta
            lambda_1 = eta
            lambda_2 = zeta
            lambda_3 = 1 - xi
            lambda_4 = xi

        Parameters
            bcs Tuple[Tensor]: ((NQ0, 3), (NQ1, 2)), the integration points.
            p (int, optional): The order of the shape function.
            index (int | slice | Tensor, optional): The index of the cell.
            variables : str, default='u'
                Variable space ('u' or 'x').
            mi (Tensor, optional): The multi-index matrix. Defaults to None.

        Returns
            'u': (NQ, ldof, GD). 
            'x': (NC, NQ, ldof, GD)
        """
        Dlambda0 = bm.array([[-1, -1], [1, 0], [0, 1]], dtype=self.ftype)       
        Dlambda1 = bm.array([[-1], [1]], dtype=self.ftype)
        
        phi0 = bm.simplex_shape_function(bcs[0], p) # (NQ0, 1/2*(p+1)*(p+2))
        phi1 = bm.simplex_shape_function(bcs[1], p) # (NQ1, (p+1))

        R0 = bm.simplex_grad_shape_function(bcs[0], p) # (NQ0, 1/2*(p+1)*(p+2), 3)
        R1 = bm.simplex_grad_shape_function(bcs[1], p) # (NQ1, (p+1), 2)
        gphi0 = bm.einsum('...ij, jn->...in', R0, Dlambda0) # (NQ0, 1/2*(p+1)*(p+2), 2)
        gphi1 = bm.einsum('...ij, jn->...in', R1, Dlambda1) # (NQ1, (p+1), 1)

        n = len(bcs[0])*len(bcs[1])
        gxy = gphi0[:, None, :, None, :] * phi1[None, :, None, :, None]
        gz  = phi0[:, None, :, None, None] * gphi1[None, :, None, :, :]      
        gphi = bm.concatenate([gxy, gz], axis=-1)                      
        gphi = gphi.reshape(n, (p+1)*(p+1)*(p+2)//2, 3)  
                          
        if variables == 'u':
            return gphi #(NQ, ldof, GD)
        elif variables == 'x':
            G, J = self.first_fundamental_form(bcs, index=index,
                    return_jacobi=True)
            G = bm.linalg.inv(G)
            gphi = bm.einsum('cqkm, cqmn, qln -> cqlk', J, G, gphi) # (NC, NQ, ldof, GD)
            # gphi = bm.einsum('cqmk, cqkn, qlm->cqln', J, G, gphi)

            return gphi

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

    def number_of_global_ipoints(self, p: int)->int:
        tri_NF = self.number_of_tri_faces()
        quad_NF = self.number_of_quad_faces()
        NC = self.number_of_cells()
        NE = self.number_of_edges()
        NN = self.number_of_nodes()
        gdof = 1/2*NC*(p-1)**2*(p-2) + 1/2*tri_NF*(p-1)*(p-2) + quad_NF*(p-1)**2 + NE*(p-1) + NN
        return int(gdof)

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
            points = bm.einsum('cim,qi->cqm', node[cell[:, [0,3,1,4,2,5]]], phi)
        
        elif isinstance(bcs, tuple) and len(bcs) == 2 and len(bcs[0] == 3):
            pass
        return points
    
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

        return ipoint

    def tri_to_ipoint(self, p: int, index: Index=_S):
        """
        Get the mapping between each triangular face in the mesh and its interpolation points.

        Parameters
            p : int
                Polynomial degree of interpolation (p ≥ 1).
            index : int, slice, or array-like, optional
                Indices of triangular faces to retrieve.

        Returns
            face2ipoint : bm.array (int)
                A 2D array of shape (number_of_tri_faces, number_of_face_dofs) containing the global indices
                of interpolation points for each triangular face.
        """
        TD = self.top_dimension()
        fdof = (p+1)*(p+2)//2

        edgeIdx = bm.zeros((2, p+1), dtype=self.itype)
        edgeIdx = bm.set_at(edgeIdx, (0, slice(None)), bm.arange(p+1))
        edgeIdx = bm.set_at(edgeIdx, (1, slice(None)), bm.flip(edgeIdx[0]))

        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NF_tri = self.number_of_tri_faces()
        face = self.entity('face')
        edge = self.entity('edge')
        flag = self.tface_flag()
        face = face[flag, :-1]
        face2edge = self.face_to_edge()[flag]
        edge2ipoint = self.edge_to_ipoint(p)
        face2ipoint = bm.zeros((NF_tri, fdof), dtype=self.itype)

        faceIdx = self.multi_index_matrix(p, TD-1, dtype=self.ftype)
        isEdgeIPoint = (faceIdx == 0)
        fe = bm.array([1, 0, 0])
        
        for i in range(3):
            I = bm.ones(NF_tri, dtype=bm.int64)
            sign = (face[:, fe[i]] == edge[face2edge[:, i], 0])
            I = bm.set_at(I, sign, 0)
            face2ipoint = bm.set_at(face2ipoint, (slice(None), isEdgeIPoint[:, i]), edge2ipoint[face2edge[:, [i]], edgeIdx[I]])
        
        if p > 2:
            base = NN + (p-1)*NE
            isInFaceIPoint = ~(isEdgeIPoint[:, 0] | isEdgeIPoint[:, 1] | isEdgeIPoint[:, 2])
            fidof = fdof - 3*p
            face2ipoint = bm.set_at(face2ipoint, (slice(None), isInFaceIPoint), base + bm.arange(NF_tri*fidof, dtype=bm.int32).reshape(NF_tri, fidof))

        return face2ipoint[index]
    
    def quad_to_ipoint(self, p, index=None):
        """Generate global indices for quadrilateral face interpolation points.

        Parameters:
            p : int
                Polynomial order of the interpolation points
            index : Index, optional
                Indices of specific faces to compute (default: all faces)

        Returns:
            TensorLike: 
                Integer array of shape (NF_q, (p+1)^2) containing global indices of 
                interpolation points for each face, where NF is number of faces.
                Points are ordered following tensor-product ordering.
        """
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NF_q = self.number_of_quad_faces()
        NF_t = self.number_of_tri_faces()
        edge = self.entity('edge')
        face = self.entity('face')
        flag = self.qface_flag()
        face = self.entity('face')[flag]
        face2edge = self.face_to_edge()[flag]
        edge2ipoint = self.edge_to_ipoint(p)

        mi = bm.repeat(bm.arange(p+1, device=bm.get_device(edge)), p+1).reshape(-1, p+1)
        multiIndex0 = mi.flatten().reshape(-1, 1);
        multiIndex1 = mi.T.flatten().reshape(-1, 1);
        multiIndex = bm.concatenate([multiIndex0, multiIndex1], axis=1)

        dofidx = bm.zeros((4, p+1), dtype=bm.int32)
        # ipdb.set_trace()
        dofidx = bm.set_at(dofidx, 0, bm.nonzero(multiIndex[:, 1]==0)[0])
        dofidx = bm.set_at(dofidx, 1, bm.nonzero(multiIndex[:, 0]==p)[0])
        dofidx = bm.set_at(dofidx, 2, bm.nonzero(multiIndex[:, 1]==p)[0])
        dofidx = bm.set_at(dofidx, 3, bm.nonzero(multiIndex[:, 0]==0)[0])
        
        face2ipoint = bm.zeros([NF_q, (p+1)**2], dtype=self.itype, device=bm.get_device(edge))
        localEdge = bm.array([[0, 1], [1, 2], [3, 2], [0, 3]], 
                            dtype=self.itype, device=bm.get_device(edge))
        for i in range(4):
            ge = face2edge[:, i]
            idx = bm.nonzero(face[:, localEdge[i, 0]] != edge[ge, 0])[0]

            face2ipoint = bm.set_at(face2ipoint, (slice(None), dofidx[i]), 
                                edge2ipoint[ge])
            face2ipoint = bm.set_at(face2ipoint, (idx[:, None], dofidx[i]), 
                                bm.flip(edge2ipoint[ge[idx]], axis=1))

        indof = bm.all(multiIndex>0, axis=-1) & bm.all(multiIndex<p, axis=-1)
        face2ipoint = bm.set_at(face2ipoint, (slice(None), indof), 
                    bm.arange(NN + NE * (p - 1) + NF_t*(p-1)*(p-2)//2,
                               NN + NE * (p - 1) + NF_t*(p-1)*(p-2)//2 + NF_q * (p - 1) ** 2, 
                    dtype=self.itype).reshape(NF_q, -1))

        return face2ipoint[index]

    def cell_to_ipoint(self, p, index=_S):
        """Generate global interpolation point indices for each cell.

        1. The interpolation point order is determined by the interpolation face, i.e., face2edge.
        2. Find the permutation of cell2edge that matches the order in face2edge, and reorder the columns of multi-indices accordingly.
        3. Use ternary (or quaternary) encoding to compute the numeric value of the reordered multi-indices, then obtain their sorting order.
        4. Use the multi-indices of the prism to extract the boundary indices and assign values; then handle the interior of the prism.

        Parameters
            p : int
                Polynomial degree (p ≥ 1), determines the number and layout of interpolation points.
            index : int, slice, or array-like, optional
                Cells to compute interpolation point mappings for.

        Returns
            cell2ipoint : bm.array (int)
                (NC, ldof) array of global interpolation point indices for each selected cell.

        """
        cell = self.entity('cell', index=index)
        if p == 1:
            return cell[:, [0, 3, 1, 4, 2, 5]]

        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()
        NF_t = self.number_of_tri_faces()
        NF_q = self.number_of_quad_faces()

        cell2face = self.cell_to_face()
        face2edge = self.face_to_edge()
        cell2edge = self.cell_to_edge()
        tface2ipoint = self.tri_to_ipoint(p)
        qface2ipoint = self.quad_to_ipoint(p)[0]
        m1 = bm.arange(p+1, device=bm.get_device(cell))
        m2 = bm.multi_index_matrix(p, 2)
        multiIndex0 = bm.repeat(m2[:, None, :], len(m1), axis=1) # (len(m2), len(m1), 3)
        multiIndex1 = bm.repeat(m1[None, :, None], len(m2), axis=0)  # (len(m2), len(m1), 1)
        multiIndex = bm.concatenate([multiIndex0, multiIndex1], axis=-1).reshape(-1,4)
        cell2ipoint = bm.zeros([NC, (p+2)*(p+1)**2//2],
                            dtype=self.itype, device=bm.get_device(cell))  
        
        e0 = bm.array([[0, 1, 2], [2, 0, 1], [1, 2, 0], [1, 0, 2], [2, 1, 0], [0, 2, 1]], dtype=bm.int32)
        a = bm.array([0, 0, 1], dtype=bm.int32)
        w = bm.array([(p+1)**2, (p+1), 1], dtype=bm.int32)

        for i in range(2):  
            tface2edge = face2edge[cell2face[:, i]][:,[2,0,1]] # (NC, 3), interpolation point order for triangle face
            idx = bm.nonzero(cell2edge[:, bm.arange(6*i,6*i+3)][:, :, None] == tface2edge[:, None, :])[2].reshape(-1, 3) # (NC, 3)
            b = a[idx[:, 1] - idx[:, 0]] # (NC,), 1 means reversed, 0 means same order
            idx = e0[3*b + idx[:, 0]] # (NC, 3)， Multi-index of swapped axes per cell
            rm = bm.permute_dims(m2[:, idx.T], (2, 0, 1)) 
            # ipdb.set_trace()
            idx1 = len(m2) - 1 - bm.argsort(bm.argsort(bm.einsum('ijk,k->ij', rm, w), axis=1),axis=1) # (NC, ldof)
            key0 = self.tface_flag()
            idx0 = bm.searchsorted(key0, cell2face[:,i])
            tf2p0 = tface2ipoint[bm.arange(NF_t)[idx0]] # Indices of interpolation points on the i-th triangle face of each cell
            tfacemultiIdx = (multiIndex[:,-1]==i*p)
            cell2ipoint = bm.set_at(cell2ipoint, (slice(None), tfacemultiIdx), tf2p0[bm.arange(NC)[:, None], idx1])         
      
        shape = (p+1, p+1)
        mi = bm.arange(p+1, device=bm.get_device(cell))
        rmi = bm.arange(p, -1, -1, device=bm.get_device(cell))
        e1 = bm.array([[0,2], [3,0], [1,3], [2,1], [1,2], [3,1], [0,3], [2,0]])
        w = bm.array([(p+1), 1])
        Index = bm.stack([
            bm.broadcast_to(mi[:, None], shape), bm.broadcast_to(rmi[:, None], shape),
            bm.broadcast_to(mi[None, :], shape), bm.broadcast_to(rmi[None, :], shape)
        ], axis=-1).reshape(-1, 4)
        Index = bm.permute_dims(Index[:, e1], (1,0,2))  # shape: (8,(p+1)**2,2)
        ridx = bm.einsum('ijk,k->ij', Index, w) # (8, (p+1)**2), Reindexing patterns for the 8 possible cases
        a = bm.array([0, 0, 0, 1])
        e = bm.array([[0, 4, 6, 3], [1, 5, 7, 4], [2, 5, 8, 3]], dtype=bm.int32)
        
        for j in range(3):
            qface2edge = face2edge[cell2face[:, j+2]]
            idx = bm.nonzero(cell2edge[:, e[j]][:, :, None] == qface2edge[:, None, :])[2].reshape(-1, 4) # (NC, 4)
            b = a[idx[:, 1] - idx[:, 0]] # (NC, )
            idx1 = ridx[4*b + idx[:, 0]] # (NC, (p+1)**2)
            key0 = self.qface_flag()
            idx0 = bm.searchsorted(key0, cell2face[:,j+2])
            tf2p0 = qface2ipoint[bm.arange(NF_q)[idx0]]
            qfacemultiIdx = (multiIndex[:, (j+2)%3] == 0) 
            cell2ipoint = bm.set_at(cell2ipoint, (slice(None), qfacemultiIdx), tf2p0[bm.arange(NC)[:, None], idx1])

        if p > 2:
            isInCellIPoint = bm.all(multiIndex[:, :-1] > 0, axis=1) & (multiIndex[:, -1]>0) & (multiIndex[:, -1]<p)
            base = NF_t*(p-1)*(p-2)//2 + NF_q*(p-1)**2 + NE*(p-1) + NN
            cell2ipoint = bm.set_at(cell2ipoint, (slice(None), isInCellIPoint), base + bm.arange(NC*(p-1)**2*(p-2)//2, dtype=bm.int32).reshape(NC, -1))

        return cell2ipoint[index]
    # boundary
  
    # refine
    def uniform_refine(self, n=1, returnim=False):
        """
        Uniform refine the prismMesh n times.

        Parameters:
            n (int): Times refine the prism mesh.
            returnirm (bool): Return the prolongation matrix list or not,from the finest to the the coarsest
        
        Returns:
            mesh: The mesh obtained after uniformly refining n times.
            List(CSRTensor): The prolongation matrix from the finest to the the coarsest
        """
        if returnim is True:
            IM = []
        
        for i in range(n):
            NN = self.number_of_nodes()
            NE = self.number_of_edges()
            NF_t = self.number_of_tri_faces()
            NF_q = self.number_of_quad_faces()
            NC = self.number_of_cells()
            node_old = self.entity('node')
            edge_old = self.entity('edge')
            cell_old = self.entity('cell')
            face_old = self.entity('face')
            qflag = self.qface_flag()

            if returnim is True:
                pass

            node = self.interpolation_points(p=2)
            cell = bm.zeros((8*NC, 6), dtype=self.itype, device=self.device)
            c2p = self.cell_to_ipoint(p=2)
            cell = bm.set_at(cell, (slice(0, None, 8)), c2p[:, [0, 3, 6, 1, 4, 7]])
            cell = bm.set_at(cell, (slice(1, None, 8)), c2p[:, [3, 12, 6, 4, 13, 7]])
            cell = bm.set_at(cell, (slice(2, None, 8)), c2p[:, [9, 12, 3, 10, 13, 4]])
            cell = bm.set_at(cell, (slice(3, None, 8)), c2p[:, [15, 6, 12, 16, 7, 13]])
            cell = bm.set_at(cell, (slice(4, None, 8)), c2p[:, [1, 4, 7, 2, 5, 8]])
            cell = bm.set_at(cell, (slice(5, None, 8)), c2p[:, [4, 13, 7, 5, 14, 8]])
            cell = bm.set_at(cell, (slice(6, None, 8)), c2p[:, [10, 13, 4, 11, 14, 5]])
            cell = bm.set_at(cell, (slice(7, None, 8)), c2p[:, [16, 7, 13, 17, 8, 14]])

            # node = bm.zeros((NN + NE + NF_q + NC, 3),
            #                 dtype=self.ftype, device=self.device)
            # cell = bm.zeros((8*NC, 6),
            #                 dtype=self.itype, device=self.device)
            
            # start = 0
            # end = NN
            # node = bm.set_at(node, (slice(start, end), slice(None)), self.entity('node'))
            # start = end
            # end = start + NE
            # node = bm.set_at(node, (slice(start, end), slice(None)), self.entity_barycenter('edge'))
            # start = end
            # end = start + NF_q
            # node = bm.set_at(node, (slice(start, end), slice(None)), bm.barycenter(face_old[qflag], node))
            # start = end

            # c2n = self.entity('cell')
            # c2e = self.cell_to_edge() + NN
            # c2f = self.cell_to_face()
            # c2f = self.face_to_qface(c2f[:,2:]) + (NN + NE)

            # cell = bm.set_at(cell, (slice(0, None, 8), 0), c2n[:, 0])
            # cell = bm.set_at(cell, (slice(0, None, 8), 1), c2e[:, 0])
            # cell = bm.set_at(cell, (slice(0, None, 8), 2), c2e[:, 2])
            # cell = bm.set_at(cell, (slice(0, None, 8), 3), c2e[:, 3])
            # cell = bm.set_at(cell, (slice(0, None, 8), 4), c2f[:, 0])  # 2 - 2 == 0
            # cell = bm.set_at(cell, (slice(0, None, 8), 5), c2f[:, 2])  # 4 - 2 == 2

            # cell = bm.set_at(cell, (slice(1, None, 8), 0), c2e[:, 0])
            # cell = bm.set_at(cell, (slice(1, None, 8), 1), c2e[:, 1])
            # cell = bm.set_at(cell, (slice(1, None, 8), 2), c2e[:, 2])
            # cell = bm.set_at(cell, (slice(1, None, 8), 3), c2f[:, 0])
            # cell = bm.set_at(cell, (slice(1, None, 8), 4), c2f[:, 1])
            # cell = bm.set_at(cell, (slice(1, None, 8), 5), c2f[:, 2])

            # cell = bm.set_at(cell, (slice(2, None, 8), 0), c2n[:, 1])
            # cell = bm.set_at(cell, (slice(2, None, 8), 1), c2e[:, 1])
            # cell = bm.set_at(cell, (slice(2, None, 8), 2), c2e[:, 0])
            # cell = bm.set_at(cell, (slice(2, None, 8), 3), c2e[:, 4])
            # cell = bm.set_at(cell, (slice(2, None, 8), 4), c2f[:, 1])  # 3 - 2 == 1
            # cell = bm.set_at(cell, (slice(2, None, 8), 5), c2f[:, 0])  # 2 - 2 == 0

            # cell = bm.set_at(cell, (slice(3, None, 8), 0), c2n[:, 2])
            # cell = bm.set_at(cell, (slice(3, None, 8), 1), c2e[:, 2])
            # cell = bm.set_at(cell, (slice(3, None, 8), 2), c2e[:, 1])
            # cell = bm.set_at(cell, (slice(3, None, 8), 3), c2e[:, 5])
            # cell = bm.set_at(cell, (slice(3, None, 8), 4), c2f[:, 2])  # 4 - 2 == 2
            # cell = bm.set_at(cell, (slice(3, None, 8), 5), c2f[:, 1])  # 3 - 2 == 1

            # cell = bm.set_at(cell, (slice(4, None, 8), 0), c2e[:, 3])
            # cell = bm.set_at(cell, (slice(4, None, 8), 1), c2f[:, 0])
            # cell = bm.set_at(cell, (slice(4, None, 8), 2), c2f[:, 2])
            # cell = bm.set_at(cell, (slice(4, None, 8), 3), c2n[:, 3])
            # cell = bm.set_at(cell, (slice(4, None, 8), 4), c2e[:, 6])
            # cell = bm.set_at(cell, (slice(4, None, 8), 5), c2e[:, 8])

            # cell = bm.set_at(cell, (slice(5, None, 8), 0), c2f[:, 0])
            # cell = bm.set_at(cell, (slice(5, None, 8), 1), c2f[:, 1])
            # cell = bm.set_at(cell, (slice(5, None, 8), 2), c2f[:, 2])
            # cell = bm.set_at(cell, (slice(5, None, 8), 3), c2e[:, 6])
            # cell = bm.set_at(cell, (slice(5, None, 8), 4), c2e[:, 7])
            # cell = bm.set_at(cell, (slice(5, None, 8), 5), c2e[:, 8])

            # cell = bm.set_at(cell, (slice(6, None, 8), 0), c2e[:, 4])
            # cell = bm.set_at(cell, (slice(6, None, 8), 1), c2f[:, 1])
            # cell = bm.set_at(cell, (slice(6, None, 8), 2), c2f[:, 0])
            # cell = bm.set_at(cell, (slice(6, None, 8), 3), c2n[:, 4])
            # cell = bm.set_at(cell, (slice(6, None, 8), 4), c2e[:, 7])
            # cell = bm.set_at(cell, (slice(6, None, 8), 5), c2e[:, 6])

            # cell = bm.set_at(cell, (slice(7, None, 8), 0), c2e[:, 5])
            # cell = bm.set_at(cell, (slice(7, None, 8), 1), c2f[:, 2])
            # cell = bm.set_at(cell, (slice(7, None, 8), 2), c2f[:, 1])
            # cell = bm.set_at(cell, (slice(7, None, 8), 3), c2n[:, 5])
            # cell = bm.set_at(cell, (slice(7, None, 8), 4), c2e[:, 8])
            # cell = bm.set_at(cell, (slice(7, None, 8), 5), c2e[:, 7])

            self.node = node
            self.cell = cell
            self.construct()

        if returnim is True:
            IM.reverse()
            return IM
        
    # jacobi
    def jacobi_matrix(self, bcs: Tuple[TensorLike], index: Index=_S, etype='cell', ftype=None, 
            return_grad=False):
        """Compute the Jacobian matrix of the mapping from the reference element (eta, zeta, xi) to the physical Lagrange triangular prism (x).
        Where:
            1. x(eta, zeta, xi) = phi_0 x_0 + phi_1 x_1 + ... + phi_{ldof-1} x_{ldof-1}
            2. For p = 1, we have ldof = 6. And
                phi_0 = (1 - eta - zeta) * (1 - xi),
                phi_0 = (1 - eta - zeta) * xi,
                phi_0 = eta * (1 - xi),
                phi_0 = eta * xi,
                phi_0 = zeta * (1 - xi),
                phi_0 = zeta * xi.
        Parameters
            bcs: Tuple[TensorLike]
                ((NQ0, 3), (NQ1, 2)), the integration points.

        Returns
            J: TensorLike
                (NC, NQ0*NQ1, 3, 3), the Jacobian [∂X/∂U] for each cell.
            gphi: TensorLike
                (NQ0*NQ1, 6, 3), gradients of the basis functions with respect to u.
        """
        node = self.entity('node')
        cell = self.entity('cell')
        
        if etype in {'cell', 3}:
            gphi = self.grad_shape_function(bcs, p=1, variables='u')  # (NQ, 6, 3)
            node_cell_flip = node[cell[index, [0, 3, 1, 4, 2, 5]]] # (NC, 6, 3)
            J = bm.einsum('cim, qin -> cqmn'
                          , node_cell_flip, gphi) # (NC, NQ, 3, 3)
        
        if return_grad is False:
            return J
        else:
            return J, gphi
    
    def first_fundamental_form(self, bcs: Tuple[TensorLike], index: Index=_S, etype='cell',
            ftype=None, return_jacobi=False, return_grad=False):
        """Compute the first fundamental form of the Lagrange mesh at integration points.

        Parameter
            bcs: Tuple[TensorLike]
                ((NQ0, 3), (NQ1, 2)), the integration points
        
        Returns
            G: TensorLike
                (NC, NQ, 3, 3)
            J: TensorLike
                (NC, NQ, 3, 3)
        """
        J, gphi = self.jacobi_matrix(bcs, index=index, return_grad=True)
        TD = J.shape[-1]
        shape = J.shape[0:-2] + (TD, TD)
        data = [[0 for i in range(TD)] for j in range(TD)]

        for i in range(TD):
            data[i][i] = bm.einsum('...d, ...d->...', J[..., i], J[..., i])
            for j in range(i+1, TD):
                data[i][j] = bm.einsum('...d, ...d->...', J[..., i], J[..., j])
                data[j][i] = data[i][j]
        data = [val.reshape(val.shape+(1,)) for data_ in data for val in data_]
        G = bm.concatenate(data, axis=-1).reshape(shape)

        if (return_jacobi is False) & (return_grad is False):
            return G
        elif (return_jacobi is True) & (return_grad is False): 
            return G, J
        elif (return_jacobi is False) & (return_grad is True): 
            return G, gphi 
        else:
            return G, J, gphi
        
    # topology
    
    ## @ingroup MeshGenerators
    @classmethod
    def from_one_prism(cls, meshtype='iso') -> 'PrismMesh':
        """Generate a single triangular prism (wedge) mesh by extruding a triangle.

        Parameters
            meshtype : str
                Base triangle type. 
                'equ' for equilateral triangle,
                'iso' for right-angled triangle (default: 'iso').

        Returns
            mesh : PrismMesh
                A PrismMesh instance with 6 nodes and 1 prism cell.
        """

        if meshtype == 'equ':
            tnode = bm.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, bm.sqrt(bm.array(3)) / 2, 0.0]
            ], dtype=bm.float64)
        elif meshtype == 'iso':
            tnode = bm.array([
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0]
            ], dtype=bm.float64)

        node = bm.concat([tnode, tnode + bm.array([0, 0, 1], dtype=bm.float64)], axis=0)  
        cell = bm.array([[0, 1, 2, 3, 4, 5]], dtype=bm.int32)

        return cls(node, cell)

    @classmethod
    def from_box(cls, box=[0, 1, 0, 1, 0, 1], nx=10, ny=10, nz=10, 
                threshold=None, device: str = None):
        """
        Generate a wedge-based mesh for a box domain.
        """
        from . import TriangleMesh
        tmesh = TriangleMesh.from_box(box, nx, ny)
        return cls.from_wedge(tmesh, 1/nz, nz)

    @classmethod
    def from_wedge(cls, tmesh: TriangleMesh, h: float, nh: int):
        """Generate a prism mesh by extruding a 2D triangle mesh along the z-axis.
        
        Parameters
            tmesh : TriangleMesh
                The 2D triangle mesh to be extruded. It provides the base geometry in the XY-plane.
            h : float
                Height of each extruded layer in the z-direction. 
            nh : int
                Number of layers (along z-axis) used in the extrusion. The mesh will have (nh) prisms.

        Returns
            mesh : PrismMesh
                A 3D prism mesh object consisting of `tmesh.number_of_cells() * nh` wedge elements.
                Each prism has 6 nodes, and total nodes are `(nh+1) * tmesh.number_of_nodes()`.
        """
        tNC = tmesh.number_of_cells()
        tNN = tmesh.number_of_nodes()
        tcell = tmesh.entity('cell')
        tnode = tmesh.entity('node')
        cell = bm.zeros([tNC*nh, 6], dtype = bm.int32)
        node2n = bm.array([0, 0, 1])
        base = bm.concat([tnode, bm.zeros((len(tnode), 1), dtype=bm.float64)], axis=1)
        k = bm.arange(nh + 1).reshape(-1, 1, 1)
        node = (base[None, :, :] + k * h * node2n).reshape(-1, 3)
        I = tNN*bm.tile(bm.arange(nh), (tNC, 1)).T.flatten()
        src = (bm.tile(tcell, (nh, 1)).T + I).T
        src = bm.astype(src, cell.dtype) 
        cell = bm.set_at(cell, (slice(None), bm.arange(3)), src)
        cell = bm.set_at(cell, (slice(None), bm.arange(3, 6)), cell[:, :3]+tNN)
        return cls(node, cell)
    
    def to_vtk(self, fname=None, etype='cell', index:Index=_S):
        pass

PrismMesh.set_ploter('3d')