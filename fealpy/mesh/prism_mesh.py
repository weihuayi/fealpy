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
            (0, 2, 1, -1), (3, 4, 5, -1), # bottom and top faces
            (0, 1, 4,  3), (1, 2, 5,  4), (0, 3, 5, 2)], **kwargs)
        self.localFace2edge = bm.array([
            (1, 0, 2, -1), (7, 8, 6, -1), 
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
            tflag = bm.where(self.face[:, -1] < 0)
            qflag = bm.where(~self.face[:, -1] < 0)
            return bm.concat([bm.barycenter(entity[tflag, :-1][0], node), bm.barycenter(entity[qflag], node)], axis=0)

        return bm.barycenter(entity, node)

    # counters
    def number_of_tri_faces(self)->int:
        flag = (self.face < 0)
        return flag.sum()
    
    def number_of_quad_faces(self)->int:
        flag = (self.face < 0)
        return len(self.face) - flag.sum()
    
    # quadrature
   
    # shape function
   
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
            bc0 = bcs[0] # (NQ0, 3)
            bc1 = bcs[1] # (NQ1, 2)
            tp = bm.stack([node[cell[:, [0, 1, 2]]], node[cell[:, [3, 4, 5]]]], axis=1)  #(NC, 2, 3, 3)
            pp = bm.einsum('im,nkmj->nikj', bc0, tp) # (NC, NQ0, 2, 3)
            p = bm.einsum('qi,nmij->nmqj', bc1, pp)             # (NC, NQ0, NQ1, 3)
            points = p.reshape(len(cell), len(bc0)*len(bc1), 3)      # (NC, NQ0*NQ1, 3)
        
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
        ipoint[c2ip] = cip

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
        edgeIdx[0, :] = bm.arange(p+1)
        edgeIdx[1, :] = bm.flip(edgeIdx[0])

        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NF_tri = self.number_of_tri_faces()
        face = self.entity('face')
        edge = self.entity('edge')
        flag = bm.where(face[:, -1] < 0)
        face = face[flag, :-1][0]
        face2edge = self.face_to_edge()[flag]
        edge2ipoint = self.edge_to_ipoint(p)
        face2ipoint = bm.zeros((NF_tri, fdof), dtype=self.itype)

        faceIdx = self.multi_index_matrix(p, TD-1, dtype=self.ftype)
        isEdgeIPoint = (faceIdx == 0)
        fe = bm.array([1, 0, 0])
        for i in range(3):
            I = bm.ones(NF_tri, dtype=bm.int64)
            sign = (face[:, fe[i]] == edge[face2edge[:, i], 0])
            I[sign] = 0
            face2ipoint[:, isEdgeIPoint[:, i]] = edge2ipoint[face2edge[:, [i]], edgeIdx[I]]
        if p > 2:
            base = NN + (p-1)*NE
            isInFaceIPoint = ~(isEdgeIPoint[:, 0] | isEdgeIPoint[:, 1] | isEdgeIPoint[:, 2])
            fidof = fdof - 3*p
            face2ipoint[:, isInFaceIPoint] = base + bm.arange(NF_tri*fidof, dtype=bm.int32).reshape(NF_tri, fidof)
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
        flag = bm.where(face[:, -1] > -0.5)
        face = self.entity('face')[flag]
        face2edge = self.face_to_edge()[flag]
        edge2ipoint = self.edge_to_ipoint(p)

        mi = bm.repeat(bm.arange(p+1, device=bm.get_device(edge)), p+1).reshape(-1, p+1)
        multiIndex0 = mi.flatten().reshape(-1, 1);
        multiIndex1 = mi.T.flatten().reshape(-1, 1);
        multiIndex = bm.concatenate([multiIndex0, multiIndex1], axis=1)

        dofidx = [0 for i in range(4)] 
        dofidx[0], = bm.nonzero(multiIndex[:, 1]==0)
        dofidx[1], = bm.nonzero(multiIndex[:, 0]==p)
        dofidx[2], = bm.nonzero(multiIndex[:, 1]==p)
        dofidx[3], = bm.nonzero(multiIndex[:, 0]==0)
        
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
        multiIndex0 = m2[:, None, :].repeat(len(m1), axis=1)        # (len(m2), len(m1), 3)
        multiIndex1 = m1[None, :, None].repeat(len(m2), axis=0)   # (len(m2), len(m1), 1)
        multiIndex = bm.concatenate([multiIndex0, multiIndex1], axis=-1).reshape(-1,4)
        cell2ipoint = bm.zeros([NC, (p+2)*(p+1)**2//2],
                            dtype=self.itype, device=bm.get_device(cell))  
        
        e0 = bm.array([[0, 1, 2], [2, 0, 1], [1, 2, 0], [1, 0, 2], [2, 1, 0], [0, 2, 1]])
        a = bm.array([0, 0, 1])
        w = bm.array([(p+1)**2, (p+1), 1])
        for i in range(2):  
            tface2edge = face2edge[cell2face[:, i]][:,[2,0,1]] # (NC, 3), interpolation point order for triangle face
            idx = bm.argmax(cell2edge[:, bm.arange(6*i,6*i+3)][:, :, None] == tface2edge[:, None, :], axis=2) # (NC, 3)
            b = a[idx[:, 1] - idx[:, 0]] # (NC,), 1 means reversed, 0 means same order
            idx = e0[3*b + idx[:, 0]] # (NC, 3)， Multi-index of swapped axes per cell
            rm = m2[:, idx.T].transpose(2, 0, 1) 
            idx1 = len(m2) - 1 - bm.argsort(bm.argsort( bm.einsum('ijk,k->ij', rm, w), axis=1),axis=1) # (NC, ldof)
            key0 = bm.where(self.face[:, -1] < 0)
            idx0 = bm.searchsorted(key0[0], cell2face[:,i])
            tf2p0 = tface2ipoint[bm.arange(NF_t)[idx0]] # Indices of interpolation points on the i-th triangle face of each cell
            tfacemultiIdx = (multiIndex[:,-1]==i*p)
            cell2ipoint[:, tfacemultiIdx] = tf2p0[bm.arange(NC)[:, None], idx1]           
        
        shape = (p+1, p+1)
        mi = bm.arange(p+1, device=bm.get_device(cell))
        rmi = bm.arange(p, -1, -1, device=bm.get_device(cell))
        e1 = bm.array([[0,2], [3,0], [1,3], [2,1], [1,2], [3,1], [0,3], [2,0]])
        w = bm.array([(p+1), 1])
        Index = bm.stack([
            bm.broadcast_to(mi[:, None], shape), bm.broadcast_to(rmi[:, None], shape),
            bm.broadcast_to(mi[None, :], shape), bm.broadcast_to(rmi[None, :], shape)
        ], axis=-1).reshape(-1, 4)
        Index = bm.take(Index, indices=e1, axis=1).transpose(1,0,2)  # shape: (8,(p+1)**2,2)
        ridx = bm.einsum('ijk,k->ij', Index, w) # (8, (p+1)**2), Reindexing patterns for the 8 possible cases
        a = bm.array([0, 0, 0, 1])
        e = bm.array([[0, 4, 6, 3], [1, 5, 7, 4], [2, 5, 8, 3]], dtype=bm.int32)
        
        for j in range(3):
            qface2edge = face2edge[cell2face[:, j+2]]
            idx = bm.argmax(cell2edge[:, e[j]][:, :, None] == qface2edge[:, None, :], axis=2) # (NC, 4)
            b = a[idx[:, 1] - idx[:, 0]] # (NC, )
            idx1 = ridx[4*b + idx[:, 0]] # (NC, (p+1)**2)
            key0 = bm.where(~(self.face[:, -1] < 0))
            idx0 = bm.searchsorted(key0[0], cell2face[:,j+2])
            tf2p0 = qface2ipoint[bm.arange(NF_q)[idx0]]
            qfacemultiIdx = (multiIndex[:, (j+2)%3] == 0) 
            cell2ipoint[:, qfacemultiIdx] = tf2p0[bm.arange(NC)[:, None], idx1]

        if p > 2:
            isInCellIPoint = bm.all(multiIndex[:, :-1] > 0, axis=1) & (multiIndex[:, -1]>0) & (multiIndex[:, -1]<p)
            base = NF_t*(p-1)*(p-2)//2 + NF_q*(p-1)**2 + NE*(p-1) + NN
            cell2ipoint[:, isInCellIPoint] = base + bm.arange(NC*(p-1)**2*(p-2)//2).reshape(NC, -1)      

        return cell2ipoint[index]
    # boundary
  
    # refine

    # jacobi
  
    # topology
    def face_to_edge(self, index: Index=_S):
        face2edge = super().face_to_edge()
        face = self.entity('face')
        flag = bm.where(face[:, -1] < 0)
        face2edge = bm.set_at(face2edge, (flag[0], -1), -1)
        return face2edge[index]
    
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