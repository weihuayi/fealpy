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
            (2, 1, 0, -1), (6, 7, 8, -1), 
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
            tflag = bm.where(self.face[:, -1] < 0)[0]
            qflag = bm.where(~self.face[:, -1] < 0)[0]
            return bm.concat([bm.barycenter(entity[tflag, :-1][0], node).reshape(1, -1) , bm.barycenter(entity[qflag], node)], axis=0)

        return bm.barycenter(entity, node)

    # counters
    def number_of_tri_faces(self)->int:
        flag = (self.face < 0)
        return flag.sum()
    
    def number_of_quad_faces(self)->int:
        flag = (self.face > 0)
        return flag.sum()
    
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
        # ipdb.set_trace()
        return int(gdof)

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


# PrismMesh.set_ploter('3d')