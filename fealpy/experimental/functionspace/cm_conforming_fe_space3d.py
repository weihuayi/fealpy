from typing import Union, TypeVar, Generic, Callable, Optional
import itertools
from urllib.request import noheaders

#from networkx.classes import number_of_nodes

from ..backend import backend_manager as bm
from ..backend import TensorLike
from ..mesh.mesh_base import Mesh
from .space import FunctionSpace
from fealpy.experimental.functionspace.bernstein_fe_space import BernsteinFESpace
from .functional import symmetry_span_array, symmetry_index, span_array
from scipy.special import factorial, comb
from fealpy.decorator import barycentric

_MT = TypeVar('_MT', bound=Mesh)
Index = Union[int, slice, TensorLike]
Number = Union[int, float]
_S = slice(None)

class CmConformingFESpace3d(FunctionSpace, Generic[_MT]):
    def __init__(self, mesh:_MT, p: int, m: int, isCornerNode:bool):
        assert(p>8*m)
        self.mesh = mesh
        self.p = p
        self.m = m
        self.isCornerNode = isCornerNode
        self.bspace = BernsteinFESpace(mesh, p)

        self.ftype = mesh.ftype
        self.itype = mesh.itype
        #self.device = mesh.device

        self.TD = mesh.top_dimension()
        self.GD = mesh.geo_dimension()

        #self.coeff = self.coefficient_matrix()
        self.multiIndex = mesh.multi_index_matrix(p, 3)
        self.dof_index = {}
        self.get_dof_index()
    
    def get_dof_index(self):
        p = self.p
        m = self.m
        midx = self.multiIndex
        ldof = midx.shape[0]

        isn_cell_dof = bm.zeros(ldof, dtype=bm.bool)

        node_dof_index = []
        for i in range(4):
            a = []
            for r in range(4*m+1):
                Dvr = midx[:, i]==p-r
                isn_cell_dof = isn_cell_dof | Dvr
                a.append(bm.where(bm.array(Dvr))[0])
            node_dof_index.append(a)

        locEdge = bm.array([[0, 1],[0, 2],[0, 3],[1, 2],[1, 3],[2, 3]], dtype=
                           bm.int32)
        dualEdge = bm.array([[2, 3],[1, 3],[1, 2],[0, 3],[0, 2],[0, 1]],
                            dtype=bm.int32)
        edge_dof_index = []  
        for i in range(6):
            a = []
            for r in range(2*m+1):
                aa = []
                for j in range(r+1):
                    Derj = (bm.sum(bm.array(midx[:, locEdge[i]]), axis=-1)==p-r) & (midx[:, dualEdge[i, 1]]==j) & (~isn_cell_dof)
                    isn_cell_dof = isn_cell_dof | Derj
                    aa.append(bm.where(Derj)[0])
                a.append(aa)
            edge_dof_index.append(a)

        face_dof_index = []
        for i in range(4):
            a = []
            for r in range(m+1):
                Dfr = (bm.array(midx[:, i]==r)) & (~isn_cell_dof)
                isn_cell_dof = isn_cell_dof | Dfr
                a.append(bm.where(Dfr)[0])
            face_dof_index.append(a)

        
        all_node_dof_index = bm.concatenate([bm.concatenate(item) if isinstance(item, list) else item for sublist in node_dof_index for item in sublist])
        all_edge_dof_index = bm.concatenate([bm.concatenate(item) if isinstance(item, list) else item for sublist in edge_dof_index for item in sublist])
        all_face_dof_index = bm.concatenate([bm.concatenate(item) if isinstance(item, list) else item for sublist in face_dof_index for item in sublist])

        self.dof_index["node"] = node_dof_index
        self.dof_index["edge"] = edge_dof_index
        self.dof_index["face"] = face_dof_index
        self.dof_index["cell"] = bm.where(isn_cell_dof==0)[0]
        self.dof_index["all"] = bm.concatenate((all_node_dof_index,
                                               all_edge_dof_index,
                                               all_face_dof_index,
                                               self.dof_index["cell"]))

    def number_of_local_dofs(self, etype, p=None) -> int: #TODO:去掉etype 2d同样
        p = self.p if p is None else p
        if etype=="cell":
            return (p+1)*(p+2)*(p+3)//6
    def number_of_internal_dofs(self, etype, p=None, m=None) -> int:
        p = self.p if p is None else p
        m = self.m if m is None else m
        if etype=='cell':
            return len(self.dof_index["cell"])
        if etype=="edge":
            N = 0
            for edof_r in self.dof_index["edge"][0]:
                for edof_r_j in edof_r:
                    N += len(edof_r_j)
            return N 
        if etype=="face":
            N = 0
            for fdof_r in self.dof_index["face"][0]:
                N += len(fdof_r)
            return N 
        if etype=='node':
            return (4*m+1)*(4*m+2)*(4*m+3)//6
    def number_of_global_dofs(self) -> int:
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()
        ndof = self.number_of_internal_dofs('node')
        eidof = self.number_of_internal_dofs('edge')
        fidof = self.number_of_internal_dofs('face')
        cidof = self.number_of_internal_dofs('cell')
        return NN*ndof +eidof*NE + fidof*NF + cidof*NC

    def node_to_dof(self):
        m = self.m
        mesh = self.mesh
        ndof = self.number_of_internal_dofs('node')
        NN = mesh.number_of_nodes()
        n2d = bm.arange(NN*ndof, dtype=self.itype).reshape(NN, ndof)
        return n2d

    def edge_to_internal_dof(self):
        p = self.p
        m = self.m
        ndof = self.number_of_internal_dofs('node') 
        eidof = self.number_of_internal_dofs('edge') 

        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        e2id = bm.arange(NN*ndof, NN*ndof + NE*eidof, dtype=self.itype).reshape(NE, eidof)
        return e2id

    def face_to_internal_dof(self):
        p = self.p
        m = self.m
        ndof = self.number_of_internal_dofs('node') 
        eidof = self.number_of_internal_dofs('edge') 
        fidof = self.number_of_internal_dofs('face') 

        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        Ndof = NN*ndof + NE*eidof
        f2id = bm.arange(Ndof, Ndof + NF*fidof, dtype=self.itype).reshape(NF, fidof)
        return f2id

    def cell_to_internal_dof(self):
        p = self.p
        m = self.m
        ldof = self.number_of_local_dofs('cell')
        ndof = self.number_of_internal_dofs('node') 
        eidof = self.number_of_internal_dofs('edge') 
        fidof = self.number_of_internal_dofs('face') 
        cidof = self.number_of_internal_dofs('cell') 

        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()
        Ndof = NN*ndof + NE*eidof + NF*fidof
        c2id = bm.arange(Ndof, Ndof + NC*cidof, dtype=self.itype).reshape(NC, cidof)
        return c2id

    def cell_to_dof(self):
        p, m = self.p, self.m
        mesh = self.mesh

        ldof = self.number_of_local_dofs('cell')
        ndof = self.number_of_internal_dofs('node')
        eidof = self.number_of_internal_dofs('edge')
        fidof = self.number_of_internal_dofs('face')
        cidof = self.number_of_internal_dofs('cell')

        n2d = self.node_to_dof()
        e2id = self.edge_to_internal_dof()
        f2id = self.face_to_internal_dof()
        c2id = self.cell_to_internal_dof()

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()
        edge = mesh.entity('edge')
        face = mesh.entity('face')
        cell = mesh.entity('cell')
        c2e = mesh.cell_to_edge()
        c2f = mesh.cell_to_face()

        c2dof = bm.zeros((NC, ldof), dtype=self.itype)
        ## node
        for v in range(4):
            c2dof[:, v*ndof:(v+1)*ndof] = n2d[cell[:, v]]

        ## edge
        localEdge = bm.array([[0, 1],[0, 2],[0, 3],[1, 2],[1, 3],[2, 3]],
                             dtype=self.itype)
        for e in range(6):
            N = ndof*4 + eidof*e
            c2dof[:, N:N+eidof] = e2id[c2e[:, e]]
            flag = edge[c2e[:, e], 0] != cell[:, localEdge[e, 0]]
            n0, n1 = 0, p-8*m-1
            for r in range(2*m+1):
                for j in range(r+1):
                    c2dof[flag, N+n0:N+n0+n1] = bm.flip(c2dof[flag, N+n0:N+n0+n1], axis=-1)
                    n0 +=n1
                n1 += 1
        ## face
        fdof_index = self.dof_index["face"]
        perms = bm.array(list(itertools.permutations([0,1,2])))
        locFace = bm.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]], dtype=self.itype)
        midx2num = lambda a: (a[:, 1]+a[:, 2])*(1+a[:, 1]+a[:, 2])//2+a[:, 2]

        indices = []
        for r in range(m+1):
            dof_fc2f = midx2num(self.multiIndex[fdof_index[0][r]][:, locFace[0]])
            midx = self.mesh.multi_index_matrix(p-r, 2)
            indices_r = bm.zeros((6, len(dof_fc2f)), dtype=self.itype)
            for i in range(6):
                indices_r[i] = bm.argsort(dof_fc2f)[bm.argsort(bm.argsort( midx2num( self.multiIndex[ fdof_index[0][r]][:, locFace[0]][:, perms[i]])))]
                indices_r[i] = bm.argsort(bm.argsort(midx2num(midx[dof_fc2f][ :, perms[i]])))
            indices.append(indices_r)
        c2fp = self.mesh.cell_to_face_permutation(locFace=locFace)
        perm2num = lambda a: a[:, 0]*2 +(a[:, 1]>a[:, 2])
        for f in range(4):
            N = ndof*4 +eidof*6 +fidof*f
            c2dof[:, N:N+fidof]  = f2id[c2f[:, f]]
            pnum = perm2num(c2fp[:, f])
            for i in range(6):
                n0 = 0
                flag = pnum == i
                for r in range(m+1):
                    idx = indices[r][i]
                    n1 = idx.shape[0]
                    c2dof[flag, N+n0:N+n0+n1] = c2dof[flag, N+n0:N+n0+n1][:, idx]
                    n0 += n1
        ## cell
        c2dof[:, ldof-cidof:] = c2id
        return c2dof

