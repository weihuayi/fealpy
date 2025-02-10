
from .lagrange_fe_space import LagrangeFESpace

from typing import Optional, TypeVar, Union, Generic, Callable
from ..typing import TensorLike, Index, _S, Threshold

from ..backend import TensorLike
from ..backend import backend_manager as bm
from ..mesh.mesh_base import Mesh
from .space import FunctionSpace
from .dofs import LinearMeshCFEDof, LinearMeshDFEDof
from .function import Function
from fealpy.decorator import barycentric, cartesian

from .scaled_monomial_space_2d import ScaledMonomialSpace2d

from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

class NCSVEDof2d():
    """
    The dof manager of non conforming vem 2d space.
    """
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        self.cell2dof = self.cell_to_dof()

        NC = mesh.number_of_cells()
        ldof = self.number_of_local_dofs(doftype='all')
        self.cell2dofLocation = bm.zeros(NC+1, dtype=mesh.itype)
        self.cell2dofLocation[1:] = bm.add.accumulate(ldof)

    def is_boundary_dof(self, threshold=None):
        """
        @brief 获取边界自由度
        """
        TD = self.mesh.top_dimension()
        if type(threshold) is bm.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_edge_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter(TD-1, index=index)
                flag = threshold(bc)
                index = index[flag]

        gdof = self.number_of_global_dofs()
        edge2dof = self.edge_to_dof(index=index) # 只获取指定的面的自由度信息
        isBdDof = bm.zeros(gdof, dtype=bm.bool_)
        isBdDof[edge2dof] = True
        return isBdDof

    def edge_to_dof(self, index=_S):
        """
        @brief 获取网格边与自由度的对应关系
        """
        if isinstance(index, slice) and index == slice(None):
            NE = self.mesh.number_of_edges()
            index = bm.arange(NE)
        elif isinstance(index, bm.ndarray) and (index.dtype == bm.bool_):
            index, = bm.nonzero(index)
            NE = len(index)
        elif isinstance(index, list) and (type(index[0]) is bm.bool_):
            index, = bm.nonzero(index)
            NE = len(index)
        else:
            NE = len(index)

        NN = self.mesh.number_of_nodes()
        p = self.p

        idx = bm.arange(p)
        edge2dof =  p*index[:, None] + idx
        return edge2dof

    face_to_dof = edge_to_dof

    def cell_to_dof(self, index=_S):
        """
        @brief 获取网格单元与自由度的对应关系
        """
        p = self.p
        mesh = self.mesh

        if p == 1:
            return mesh.ds.cell_to_edge()
        else:
            NC = mesh.number_of_cells()

            ldof = self.number_of_local_dofs()
            cell2dofLocation = bm.zeros(NC+1, dtype=bm.int_)
            cell2dofLocation[1:] = bm.add.accumulate(ldof)
            cell2dof = bm.zeros(cell2dofLocation[-1], dtype=bm.int_)

            edge2dof = self.edge_to_dof()
            edge2cell = mesh.ds.edge_to_cell()
            idx = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + bm.arange(p)
            cell2dof[idx] = edge2dof

            isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
            idx = (cell2dofLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*p).reshape(-1, 1) + bm.arange(p)
            cell2dof[idx] = edge2dof[isInEdge, p-1::-1]

            NV = mesh.ds.number_of_vertices_of_cells()
            NE = mesh.number_of_edges()
            idof = (p-1)*p//2
            idx = (cell2dofLocation[:-1] + NV*p).reshape(-1, 1) + bm.arange(idof)
            cell2dof[idx] = NE*p + bm.arange(NC*idof).reshape(NC, idof)
            return bm.hsplit(cell2dof, cell2dofLocation[1:-1])[index]

    def number_of_global_dofs(self):
        """
        @brief 获取全部自由度的个数
        """
        p = self.p
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        gdof = NE*p + NC*(p-1)*p//2
        return gdof

    def number_of_local_dofs(self, doftype: Union[int, str]='all') -> Union[TensorLike, int]:
        """
        @brief 获取局部插值点的个数 
        """
        p = self.p
        mesh = self.mesh
        if doftype == 'all':
            NCE = mesh.ds.number_of_edges_of_cells()
            return NCE*p + (p-1)*p//2
        elif doftype in {'cell', 2}:
            return (p-1)*p//2
        elif doftype in {'edge', 'face', 1}:
            return p
        elif doftype in {'node', 0}:
            return 0 

    def interpolation_points(self, scale:float=0.3):
        """
        Get the node-value-type interpolation points.

        On every edge, there exist p points
        """
        p = self.p
        mesh = self.mesh
        gdof = self.number_of_global_dofs()
        node = mesh.entity('node')
        edge = mesh.entity('edge')

        GD = mesh.geo_dimension()
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        qf = GaussLegendreQuadrature(p)
        bcs, ws = qf.get_quadrature_points_and_weights()
        ipoint = bm.zeros((gdof, GD),dtype=bm.float_)
        if p==1:
            ipoint = bm.einsum(
                    'ij, ...jm->...im',
                    bcs, node[edge, :]).reshape(-1, GD)
            return ipoint

        ipoint[:NE*p, :] =  bm.einsum(
                    'ij, ...jm->...im',
                    bcs, node[edge, :]).reshape(-1, GD)
        if p == 2:
            ipoint[NE*p:, :] = mesh.entity_barycenter('cell')
            return ipoint

        h = bm.sqrt(mesh.cell_area())[:, None]*scale
        bc = mesh.entity_barycenter('cell')
        t = bm.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, bm.sqrt(3)/2]], dtype=bm.float_)
        t -= bm.array([0.5, bm.sqrt(3)/6.0], dtype=bm.float_)

        tri = bm.zeros((NC, 3, GD), dtype=bm.float_)
        tri[:, 0, :] = bc + t[0]*h
        tri[:, 1, :] = bc + t[1]*h
        tri[:, 2, :] = bc + t[2]*h

        TD = mesh.top_dimension()
        bcs = mesh.multi_index_matrix(p-2, TD)/(p-2)
        ipoint[NE*p:, :] = bm.einsum('ij, ...jm->...im', bcs, tri).reshape(-1, GD)
        return ipoint


class NonConformingScalarVESpace2d():
    def __init__(self, mesh, p=1):
        """
        p: the space order
        q: the index of integral formular
        bc: user can give a barycenter for every mesh cell
        """
        self.mesh = mesh
        self.p = p
        self.smspace = ScaledMonomialSpace2d(mesh, p)
        self.cellmeasure = self.smspace.cellmeasure
        self.dof = NCSVEDof2d(mesh, p)

        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype
        self.stype = 'ncsvem' # 空间类型

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='all'):
        return self.dof.number_of_local_dofs(doftype=doftype)

    def cell_to_dof(self, index=_S):
        return self.dof.cell2dof[index]

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def project_to_smspace(self, uh, PI):
        """
        @brief Project a non conforming vem function uh into polynomial space.

        @param[in] uh
        @param[in] PI
        """
        p = self.p
        cell2dof = self.cell_to_dof()
        g = lambda x: x[0]@uh[x[1]]
        S = self.smspace.function()
        S[:] = bm.concatenate(list(map(g, zip(PI, cell2dof))))
        return S

    def array(self, dim=None, dtype=bm.float64):
        gdof = self.number_of_global_dofs()
        if dim is None:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return bm.zeros(shape, dtype=dtype)

    def function(self, dim=None, array=None, dtype=bm.float64):
        return Function(self, dim=dim, array=array, coordtype='cartesian', dtype=dtype)

    def set_dirichlet_bc(self, gD, uh, threshold=None):
        """
        初始化解 uh  的第一类边界条件。
        """
        p = self.p
        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        ipoints = self.interpolation_points()
        isDDof = self.dof.is_boundary_dof(threshold=threshold)
        uh[isDDof] = gD(ipoints[isDDof])
        return isDDof

    def interpolation(self, u, iptype=True):
        """
        @brief 把函数 u 插值到非协调空间当中
        """
        p = self.p
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        ipoint = self.dof.interpolation_points()
        uI = self.function()
        uI[:NE*p] = u(ipoint)
        if p > 1:
            phi = self.smspace.basis

            def f(x, index):
                return bm.einsum('ij, ij...->ij...', u(x), phi(x, index=index, p=p-2))

            bb = self.mesh.integral(f,
                    celltype=True)/self.smspace.cellmeasure[..., bm.newaxis]
            uI[p*NE:] = bb.reshape(-1)
        return uI
