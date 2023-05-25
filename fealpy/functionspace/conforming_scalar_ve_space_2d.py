import numpy as np
from numpy.linalg import inv
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from .Function import Function
from ..quadrature import GaussLobattoQuadrature
from ..quadrature import GaussLegendreQuadrature
from ..quadrature import PolygonMeshIntegralAlg
from .ScaledMonomialSpace2d import ScaledMonomialSpace2d

class CSVEDof2d():
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        self.cell2dof = self.cell_to_dof() # 初始化的时候就构建出 cell2dof 数组

    def is_boundary_dof(self, threshold=None):
        idx = self.mesh.ds.boundary_edge_index()
        if threshold is not None:
            bc = self.mesh.entity_barycenter('edge', index=idx)
            flag = threshold(bc)
            idx  = idx[flag]
        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        edge2dof = self.edge_to_dof()
        isBdDof[edge2dof[idx]] = True
        return isBdDof

    def edge_to_dof(self, index=np.s_[:]):
        return self.mesh.edge_to_ipoint(self.p, index=index)

    face_to_dof = edge_to_dof

    def cell_to_dof(self):
        return self.mesh.cell_to_ipoint(self.p)

    def number_of_global_dofs(self):
        return self.mesh.number_of_global_ipoints(self.p)

    def number_of_local_dofs(self, doftype='all'):
        return self.mesh.number_of_local_ipoints(self.p, iptype=doftype)

    def interpolation_points(self, index=np.s_[:]):
        return self.mesh.interpolation_points(self.p, scale=0.3)


class ConformingScalarVESpace2d():
    def __init__(self, mesh, p=1, q=None, bc=None):
        """
        p: the space order
        q: the index of integral formular
        bc: user can give a barycenter for every mesh cell
        """
        self.mesh = mesh
        self.p = p
        self.smspace = ScaledMonomialSpace2d(mesh, p, q=q, bc=bc)
        self.cellmeasure = self.smspace.cellmeasure
        self.dof = CSVEDof2d(mesh, p)

        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype
        self.stype = 'cvem' # 空间类型

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='all'):
        return self.dof.number_of_local_dofs(doftype=doftype)

    def cell_to_dof(self, index=np.s_[:]):
        return self.dof.cell2dof[index]

    def interpolation_points(self, index=np.s_[:]):
        return self.dof.interpolation_points()

    def array(self, dim=None, dtype=np.float64):
        gdof = self.number_of_global_dofs()
        if dim is None:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=dtype)
