import numpy as np
from numpy.linalg import inv
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from .Function import Function
from ..quadrature import GaussLobattoQuadrature
from ..quadrature import GaussLegendreQuadrature
from ..quadrature import PolygonMeshIntegralAlg
from .ScaledMonomialSpace2d import ScaledMonomialSpace2d

class CVEMDof2d():
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        #self.cell2dof, self.cell2dofLocation = self.cell_to_dof()

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

    def cell_to_dof(self, index=np.s_[:]):
        return self.mesh.cell_to_ipoint(self.p, index=index)

    def number_of_global_dofs(self, index=np.s_[:]):
        return self.mesh.number_of_global_ipoints(self.p)

    def number_of_local_dofs(self, index=np.s_[:]):
        return self.mesh.number_of_local_ipoints(self.p)

    def interpolation_points(self, index=np.s_[:]):
        return self.mesh.interpolation_points(self.p,scale=0.3)


