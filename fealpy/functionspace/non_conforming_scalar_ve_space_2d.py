import numpy as np
from numpy.linalg import inv
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from .Function import Function
from ..quadrature import GaussLegendreQuadrature
from ..quadrature import PolygonMeshIntegralAlg
from .scaled_monomial_space_2d import ScaledMonomialSpace2d

class NCSVEDof2d():
    """
    The dof manager of non conforming vem 2d space.
    """
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        #self.cell2dof = self.cell_to_dof()

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

    def edge_to_dof(self):
        p = self.p
        mesh = self.mesh
        NE = mesh.number_of_edges()
        edge2dof = np.arange(NE*p).reshape(NE, p)
        return edge2dof

    def cell_to_dof(self):
        """
        Construct the cell2dof array which are 1D array with a location array
        cell2dofLocation. 

        The following code give the dofs of i-th cell.

        cell2dof[cell2dofLocation[i]:cell2dofLocation[i+1]]
        """
        p = self.p
        mesh = self.mesh

        if p == 1:
            cell2edge, _ = mesh.ds.cell_to_edge(return_sparse=False)
            return cell2edge, cellLocation
        else:
            NC = mesh.number_of_cells()

            ldof = self.number_of_local_dofs()
            cell2dofLocation = np.zeros(NC+1, dtype=np.int_)
            cell2dofLocation[1:] = np.add.accumulate(ldof)
            cell2dof = np.zeros(cell2dofLocation[-1], dtype=np.int_)

            edge2dof = self.edge_to_dof()
            edge2cell = mesh.ds.edge_to_cell()
            idx = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + np.arange(p)
            cell2dof[idx] = edge2dof

            isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
            idx = (cell2dofLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*p).reshape(-1, 1) + np.arange(p)
            cell2dof[idx] = edge2dof[isInEdge, p-1::-1]

            NV = mesh.ds.number_of_vertices_of_cells()
            NE = mesh.number_of_edges()
            idof = (p-1)*p//2
            idx = (cell2dofLocation[:-1] + NV*p).reshape(-1, 1) + np.arange(idof)
            cell2dof[idx] = NE*p + np.arange(NC*idof).reshape(NC, idof)
            return np.hsplit(cell2dof, cell2dofLocation[1:-1])


    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        gdof = NE*p + NC*(p-1)*p//2
        return gdof

    def number_of_local_dofs(self):
        p = self.p
        mesh = self.mesh
        NCE = mesh.ds.number_of_edges_of_cells()
        ldofs = NCE*p + (p-1)*p//2
        return ldofs

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
        ipoint = np.zeros((gdof, GD),dtype=np.float_)
        if p==1:
            ipoint = np.einsum(
                    'ij, ...jm->...im',
                    bcs, node[edge, :]).reshape(-1, GD)
            return ipoint

        ipoint[:NE*p, :] =  np.einsum(
                    'ij, ...jm->...im',
                    bcs, node[edge, :]).reshape(-1, GD)
        if p == 2:
            ipoint[NE*p:, :] = mesh.entity_barycenter('cell')
            return ipoint

        h = np.sqrt(mesh.cell_area())[:, None]*scale
        bc = mesh.entity_barycenter('cell')
        t = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2]], dtype=np.float_)
        t -= np.array([0.5, np.sqrt(3)/6.0], dtype=np.float_)

        tri = np.zeros((NC, 3, GD), dtype=np.float_)
        tri[:, 0, :] = bc + t[0]*h
        tri[:, 1, :] = bc + t[1]*h
        tri[:, 2, :] = bc + t[2]*h

        bcs = mesh.multi_index_matrix(p-2)/(p-2)
        ipoint[NE*p:, :] = np.einsum('ij, ...jm->...im', bcs, tri).reshape(-1, GD)
        return ipoint






