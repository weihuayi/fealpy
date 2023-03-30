import numpy as np
from ..quadrature import GaussLegendreQuadrature
from ..functionspace import ScaledMonomialSpace2d, ScaledMonomialSpace3d


class ScaledMonomialSpaceMassIntegrator2d:

    def assembly_cell_matrix(self, space: ScaledMonomialSpace2d):
        """
        """
        p = space.p
        mesh = space.mesh
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        NC = mesh.number_of_cells()

        qf = GaussLegendreQuadrature(p + 1)
        bcs, ws = qf.quadpts, qf.weights
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        phi0 = space.basis(ps, index=edge2cell[:, 0], p=p)
        phi1 = space.basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1], p=p)
        H0 = np.einsum('i, ijk, ijm->jkm', ws, phi0, phi0)
        H1 = np.einsum('i, ijk, ijm->jkm', ws, phi1, phi1)

        nm = mesh.edge_normal()
        b = node[edge[:, 0]] - space.cellbarycenter[edge2cell[:, 0]]
        H0 = np.einsum('ij, ij, ikm->ikm', b, nm, H0)
        b = node[edge[isInEdge, 0]] - space.cellbarycenter[edge2cell[isInEdge, 1]]
        H1 = np.einsum('ij, ij, ikm->ikm', b, -nm[isInEdge], H1)

        ldof = space.number_of_local_dofs(p=p, doftype='cell')
        H = np.zeros((NC, ldof, ldof), dtype=np.float)
        np.add.at(H, edge2cell[:, 0], H0)
        np.add.at(H, edge2cell[isInEdge, 1], H1)

        multiIndex = space.dof.multi_index_matrix(p=p)
        q = np.sum(multiIndex, axis=1)
        H /= q + q.reshape(-1, 1) + 2
        return H

    def assembly_cell_matrix_numba(self, space):
        pass

class ScaledMonomialSpaceMassIntegrator3d:
    def assembly_cell_matrix(self, space: ScaledMonomialSpace3d):
        """
        """
        pass

    def assembly_cell_matrix_numba(self, space):
        """
        """
        pass
