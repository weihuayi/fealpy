import ipdb
import numpy as np
from ..quadrature import GaussLegendreQuadrature
from ..functionspace import ScaledMonomialSpace2d, ScaledMonomialSpace3d


class ScaledMonomialSpaceMassIntegrator2d:
    def __init__(self, q=3):
        self.q = q

    def assembly_cell_matrix(self, space: ScaledMonomialSpace2d, p=None, index=np.s_[:]):
        """
        @brief 组装缩放单项式空间每个单元上的质量矩阵
        """
        p = space.p if p is None else p
        mesh = space.mesh
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        NC = mesh.number_of_cells()

        qf = mesh.integrator(p+1, 'edge')
        bcs, ws = qf.get_quadrature_points_and_weights()

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
        H = np.zeros((NC, ldof, ldof), dtype=space.ftype)
        np.add.at(H, edge2cell[:, 0], H0)
        np.add.at(H, edge2cell[isInEdge, 1], H1)

        multiIndex = space.dof.multi_index_matrix(p=p)
        q = np.sum(multiIndex, axis=1)
        H /= q + q.reshape(-1, 1) + 2
        return H[index]

    def assembly_cell_matrix_0(self, space: ScaledMonomialSpace2d, p=None, index=np.s_[:]):
        """
        @brief 组装缩放单项式空间每个单元上的质量矩阵
        """
        p = space.p if p is None else p
        mesh = space.mesh
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()

        NC = mesh.number_of_cells()

        isMarkedCell = np.zeros(NC, dtype=np.bool_)
        isMarkedCell[index] = True

        isLeftEdge = isMarkedCell[edge2cell[:, 0]]
        isRightEdge = (edge2cell[:, 0] != edge2cell[:, 1])&(isMarkedCell[edge2cell[:, 1]])

        qf = mesh.integrator(p+1, 'edge')
        bcs, ws = qf.get_quadrature_points_and_weights()

        ps = np.einsum('ij, kjm->ikm', bcs, node[edge[isLeftEdge]])
        phi0 = space.basis(ps, index=edge2cell[isLeftEdge, 0], p=p)

        ps = np.einsum('ij, kjm->ikm', bcs, node[edge[isRightEdge]])
        phi1 = space.basis(ps, index=edge2cell[isRightEdge, 1], p=p)

        H0 = np.einsum('i, ijk, ijm->jkm', ws, phi0, phi0)
        H1 = np.einsum('i, ijk, ijm->jkm', ws, phi1, phi1)

        nm = mesh.edge_normal()
        b = node[edge[isLeftEdge, 0]] - space.cellbarycenter[edge2cell[isLeftEdge, 0]]
        H0 = np.einsum('ij, ij, ikm->ikm', b, nm[isLeftEdge], H0)
        b = node[edge[isRightEdge, 0]] - space.cellbarycenter[edge2cell[isRightEdge, 1]]
        H1 = np.einsum('ij, ij, ikm->ikm', b, -nm[isRightEdge], H1)

        ldof = space.number_of_local_dofs(p=p, doftype='cell')
        H = np.zeros((NC, ldof, ldof), dtype=space.ftype)
        np.add.at(H, edge2cell[isLeftEdge, 0], H0)
        np.add.at(H, edge2cell[isRightEdge, 1], H1)
        H = H[index]

        multiIndex = space.dof.multi_index_matrix(p=p)
        q = np.sum(multiIndex, axis=1)
        H /= q + q.reshape(-1, 1) + 2
        return H

    def assembly_cell_matrix_numba(self, space: ScaledMonomialSpace2d, index=np.s_[:]):
        pass

class ScaledMonomialSpaceMassIntegrator3d:
    def __init__(self, q=3):
        self.q = 3

    def assembly_cell_matrix(self, space: ScaledMonomialSpace3d):
        """
        """
        pass

    def assembly_cell_matrix_numba(self, space):
        """
        """
        pass
