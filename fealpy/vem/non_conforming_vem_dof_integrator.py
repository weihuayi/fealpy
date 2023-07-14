import numpy as np
from ..functionspace import NonConformingScalarVESpace2d
from ..quadrature import GaussLegendreQuadrature


class NonConformingVEMDoFIntegrator2d:
    def assembly_cell_matrix(self, space: NonConformingScalarVESpace2d, M):
        """
        @brief 非协调虚单元方法的自由度矩阵
        @param sapce 
        @param M the cell mass matrix of the 2d scaled monomial space
        """
        p = space.p
        mesh = space.mesh
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        cell2dofLocation = space.dof.cell2dofLocation
        smldof = space.smspace.number_of_local_dofs()
        D = np.ones((cell2dofLocation[-1], smldof), dtype=space.ftype)

        qf = GaussLegendreQuadrature(p)
        bcs = qf.quadpts
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        phi0 = space.smspace.basis(ps, index=edge2cell[:, 0])
        phi1 = space.smspace.basis(ps[p-1::-1, isInEdge, :], index=edge2cell[isInEdge, 1])

        idx = cell2dofLocation[edge2cell[:, 0]] + edge2cell[:, 2]*p + np.arange(p).reshape(-1, 1)
        D[idx, :] = phi0

        idx = cell2dofLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*p + np.arange(p).reshape(-1, 1)
        D[idx, :] = phi1
        if p > 1:
            idof = (p-1)*p//2  # the number of dofs of scale polynomial space with degree p-2
            idx = cell2dofLocation[1:].reshape(-1, 1) + np.arange(-idof, 0)
            D[idx, :] = M[:, :idof, :]/space.smspace.cellmeasure.reshape(-1, 1, 1)

        return np.vsplit(D, cell2dofLocation[1:-1])
