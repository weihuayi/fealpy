import numpy as np
from ..functionspace import ConformingVirtualElementSpace2d
from fealpy.quadrature import GaussLobattoQuadrature


class ConformingVEMDoFIntegrator2d:

    def assembly_cell_matrix(self, space: ConformingVirtualElementSpace2d, M):
        """

        @param sapce 
        @param M the cell mass matrix of the 2d scaled monomial space
        """
        p = space.p
        smldof = space.smspace.number_of_local_dofs()
        mesh = space.mesh
        NV = mesh.ds.number_of_vertices_of_cells()
        h = space.smspace.cellsize
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        cell = np.concatenate(mesh.entity('cell'))
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        cell2dofLocation = space.dof.cell2dofLocation
        D = np.ones((cell2dofLocation[-1], smldof), dtype=np.float64)

        if p == 1:
            bc = np.repeat(space.smspace.cellbarycenter, NV, axis=0)
            D[:, 1:] = (node[cell, :] - bc)/np.repeat(h, NV).reshape(-1, 1)
            return np.vsplit(D, cell2dofLocation[1:-1])

        qf = GaussLobattoQuadrature(p+1)
        bcs, ws = qf.quadpts, qf.weights
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        phi0 = space.smspace.basis(ps[:-1], index=edge2cell[:, 0])
        phi1 = space.smspace.basis(ps[p:0:-1, isInEdge, :], index=edge2cell[isInEdge, 1])
        idx = cell2dofLocation[edge2cell[:, 0]] + edge2cell[:, 2]*p + np.arange(p).reshape(-1, 1)
        D[idx, :] = phi0
        idx = cell2dofLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*p + np.arange(p).reshape(-1, 1)
        D[idx, :] = phi1
        if p > 1:
            area = space.smspace.cellmeasure
            idof = (p-1)*p//2 # the number of dofs of scale polynomial space with degree p-2
            idx = cell2dofLocation[1:].reshape(-1, 1) + np.arange(-idof, 0)
            D[idx, :] = M[:, :idof, :]/area.reshape(-1, 1, 1)

        return np.vsplit(D, cell2dofLocation[1:-1])
