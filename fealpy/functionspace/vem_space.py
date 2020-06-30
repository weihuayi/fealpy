"""
Virtual Element Space

"""

import numpy as np
from numpy.linalg import inv
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from .Function import Function
from ..quadrature import GaussLobattoQuadrature
from ..quadrature import GaussLegendreQuadrature
from .ScaledMonomialSpace2d import ScaledMonomialSpace2d


class VEMDof2d():
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        self.cell2dof, self.cell2dofLocation = self.cell_to_dof()

    def boundary_dof(self):
        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool)
        edge2dof = self.edge_to_dof()
        isBdEdge = self.mesh.ds.boundary_edge_flag()
        isBdDof[edge2dof[isBdEdge]] = True
        return isBdDof

    def edge_to_dof(self):
        p = self.p
        mesh = self.mesh

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()

        edge = mesh.ds.edge
        edge2dof = np.zeros((NE, p+1), dtype=np.int)
        edge2dof[:, [0, p]] = edge
        if p > 1:
            edge2dof[:, 1:-1] = np.arange(NN, NN + NE*(p-1)).reshape(NE, p-1)
        return edge2dof

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh
        cell = mesh.ds.cell
        cellLocation = mesh.ds.cellLocation

        if p == 1:
            return cell, cellLocation
        else:
            NC = mesh.number_of_cells()

            ldof = self.number_of_local_dofs()
            cell2dofLocation = np.zeros(NC+1, dtype=np.int)
            cell2dofLocation[1:] = np.add.accumulate(ldof)
            cell2dof = np.zeros(cell2dofLocation[-1], dtype=np.int)

            edge2dof = self.edge_to_dof()
            edge2cell = mesh.ds.edge_to_cell()

            idx = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + np.arange(p)
            cell2dof[idx] = edge2dof[:, 0:p]
 
            isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
            idx = (cell2dofLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*p).reshape(-1, 1) + np.arange(p)
            cell2dof[idx] = edge2dof[isInEdge, p:0:-1]

            NN = mesh.number_of_nodes()
            NV = mesh.number_of_vertices_of_cells()
            NE = mesh.number_of_edges()
            idof = (p-1)*p//2
            idx = (cell2dofLocation[:-1] + NV*p).reshape(-1, 1) + np.arange(idof)
            cell2dof[idx] = NN + NE*(p-1) + np.arange(NC*idof).reshape(NC, idof)
            return cell2dof, cell2dofLocation

    def number_of_global_dofs(self):
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        gdof = NN
        p = self.p
        if p > 1:
            gdof += NE*(p-1) + NC*(p-1)*p//2
        return gdof

    def number_of_local_dofs(self):
        mesh = self.mesh
        NV = mesh.number_of_vertices_of_cells()
        ldofs = NV
        p = self.p
        if p > 1:
            ldofs += NV*(p-1) + (p-1)*p//2
        return ldofs

    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        node = mesh.entity('node')

        if p == 1:
            return node
        if p > 1:
            NN = mesh.number_of_nodes()
            GD = mesh.geo_dimension()
            NE = mesh.number_of_edges()

            ipoint = np.zeros((NN+(p-1)*NE, GD), dtype=np.float)
            ipoint[:NN, :] = node
            edge = mesh.entity('edge')

            qf = GaussLobattoQuadrature(p + 1)

            bcs = qf.quadpts[1:-1, :]
            ipoint[NN:NN+(p-1)*NE, :] = np.einsum('ij, ...jm->...im', bcs, node[edge, :]).reshape(-1, GD)
            return ipoint


class VirtualElementSpace2d():
    def __init__(self, mesh, p=1):
        self.mesh = mesh
        self.p = p
        self.smspace = ScaledMonomialSpace2d(mesh, p)
        self.dof = VEMDof2d(mesh, p)

    def cell_to_dof(self):
        return self.dof.cell2dof, self.dof.cell2dofLocation

    def boundary_dof(self):
        return self.dof.boundary_dof()

    def basis(self, bc):
        pass

    def grad_basis(self, bc):
        pass

    def hessian_basis(self, bc):
        pass

    def dual_basis(self, u):
        pass

    def value(self, uh, bc):
        pass

    def grad_value(self, uh, bc):
        pass

    def hessian_value(self, uh, bc):
        pass

    def div_value(self, uh, bc):
        pass

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array)
        return f

    def interpolation(self, u, integral=None):
        mesh = self.mesh
        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()
        p = self.p
        ipoint = self.dof.interpolation_points()
        uI = self.function()
        uI[:NN+(p-1)*NE] = u(ipoint)
        if p > 1:
            phi = self.smspace.basis

            def f(x, cellidx):
                return np.einsum(
                        'ij, ij...->ij...',
                        u(x), phi(x, cellidx=cellidx, p=p-2))

            bb = integral(f, celltype=True)/self.smspace.area[..., np.newaxis]
            uI[NN+(p-1)*NE:] = bb.reshape(-1)
        return uI

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def projection(self, u, up):
        pass

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim is None:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=np.float)


