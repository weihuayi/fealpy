"""
Virtual Element Space

"""

import numpy as np
from .function import Function
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


class NCVEMDof2d():
    """
    The dof manager of non conforming vem space.
    """
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        self.cell2dof, self.cell2dofLocation = self.cell_to_dof()
        print(self.cell2dof)
        print(self.cell2dofLocation)

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
        NE = mesh.number_of_edges()
        edge2dof = np.arange(NE*p).reshape(NE, p)
        return edge2dof

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh
        cellLocation = mesh.ds.cellLocation
        cell2edge = mesh.ds.cell_to_edge(sparse=False)

        if p == 1:
            return cell2edge, cellLocation
        else:
            NC = mesh.number_of_cells()

            ldof = self.number_of_local_dofs()
            cell2dofLocation = np.zeros(NC+1, dtype=np.int)
            cell2dofLocation[1:] = np.add.accumulate(ldof)
            cell2dof = np.zeros(cell2dofLocation[-1], dtype=np.int)

            edge2dof = self.edge_to_dof()
            edge2cell = mesh.ds.edge_to_cell()
            idx = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + np.arange(p)
            cell2dof[idx] = edge2dof

            isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
            idx = (cell2dofLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*p).reshape(-1, 1) + np.arange(p)
            cell2dof[idx] = edge2dof[isInEdge, p-1::-1]

            NV = mesh.number_of_vertices_of_cells()
            NE = mesh.number_of_edges()
            idof = (p-1)*p//2
            idx = (cell2dofLocation[:-1] + NV*p).reshape(-1, 1) + np.arange(idof)
            cell2dof[idx] = NE*p + np.arange(NC*idof).reshape(NC, idof)
            return cell2dof, cell2dofLocation

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
        NV = mesh.number_of_vertices_of_cells()
        ldofs = NV*p + (p-1)*p//2
        return ldofs

    def interpolation_points(self):
        p = self.p
        mesh = self.mesh
        node = mesh.entity('node')
        edge = mesh.entity('edge')

        GD = mesh.geo_dimension()

        qf = GaussLegendreQuadrature(p)
        bcs, ws = qf.get_quadrature_points_and_weights()

        ipoint = np.einsum(
                'ij, ...jm->...im',
                bcs, node[edge, :]).reshape(-1, GD)
        return ipoint


class NonConformingVirtualElementSpace2d():
    def __init__(self, mesh, p=1):
        self.mesh = mesh
        self.p = p
        self.smspace = ScaledMonomialSpace2d(mesh, p)
        self.dof = NCVEMDof2d(mesh, p)

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
                return np.einsum('ij, ij...->ij...', u(x), phi(x, cellidx=cellidx, p=p-2))

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

    def matrix_H(self):
        p = self.p
        mesh = self.mesh
        node = mesh.entity('node')

        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        NC = mesh.number_of_cells()

        qf = GaussLegendreQuadrature(p + 1)
        bcs, ws = qf.quadpts, qf.weights
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        phi0 = self.smspace.basis(ps, cellidx=edge2cell[:, 0])
        phi1 = self.smspace.basis(ps[:, isInEdge, :], cellidx=edge2cell[isInEdge, 1])
        H0 = np.einsum('i, ijk, ijm->jkm', ws, phi0, phi0)
        H1 = np.einsum('i, ijk, ijm->jkm', ws, phi1, phi1)

        nm = mesh.edge_normal()
        b = node[edge[:, 0]] - self.smspace.barycenter[edge2cell[:, 0]]
        H0 = np.einsum('ij, ij, ikm->ikm', b, nm, H0)
        b = node[edge[isInEdge, 0]] - self.smspace.barycenter[edge2cell[isInEdge, 1]]
        H1 = np.einsum('ij, ij, ikm->ikm', b, -nm[isInEdge], H1)

        ldof = self.smspace.number_of_local_dofs()
        H = np.zeros((NC, ldof, ldof), dtype=np.float)
        np.add.at(H, edge2cell[:, 0], H0)
        np.add.at(H, edge2cell[isInEdge, 1], H1)

        multiIndex = self.smspace.dof.multiIndex
        q = np.sum(multiIndex, axis=1)
        H /= q + q.reshape(-1, 1) + 2
        return H

    def matrix_D(self, H):
        p = self.p
        smldof = self.smspace.number_of_local_dofs()
        mesh = self.mesh
        node = mesh.node
        edge = mesh.ds.edge
        edge2cell = mesh.ds.edge2cell
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        D = np.ones((len(cell2dof), smldof), dtype=np.float)

        qf = GaussLegendreQuadrature(p)
        bcs = qf.quadpts
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        phi0 = self.smspace.basis(ps, cellidx=edge2cell[:, 0])
        phi1 = self.smspace.basis(ps[p-1::-1, isInEdge, :], cellidx=edge2cell[isInEdge, 1])
        idx = cell2dofLocation[edge2cell[:, 0]] + edge2cell[:, 2]*p + np.arange(p).reshape(-1, 1)
        D[idx, :] = phi0
        idx = cell2dofLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*p + np.arange(p).reshape(-1, 1)
        D[idx, :] = phi1
        if p > 1:
            area = self.smspace.area
            idof = (p-1)*p//2  # the number of dofs of scale polynomial space with degree p-2
            idx = cell2dofLocation[1:].reshape(-1, 1) + np.arange(-idof, 0)
            D[idx, :] = H[:, :idof, :]/area.reshape(-1, 1, 1)
        return D

    def matrix_B(self):
        p = self.p
        smldof = self.smspace.number_of_local_dofs()

        mesh = self.mesh
        node = mesh.node
        edge = mesh.ds.edge
        edge2cell = mesh.ds.edge2cell
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation

        qf = GaussLegendreQuadrature(p)
        bcs, ws = qf.get_quadrature_points_and_weights()

        B = np.zeros((smldof, cell2dof.shape[0]), dtype=np.float)

        # the normal deriveration part
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        gphi0 = self.smspace.grad_basis(ps, cellidx=edge2cell[:, 0])
        gphi1 = self.smspace.grad_basis(
                ps[-1::-1, isInEdge, :],
                cellidx=edge2cell[isInEdge, 1])
        nm = mesh.edge_normal()
        # m: the scaled basis number,
        # j: the edge number,
        # i: the virtual basis number
        val = np.einsum('ijmk, jk->mji', gphi0, nm)
        val = np.einsum('i, mji->mji', ws, val)
        idx = (
                cell2dofLocation[edge2cell[:, 0]] + edge2cell[:, 2]*p
                ).reshape(-1, 1) + np.arange(p)
        B[:, idx] += val
        B[0, idx] += ws.reshape(1, -1)

        val = np.einsum('ijmk, jk->mji', gphi1, -nm[isInEdge])
        val = np.einsum('i, mji->mji', ws, val)
        idx = (
                cell2dofLocation[edge2cell[isInEdge, 1]]
                + edge2cell[isInEdge, 3]*p).reshape(-1, 1) + np.arange(p)
        B[:, idx] += val
        B[0, idx] += ws.reshape(1, -1)

        # the internal part

        start = 3
        r = np.arange(1, p+1)
        r = r[0:-1]*r[1:]
        for i in range(2, p+1):
            idx0 = np.arange(start, start+i-1)
            idx1 = np.arange(start-2*i+1, start-i)
            idx1 = idx.reshape(-1, 1) + idx1
            B[idx0, idx1] -= r[i-2::-1]
            B[idx0+2, idx1] -= r[0:i-1]
            start += i+1

        return B

    def matrix_G(self, B, D):
        p = self.p
        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        BB = np.hsplit(B, cell2dofLocation[1:-1])
        DD = np.vsplit(D, cell2dofLocation[1:-1])
        g = lambda x: x[0]@x[1]
        G = list(map(g, zip(BB, DD)))
        return G

    def matrix_C(self, G, H):
        p = self.p

        smldof = self.smspace.number_of_local_dofs()
        area = self.smspace.area
        idof = (p-1)*p//2

        mesh = self.mesh
        NV = mesh.number_of_vertices_of_cells()
        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        BB = np.hsplit(B, cell2dofLocation[1:-1])
        DD = np.vsplit(D, cell2dofLocation[1:-1])
        g = lambda x: x[0]@x[1]
        d = lambda x: x[0]@inv(x[1])@x[2]
        C = list(map(d, zip(H, G, BB)))
        if p == 1:
            return C
        else:
            l = lambda x: np.r_[
                    '0',
                    np.r_['1', np.zeros((idof, p*x[0])), x[1]*np.eye(idof)],
                    x[2][idof:, :]]
            return list(map(l, zip(NV, area, C)))

    def matrix_PI_0(self, H, C):
        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        pi0 = lambda x: inv(x[0])@x[1]
        return list(map(pi0, zip(H, C)))

    def matrix_PI_1(self, G, B):
        p = self.p
        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        BB = np.hsplit(B, cell2dofLocation[1:-1])
        g = lambda x: inv(x[0])@x[1]
        return list(map(g, zip(G, BB)))
