import numpy as np
from numpy.linalg import inv
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from .Function import Function
from ..quadrature import GaussLegendreQuadrature
from ..quadrature import PolygonMeshIntegralAlg
from .ScaledMonomialSpace2d import ScaledMonomialSpace2d

class NCVEMDof2d():
    """
    The dof manager of non conforming vem 2d space.
    """
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        self.cell2dof, self.cell2dofLocation = self.cell_to_dof()

    def boundary_dof(self, threshold=None):
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
        cell, cellLocation = mesh.entity('cell')

        if p == 1:
            cell2edge, _ = mesh.ds.cell_to_edge(return_sparse=False)
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
        NCE = mesh.number_of_edges_of_cells()
        ldofs = NCE*p + (p-1)*p//2
        return ldofs

    def interpolation_points(self):
        """
        Get the node-value-type interpolation points.

        On every edge, there exist p points
        """
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
    """
    The non conforming 2d vem space.
    """
    def __init__(self, mesh, p=1, q=None):
        self.p = p
        self.smspace = ScaledMonomialSpace2d(mesh, p, q=q)
        self.mesh = mesh
        self.dof = NCVEMDof2d(mesh, p)

        self.integralalg = self.smspace.integralalg

        self.H = self.matrix_H()
        self.D = self.matrix_D(self.H)
        self.B = self.matrix_B()
        self.G = self.matrix_G(self.B, self.D)

        self.PI1 = self.matrix_PI_1(self.G, self.B)
        self.C = self.matrix_C(self.H, self.PI1)

        self.PI0 = self.matrix_PI_0(self.H, self.C)

    def project_to_smspace(self, uh):
        """
        Project a non conforming vem function uh into polynomial space.
        """
        p = self.p
        cell2dof = self.dof.cell2dof
        cell2dofLocation = self.dof.cell2dofLocation
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        g = lambda x: x[0]@uh[x[1]]
        S = self.smspace.function()
        S[:] = np.concatenate(list(map(g, zip(self.PI1, cd))))
        return S

    def project_to_smspace_L2(self, uh):
        """
        Project a non conforming vem function uh into polynomial space.
        """
        p = self.p
        cell2dof = self.dof.cell2dof
        cell2dofLocation = self.dof.cell2dofLocation
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        g = lambda x: x[0]@uh[x[1]]
        S = self.smspace.function()
        S[:] = np.concatenate(list(map(g, zip(self.PI0, cd))))
        return S

    def stiff_matrix(self):
        p = self.p
        G = self.G
        D = self.D
        PI1 = self.PI1

        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])

        DD = np.vsplit(D, cell2dofLocation[1:-1])

        def f(x):
            x[0, :] = 0
            return x

        tG = list(map(f, G))

        f1 = lambda x: x[1].T@x[2]@x[1] + (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1])
        K = list(map(f1, zip(DD, PI1, tG)))

        f2 = lambda x: np.repeat(x, x.shape[0])
        f3 = lambda x: np.tile(x, x.shape[0])
        f4 = lambda x: x.flatten()

        I = np.concatenate(list(map(f2, cd)))
        J = np.concatenate(list(map(f3, cd)))
        val = np.concatenate(list(map(f4, K)))
        gdof = self.number_of_global_dofs()
        A = csr_matrix((val, (I, J)), shape=(gdof, gdof), dtype=np.float)
        return A

    def mass_matrix(self):
        p = self.p

        # the projector matrix
        D = self.D
        H = self.H
        C = self.C

        # the dof arrays
        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])

        f1 = lambda x: x[0]@x[1]
        PI0 = self.PI0
        DD = np.vsplit(D, cell2dofLocation[1:-1])
        PIS = list(map(f1, zip(DD, PI0)))

        f1 = lambda x: x[0].T@x[1]@x[0] + x[3]*(np.eye(x[2].shape[1]) - x[2]).T@(np.eye(x[2].shape[1]) - x[2])
        K = list(map(f1, zip(PI0, H, PIS, self.smspace.cellmeasure)))

        f2 = lambda x: np.repeat(x, x.shape[0])
        f3 = lambda x: np.tile(x, x.shape[0])
        f4 = lambda x: x.flatten()

        I = np.concatenate(list(map(f2, cd)))
        J = np.concatenate(list(map(f3, cd)))
        val = np.concatenate(list(map(f4, K)))
        gdof = self.number_of_global_dofs()
        M = csr_matrix((val, (I, J)), shape=(gdof, gdof), dtype=np.float)
        return M


    def source_vector(self, f):
        phi = self.smspace.basis
        def u(x, index):
            return np.einsum('ij, ijm->ijm', f(x), phi(x, index=index))
        bb = self.integralalg.integral(u, celltype=True)
        g = lambda x: x[0].T@x[1]
        bb = np.concatenate(list(map(g, zip(self.PI0, bb))))
        gdof = self.number_of_global_dofs()
        b = np.bincount(self.dof.cell2dof, weights=bb, minlength=gdof)
        return b

    def set_dirichlet_bc(self, gD, uh, threshold=None):
        """
        初始化解 uh  的第一类边界条件。
        """
        p = self.p
        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        end = p*NE
        ipoints = self.interpolation_points()
        isDDof = self.boundary_dof(threshold=threshold)
        uh[isDDof] = gD(ipoints[isDDof[:end]])
        return isDDof

    def cell_to_dof(self):
        return self.dof.cell2dof, self.dof.cell2dofLocation

    def edge_basis_to_integral_basis(self):
        p = self.p
        qf = GaussLegendreQuadrature(p)
        bcs, ws = qf.get_quadrature_points_and_weights()

        A = np.ones((p, p), dtype=np.float_)
        A[:, 1:] = bcs[:, 1, None]-0.5
        A[:] = np.cumprod(A, axis = 1)*ws[:, None]
        return A 

    def boundary_dof(self, threshold=None):
        return self.dof.boundary_dof(threshold=threshold)

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

    def interpolation(self, u):
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
                return np.einsum('ij, ij...->ij...', u(x), phi(x, index=index, p=p-2))

            bb = self.integralalg.integral(f,
                    celltype=True)/self.smspace.cellmeasure[..., np.newaxis]
            uI[p*NE:] = bb.reshape(-1)
        return uI

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def projection(self, u, up):
        pass

    def array(self, dim=None, dtype=np.float_):
        gdof = self.number_of_global_dofs()
        if dim is None:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=dtype)

    def matrix_H(self):
        p = self.p
        mesh = self.mesh
        node = mesh.entity('node')

        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        NC = mesh.number_of_cells()

        qf = GaussLegendreQuadrature(p + 1)
        bcs, ws = qf.get_quadrature_points_and_weights()

        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        phi0 = self.smspace.basis(ps, index=edge2cell[:, 0])
        phi1 = self.smspace.basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1])
        H0 = np.einsum('i, ijk, ijm->jkm', ws, phi0, phi0)
        H1 = np.einsum('i, ijk, ijm->jkm', ws, phi1, phi1)

        nm = mesh.edge_normal()
        b = node[edge[:, 0]] - self.smspace.cellbarycenter[edge2cell[:, 0]]
        H0 = np.einsum('ij, ij, ikm->ikm', b, nm, H0)
        b = node[edge[isInEdge, 0]] - self.smspace.cellbarycenter[edge2cell[isInEdge, 1]]
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
        node = mesh.entity('node')

        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        cell2dof, cell2dofLocation = self.cell_to_dof()
        D = np.ones((len(cell2dof), smldof), dtype=np.float)

        qf = GaussLegendreQuadrature(p)
        bcs = qf.quadpts
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        phi0 = self.smspace.basis(ps, index=edge2cell[:, 0])
        phi1 = self.smspace.basis(ps[p-1::-1, isInEdge, :], index=edge2cell[isInEdge, 1])

        idx = cell2dofLocation[edge2cell[:, 0]] + edge2cell[:, 2]*p + np.arange(p).reshape(-1, 1)
        D[idx, :] = phi0

        idx = cell2dofLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*p + np.arange(p).reshape(-1, 1)
        D[idx, :] = phi1
        if p > 1:
            idof = (p-1)*p//2  # the number of dofs of scale polynomial space with degree p-2
            idx = cell2dofLocation[1:].reshape(-1, 1) + np.arange(-idof, 0)
            D[idx, :] = H[:, :idof, :]/self.smspace.cellmeasure.reshape(-1, 1, 1)
        return D

    def matrix_B(self):
        p = self.p
        smldof = self.smspace.number_of_local_dofs()

        mesh = self.mesh
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation

        qf = GaussLegendreQuadrature(p)
        bcs, ws = qf.get_quadrature_points_and_weights()

        B = np.zeros((smldof, cell2dof.shape[0]), dtype=np.float)

        # the internal part
        if p > 1:
            NCE = mesh.number_of_edges_of_cells()
            idx = cell2dofLocation[0:-1] + NCE*p
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

        # the normal deriveration part
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        gphi0 = self.smspace.grad_basis(ps, index=edge2cell[:, 0])
        gphi1 = self.smspace.grad_basis(
                ps[-1::-1, isInEdge, :],
                index=edge2cell[isInEdge, 1])
        nm = mesh.edge_normal()
        h = np.sqrt(np.sum(nm**2, axis=-1))
        # m: the scaled basis number,
        # j: the edge number,
        # i: the virtual element basis number
        val = np.einsum('i, ijmk, jk->mji', ws, gphi0, nm, optimize=True)
        idx = (cell2dofLocation[edge2cell[:, 0]]
                + edge2cell[:, 2]*p).reshape(-1, 1) + np.arange(p)
        B[:, idx] += val
        B[0, idx] = h.reshape(-1, 1)*ws

        val = np.einsum('i, ijmk, jk->mji', ws, gphi1, -nm[isInEdge],
                optimize=True)
        idx = ( cell2dofLocation[edge2cell[isInEdge, 1]]
                + edge2cell[isInEdge, 3]*p).reshape(-1, 1) + np.arange(p)
        B[:, idx] += val
        B[0, idx] = h[isInEdge].reshape(-1, 1)*ws
        return B

    def matrix_G(self, B, D):
        p = self.p
        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        BB = np.hsplit(B, cell2dofLocation[1:-1])
        DD = np.vsplit(D, cell2dofLocation[1:-1])
        g = lambda x: x[0]@x[1]
        G = list(map(g, zip(BB, DD)))
        return G

    def matrix_G_test(self, integralalg):
        def u(x, index=None):
            gphi = self.smspace.grad_basis(x, index=index)
            return np.einsum('ijkm, ijpm->ijkp', gphi, gphi, optimize=True)

        G = integralalg.integral(u, celltype=True)
        return G


    def matrix_C(self, H, PI1):
        p = self.p

        smldof = self.smspace.number_of_local_dofs()
        idof = (p-1)*p//2

        mesh = self.mesh
        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        d = lambda x: x[0]@x[1]
        C = list(map(d, zip(H, PI1)))
        if p == 1:
            return C
        else:
            NV = mesh.number_of_vertices_of_cells()
            l = lambda x: np.r_[
                    '0',
                    np.r_['1', np.zeros((idof, p*x[0])), x[1]*np.eye(idof)],
                    x[2][idof:, :]]
            return list(map(l, zip(NV, self.smspace.cellmeasure, C)))

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

