import numpy as np
from numpy.linalg import inv
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from .function import Function
from ..quadrature import GaussLobattoQuadrature
from ..quadrature import GaussLegendreQuadrature
from ..quadrature import PolygonMeshIntegralAlg
from .ScaledMonomialSpace2d import ScaledMonomialSpace2d


class CVEMDof2d():
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


class ConformingVirtualElementSpace2d():
    def __init__(self, mesh, p=1, q=None, bc=None):
        """
        p: the space order
        q: the index of integral formular
        bc: user can give a barycenter for every mesh cell
        """
        self.mesh = mesh
        self.p = p
        self.smspace = ScaledMonomialSpace2d(mesh, p, bc=bc)
        self.area = self.smspace.area
        self.dof = CVEMDof2d(mesh, p)

        self.H = self.smspace.matrix_H()
        self.D = self.matrix_D(self.H)
        self.B = self.matrix_B()
        self.G = self.matrix_G(self.B, self.D)

        self.PI1 = self.matrix_PI_1(self.G, self.B)
        self.C = self.matrix_C(self.H, self.PI1)

        self.PI0 = self.matrix_PI_0(self.H, self.C)

        if q is None:
            self.integrator = mesh.integrator(p+3)
        else:
            self.integrator = mesh.integrator(q)

        self.integralalg = PolygonMeshIntegralAlg(
                self.integrator,
                self.mesh,
                area=self.area,
                barycenter=self.smspace.barycenter)

    def project_to_smspace(self, uh):
        """
        Project a conforming vem function uh into polynomial space.
        """
        p = self.p
        cell2dof = self.dof.cell2dof
        cell2dofLocation = self.dof.cell2dofLocation
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        g = lambda x: x[0]@uh[x[1]]
        S = self.smspace.function()
        S[:] = np.concatenate(list(map(g, zip(self.PI1, cd))))
        return S

    def project(self, F, space1):
        """
        S is a function in ScaledMonomialSpace2d, this function project  S to 
        MonomialSpace2d.
        """
        space0 = F.space
        def f(x, cellidx):
            return np.einsum(
                    '...im, ...in->...imn',
                    mspace.basis(x, cellidx),
                    smspace.basis(x, cellidx)
                    )
        C = self.integralalg.integral(f, celltype=True)
        H = space1.matrix_H()
        PI0 = inv(H)@C
        SS = mspace.function()
        SS[:] = np.einsum('ikj, ij->ik', PI0, S[smspace.cell_to_dof()]).reshape(-1)
        return SS

    def stiff_matrix(self, cfun=None):
        area = self.smspace.area

        def f(x):
            x[0, :] = 0
            return x

        p = self.p
        G = self.G
        D = self.D
        PI1 = self.PI1

        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        NC = len(cell2dofLocation) - 1
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        DD = np.vsplit(D, cell2dofLocation[1:-1])

        if p == 1:
            tG = np.array([(0, 0, 0), (0, 1, 0), (0, 0, 1)])
            if cfun is None:
                f1 = lambda x: x[1].T@tG@x[1] + (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1])
                K = list(map(f1, zip(DD, PI1)))
            else:
                barycenter = V.smspace.barycenter
                k = cfun(barycenter)
                f1 = lambda x: (x[1].T@tG@x[1] + (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1]))*x[2]
                K = list(map(f1, zip(DD, PI1, k)))
        else:
            tG = list(map(f, G))
            if cfun is None:
                f1 = lambda x: x[1].T@x[2]@x[1] + (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1])
                K = list(map(f1, zip(DD, PI1, tG)))
            else:
                barycenter = V.smspace.barycenter
                k = cfun(barycenter)
                f1 = lambda x: (x[1].T@x[2]@x[1] + (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1]))*x[3]
                K = list(map(f1, zip(DD, PI1, tG, k)))

        f2 = lambda x: np.repeat(x, x.shape[0])
        f3 = lambda x: np.tile(x, x.shape[0])
        f4 = lambda x: x.flatten()

        I = np.concatenate(list(map(f2, cd)))
        J = np.concatenate(list(map(f3, cd)))
        val = np.concatenate(list(map(f4, K)))
        gdof = self.number_of_global_dofs()
        A = csr_matrix((val, (I, J)), shape=(gdof, gdof), dtype=np.float)
        return A

    def mass_matrix(self, cfun=None):
        area = self.smspace.area
        p = self.p

        PI0 = self.PI0
        D = self.D
        H = self.H
        C = self.C

        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        NC = len(cell2dofLocation) - 1
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        DD = np.vsplit(D, cell2dofLocation[1:-1])

        f1 = lambda x: x[0]@x[1]
        PIS = list(map(f1, zip(DD, PI0)))

        f1 = lambda x: x[0].T@x[1]@x[0] + x[3]*(np.eye(x[2].shape[1]) - x[2]).T@(np.eye(x[2].shape[1]) - x[2])
        K = list(map(f1, zip(PI0, H, PIS, area)))

        f2 = lambda x: np.repeat(x, x.shape[0])
        f3 = lambda x: np.tile(x, x.shape[0])
        f4 = lambda x: x.flatten()

        I = np.concatenate(list(map(f2, cd)))
        J = np.concatenate(list(map(f3, cd)))
        val = np.concatenate(list(map(f4, K)))
        gdof = self.number_of_global_dofs()
        M = csr_matrix((val, (I, J)), shape=(gdof, gdof), dtype=np.float)
        return M

    def cross_mass_matrix(self, wh):
        p = self.p
        mesh = self.mesh

        area = self.smspace.area
        PI0 = self.PI0

        phi = self.smspace.basis
        def u(x, cellidx):
            val = phi(x, cellidx=cellidx)
            wval = wh(x, cellidx=cellidx)
            return np.einsum('ij, ijm, ijn->ijmn', wval, val, val)
        H = self.integralalg.integral(u, celltype=True)

        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        NC = len(cell2dofLocation) - 1
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])

        f1 = lambda x: x[0].T@x[1]@x[0]
        K = list(map(f1, zip(PI0, H)))

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
        PI0 = self.PI0
        phi = self.smspace.basis
        def u(x, cellidx):
            return np.einsum('ij, ijm->ijm', f(x), phi(x, cellidx=cellidx))
        bb = self.integralalg.integral(u, celltype=True)
        g = lambda x: x[0].T@x[1]
        bb = np.concatenate(list(map(g, zip(PI0, bb))))
        gdof = self.number_of_global_dofs()
        b = np.bincount(self.dof.cell2dof, weights=bb, minlength=gdof)
        return b

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

    def interpolation(self, u):
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
            bb = self.integralalg.integral(f, celltype=True)/self.smspace.area[..., np.newaxis]
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

    def matrix_D(self, H):
        p = self.p
        smldof = self.smspace.number_of_local_dofs()
        mesh = self.mesh
        NV = mesh.number_of_vertices_of_cells()
        h = self.smspace.h
        node = mesh.node
        edge = mesh.ds.edge
        edge2cell = mesh.ds.edge2cell
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        D = np.ones((len(cell2dof), smldof), dtype=np.float)

        if p == 1:
            bc = np.repeat(self.smspace.barycenter, NV, axis=0)
            D[:, 1:] = (node[mesh.ds.cell, :] - bc)/np.repeat(h, NV).reshape(-1, 1)
            return D

        qf = GaussLobattoQuadrature(p+1)
        bcs, ws = qf.quadpts, qf.weights
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        phi0 = self.smspace.basis(ps[:-1], cellidx=edge2cell[:, 0])
        phi1 = self.smspace.basis(ps[p:0:-1, isInEdge, :], cellidx=edge2cell[isInEdge, 1])
        idx = cell2dofLocation[edge2cell[:, 0]] + edge2cell[:, 2]*p + np.arange(p).reshape(-1, 1)
        D[idx, :] = phi0
        idx = cell2dofLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*p + np.arange(p).reshape(-1, 1)
        D[idx, :] = phi1
        if p > 1:
            area = self.smspace.area
            idof = (p-1)*p//2 # the number of dofs of scale polynomial space with degree p-2
            idx = cell2dofLocation[1:].reshape(-1, 1) + np.arange(-idof, 0)
            D[idx, :] = H[:, :idof, :]/area.reshape(-1, 1, 1)
        return D

    def matrix_B(self):
        p = self.p
        smldof = self.smspace.number_of_local_dofs()
        mesh = self.mesh
        NV = mesh.number_of_vertices_of_cells()
        h = self.smspace.h
        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        B = np.zeros((smldof, cell2dof.shape[0]), dtype=np.float)
        if p == 1:
            B[0, :] = 1/np.repeat(NV, NV)
            B[1:, :] = mesh.node_normal().T/np.repeat(h, NV).reshape(1, -1)
            return B
        else:
            idx = cell2dofLocation[0:-1] + NV*p
            B[0, idx] = 1
            idof = (p-1)*p//2
            start = 3
            r = np.arange(1, p+1)
            r = r[0:-1]*r[1:]
            for i in range(2, p+1):
                idx0 = np.arange(start, start+i-1)
                idx1 =  np.arange(start-2*i+1, start-i)
                idx1 = idx.reshape(-1, 1) + idx1
                B[idx0, idx1] -= r[i-2::-1]
                B[idx0+2, idx1] -= r[0:i-1]
                start += i+1

            node = mesh.node
            edge = mesh.ds.edge
            edge2cell = mesh.ds.edge2cell
            isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

            qf = GaussLobattoQuadrature(p + 1)
            bcs, ws = qf.quadpts, qf.weights
            ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
            gphi0 = self.smspace.grad_basis(ps, cellidx=edge2cell[:, 0])
            gphi1 = self.smspace.grad_basis(ps[-1::-1, isInEdge, :], cellidx=edge2cell[isInEdge, 1])
            nm = mesh.edge_normal()

            # m: the scaled basis number,
            # j: the edge number,
            # i: the virtual element basis number

            NV = mesh.number_of_vertices_of_cells()

            val = np.einsum('i, ijmk, jk->mji', ws, gphi0, nm, optimize=True)
            idx = cell2dofLocation[edge2cell[:, [0]]] + \
                    (edge2cell[:, [2]]*p + np.arange(p+1))%(NV[edge2cell[:, [0]]]*p)
            np.add.at(B, (np.s_[:], idx), val)


            if isInEdge.sum() > 0:
                val = np.einsum('i, ijmk, jk->mji', ws, gphi1, -nm[isInEdge], optimize=True)
                idx = cell2dofLocation[edge2cell[isInEdge, 1]].reshape(-1, 1) + \
                        (edge2cell[isInEdge, 3].reshape(-1, 1)*p + np.arange(p+1)) \
                        %(NV[edge2cell[isInEdge, 1]].reshape(-1, 1)*p)
                np.add.at(B, (np.s_[:], idx), val)
            return B

    def matrix_G(self, B, D):
        p = self.p
        if p == 1:
            G = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        else:
            cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
            BB = np.hsplit(B, cell2dofLocation[1:-1])
            DD = np.vsplit(D, cell2dofLocation[1:-1])
            g = lambda x: x[0]@x[1]
            G = list(map(g, zip(BB, DD)))
        return G

    def matrix_C(self, H, PI1):
        p = self.p

        smldof = self.smspace.number_of_local_dofs()
        idof = (p-1)*p//2

        mesh = self.mesh
        NV = mesh.number_of_vertices_of_cells()
        d = lambda x: x[0]@x[1]
        C = list(map(d, zip(H, PI1)))
        if p == 1:
            return C
        else:
            l = lambda x: np.r_[
                    '0',
                    np.r_['1', np.zeros((idof, p*x[0])), x[1]*np.eye(idof)],
                    x[2][idof:, :]]
            return list(map(l, zip(NV, self.smspace.area, C)))

    def matrix_PI_0(self, H, C):
        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        pi0 = lambda x: inv(x[0])@x[1]
        return list(map(pi0, zip(H, C)))

    def matrix_PI_1(self, G, B):
        p = self.p
        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        if p == 1:
            return np.hsplit(B, cell2dofLocation[1:-1])
        else:
            BB = np.hsplit(B, cell2dofLocation[1:-1])
            g = lambda x: inv(x[0])@x[1]
            return list(map(g, zip(G, BB)))

