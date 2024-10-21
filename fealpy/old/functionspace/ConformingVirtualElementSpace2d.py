import numpy as np
from numpy.linalg import inv
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from .Function import Function
from ..quadrature import GaussLobattoQuadrature
from ..quadrature import GaussLegendreQuadrature
from ..quadrature import PolygonMeshIntegralAlg
from .scaled_monomial_space_2d import ScaledMonomialSpace2d


class CVEMDof2d():
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

        NN = mesh.number_of_nodes()
        NE = mesh.number_of_edges()

        edge = mesh.entity('edge')
        edge2dof = np.zeros((NE, p+1), dtype=np.int_)
        edge2dof[:, [0, p]] = edge
        if p > 1:
            edge2dof[:, 1:-1] = np.arange(NN, NN + NE*(p-1)).reshape(NE, p-1)
        return edge2dof

    def cell_to_dof(self):
        p = self.p
        mesh = self.mesh
        cell, cellLocation = mesh.entity('cell')
        #cell = mesh.ds._cell
        #cellLocation = mesh.ds.cellLocation

        if p == 1:
            return cell, cellLocation
        else:
            NC = mesh.number_of_cells()

            ldof = self.number_of_local_dofs()
            cell2dofLocation = np.zeros(NC+1, dtype=np.int_)
            cell2dofLocation[1:] = np.add.accumulate(ldof)
            cell2dof = np.zeros(cell2dofLocation[-1], dtype=np.int_)

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

            ipoint = np.zeros((NN+(p-1)*NE, GD), dtype=np.float64)
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
        self.smspace = ScaledMonomialSpace2d(mesh, p, q=q, bc=bc)
        self.cellmeasure = self.smspace.cellmeasure
        self.dof = CVEMDof2d(mesh, p)

        self.H = self.smspace.matrix_H()
        self.D = self.matrix_D(self.H)
        self.B = self.matrix_B()
        self.G = self.matrix_G(self.B, self.D)

        self.PI1 = self.matrix_PI_1(self.G, self.B)
        self.C = self.matrix_C(self.H, self.PI1)

        self.PI0 = self.matrix_PI_0(self.H, self.C)

        self.integralalg = self.smspace.integralalg
        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype
        self.stype = 'cvem' # 空间类型

    def integral(self, uh):
        """
        计算虚单元函数的积分 \int_\Omega uh dx
        """
 
        p = self.p
        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        if p == 1:
            cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
            f = lambda x: sum(uh[x[0]]*x[0, :])
            val = sum(map(f, zip(cd, self.C)))
            return val
        else:
            NV = self.mesh.number_of_vertices_of_cells()
            idx =cell2dof[cell2dofLocation[0:-1]+NV*p]
            val = np.sum(uh[idx]*self.area)
            return val

    def project_to_smspace(self, uh):
        """
        Project a conforming vem function uh into polynomial space.
        """
        dim = len(uh.shape)
        p = self.p
        cell2dof = self.dof.cell2dof
        cell2dofLocation = self.dof.cell2dofLocation
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        g = lambda x: x[0]@uh[x[1]]
        S = self.smspace.function(dim=dim)
        S[:] = np.concatenate(list(map(g, zip(self.PI1, cd))))
        return S

    def grad_recovery(self, uh):

        p = self.p
        smldof = self.smspace.number_of_local_dofs()
        NC = self.mesh.number_of_cells()
        h = self.smspace.cellsize

        s = self.project_to_smspace(uh).reshape(-1, smldof)
        sx = np.zeros((NC, smldof), dtype=self.ftype)
        sy = np.zeros((NC, smldof), dtype=self.ftype)

        start = 1
        r = np.arange(1, p+1)
        for i in range(p):
            sx[:, start-i-1:start] = r[i::-1]*s[:, start:start+i+1]
            sy[:, start-i-1:start] = r[0:i+1]*s[:, start+1:start+i+2]
            start += i+2

        sx /= h.reshape(-1, 1)
        sy /= h.reshape(-1, 1)

        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        NC = len(cell2dofLocation) - 1
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        DD = np.vsplit(self.D, cell2dofLocation[1:-1])

        f1 = lambda x: x[0]@x[1]
        sx = np.concatenate(list(map(f1, zip(DD, sx))))
        sy = np.concatenate(list(map(f1, zip(DD, sy))))


        ldof = self.number_of_local_dofs()
        w = np.repeat(1/self.smspace.cellsize, ldof)
        sx *= w
        sy *= w

        uh = self.function(dim=2)
        ws = np.zeros(uh.shape[0], dtype=self.ftype)
        np.add.at(uh[:, 0], cell2dof, sx)
        np.add.at(uh[:, 1], cell2dof, sy)
        np.add.at(ws, cell2dof, w)
        uh /=ws.reshape(-1, 1)
        return uh

    def recovery_estimate(self, uh, pde, method='simple', residual=True,
            returnsup=False):
        """
        estimate the recover-type error

        Parameters
        ----------
        self : PoissonVEMModel object
        rtype : str
            'simple':
            'area'
            'inv_area'

        See Also
        --------

        Notes
        -----

        """
        mesh = self.mesh
        NC = mesh.number_of_cells()
        NV = mesh.number_of_vertices_of_cells()
        cell, cellLocation = mesh.entity('cell')
        barycenter = self.smspace.cellbarycenter

        h = self.smspace.cellsize
        area = self.cellmeasure
        ldof = self.smspace.number_of_local_dofs()

        # project the vem solution into linear polynomial space
        idx = np.repeat(range(NC), NV)
        S = self.project_to_smspace(uh)

        grad = S.grad_value(barycenter)
        S0 = self.smspace.function()
        S1 = self.smspace.function()
        n2c = mesh.ds.node_to_cell()

        if method == 'simple':
            d = n2c.sum(axis=1)
            ruh = np.asarray((n2c@grad)/d.reshape(-1, 1))
        elif method == 'area':
            d = n2c@area
            ruh = np.asarray((n2c@(grad*area.reshape(-1, 1)))/d.reshape(-1, 1))
        elif method == 'inv_area':
            d = n2c@(1/area)
            ruh = np.asarray((n2c@(grad/area.reshape(-1,1)))/d.reshape(-1, 1))
        else:
            raise ValueError("I have note code method: {}!".format(rtype))

        for i in range(ldof):
            S0[i::ldof] = np.bincount(
                    idx,
                    weights=self.B[i, :]*ruh[cell, 0],
                    minlength=NC)
            S1[i::ldof] = np.bincount(
                    idx,
                    weights=self.B[i, :]*ruh[cell, 1],
                    minlength=NC)

        k = 1 # TODO: for general diffusion coef

        node = mesh.node
        gx = S0.value(node[cell], idx) - np.repeat(grad[:, 0], NV)
        gy = S1.value(node[cell], idx) - np.repeat(grad[:, 1], NV)
        eta = k*np.bincount(idx, weights=gx**2+gy**2)/NV*area


        if residual is True:
            fh = self.integralalg.fun_integral(pde.source, True)/area
            g0 = S0.grad_value(barycenter)
            g1 = S1.grad_value(barycenter)
            eta += (fh + k*(g0[:, 0] + g1[:, 1]))**2*area**2

        return np.sqrt(eta)

    def smooth_estimator(self, eta):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()
        NV = mesh.number_of_vertices_of_cells()

        nodeEta = np.zeros(NN, dtype=np.float64)

        cell, cellLocation = mesh.entity('cell')
        NNC = cellLocation[1:] - cellLocation[:-1] #number_of_node_per_cell
        NCN = np.zeros(NN, dtype=np.int_) #number_of_cell_around_node

        number = np.ones(NC, dtype=np.int_)

        
        for i in range(3):
            nodeEta[:]=0
            NCN[:]=0
            k = 0
            while True:
                flag = NNC > k
                if np.all(~flag):
                    break
                np.add.at(nodeEta, cell[cellLocation[:-1][flag]+k], eta[flag])
                np.add.at(NCN, cell[cellLocation[:-1][flag]+k], number[flag])
                k += 1
            nodeEta = nodeEta/NCN
            eta[:] = 0

            k = 0
            while True:
                flag = NNC > k
                if np.all(~flag):
                    break
                eta[flag] = eta[flag] + nodeEta[cell[cellLocation[:-1][flag]+k]]
                k += 1
            eta = eta/NNC
        return eta
        
        

    def project(self, F, space1):
        """
        S is a function in ScaledMonomialSpace2d, this function project  S to 
        MonomialSpace2d.
        """
        space0 = F.space
        def f(x, index):
            return np.einsum(
                    '...im, ...in->...imn',
                    mspace.basis(x, index),
                    smspace.basis(x, index)
                    )
        C = self.integralalg.integral(f, celltype=True)
        H = space1.matrix_H()
        PI0 = inv(H)@C
        SS = mspace.function()
        SS[:] = np.einsum('ikj, ij->ik', PI0, S[smspace.cell_to_dof()]).reshape(-1)
        return SS

    def stiff_matrix(self, cfun=None):
        area = self.smspace.cellmeasure

        def f(x):
            x[0, :] = 0
            return x

        p = self.p
        G = self.G
        D = self.D
        PI1 = self.PI1

        cell2dof, cell2dofLocation = self.cell_to_dof()
        NC = len(cell2dofLocation) - 1
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        DD = np.vsplit(D, cell2dofLocation[1:-1])

        if p == 1:
            tG = np.array([(0, 0, 0), (0, 1, 0), (0, 0, 1)])
            if cfun is None:
                def f1(x):
                    M = np.eye(x[1].shape[1])
                    M -= x[0]@x[1]
                    N = x[1].shape[1]
                    A = np.zeros((N, N))
                    idx = np.arange(N)
                    A[idx, idx] = 2
                    A[idx[:-1], idx[1:]] = -1
                    A[idx[1:], idx[:-1]] = -1
                    A[0, -1] = -1
                    A[-1, 0] = -1
                    return x[1].T@tG@x[1] + M.T@A@M
                f1 = lambda x: x[1].T@tG@x[1] + (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1])
                K = list(map(f1, zip(DD, PI1)))
            else:
                cellbarycenter = self.smspace.cellbarycenter
                k = cfun(cellbarycenter)
                f1 = lambda x: (x[1].T@tG@x[1] + (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1]))*x[2]
                K = list(map(f1, zip(DD, PI1, k)))
        else:
            tG = list(map(f, G))
            if cfun is None:
                f1 = lambda x: x[1].T@x[2]@x[1] + (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1])
                K = list(map(f1, zip(DD, PI1, tG)))
            else:
                cellbarycenter = self.smspace.cellbarycenter
                k = cfun(cellbarycenter)
                f1 = lambda x: (x[1].T@x[2]@x[1] + (np.eye(x[1].shape[1]) - x[0]@x[1]).T@(np.eye(x[1].shape[1]) - x[0]@x[1]))*x[3]
                K = list(map(f1, zip(DD, PI1, tG, k)))

        f2 = lambda x: np.repeat(x, x.shape[0])
        f3 = lambda x: np.tile(x, x.shape[0])
        f4 = lambda x: x.flatten()

        I = np.concatenate(list(map(f2, cd)))
        J = np.concatenate(list(map(f3, cd)))
        val = np.concatenate(list(map(f4, K)))
        gdof = self.number_of_global_dofs()
        A = csr_matrix((val, (I, J)), shape=(gdof, gdof), dtype=np.float64)
        return A

    def mass_matrix(self, cfun=None):
        area = self.smspace.cellmeasure
        p = self.p

        PI0 = self.PI0
        D = self.D
        H = self.H
        C = self.C

        cell2dof, cell2dofLocation = self.cell_to_dof()
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
        M = csr_matrix((val, (I, J)), shape=(gdof, gdof), dtype=np.float64)
        return M

    def cross_mass_matrix(self, wh):
        p = self.p
        mesh = self.mesh

        area = self.smspace.cellmeasure
        PI0 = self.PI0

        phi = self.smspace.basis
        def u(x, index):
            val = phi(x, index=index)
            wval = wh(x, index=index)
            return np.einsum('ij, ijm, ijn->ijmn', wval, val, val)
        H = self.integralalg.integral(u, celltype=True)

        cell2dof, cell2dofLocation = self.cell_to_dof()
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
        M = csr_matrix((val, (I, J)), shape=(gdof, gdof), dtype=np.float64)
        return M

    def source_vector(self, f):
        PI0 = self.PI0
        phi = self.smspace.basis
        def u(x, index):
            return np.einsum('ij, ijm->ijm', f(x), phi(x, index=index))
        bb = self.integralalg.integral(u, celltype=True)
        g = lambda x: x[0].T@x[1]
        bb = np.concatenate(list(map(g, zip(PI0, bb))))
        gdof = self.number_of_global_dofs()
        b = np.bincount(self.dof.cell2dof, weights=bb, minlength=gdof)
        return b

    def chen_stability_term(self):
        area = self.smspace.cellmeasure

        p = self.p
        G = self.G
        D = self.D
        PI1 = self.PI1

        cell2dof, cell2dofLocation = self.cell_to_dof()
        NC = len(cell2dofLocation) - 1
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        DD = np.vsplit(D, cell2dofLocation[1:-1])

        tG = np.array([(0, 0, 0), (0, 1, 0), (0, 0, 1)])
        def f1(x):
            M = np.eye(x[1].shape[1])
            M -= x[0]@x[1]
            N = x[1].shape[1]
            A = np.zeros((N, N))
            idx = np.arange(N)
            A[idx, idx] = 2
            A[idx[:-1], idx[1:]] = -1
            A[idx[1:], idx[:-1]] = -1
            A[0, -1] = -1
            A[-1, 0] = -1
            return x[1].T@tG@x[1],  M.T@A@M
        K = list(map(f1, zip(DD, PI1)))
        f2 = lambda x: np.repeat(x, x.shape[0])
        f3 = lambda x: np.tile(x, x.shape[0])
        f4 = lambda x: x[0].flatten()
        f5 = lambda x: x[1].flatten()

        I = np.concatenate(list(map(f2, cd)))
        J = np.concatenate(list(map(f3, cd)))
        val0 = np.concatenate(list(map(f4, K)))
        val1 = np.concatenate(list(map(f5, K)))
        gdof = self.number_of_global_dofs()
        A = csr_matrix((val0, (I, J)), shape=(gdof, gdof), dtype=np.float64)
        S = csr_matrix((val1, (I, J)), shape=(gdof, gdof), dtype=np.float64)
        return A, S

    def cell_to_dof(self):
        return self.dof.cell2dof, self.dof.cell2dofLocation

    def boundary_dof(self, threshold=None):
        return self.dof.boundary_dof(threshold=threshold)

    def edge_basis(self, bc):
        pass

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

    def set_dirichlet_bc(self, gD, uh, threshold=None):
        """
        初始化解 uh  的第一类边界条件。
        """
        p = self.p
        NN = self.mesh.number_of_nodes()
        NE = self.mesh.number_of_edges()
        end = NN + (p - 1)*NE
        ipoints = self.interpolation_points()
        isDDof = self.boundary_dof(threshold=threshold)
        uh[isDDof] = gD(ipoints[isDDof[:end]])
        return isDDof

    def interpolation(self, u, HB=None):
        """
        u: 可以是一个连续函数， 也可以是一个缩放单项式函数
        """
        if HB is None:
            mesh = self.mesh
            NN = mesh.number_of_nodes()
            NE = mesh.number_of_edges()
            p = self.p
            ipoint = self.dof.interpolation_points()
            uI = self.function()
            uI[:NN+(p-1)*NE] = u(ipoint)
            if p > 1:
                phi = self.smspace.basis
                def f(x, index):
                    return np.einsum(
                            'ij, ij...->ij...',
                            u(x), phi(x, index=index, p=p-2))
                bb = self.integralalg.integral(f, celltype=True)/self.smspace.cellmeasure[..., np.newaxis]
                uI[NN+(p-1)*NE:] = bb.reshape(-1)
            return uI
        else:
            uh = self.smspace.interpolation(u, HB)

            cell2dof, cell2dofLocation = self.cell_to_dof()
            NC = len(cell2dofLocation) - 1
            cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
            DD = np.vsplit(self.D, cell2dofLocation[1:-1])

            smldof = self.smspace.number_of_local_dofs()
            f1 = lambda x: x[0]@x[1]
            uh = np.concatenate(list(map(f1, zip(DD, uh.reshape(-1, smldof)))))

            ldof = self.number_of_local_dofs()
            w = np.repeat(1/self.smspace.cellmeasure, ldof)
            uh *= w

            uI = self.function()
            ws = np.zeros(uI.shape[0], dtype=self.ftype)
            np.add.at(uI, cell2dof, uh)
            np.add.at(ws, cell2dof, w)
            uI /=ws
            return uI


    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

    def interpolation_points(self):
        return self.dof.interpolation_points()

    def projection(self, u, up):
        pass

    def array(self, dim=None, dtype=np.float64):
        gdof = self.number_of_global_dofs()
        if dim is None:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=dtype)

    def matrix_D(self, H):
        p = self.p
        smldof = self.smspace.number_of_local_dofs()
        mesh = self.mesh
        NV = mesh.number_of_vertices_of_cells()
        h = self.smspace.cellsize
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        cell, cellLocation = mesh.entity('cell')
        #cell = mesh.ds._cell
        #cellLocation = mesh.ds.cellLocation
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        cell2dof, cell2dofLocation = self.cell_to_dof()
        D = np.ones((len(cell2dof), smldof), dtype=np.float64)

        if p == 1:
            bc = np.repeat(self.smspace.cellbarycenter, NV, axis=0)
            D[:, 1:] = (node[cell, :] - bc)/np.repeat(h, NV).reshape(-1, 1)
            return D

        qf = GaussLobattoQuadrature(p+1)
        bcs, ws = qf.quadpts, qf.weights
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        phi0 = self.smspace.basis(ps[:-1], index=edge2cell[:, 0])
        phi1 = self.smspace.basis(ps[p:0:-1, isInEdge, :], index=edge2cell[isInEdge, 1])
        idx = cell2dofLocation[edge2cell[:, 0]] + edge2cell[:, 2]*p + np.arange(p).reshape(-1, 1)
        D[idx, :] = phi0
        idx = cell2dofLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*p + np.arange(p).reshape(-1, 1)
        D[idx, :] = phi1
        if p > 1:
            area = self.smspace.cellmeasure
            idof = (p-1)*p//2 # the number of dofs of scale polynomial space with degree p-2
            idx = cell2dofLocation[1:].reshape(-1, 1) + np.arange(-idof, 0)
            D[idx, :] = H[:, :idof, :]/area.reshape(-1, 1, 1)
        return D

    def matrix_B(self):
        p = self.p
        smldof = self.smspace.number_of_local_dofs()
        mesh = self.mesh
        NV = mesh.number_of_vertices_of_cells()
        h = self.smspace.cellsize
        cell2dof, cell2dofLocation = self.cell_to_dof()
        B = np.zeros((smldof, cell2dof.shape[0]), dtype=np.float64)
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

            node = mesh.entity('node')
            edge = mesh.entity('edge')
            edge2cell = mesh.ds.edge_to_cell()
            isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

            qf = GaussLobattoQuadrature(p + 1)
            bcs, ws = qf.quadpts, qf.weights
            ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
            gphi0 = self.smspace.grad_basis(ps, index=edge2cell[:, 0])
            gphi1 = self.smspace.grad_basis(ps[-1::-1, isInEdge, :], index=edge2cell[isInEdge, 1])
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
            cell2dof, cell2dofLocation = self.cell_to_dof()
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
            return list(map(l, zip(NV, self.smspace.cellmeasure, C)))

    def matrix_PI_0(self, H, C):
        cell2dof, cell2dofLocation = self.cell_to_dof()
        pi0 = lambda x: inv(x[0])@x[1]
        return list(map(pi0, zip(H, C)))

    def matrix_PI_1(self, G, B):
        p = self.p
        cell2dof, cell2dofLocation = self.cell_to_dof()
        if p == 1:
            return np.hsplit(B, cell2dofLocation[1:-1])
        else:
            BB = np.hsplit(B, cell2dofLocation[1:-1])
            g = lambda x: inv(x[0])@x[1]
            return list(map(g, zip(G, BB)))

