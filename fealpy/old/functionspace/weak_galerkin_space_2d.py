
import numpy as np
from numpy.linalg import inv
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from .Function import Function
from ..quadrature import GaussLegendreQuadrature
from ..quadrature import PolygonMeshIntegralAlg
from ..decorator import cartesian
from .scaled_monomial_space_2d import ScaledMonomialSpace2d


class WGDof2d():
    """
    The dof manager of weak galerkin 2d space.
    """
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        self.cell2dof, self.cell2dofLocation = self.cell_to_dof()
        self.multiIndex1d = self.multi_index_matrix1d()

    def multi_index_matrix1d(self):
        p = self.p
        ldof = p + 1
        multiIndex = np.zeros((ldof, 2), dtype=np.int)
        multiIndex[:, 0] = np.arange(p, -1, -1)
        multiIndex[:, 1] = p - multiIndex[:, 0]
        return multiIndex

    def is_boundary_dof(self, threshold=None):
        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_edge_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter('edge', index=index)
                flag = threshold(bc)
                index = index[flag]
        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        edge2dof = self.edge_to_dof()
        isBdDof[edge2dof[index]] = True
        return isBdDof

    def boundary_dof(self, threshold=None):
        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_edge_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter('edge', index=index)
                flag = threshold(bc)
                index = index[flag]
        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        edge2dof = self.edge_to_dof()
        isBdDof[edge2dof[index]] = True
        return isBdDof

    def edge_to_dof(self):
        p = self.p
        mesh = self.mesh
        NE = mesh.number_of_edges()
        edge2dof = np.arange(NE*(p+1)).reshape(NE, p+1)
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
        cellLocation = mesh.ds.cellLocation
        cell2edge = mesh.ds.cell_to_edge(return_sparse=False)

        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()
        cell2dofLocation = np.zeros(NC+1, dtype=np.int)
        cell2dofLocation[1:] = np.add.accumulate(ldof)
        cell2dof = np.zeros(cell2dofLocation[-1], dtype=np.int)

        edge2dof = self.edge_to_dof()
        edge2cell = mesh.ds.edge_to_cell()
        idx = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]]*(p+1) + np.arange(p+1)
        cell2dof[idx] = edge2dof

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        idx = (cell2dofLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge,
            3]*(p+1)).reshape(-1, 1) + np.arange(p+1)
        cell2dof[idx] = edge2dof[isInEdge]

        NV = mesh.number_of_vertices_of_cells()
        NE = mesh.number_of_edges()
        idof = (p+1)*(p+2)//2
        idx = (cell2dofLocation[:-1] + NV*(p+1)).reshape(-1, 1) + np.arange(idof)
        cell2dof[idx] = NE*(p+1) + np.arange(NC*idof).reshape(NC, idof)
        return cell2dof, cell2dofLocation

    def cell_to_dof_1(self):
        """
        Construct the cell2dof array which are 1D array with a location array
        cell2dofLocation.

        The following code give the dofs of i-th cell.

        cell2dof[cell2dofLocation[i]:cell2dofLocation[i+1]]
        """
        p = self.p
        mesh = self.mesh
        cellLocation = mesh.ds.cellLocation
        cell2edge = mesh.ds.cell_to_edge(return_sparse=False)

        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs()
        cell2dofLocation = np.zeros(NC+1, dtype=np.int)
        cell2dofLocation[1:] = np.add.accumulate(ldof)
        cell2dof = np.zeros(cell2dofLocation[-1], dtype=np.int)

        edge2dof = self.edge_to_dof()
        edge2cell = mesh.ds.edge_to_cell()
        idx = cell2dofLocation[edge2cell[:, [0]]] + edge2cell[:, [2]]*(p+1) + np.arange(p+1)
        cell2dof[idx] = edge2dof

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        idx = (cell2dofLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge,
            3]*(p+1)).reshape(-1, 1) + np.arange(p+1)
        cell2dof[idx] = edge2dof[isInEdge, p::-1]

        NV = mesh.number_of_vertices_of_cells()
        NE = mesh.number_of_edges()
        idof = (p+1)*(p+2)//2
        idx = (cell2dofLocation[:-1] + NV*(p+1)).reshape(-1, 1) + np.arange(idof)
        cell2dof[idx] = NE*(p+1) + np.arange(NC*idof).reshape(NC, idof)
        return cell2dof, cell2dofLocation

    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        gdof = NE*(p+1) + NC*(p+1)*(p+2)//2
        return gdof

    def number_of_local_dofs(self):
        p = self.p
        mesh = self.mesh
        NCE = mesh.number_of_edges_of_cells()
        ldofs = NCE*(p+1) + (p+1)*(p+2)//2
        return ldofs


class WeakGalerkinSpace2d:
    def __init__(self, mesh, p=1, q=None):
        self.p = p
        self.smspace = ScaledMonomialSpace2d(mesh, p, q=q)
        self.mesh = mesh
        self.cellsize = self.smspace.cellsize

        self.ftype = self.mesh.ftype
        self.itype = self.mesh.itype

        self.dof = WGDof2d(mesh, p)

        self.integralalg = self.smspace.integralalg

        self.CM = self.smspace.cell_mass_matrix()
        self.EM = self.smspace.edge_mass_matrix()

        self.H0 = inv(self.CM)
        self.H1 = inv(self.EM)
        self.R = self.left_weak_matrix()

        self.stype = 'wg' # 空间类型

    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def is_boundary_dof(self, threshold=None):
        return self.dof.is_boundary_dof(threshold=None)

    def boundary_dof(self, threshold=None):
        return self.dof.boundary_dof(threshold=None)

    def edge_to_dof(self):
        return self.dof.edge_to_dof()

    def cell_to_dof(self, doftype='all'):
        if doftype == 'all':
            return self.dof.cell2dof, self.dof.cell2dofLocation
        elif doftype == 'cell':
            p = self.p
            NE = self.mesh.number_of_edges()
            NC = self.mesh.number_of_cells()
            idof = (p+1)*(p+2)//2
            cell2dof = NE*(p+1) + np.arange(NC*idof).reshape(NC, idof)
            return cell2dof

    def weak_grad(self, uh):
        cell2dof, cell2dofLocation = self.cell_to_dof()
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        R0 = np.hsplit(self.R[0], cell2dofLocation[1:-1])
        R1 = np.hsplit(self.R[1], cell2dofLocation[1:-1])
        ph = self.smspace.function(dim=2)

        f0 = lambda x: x[0]@(x[1]@uh[x[2]])
        ph[:, 0] = np.concatenate(list(map(f0, zip(self.H0, R0, cd))))
        ph[:, 1] = np.concatenate(list(map(f0, zip(self.H0, R1, cd))))
        return ph

    def weak_div(self, ph):
        cell2dof, cell2dofLocation = self.cell_to_dof()
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        R0 = np.hsplit(self.R[0], cell2dofLocation[1:-1])
        R1 = np.hsplit(self.R[1], cell2dofLocation[1:-1])
        dh = self.smspace.function()
        f0 = lambda x: x[0]@(x[1]@ph[x[3], 0] + x[2]@ph[x[3], 1])
        dh[:] = np.concatenate(list(map(f0, zip(self.H0, R0, R1, cd))))
        return dh

    def left_weak_matrix(self):
        """
        计算单元上的弱梯度和弱散度投影算子的右端矩阵
        """
        p = self.p
        mesh = self.mesh
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        h = self.integralalg.edgemeasure
        n = mesh.edge_unit_normal()

        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        phi0 = self.smspace.basis(ps, index=edge2cell[:, 0]) # (NQ, NE, ldof)
        phi1 = self.smspace.basis(
                ps[:, isInEdge, :],
                index=edge2cell[isInEdge, 1]
                ) # (NQ, NE, ldof)
        phi = self.edge_basis(ps)

        F0 = np.einsum('i, ijm, ijn, j->mjn', ws, phi0, phi, h)
        F1 = np.einsum('i, ijm, ijn, j->mjn', ws, phi1, phi[:, isInEdge, :], h[isInEdge])

        #F0 = np.einsum('i, ijm, in, j->mjn', ws, phi0, phi, h)
        #F1 = np.einsum('i, ijm, in, j->mjn', ws, phi1, phi[:, -1::-1], h[isInEdge])

        smldof = self.smspace.number_of_local_dofs()
        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        R0 = np.zeros((smldof, len(cell2dof)), dtype=np.float)
        R1 = np.zeros((smldof, len(cell2dof)), dtype=np.float)

        idx = cell2dofLocation[edge2cell[:, [0]]] + \
                edge2cell[:, [2]]*(p+1) + np.arange(p+1)
        R0[:, idx] = n[np.newaxis, :, [0]]*F0
        R1[:, idx] = n[np.newaxis, :, [1]]*F0
        if isInEdge.sum() > 0:
            idx = cell2dofLocation[edge2cell[isInEdge, 1]].reshape(-1, 1) + \
                    (p+1)*edge2cell[isInEdge, [3]].reshape(-1, 1) + np.arange(p+1)
            n = n[isInEdge]
            R0[:, idx] = -n[np.newaxis, :, [0]]*F1 # 这里应该加上负号
            R1[:, idx] = -n[np.newaxis, :, [1]]*F1 # 这里应该加上负号

        def f(x, index):
            gphi = self.grad_basis(x, index)
            phi = self.basis(x, index)
            return np.einsum(
                    '...mn, ...k->...nmk',
                    gphi, phi)
        M = self.integralalg.integral(f, celltype=True)
        idof = (p+1)*(p+2)//2
        idx = cell2dofLocation[1:].reshape(-1, 1) + np.arange(-idof, 0)
        R0[:, idx] = -M[:, 0].swapaxes(0, 1)
        R1[:, idx] = -M[:, 1].swapaxes(0, 1)
        return R0, R1

    def stiff_matrix(self):
        gdof = self.number_of_global_dofs()
        cell2dof, cell2dofLocation = self.cell_to_dof()
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        H0 = self.H0
        R = self.R
        def f0(i):
            R0 = R[0][:, cell2dofLocation[i]:cell2dofLocation[i+1]]
            R1 = R[1][:, cell2dofLocation[i]:cell2dofLocation[i+1]]
            return R0.T@H0[i]@R0, R1.T@H0[i]@R1, R0.T@H0[i]@R1

        NC = self.mesh.number_of_cells()
        M = list(map(f0, range(NC)))

        idx = list(map(np.meshgrid, cd, cd))
        I = np.concatenate(list(map(lambda x: x[1].flat, idx)))
        J = np.concatenate(list(map(lambda x: x[0].flat, idx)))

        val = np.concatenate(list(map(lambda x: x[0].flat, M)))
        M00 = csr_matrix((val, (I, J)), shape=(gdof, gdof))

        val = np.concatenate(list(map(lambda x: x[1].flat, M)))
        M11 = csr_matrix((val, (I, J)), shape=(gdof, gdof))

        A = M00 + M11 # weak gradient matrix
        return A

    def mass_matrix(self):
        cell2dof = self.cell_to_dof(doftype='cell') # only get the dofs in cell
        ldof = cell2dof.shape[1]
        gdof = self.number_of_global_dofs()
        I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
        J = I.swapaxes(-1, -2)
        M = csr_matrix(
                (self.CM.flat, (I.flat, J.flat)), shape=(gdof, gdof)
                )
        return M

    def weak_grad_matrix(self):
        pass

    def weak_div_matrix(self):
        pass

    def stabilizer_matrix(self):
        """
        Note
        ----
            WG 方法的稳定子矩阵
        """

        mesh = self.mesh

        qf = self.integralalg.edgeintegrator
        bcs, ws = qf.quadpts, qf.weights

        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        ps = self.mesh.edge_bc_to_point(bcs) # (NQ, NE, 2)
        phi0 = self.basis(ps, index=edge2cell[:, 0]) # (NQ, NE, cldof)
        phi1 = self.basis(
                ps[:, isInEdge, :],
                index=edge2cell[isInEdge, 1]
                )

        phi = self.edge_basis(ps) # (NQ, NE, eldof)

        h = mesh.entity_measure('edge')
        cellsize = self.cellsize
        h0 = cellsize[edge2cell[:, 0]].reshape(-1, 1, 1)
        h1 = cellsize[edge2cell[isInEdge, 1]].reshape(-1, 1, 1)
        F0 = np.einsum('i, ijm, ijn, j->jmn', ws, phi0, phi, h)/h0 # (NE, cldof, eldof)
        F1 = np.einsum('i, ijm, ijn, j->jmn', ws, phi1, phi[:, isInEdge, :], h[isInEdge])/h1

        F2 = np.einsum('i, ijm, ijn, j->jmn', ws, phi0, phi0, h)/h0
        F3 = np.einsum('i, ijm, ijn, j->jmn', ws, phi1, phi1, h[isInEdge])/h1

        F4 = np.einsum('i, ijm, ijn, j->jmn', ws, phi, phi, h)
        F5 = F4[isInEdge]/h1
        F4 /=h0


        edge2dof = self.edge_to_dof()
        cell2dof = self.cell_to_dof(doftype='cell')

        cdof = cell2dof.shape[1]
        edof = edge2dof.shape[1]
        gdof = self.number_of_global_dofs()
        S = csr_matrix((gdof, gdof), dtype=self.ftype)

        I = np.einsum('ij, k->ijk', cell2dof[edge2cell[:, 0]], np.ones(edof))
        J = np.einsum('ik, j->ijk', edge2dof, np.ones(cdof))
        S -= csr_matrix((F0.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        S -= csr_matrix((F0.flat, (J.flat, I.flat)), shape=(gdof, gdof))

        I = np.einsum('ij, k->ijk', cell2dof[edge2cell[isInEdge, 1]], np.ones(edof))
        J = np.einsum('ik, j->ijk', edge2dof[isInEdge], np.ones(cdof))
        S -= csr_matrix((F1.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        S -= csr_matrix((F1.flat, (J.flat, I.flat)), shape=(gdof, gdof))

        I = np.einsum('ij, k->ijk', cell2dof[edge2cell[:, 0]], np.ones(cdof))
        J = I.swapaxes(-1, -2)
        S += csr_matrix((F2.flat, (I.flat, J.flat)), shape=(gdof, gdof))

        I = np.einsum('ij, k->ijk', cell2dof[edge2cell[isInEdge, 1]], np.ones(cdof))
        J = I.swapaxes(-1, -2)
        S += csr_matrix((F3.flat, (I.flat, J.flat)), shape=(gdof, gdof))

        I = np.einsum('ij, k->ijk', edge2dof, np.ones(edof))
        J = I.swapaxes(-1, -2)
        S += csr_matrix((F4.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        I = np.einsum('ij, k->ijk', edge2dof[isInEdge], np.ones(edof))
        J = I.swapaxes(-1, -2)
        S += csr_matrix((F5.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return S

    def set_dirichlet_bc(self, gD, uh, threshold=None):
        """
        初始化解 uh  的第一类边界条件。
        """
        mesh = self.mesh
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        isBdEdge = mesh.ds.boundary_edge_flag()
        isBdDof = self.boundary_dof()

        qf = GaussLegendreQuadrature(self.p + 3)
        bcs, ws = qf.quadpts, qf.weights
        ps = mesh.edge_bc_to_point(bcs, index=isBdEdge)
        gI = gD(ps)
        ephi = self.edge_basis(ps, index=isBdEdge)
        h = mesh.entity_measure('edge')
        b = np.einsum('i, ij, ijk, j->jk', ws, gI, ephi, h[isBdEdge])
        uh[isBdDof] = np.einsum('ijk, ik->ij', self.H1[isBdEdge], b).flat
        return isBdDof

    @cartesian
    def basis(self, point, index=None):
        return self.smspace.basis(point, index=index)

    @cartesian
    def grad_basis(self, point, index=None):
        return self.smspace.grad_basis(point, index=index)

    @cartesian
    def value(self, uh, point, index=None):
        NE = self.mesh.number_of_edges()
        p = self.p
        return self.smspace.value(uh[NE*(p+1):, ...], point, index=index)

    @cartesian
    def grad_value(self, uh, point, index=None):
        NE = self.mesh.number_of_edges()
        p = self.p
        return self.smspace.grad_value(uh[NE*(p+1):, ...], point, index=index)

    @cartesian
    def edge_basis(self, point, index=None):
        return self.smspace.edge_basis(point, index=index)

    def lagrange_edge_basis(self, bc):
        p = self.p   # the degree of polynomial basis function
        multiIndex = self.dof.multiIndex1d
        c = np.arange(1, p+1, dtype=np.int)
        P = 1.0/np.multiply.accumulate(c)
        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, 2)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = np.arange(2)
        phi = np.prod(A[..., multiIndex, idx], axis=-1)
        return phi

    def edge_value(self, uh, bcs):
        NE = self.mesh.number_of_edges()
        p = self.p
        phi = self.edge_basis(bcs)
        edge2dof = self.dof.edge_to_dof()

        dim = len(uh.shape) - 1
        s0 = 'abcdefg'[:dim]
        s1 = '...ij, ij{}->...i{}'.format(s0, s0)
        #s1 = '...j, ij{}->...i{}'.format(s0, s0)
        val = np.einsum(s1, phi, uh[edge2dof])
        return val

    def source_vector(self, f):
        phi = self.basis
        def u(x, index):
            return np.einsum('ij, ijm->ijm', f(x), phi(x, index=index))
        bb = self.integralalg.integral(u, celltype=True)
        gdof = self.number_of_global_dofs()
        cell2dof = self.cell_to_dof(doftype='cell')
        b = np.bincount(cell2dof.flat, weights=bb.flat, minlength=gdof)
        return b

    def project(self, u, dim=1):
        p = self.p
        mesh = self.mesh
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        h = mesh.entity_measure('edge')
        NE = mesh.number_of_edges()

        uh = self.function(dim=dim)

        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        uI = u(ps)

        ephi = self.edge_basis(ps)
        b = np.einsum('i, ij..., ijk, j->jk...', ws, uI, ephi, h)
        if dim == 1:
            uh[:NE*(p+1), ...].flat = (self.H1@b[:, :, np.newaxis]).flat
        else:
            uh[:NE*(p+1), ...].flat = (self.H1@b).flat

        t = 'd'
        s = '...{}, ...m->...m{}'.format(t[:dim>1], t[:dim>1])
        def f1(x, index):
            phi = self.basis(x, index)
            return np.einsum(s, u(x), phi)

        b = self.integralalg.integral(f1, celltype=True)
        if dim == 1:
            uh[NE*(p+1):, ...].flat = (self.H0@b[:, :, np.newaxis]).flat
        else:
            uh[NE*(p+1):, ...].flat = (self.H0@b).flat
        return uh


    def function(self, dim=None, array=None, dtype=np.float64):
        f = Function(self, dim=dim, array=array, coordtype='cartesian',
                dtype=dtype)
        return f

    def array(self, dim=None, dtype=np.float64):
        gdof = self.number_of_global_dofs()
        if dim in {None, 1}:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=dtype)

