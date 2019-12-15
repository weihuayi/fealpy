
import numpy as np
from numpy.linalg import inv
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from .function import Function
from ..quadrature import GaussLegendreQuadrature
from ..quadrature import PolygonMeshIntegralAlg
from .ScaledMonomialSpace2d import ScaledMonomialSpace2d


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
        cell2edge = mesh.ds.cell_to_edge(sparse=False)

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
    def __init__(self, mesh, p=1):
        self.p = p
        self.smspace = ScaledMonomialSpace2d(mesh, p)
        self.mesh = mesh

        self.ftype = self.mesh.ftype
        self.itype = self.mesh.itype

        self.dof = WGDof2d(mesh, p)

        self.integrator = self.smspace.integrator
        self.integralalg = self.smspace.integralalg
        self.H0 = inv(self.smspace.mass_matrix())
        self.H1 = inv(self.edge_mass_matrix())
        self.R0, self.R1 = self.weak_matrix()

    def number_of_local_dofs(self):
        return self.dof.number_of_local_dofs()

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def edge_mass_matrix(self):
        p = self.p
        mesh = self.mesh
        edge = mesh.entity('edge')
        measure = mesh.entity_measure('edge')
        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights
        phi = self.edge_basis(bcs)
        H = np.einsum('i, ik, im, j->jkm', ws, phi, phi, measure, optimize=True)
        return H

    def cell_to_dof(self):
        return self.dof.cell2dof, self.dof.cell2dofLocation

    def weak_grad(self, uh):
        cell2dof, cell2dofLocation = self.cell_to_dof()
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        R0 = np.hsplit(self.R0, cell2dofLocation[1:-1])
        R1 = np.hsplit(self.R1, cell2dofLocation[1:-1])
        ph = self.smspace.function(dim=2)

        f0 = lambda x: x[0]@(x[1]@uh[x[2]])
        ph[:, 0] = np.concatenate(list(map(f0, zip(self.H0, R0, cd))))
        ph[:, 1] = np.concatenate(list(map(f0, zip(self.H0, R1, cd))))
        return ph

    def weak_div(self, ph):
        cell2dof, cell2dofLocation = self.cell_to_dof()
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        R0 = np.hsplit(self.R0, cell2dofLocation[1:-1])
        R1 = np.hsplit(self.R1, cell2dofLocation[1:-1])
        dh = self.smspace.function()
        f0 = lambda x: x[0]@(x[1]@ph[x[3], 0] + x[2]@ph[x[3], 1])
        dh[:] = np.concatenate(list(map(f0, zip(self.H0, R0, R1, cd))))
        return dh

    def weak_matrix(self):
        """
        计算单元上的弱梯度和弱散度算子的右端矩阵
        """
        p = self.p
        mesh = self.mesh
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        h = mesh.entity_measure('edge')
        n = mesh.edge_unit_normal()

        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        phi0 = self.smspace.basis(ps, cellidx=edge2cell[:, 0])
        phi1 = self.smspace.basis(
                ps[:, isInEdge, :],
                cellidx=edge2cell[isInEdge, 1]
                )
        phi = self.edge_basis(bcs)

        F0 = np.einsum('i, ijm, in, j->mjn', ws, phi0, phi, h)
        F1 = np.einsum('i, ijm, in, j->mjn', ws, phi1, phi[:, -1::-1], h[isInEdge])

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

        def f(x, cellidx):
            gphi = self.grad_basis(x, cellidx)
            phi = self.basis(x, cellidx)
            return np.einsum(
                    '...mn, ...k->...nmk',
                    gphi, phi)
        M = self.integralalg.integral(f, celltype=True)
        idof = (p+1)*(p+2)//2
        idx = cell2dofLocation[1:].reshape(-1, 1) + np.arange(-idof, 0)
        R0[:, idx] = -M[:, 0].swapaxes(0, 1)
        R1[:, idx] = -M[:, 1].swapaxes(0, 1)
        return R0, R1

    def basis(self, point, cellidx=None):
        return self.smspace.basis(point, cellidx=cellidx)

    def grad_basis(self, point, cellidx=None):
        return self.smspace.grad_basis(point, cellidx=cellidx)

    def value(self, uh, point, cellidx=None):
        NE = self.mesh.number_of_edges()
        p = self.p
        return self.smspace.value(uh[NE*(p+1):, ...], point, cellidx=cellidx)

    def grad_value(self, uh, point, cellidx=None):
        NE = self.mesh.number_of_edges()
        p = self.p
        return self.smspace.grad_value(uh[NE*(p+1):, ...], point, cellidx=cellidx)

    def edge_basis(self, bc):
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
        s1 = '...j, ij{}->...i{}'.format(s0, s0)
        val = np.einsum(s1, phi, uh[edge2dof])
        return val

    def projection(self, u, dim=1):
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

        ephi = self.edge_basis(bcs)
        b = np.einsum('i, ij..., ik, j->jk...', ws, uI, ephi, h)
        if dim == 1:
            uh[:NE*(p+1), ...].flat = (self.H1@b[:, :, np.newaxis]).flat
        else:
            uh[:NE*(p+1), ...].flat = (self.H1@b).flat

        t = 'd'
        s = '...{}, ...m->...m{}'.format(t[:dim>1], t[:dim>1])
        def f1(x, cellidx):
            phi = self.basis(x, cellidx)
            return np.einsum(s, u(x), phi)

        b = self.integralalg.integral(f1, celltype=True)
        if dim == 1:
            uh[NE*(p+1):, ...].flat = (self.H0@b[:, :, np.newaxis]).flat
        else:
            uh[NE*(p+1):, ...].flat = (self.H0@b).flat
        return uh


    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array)
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim in [None, 1]:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=np.float)

