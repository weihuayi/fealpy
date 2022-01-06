
import numpy as np
from numpy.linalg import inv
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from .Function import Function
from ..decorator import cartesian
from .ScaledMonomialSpace3d import ScaledMonomialSpace3d


class WGDof3d():
    """
    The dof manager of weak galerkin 2d space.
    """
    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        self.cell2dof, self.cell2dofLocation = self.cell_to_dof()
        self.multiIndex2d = self.multi_index_matrix2d()

    def multi_index_matrix2d(self):
        ldof = (p+1)*(p+2)//2
        idx = np.arange(0, ldof)
        idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
        multiIndex = np.zeros((ldof, 3), dtype=np.int_)
        multiIndex[:,2] = idx - idx0*(idx0 + 1)/2
        multiIndex[:,1] = idx0 - multiIndex[:,2]
        multiIndex[:,0] = p - multiIndex[:, 1] - multiIndex[:, 2]
        return multiIndex

    def is_boundary_dof(self, threshold=None):
        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_face_index()
            if callable(threshold):
                bc = self.mesh.entity_barycenter('face', index=index)
                flag = threshold(bc)
                index = index[flag]
        gdof = self.number_of_global_dofs()
        isBdDof = np.zeros(gdof, dtype=np.bool)
        edge2dof = self.face_to_dof()
        isBdDof[face2dof[index]] = True
        return isBdDof

    def face_to_dof(self):
        p = self.p
        mesh = self.mesh
        NF = mesh.number_of_faces()
        edge2dof = np.arange(NF*(p+1)).reshape(NE, p+1)
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
        cell2edge = mesh.ds.cell_to_face(return_sparse=False)

        NC = mesh.number_of_cells()

        ldof = self.number_of_local_dofs(doftype='all')
        cdof = self.number_of_local_dofs(doftype='cell')
        fdof = self.number_of_local_dofs(doftype='face')
        cell2dofLocation = np.zeros(NC+1, dtype=np.int)
        cell2dofLocation[1:] = np.add.accumulate(ldof)
        cell2dof = np.zeros(cell2dofLocation[-1], dtype=np.int_)

        face2dof = self.face_to_dof()
        face2cell = mesh.ds.face_to_cell()
        idx = cell2dofLocation[face2cell[:, [0]]] + face2cell[:, [2]]*fdof + np.arange(fdof)
        cell2dof[idx] = face2dof

        isInFace = (face2cell[:, 0] != face2cell[:, 1])
        idx = (cell2dofLocation[face2cell[isInFace, 1]] + face2cell[isInFace,
            3]*fdof).reshape(-1, 1) + np.arange(fdof)
        cell2dof[idx] = face2dof[isInFace]

        NFC = mesh.number_of_faces_of_cells()
        NF = mesh.number_of_faces()
        idx = (cell2dofLocation[:-1] + NFC*fdof).reshape(-1, 1) + np.arange(cdof)
        cell2dof[idx] = NF*fdof + np.arange(NC*cdof).reshape(NC, cdof)
        return cell2dof, cell2dofLocation


    def number_of_global_dofs(self):
        p = self.p
        mesh = self.mesh
        NF = mesh.number_of_edges()
        NC = mesh.number_of_cells()
        cdof = self.number_of_local_dofs(doftype='cell') # 单元内部的自由度
        fdof = self.number_of_local_dofs(doftype='face') # 面内部的自由度
        gdof = NF*fdof + NC*cdof
        return gdof

    def number_of_local_dofs(self, doftype='all'):
        p = self.p
        if doftype == 'all': # number of all dofs on a cell 
            return (p+1)*(p+3) 
        elif doftype in {'cell', 3}: # number of dofs inside the cell 
            return (p+1)*(p+2)*(p+3)//6 
        elif doftype in {'face', 2}: # number of dofs on each face 
            return (p+1)*(p+2)//2
        elif doftype in {'edge', 1}: # number of dofs on each edge
            return 0
        elif doftype in {'node', 0}: # number of dofs on each node
            return 0


class WeakGalerkinSpace3d:
    def __init__(self, mesh, p=1, q=None):
        self.p = p
        self.smspace = ScaledMonomialSpace3d(mesh, p, q=q)
        self.mesh = mesh
        self.cellsize = self.smspace.cellsize

        self.ftype = self.mesh.ftype
        self.itype = self.mesh.itype

        self.dof = WGDof3d(mesh, p)

        self.integralalg = self.smspace.integralalg

        self.CM = self.smspace.cell_mass_matrix()
        self.FM = self.smspace.face_mass_matrix()

        self.H0 = inv(self.CM)
        self.H1 = inv(self.FM)
        self.R = self.left_weak_matrix()

        self.stype = 'wg3d' # 空间类型

    def number_of_local_dofs(self, doftype='all'):
        return self.dof.number_of_local_dofs(doftype=doftype)

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def is_boundary_dof(self, threshold=None):
        return self.dof.is_boundary_dof(threshold=None)

    def face_to_dof(self):
        return self.dof.face_to_dof()

    def cell_to_dof(self, doftype='all'):
        if doftype == 'all':
            return self.dof.cell2dof, self.dof.cell2dofLocation
        elif doftype == 'cell':
            p = self.p
            NF = self.mesh.number_of_faces()
            NC = self.mesh.number_of_cells()
            fdof = self.number_of_local_dofs(doftype='face')
            cdof = self.number_of_local_dofs(doftype='cell')
            cell2dof = NF*fdof + np.arange(NC*cdof).reshape(NC, cdof)
            return cell2dof

    def weak_grad(self, uh):
        cell2dof, cell2dofLocation = self.cell_to_dof()
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        R0 = np.hsplit(self.R[0], cell2dofLocation[1:-1])
        R1 = np.hsplit(self.R[1], cell2dofLocation[1:-1])
        R2 = np.hsplit(self.R[2], cell2dofLocation[1:-1])
        ph = self.smspace.function(dim=2)

        f0 = lambda x: x[0]@(x[1]@uh[x[2]])
        ph[:, 0] = np.concatenate(list(map(f0, zip(self.H0, R0, cd))))
        ph[:, 1] = np.concatenate(list(map(f0, zip(self.H0, R1, cd))))
        ph[:, 2] = np.concatenate(list(map(f0, zip(self.H0, R2, cd))))
        return ph

    def weak_div(self, ph):
        cell2dof, cell2dofLocation = self.cell_to_dof()
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        R0 = np.hsplit(self.R[0], cell2dofLocation[1:-1])
        R1 = np.hsplit(self.R[1], cell2dofLocation[1:-1])
        R2 = np.hsplit(self.R[2], cell2dofLocation[1:-1])
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
        mesh = self.mesh

        qf = self.integralalg.edgeintegrator
        bcs, ws = qf.quadpts, qf.weights

        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        ps = self.mesh.edge_bc_to_point(bcs)
        phi0 = self.basis(ps, index=edge2cell[:, 0])
        phi1 = self.basis(
                ps[:, isInEdge, :],
                index=edge2cell[isInEdge, 1]
                )
        phi = self.edge_basis(ps)

        edge2dof = self.edge_to_dof()
        cell2dof = self.cell_to_dof(doftype='cell')

        h = mesh.entity_measure('edge')
        cellsize = self.cellsize
        h0 = cellsize[edge2cell[:, 0]].reshape(-1, 1, 1)
        h1 = cellsize[edge2cell[isInEdge, 1]].reshape(-1, 1, 1)
        F0 = np.einsum('i, ijm, ijn, j->jmn', ws, phi0, phi, h)/h0
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

