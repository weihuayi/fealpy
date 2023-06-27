
import numpy as np
from numpy.linalg import inv
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, spdiags, eye

from .Function import Function
from ..decorator import cartesian
from .scaled_monomial_space_3d import ScaledMonomialSpace3d

class WGDof3d:
    """
    The dof manager of weak galerkin 3d space.
    """

    def __init__(self, mesh, p):
        self.p = p
        self.mesh = mesh
        self.cell2dof, self.cell2dofLocation = self.cell_to_dof()
        self.multiIndex2d = self.multi_index_matrix2d()

    def multi_index_matrix2d(self):
        p = self.p
        ldof = self.number_of_local_dofs(doftype='face')
        idx = np.arange(0, ldof)
        idx0 = np.floor((-1 + np.sqrt(1 + 8 * idx)) / 2)
        multiIndex = np.zeros((ldof, 3), dtype=np.int_)
        multiIndex[:, 2] = idx - idx0 * (idx0 + 1) / 2
        multiIndex[:, 1] = idx0 - multiIndex[:, 2]
        multiIndex[:, 0] = p - multiIndex[:, 1] - multiIndex[:, 2]
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
        isBdDof = np.zeros(gdof, dtype=np.bool_)
        face2dof = self.face_to_dof()
        isBdDof[face2dof[index]] = True
        return isBdDof

    def face_to_dof(self):
        mesh = self.mesh
        fdof = self.number_of_local_dofs(doftype='face')
        NF = mesh.number_of_faces()
        face2dof = np.arange(NF * fdof).reshape(NF, fdof)
        return face2dof

    def cell_to_dof(self):
        """
        Construct the cell2dof array which are 1D array with a location array
        cell2dofLocation.

        The following code give the dofs of i-th cell.

        cell2dof[cell2dofLocation[i]:cell2dofLocation[i+1]]
        """
        mesh = self.mesh
        ldof = self.number_of_local_dofs(doftype='all')
        cdof = self.number_of_local_dofs(doftype='cell')
        fdof = self.number_of_local_dofs(doftype='face')

        NC = mesh.number_of_cells()

        cell2dofLocation = np.zeros(NC + 1, dtype=np.int_)  # NC-1 + 2 中间+两头 用于记录每个单元自由度的起始位置
        cell2dofLocation[1:] = np.add.accumulate(ldof)
        cell2dof = np.zeros(cell2dofLocation[-1], dtype=np.int_)

        face2dof = self.face_to_dof()
        face2cell = mesh.ds.face_to_cell()
        idx = cell2dofLocation[face2cell[:, [0]]] + face2cell[:, [2]] * fdof + np.arange(fdof)
        cell2dof[idx] = face2dof

        isInFace = (face2cell[:, 0] != face2cell[:, 1])
        idx = (cell2dofLocation[face2cell[isInFace, 1]] + face2cell[isInFace,
                                                                    3] * fdof).reshape(-1, 1) + np.arange(fdof)
        cell2dof[idx] = face2dof[isInFace]

        NFC = mesh.number_of_faces_of_cells()
        NF = mesh.number_of_faces()
        idx = (cell2dofLocation[:-1] + NFC * fdof).reshape(-1, 1) + np.arange(cdof)
        cell2dof[idx] = NF * fdof + np.arange(NC * cdof).reshape(NC, cdof)
        return cell2dof, cell2dofLocation

    def number_of_global_dofs(self):
        mesh = self.mesh
        NF = mesh.number_of_faces()
        NC = mesh.number_of_cells()
        cdof = self.number_of_local_dofs(doftype='cell')  # 单元内部的自由度
        fdof = self.number_of_local_dofs(doftype='face')  # 面上的自由度
        gdof = NF * fdof + NC * cdof
        return gdof

    def number_of_local_dofs(self, doftype='all'):
        p = self.p
        mesh = self.mesh
        NCF = np.ones(mesh.number_of_cells(), dtype=int) * mesh.number_of_faces_of_cells()
        if doftype == 'all':  # number of all dofs on a cell
            return NCF * (p + 1) * (p + 2) // 2 + (p + 1) * (p + 2) * (p + 3) // 6
        elif doftype in {'cell', 3}:  # number of dofs inside the cell
            return (p + 1) * (p + 2) * (p + 3) // 6
        elif doftype in {'face', 2}:  # number of dofs on each face
            return (p + 1) * (p + 2) // 2
        elif doftype in {'edge', 1}:  # number of dofs on each edge
            return 0
        elif doftype in {'node', 0}:  # number of dofs on each node
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

        self.stype = 'wg3d'  # 空间类型

    def number_of_local_dofs(self, doftype='all'):
        return self.dof.number_of_local_dofs(doftype=doftype)

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def is_boundary_dof(self):
        return self.dof.is_boundary_dof(threshold=None)

    def face_to_dof(self):
        return self.dof.face_to_dof()

    def cell_to_dof(self, doftype='all'):
        if doftype == 'all':
            return self.dof.cell2dof, self.dof.cell2dofLocation
        elif doftype == 'cell':
            NF = self.mesh.number_of_faces()
            NC = self.mesh.number_of_cells()
            fdof = self.number_of_local_dofs(doftype='face')
            cdof = self.number_of_local_dofs(doftype='cell')
            cell2dof = NF * fdof + np.arange(NC * cdof).reshape(NC, cdof)
            return cell2dof

    def weak_grad(self, uh):
        cell2dof, cell2dofLocation = self.cell_to_dof()
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        R0 = np.hsplit(self.R[0], cell2dofLocation[1:-1])
        R1 = np.hsplit(self.R[1], cell2dofLocation[1:-1])
        R2 = np.hsplit(self.R[2], cell2dofLocation[1:-1])  # 增添了一行

        ph = self.smspace.function(dim=3)

        def f0(x):
            return x[0] @ (x[1] @ uh[x[2]])

        ph[:, 0] = np.concatenate(list(map(f0, zip(self.H0, R0, cd))))
        ph[:, 1] = np.concatenate(list(map(f0, zip(self.H0, R1, cd))))
        ph[:, 2] = np.concatenate(list(map(f0, zip(self.H0, R2, cd))))  # 增添了一行
        return ph

    def weak_div(self, ph):
        cell2dof, cell2dofLocation = self.cell_to_dof()
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        R0 = np.hsplit(self.R[0], cell2dofLocation[1:-1])
        R1 = np.hsplit(self.R[1], cell2dofLocation[1:-1])
        #        R2 = np.hsplit(self.R[2], NC)
        dh = self.smspace.function()

        def f0(x):
            return x[0] @ (x[1] @ ph[x[3], 0] + x[2] @ ph[x[3], 1])

        dh[:] = np.concatenate(list(map(f0, zip(self.H0, R0, R1, cd))))
        return dh

    def left_weak_matrix(self):
        """
        计算单元上的弱梯度和弱散度投影算子的右端矩阵
        """
        p = self.p
        mesh = self.mesh
        node = mesh.entity('node')
        face = mesh.entity('face')
        face2cell = mesh.ds.face_to_cell()
        isInFace = (face2cell[:, 0] != face2cell[:, 1])  # 内部面

        h = self.integralalg.facemeasure
        n = mesh.face_unit_normal()  # 法向量

        qf = mesh.integrator(k=p+3, etype='face')
        bcs, ws = qf.quadpts, qf.weights
        ps = np.einsum('ij, kjm->ikm', bcs, node[face])
        phi0 = self.smspace.basis(ps, index=face2cell[:, 0])  # (NQ, NE, ldof)
        phi1 = self.smspace.basis(
            ps[:, isInFace, :],
            index=face2cell[isInFace, 1]
        )  # (NQ, NINE, ldof)
        phi = self.face_basis(ps)

        F0 = np.einsum('i, ijm, ijn, j->mjn', ws, phi0, phi, h)
        F1 = np.einsum('i, ijm, ijn, j->mjn', ws, phi1, phi[:, isInFace, :], h[isInFace])

        smldof = self.smspace.number_of_local_dofs()
        cell2dof, cell2dofLocation = self.dof.cell2dof, self.dof.cell2dofLocation
        R0 = np.zeros((smldof, len(cell2dof)), dtype=np.float_)
        R1 = np.zeros((smldof, len(cell2dof)), dtype=np.float_)
        R2 = np.zeros((smldof, len(cell2dof)), dtype=np.float_)  # 这里增加了一行

        fdof = self.dof.number_of_local_dofs(doftype='face')
        idx = cell2dofLocation[face2cell[:, [0]]] + \
              face2cell[:, [2]] * fdof + np.arange(fdof)
        R0[:, idx] = n[np.newaxis, :, [0]] * F0  # 带上法向量   v_b 对应的矩阵
        R1[:, idx] = n[np.newaxis, :, [1]] * F0
        R2[:, idx] = n[np.newaxis, :, [2]] * F0  # 这里增加了一行
        if isInFace.sum() > 0:
            idx = np.array(fdof * face2cell[isInFace, 1]).reshape(-1, 1) + \
                  fdof * face2cell[isInFace, [3]].reshape(-1, 1) + np.arange(fdof)
            n = n[isInFace]
            R0[:, idx] = -n[np.newaxis, :, [0]] * F1  # 这里应该加上负号
            R1[:, idx] = -n[np.newaxis, :, [1]] * F1
            R2[:, idx] = -n[np.newaxis, :, [2]] * F1

        def f(x, index=np.s_[:]):
            gphi = self.grad_basis(x, index)
            phi = self.basis(x, index)
            return np.einsum(
                '...mn, ...k->...nmk',
                gphi, phi)

        M = self.integralalg.integral(f, celltype=True, barycenter=False)  # v_0 对应的矩阵

        cdof = self.number_of_local_dofs(doftype='cell')
        idx = cell2dofLocation[1:].reshape(-1, 1) + np.arange(-cdof, 0)
        R0[:, idx] = -M[:, 0].swapaxes(0, 1)  # 将 M 填充进去
        R1[:, idx] = -M[:, 1].swapaxes(0, 1)
        R2[:, idx] = -M[:, 2].swapaxes(0, 1)  # 这里增加了一行
        return R0, R1, R2

    def stiff_matrix(self):
        gdof = self.number_of_global_dofs()
        cell2dof, cell2dofLocation = self.cell_to_dof()
        cd = np.hsplit(cell2dof, cell2dofLocation[1:-1])
        H0 = self.H0
        R = self.R

        def f0(i):
            R0 = R[0][:, cell2dofLocation[i]:cell2dofLocation[i + 1]]
            R1 = R[1][:, cell2dofLocation[i]:cell2dofLocation[i + 1]]
            R2 = R[2][:, cell2dofLocation[i]:cell2dofLocation[i + 1]]  # 这里添加了一行
            return R0.T @ H0[i] @ R0, R1.T @ H0[i] @ R1, R2.T @ H0[i] @ R2  # 这里添加了一个输出

        NC = self.mesh.number_of_cells()
        M = list(map(f0, range(NC)))  # (3, NC, cdof, codf)

        idx = list(map(np.meshgrid, cd, cd))

        I = np.concatenate(list(map(lambda x: x[1].flat, idx)))
        J = np.concatenate(list(map(lambda x: x[0].flat, idx)))

        val = np.concatenate(list(map(lambda x: x[0].flat, M)))
        M00 = csr_matrix((val, (I, J)), shape=(gdof, gdof))

        val = np.concatenate(list(map(lambda x: x[1].flat, M)))
        M11 = csr_matrix((val, (I, J)), shape=(gdof, gdof))

        val = np.concatenate(list(map(lambda x: x[2].flat, M)))
        M22 = csr_matrix((val, (I, J)), shape=(gdof, gdof))

        A = M00 + M11 + M22  # weak gradient matrix

        return A

    def mass_matrix(self):
        cell2dof = self.cell_to_dof(doftype='cell')  # only get the dofs in cell
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

        qf = TriangleQuadrature(self.p+3)
        bcs, ws = qf.quadpts, qf.weights

        face2cell = mesh.ds.face_to_cell()
        isInFace = (face2cell[:, 0] != face2cell[:, 1])

        ps = self.mesh.bc_to_point(bcs, etype='face')
        phi0 = self.basis(ps, index=face2cell[:, 0])
        phi1 = self.basis(
            ps[:, isInFace, :],
            index=face2cell[isInFace, 1]
        )
        phi = self.face_basis(ps)

        h = mesh.entity_measure('face')
        cellsize = self.cellsize
        h0 = cellsize[face2cell[:, 0]].reshape(-1, 1, 1)
        h1 = cellsize[face2cell[isInFace, 1]].reshape(-1, 1, 1)
        F0 = np.einsum('i, ijm, ijn, j->jmn', ws, phi0, phi, h) / h0
        F1 = np.einsum('i, ijm, ijn, j->jmn', ws, phi1, phi[:, isInFace, :], h[isInFace]) / h1

        F2 = np.einsum('i, ijm, ijn, j->jmn', ws, phi0, phi0, h) / h0
        F3 = np.einsum('i, ijm, ijn, j->jmn', ws, phi1, phi1, h[isInFace]) / h1

        F4 = np.einsum('i, ijm, ijn, j->jmn', ws, phi, phi, h)
        F5 = F4[isInFace] / h1
        F4 /= h0

        face2dof = self.face_to_dof()
        cell2dof = self.cell_to_dof(doftype='cell')

        cdof = self.dof.number_of_local_dofs(doftype='cell')
        fdof = self.dof.number_of_local_dofs(doftype='face')
        gdof = self.number_of_global_dofs()
        S = csr_matrix((gdof, gdof), dtype=self.ftype)

        I = np.einsum('ij, k->ijk', cell2dof[face2cell[:, 0]], np.ones(fdof))
        J = np.einsum('ik, j->ijk', face2dof, np.ones(cdof))
        S -= csr_matrix((F0.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        S -= csr_matrix((F0.flat, (J.flat, I.flat)), shape=(gdof, gdof))

        I = np.einsum('ij, k->ijk', cell2dof[face2cell[isInFace, 1]], np.ones(fdof))
        J = np.einsum('ik, j->ijk', face2dof[isInFace], np.ones(cdof))
        S -= csr_matrix((F1.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        S -= csr_matrix((F1.flat, (J.flat, I.flat)), shape=(gdof, gdof))

        I = np.einsum('ij, k->ijk', cell2dof[face2cell[:, 0]], np.ones(cdof))
        J = I.swapaxes(-1, -2)
        S += csr_matrix((F2.flat, (I.flat, J.flat)), shape=(gdof, gdof))

        I = np.einsum('ij, k->ijk', cell2dof[face2cell[isInFace, 1]], np.ones(cdof))
        J = I.swapaxes(-1, -2)
        S += csr_matrix((F3.flat, (I.flat, J.flat)), shape=(gdof, gdof))

        I = np.einsum('ij, k->ijk', face2dof, np.ones(fdof))
        J = I.swapaxes(-1, -2)
        S += csr_matrix((F4.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        I = np.einsum('ij, k->ijk', face2dof[isInFace], np.ones(fdof))
        J = I.swapaxes(-1, -2)
        S += csr_matrix((F5.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return S

    def set_dirichlet_bc(self, gD, uh, threshold=None):    # threshold=None 保持接口一致
        """
        初始化解 uh  的第一类边界条件。
        """
        mesh = self.mesh
        isBdFace = mesh.ds.boundary_face_flag()
        isBdDof = self.is_boundary_dof()

        qf = mesh.integrator(k=self.p+3, etype='face')
        bcs, ws = qf.quadpts, qf.weights
        ps = mesh.bc_to_point(bcs, etype='face', index=isBdFace)
        gI = gD(ps)
        fphi = self.face_basis(ps, index=isBdFace)
        h = mesh.entity_measure('face')
        b = np.einsum('i, ij, ijk, j->jk', ws, gI, fphi, h[isBdFace])
        uh[isBdDof] = np.einsum('ijk, ik->ij', self.H1[isBdFace], b).flat
        return isBdDof

    @cartesian
    def basis(self, point, index=None):
        return self.smspace.basis(point, index=index)

    @cartesian
    def grad_basis(self, point, index=None):
        return self.smspace.grad_basis(point, index=index)

    @cartesian
    def value(self, uh, point, index=np.s_[:]):
        NF = self.mesh.number_of_faces()
        fdof = self.number_of_local_dofs(doftype='face')
        return self.smspace.value(uh[NF * fdof:, ...], point, index=index)

    @cartesian
    def grad_value(self, uh, point, index=np.s_[:]):
        NF = self.mesh.number_of_faces()
        fdof = self.number_of_local_dofs(doftype='face')
        return self.smspace.grad_value(uh[NF * fdof:, ...], point, index=index)

    @cartesian
    def face_basis(self, point, index=np.s_[:]):
        return self.smspace.face_basis(point, index=index)

    def lagrange_edge_basis(self, bc):
        p = self.p  # the degree of polynomial basis function
        multiIndex = self.dof.multiIndex2d  # TODO:未修正
        fdof = self.number_of_local_dofs(doftype='face')
        c = np.arange(1, fdof, dtype=np.int)
        P = 1.0 / np.multiply.accumulate(c)
        t = np.arange(0, p)
        shape = bc.shape[:-1] + (fdof, 2)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p * bc[..., np.newaxis, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = np.arange(2)
        phi = np.prod(A[..., multiIndex, idx], axis=-1)
        return phi

    def source_vector(self, f):
        phi = self.basis

        def u(x, index=np.s_[:]):
            return np.einsum('ij, ijm->ijm', f(x), phi(x, index=index))

        bb = self.integralalg.integral(u, celltype=True, barycenter=False)
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

        qf = TriangleQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        uI = u(ps)

        fphi = self.face_basis(ps)
        b = np.einsum('i, ij..., ijk, j->jk...', ws, uI, fphi, h)
        if dim == 1:
            uh[:NE * (p + 1), ...].flat = (self.H1 @ b[:, :, np.newaxis]).flat
        else:
            uh[:NE * (p + 1), ...].flat = (self.H1 @ b).flat

        t = 'd'
        s = '...{}, ...m->...m{}'.format(t[:dim > 1], t[:dim > 1])

        def f1(x, index):
            phi = self.basis(x, index)
            return np.einsum(s, u(x), phi)

        b = self.integralalg.integral(f1, celltype=True)
        if dim == 1:
            uh[NE * (p + 1):, ...].flat = (self.H0 @ b[:, :, np.newaxis]).flat
        else:
            uh[NE * (p + 1):, ...].flat = (self.H0 @ b).flat
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
            shape = (gdof,) + dim
        return np.zeros(shape, dtype=dtype)
