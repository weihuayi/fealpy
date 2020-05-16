import numpy as np
from numpy.linalg import inv
from .function import Function
from ..quadrature import GaussLobattoQuadrature
from ..quadrature import GaussLegendreQuadrature
from ..quadrature import PolygonMeshIntegralAlg
from ..common import ranges


class SMDof2d():
    """
    缩放单项式空间自由度管理类
    """
    def __init__(self, mesh, p):
        self.mesh = mesh
        self.p = p # 默认的空间次数
        self.multiIndex = self.multi_index_matrix() # 默认的多重指标
        self.cell2dof = self.cell_to_dof() # 默认的自由度数组

    def multi_index_matrix(self, p=None):
        """
        Compute the natural correspondence from the one-dimensional index
        starting from 0.

        Notes
        -----

        0<-->(0, 0), 1<-->(1, 0), 2<-->(0, 1), 3<-->(2, 0), 4<-->(1, 1),
        5<-->(0, 2), .....

        """
        ldof = self.number_of_local_dofs(p=p)
        idx = np.arange(0, ldof)
        idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
        multiIndex = np.zeros((ldof, 2), dtype=np.int)
        multiIndex[:, 1] = idx - idx0*(idx0 + 1)/2
        multiIndex[:, 0] = idx0 - multiIndex[:, 1]
        return multiIndex

    def cell_to_dof(self, p=None):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        ldof = self.number_of_local_dofs(p=p)
        cell2dof = np.arange(NC*ldof).reshape(NC, ldof)
        return cell2dof

    def number_of_local_dofs(self, p=None):
        p = self.p if p is None else p
        return (p+1)*(p+2)//2

    def number_of_global_dofs(self, p=None):
        ldof = self.number_of_local_dofs(p=p)
        NC = self.mesh.number_of_cells()
        return NC*ldof


class ScaledMonomialSpace2d():
    def __init__(self, mesh, p, q=None, bc=None):
        """
        The Scaled Momomial Space in R^2
        """

        self.mesh = mesh
        self.cellbarycenter = mesh.entity_barycenter('cell') if bc is None else bc
        self.p = p
        self.cellmeasure = mesh.entity_measure('cell')
        self.cellsize = np.sqrt(self.cellmeasure)
        self.dof = SMDof2d(mesh, p)
        self.GD = 2

        q = q if q is not None else p+3
        self.integralalg = PolygonMeshIntegralAlg(
                self.mesh, q,
                cellmeasure=self.cellmeasure,
                cellbarycenter=self.cellbarycenter)

        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype

    def index1(self, p=None):
        """
        缩放单项式基函数求一次导数后，非零的基函数编号，及因求导出现的系数
        """
        p = self.p if p is None else p
        n = (p+1)*(p+2)//2
        idx1 = np.cumsum(np.arange(p+1))
        idx0 = np.arange(p+1) + idx1

        mask0 = np.ones(n, dtype=np.bool)
        mask1 = np.ones(n, dtype=np.bool)
        mask0[idx0] = False
        mask1[idx1] = False

        idx = np.arange(n)
        idx0 = idx[mask0]
        idx1 = idx[mask1]

        idx = np.repeat(range(2, p+2), range(1, p+1))
        idx3 = ranges(range(p+1), start=1)
        idx2 = idx - idx3
        # idx0: 关于 x 求一阶导数后不为零的基函数编号
        # idx1：关于 y 求一阶导数后不为零的基函数的编号
        # idx2: 关于 x 求一阶导数后不为零的基函数的整数系数
        # idx3: 关于 y 求一阶导数后不为零的基函数的整数系数
        return {'x':(idx0, idx2), 'y':(idx1, idx3)}

    def index2(self, p=None):
        """
        缩放单项式基函数求两次导数后，非零的编号及因求导出现的系数
        """
        p = self.p if p is None else p
        n = (p+1)*(p+2)//2
        mask0 = np.ones(n, dtype=np.bool)
        mask1 = np.ones(n, dtype=np.bool)
        mask2 = np.ones(n, dtype=np.bool)

        idx1 = np.cumsum(np.arange(p+1))
        idx0 = np.arange(p+1) + idx1
        mask0[idx0] = False
        mask1[idx1] = False

        mask2[idx0] = False
        mask2[idx1] = False

        idx0 = np.cumsum([1]+list(range(3, p+2)))
        idx1 = np.cumsum([2]+list(range(2, p+1)))
        mask0[idx0] = False
        mask1[idx1] = False

        idx = np.arange(n)
        idx0 = idx[mask0]
        idx1 = idx[mask1]
        idx2 = idx[mask2]

        idxa = np.repeat(range(2, p+1), range(1, p))
        idxb = np.repeat(range(4, p+3), range(1, p))

        idxc = ranges(range(p), start=1)
        idxd = ranges(range(p), start=2)

        idx3 = (idxa - idxc)*(idxb - idxd)
        idx4 = idxc*idxd
        idx5 = idxc*(idxa - idxc)

        # idx0: 关于 x 求二阶导数后不为零的基函数编号
        # idx1：关于 y 求二阶导数后不为零的基函数的编号
        # idx2：关于 x 和 y 求混合导数后不为零的基函数的编号
        # idx3: 关于 x 求二阶导数后不为零的基函数的整数系数
        # idx4：关于 y 求二阶导数后不为零的基函数的整数系数
        # idx5：关于 x 和 y 求混合导数扣不为零的基函数的整数系数
        return {'xx': (idx0, idx3), 'yy': (idx1, idx4), 'xy': (idx2, idx5)}

    def geo_dimension(self):
        return self.GD

    def cell_to_dof(self, p=None):
        return self.dof.cell_to_dof(p=p)

    def edge_basis(self, point, index=None, p=None):
        p = self.p if p is None else p
        index = index if index is not None else np.s_[:]
        center = self.integralalg.edgebarycenter
        h = self.integralalg.edgemeasure
        t = self.mesh.edge_unit_tagent()
        val = np.sum((point - center[index])*t[index], axis=-1)/h[index]
        phi = np.ones(val.shape + (p+1,), dtype=self.ftype)
        if p == 1:
            phi[..., 1] = val
        else:
            phi[..., 1:] = val[..., np.newaxis]
            np.multiply.accumulate(phi, axis=-1, out=phi)
        return phi

    def basis(self, point, index=None, p=None):
        """
        Compute the basis values at point

        Parameters
        ----------
        point : ndarray
            The shape of point is (..., M, 2), M is the number of cells

        Returns
        -------
        phi : ndarray
            The shape of `phi` is (..., M, ldof)

        """
        p = self.p if p is None else p
        h = self.cellsize
        NC = self.mesh.number_of_cells()

        ldof = self.number_of_local_dofs(p=p)
        if p == 0:
            shape = point.shape[:-1] + (1, )
            return np.ones(shape, dtype=np.float)

        shape = point.shape[:-1]+(ldof,)
        phi = np.ones(shape, dtype=np.float)  # (..., M, ldof)
        index = index if index is not None else np.s_[:] 
        phi[..., 1:3] = (point - self.cellbarycenter[index])/h[index].reshape(-1, 1)
        if p > 1:
            start = 3
            for i in range(2, p+1):
                phi[..., start:start+i] = phi[..., start-i:start]*phi[..., [1]]
                phi[..., start+i] = phi[..., start-1]*phi[..., 2]
                start += i+1
        return phi


    def grad_basis(self, point, index=None, p=None):

        p = self.p if p is None else p
        h = self.cellsize
        num = len(h) if index is  None else len(index)
        index = np.s_[:] if index is None else index 

        ldof = self.number_of_local_dofs(p=p)
        shape = point.shape[:-1]+(ldof, 2)
        phi = self.basis(point, index=index, p=p-1)
        idx = self.index1(p=p)
        gphi = np.zeros(shape, dtype=np.float)
        xidx = idx['x']
        yidx = idx['y']
        gphi[..., xidx[0], 0] = np.einsum('i, ...i->...i', xidx[1], phi) 
        gphi[..., yidx[0], 1] = np.einsum('i, ...i->...i', yidx[1], phi)
        if point.shape[-2] == num:
            return gphi/h[index].reshape(-1, 1, 1)
        elif point.shape[0] == num:
            return gphi/h[index].reshape(-1, 1, 1, 1)

    def laplace_basis(self, point, index=None, p=None):
        p = self.p if p is None else p
        index = index if index is not None else np.s_[:]

        area = self.cellmeasure
        ldof = self.number_of_local_dofs(p=p)
        shape = point.shape[:-1]+(ldof,)
        lphi = np.zeros(shape, dtype=np.float)
        if p > 1:
            phi = self.basis(point, index=index, p=p-2)
            idx = self.index2(p=p)
            lphi[..., idx['xx'][0]] += np.einsum('i, ...i->...i', idx['xx'][1], phi)
            lphi[..., idx['yy'][0]] += np.einsum('i, ...i->...i', idx['yy'][1], phi)
        return lphi/area[index].reshape(-1, 1)

    def hessian_basis(self, point, index=None, p=None):
        """
        Compute the value of the hessian of the basis at a set of 'point'

        Parameters
        ----------
        point : numpy array
            The shape of point is (..., NC, 2)

        Returns
        -------
        hphi : numpy array
            the shape of hphi is (..., NC, ldof, 2, 2)
        """
        p = self.p if p is None else p
        index = index if index is not None else np.s_[:]

        area = self.cellmeasure
        ldof = self.number_of_local_dofs(p=p)
        shape = point.shape[:-1]+(ldof, 2, 2)
        hphi = np.zeros(shape, dtype=np.float)
        if p > 1:
            phi = self.basis(point, index=index, p=p-2)
            idx = self.index2(p=p)
            hphi[..., idx['xx'][0], 0, 0] = np.einsum('i, ...i->...i', idx['xx'][1], phi)
            hphi[..., idx['xy'][0], 0, 1] = np.einsum('i, ...i->...i', idx['xy'][1], phi)
            hphi[..., idx['yy'][0], 1, 1] = np.einsum('i, ...i->...i', idx['yy'][1], phi)
            hphi[..., 1, 0] = hphi[..., 0, 1] 
        return hphi/area[index].reshape(-1, 1, 1, 1)

    def value(self, uh, point, index=None):
        phi = self.basis(point, index=index)
        cell2dof = self.dof.cell2dof
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ij, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        index = index if index is not None else np.s_[:]
        return np.einsum(s1, phi, uh[cell2dof[index]])

    def grad_value(self, uh, point, index=None):
        gphi = self.grad_basis(point, index=index)
        cell2dof = self.dof.cell2dof
        index = index if index is not None else np.s_[:]
        if (type(index) is np.ndarray) and (index.dtype.name == 'bool'):
            N = np.sum(index)
        elif type(index) is slice:
            N = len(cell2dof)
        else:
            N = len(index)
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        if point.shape[-2] == N:
            s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
            return np.einsum(s1, gphi, uh[cell2dof[index]])
        elif point.shape[0] == N:
            return np.einsum('ikjm, ij->ikm', gphi, uh[cell2dof[index]])

    def laplace_value(self, uh, point, index=None):
        lphi = self.laplace_basis(point, index=index)
        cell2dof = self.dof.cell2dof
        index = index if index is not None else np.s_[:]
        return np.einsum('...ij, ij->...i', lphi, uh[cell2dof[index]])

    def hessian_value(self, uh, point, index=None):
        #TODO:
        pass

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

    def number_of_local_dofs(self, p=None):
        return self.dof.number_of_local_dofs(p=p)

    def number_of_global_dofs(self, p=None):
        return self.dof.number_of_global_dofs(p=p)

    def cell_mass_matrix(self):
        return self.matrix_H()

    def edge_mass_matrix(self, p=None):
        p = self.p if p is None else p
        mesh = self.mesh
        NE = mesh.number_of_edges()
        edge = mesh.entity('edge')
        eh = mesh.entity_measure('edge')
        ldof = p + 1
        q = np.arange(p+1)
        Q = q.reshape(-1, 1) + q + 1
        flag = Q%2==1
        H = np.zeros((NE, ldof, ldof), dtype=self.ftype)
        H[:, flag] = eh.reshape(-1, 1)
        H[:, flag] /= Q[flag]
        H[:, flag] /=2**(Q[flag]-1)
        return H

    def edge_mass_matrix_1(self, p=None):
        p = self.p if p is None else p
        mesh = self.mesh
        edge = mesh.entity('edge')
        measure = mesh.entity_measure('edge')
        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights
        ps = self.mesh.edge_bc_to_point(bcs)
        phi = self.edge_basis(ps, p=p)
        H = np.einsum('i, ijk, ijm, j->jkm', ws, phi, phi, measure, optimize=True)
        return H

    def mass_matrix(self, p=None):
        return self.matrix_H(p=p)

    def stiff_matrix(self, p=None):
        p = self.p if p is None else p
        def f(x, index):
            gphi = self.grad_basis(x, index=index, p=p)
            return np.einsum('ijkm, ijpm->ijkp', gphi, gphi)
    
        A = self.integralalg.integral(f, celltype=True, q=p+3)
        cell2dof = self.cell_to_dof(p=p)
        ldof = self.number_of_local_dofs(p=p)
        I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
        J = I.swapaxes(-1, -2)
        gdof = self.number_of_global_dofs(p=p)

        # Construct the stiffness matrix
        A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return A

    def source_vector(self, f, dim=None, p=None):
        def u(x, index):
            return np.einsum('ij, ijm->ijm', f(x), self.basis(x, index=index, p=None))
        bb = self.integralalg.integral(u, celltype=True, q=p+3)
        gdof = self.number_of_global_dofs(p=p)
        cell2dof = self.cell_to_dof(p=p)
        b = np.bincount(cell2dof.flat, weights=bb.flat, minlength=gdof)
        return b

    def matrix_H(self, p=None):
        p = self.p if p is None else p
        mesh = self.mesh
        node = mesh.entity('node')

        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        NC = mesh.number_of_cells()

        qf = GaussLegendreQuadrature(p + 1)
        bcs, ws = qf.quadpts, qf.weights
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge])
        phi0 = self.basis(ps, index=edge2cell[:, 0], p=p)
        phi1 = self.basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1], p=p)
        H0 = np.einsum('i, ijk, ijm->jkm', ws, phi0, phi0)
        H1 = np.einsum('i, ijk, ijm->jkm', ws, phi1, phi1)

        nm = mesh.edge_normal()
        b = node[edge[:, 0]] - self.cellbarycenter[edge2cell[:, 0]]
        H0 = np.einsum('ij, ij, ikm->ikm', b, nm, H0)
        b = node[edge[isInEdge, 0]] - self.cellbarycenter[edge2cell[isInEdge, 1]]
        H1 = np.einsum('ij, ij, ikm->ikm', b, -nm[isInEdge], H1)

        ldof = self.number_of_local_dofs(p=p)
        H = np.zeros((NC, ldof, ldof), dtype=np.float)
        np.add.at(H, edge2cell[:, 0], H0)
        np.add.at(H, edge2cell[isInEdge, 1], H1)

        multiIndex = self.dof.multi_index_matrix(p=p)
        q = np.sum(multiIndex, axis=1)
        H /= q + q.reshape(-1, 1) + 2
        return H

    def projection(self, F):
        """
        F is a function in MonomialSpace2d, this function project  F to 
        ScaledMonomialSpace2d.
        """
        mspace = F.space
        C = self.matrix_C(mspace)
        H = self.matrix_H()
        PI0 = inv(H)@C
        SS = self.function()
        SS[:] = np.einsum('ikj, ij->ik', PI0, F[self.cell_to_dof()]).reshape(-1)
        return SS

    def matrix_C(self, mspace):
        def f(x, index):
            return np.einsum(
                    '...im, ...in->...imn',
                    self.basis(x, index),
                    mspace.basis(x, index)
                    )
        C = self.integralalg.integral(f, celltype=True)
        return C

    def interpolation(self, sh0, HB):
        """
         interpolation sh in space into self space.
        """
        p = self.p
        ldofs = self.number_of_local_dofs()
        mesh = self.mesh
        NC = mesh.number_of_cells()

        space0 = sh0.space
        h0 = space0.cellsize

        space1 = self
        h1 = space1.cellsize
        sh1 = space1.function()

        bc = (space1.cellbarycenter[HB[:, 0]] - space0.cellbarycenter[HB[:,
            1]])/h0[HB[:, [1]]]
        h = h1[HB[:, 0]]/h0[HB[:, 1]]

        c = sh0.reshape(-1, ldofs)
        d = sh1.reshape(-1, ldofs)

        num = np.zeros(NC, dtype=self.itype)
        np.add.at(num, HB[:, 0], 1)

        m = HB.shape[0]
        td = np.zeros((m, ldofs), dtype=self.ftype)

        td[:, 0] = c[HB[:, 1], 0] + c[HB[:, 1], 1]*bc[:, 0] + c[HB[:, 1], 2]*bc[:, 1]
        td[:, 1] = h*c[HB[:, 1], 1]
        td[:, 2] = h*c[HB[:, 1], 2]

        if p > 1:
            td[:, 0] += c[HB[:, 1], 3]*bc[:, 0]**2 + c[HB[:, 1], 4]*bc[:, 0]*bc[:, 1] + c[HB[:, 1], 5]*bc[:, 1]**2
            td[:, 1] += 2*c[HB[:, 1], 3]*bc[:, 0]*h + c[HB[:, 1], 4]*bc[:, 1]*h
            td[:, 2] += c[HB[:, 1], 4]*bc[:, 0]*h + 2*c[HB[:, 1], 5]*bc[:, 1]*h
            td[:, 3] = c[HB[:, 1], 3]*h**2
            td[:, 4] = c[HB[:, 1], 4]*h**2
            td[:, 5] = c[HB[:, 1], 5]*h**2

        np.add.at(d, (HB[:, 0], np.s_[:]), td)
        d /= num.reshape(-1, 1)
        return sh1
