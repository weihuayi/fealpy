import numpy as np
from scipy.sparse import csr_matrix
from numpy.linalg import inv
from .Function import Function
from ..decorator import cartesian
from ..quadrature import GaussLobattoQuadrature
from ..quadrature import GaussLegendreQuadrature
from ..quadrature import PolygonMeshIntegralAlg
from ..quadrature import FEMeshIntegralAlg
from ..common import ranges

from .femdof import multi_index_matrix2d, multi_index_matrix1d
from .LagrangeFiniteElementSpace import LagrangeFiniteElementSpace

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
        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        idx = np.arange(0, ldof)
        idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
        multiIndex = np.zeros((ldof, 2), dtype=np.int)
        multiIndex[:, 1] = idx - idx0*(idx0 + 1)/2
        multiIndex[:, 0] = idx0 - multiIndex[:, 1]
        return multiIndex

    def cell_to_dof(self, p=None):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        cell2dof = np.arange(NC*ldof).reshape(NC, ldof)
        return cell2dof

    def number_of_local_dofs(self, p=None, doftype='cell'):
        p = self.p if p is None else p
        if doftype in {'cell', 2}:
            return (p+1)*(p+2)//2
        elif doftype in {'face', 'edge', 1}:
            return (p+1)
        elif doftype in {'node', 0}:
            return 0 

    def number_of_global_dofs(self, p=None, doftype='cell'):
        ldof = self.number_of_local_dofs(p=p, doftype=doftype)
        if doftype in {'cell', 2}:
            N = self.mesh.number_of_cells()
        elif doftype in {'face', 'edge', 1}:
            N = self.mesh.number_of_edges()
        return N*ldof


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

        mtype = mesh.meshtype
        if mtype in {'polygon', 'halfedge', 'halfedge2d'}:
            self.integralalg = PolygonMeshIntegralAlg(
                    self.mesh, q,
                    cellmeasure=self.cellmeasure,
                    cellbarycenter=self.cellbarycenter)
        else:
            self.integralalg = FEMeshIntegralAlg(
                    self.mesh, q,
                    cellmeasure=self.cellmeasure)

        self.integrator = self.integralalg.integrator

        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype


    def diff_index_1(self, p=None):
        """

        Notes
        -----
        对基函数求一阶导后非零项的编号，及系数
        """
        p = self.p if p is None else p
        index = multi_index_matrix2d(p)

        x, = np.nonzero(index[:, 1] > 0) # 关于 x 求导非零的缩放单项式编号
        y, = np.nonzero(index[:, 2] > 0) # 关于 y 求导非零的缩放单项式编号

        return {'x':(x, index[x, 1]),
                'y':(y, index[y, 2]),
                }

    def diff_index_2(self, p=None):
        """

        Notes
        -----
        对基函数求二阶导后非零项的编号，及系数
        """
        p = self.p if p is None else p
        index = multi_index_matrix2d(p)

        xx, = np.nonzero(index[:, 1] > 1)
        yy, = np.nonzero(index[:, 2] > 1)

        xy, = np.nonzero((index[:, 1] > 0) & (index[:, 2] > 0))

        return {'xx':(xx, index[xx, 1]*(index[xx, 1]-1)),
                'yy':(yy, index[yy, 2]*(index[yy, 2]-1)),
                'xy':(xy, index[xy, 1]*index[xy, 2]),
                }

    def face_index_1(self, p=None):
        """
        Parameters
        ----------
        p : >= 1
        """
        p = self.p if p is None else p
        index = multi_index_matrix1d(p)
        x, = np.nonzero(index[:, 0] > 0)
        y, = np.nonzero(index[:, 1] > 0)
        return {'x': x, 'y':y}

    def edge_index_1(self, p=None):
        """
        Parameters
        ----------
        p : >= 1
        """
        p = self.p if p is None else p
        index = multi_index_matrix1d(p)
        x, = np.nonzero(index[:, 0] > 0)
        y, = np.nonzero(index[:, 1] > 0)
        return {'x': x, 'y':y}

    def geo_dimension(self):
        return self.GD

    def cell_to_dof(self, p=None):
        return self.dof.cell_to_dof(p=p)

    @cartesian
    def edge_basis(self, point, index=np.s_[:], p=None):
        p = self.p if p is None else p
        if p == 0:
            shape = len(point.shape)*(1, )
            return np.array([1.0], dtype=self.ftype).reshape(shape)

        ec = self.integralalg.edgebarycenter
        eh = self.integralalg.edgemeasure
        et = self.mesh.edge_unit_tangent()
        val = np.sum((point - ec[index])*et[index], axis=-1)/eh[index]
        phi = np.ones(val.shape + (p+1,), dtype=self.ftype)
        if p == 1:
            phi[..., 1] = val
        else:
            phi[..., 1:] = val[..., np.newaxis]
            np.multiply.accumulate(phi, axis=-1, out=phi)
        return phi

    @cartesian
    def basis(self, point, index=np.s_[:], p=None):
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

        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        if p == 0:
            shape = len(point.shape)*(1, )
            return np.array([1.0], dtype=self.ftype).reshape(shape)

        shape = point.shape[:-1]+(ldof,)
        phi = np.ones(shape, dtype=np.float)  # (..., M, ldof)
        phi[..., 1:3] = (point - self.cellbarycenter[index])/h[index].reshape(-1, 1)
        if p > 1:
            start = 3
            for i in range(2, p+1):
                phi[..., start:start+i] = phi[..., start-i:start]*phi[..., [1]]
                phi[..., start+i] = phi[..., start-1]*phi[..., 2]
                start += i+1
        return phi

    @cartesian
    def grad_basis(self, point, index=np.s_[:], p=None, scaled=True):
        """

        p >= 0
        """

        p = self.p if p is None else p 
        h = self.cellsize

        num = len(h) if type(index) is slice else len(index)

        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        shape = point.shape[:-1]+(ldof, 2)
        gphi = np.zeros(shape, dtype=np.float)

        if p == 0:
            return gphi

        phi = self.basis(point, index=index, p=p-1)
        idx = self.diff_index_1(p=p)
        xidx = idx['x']
        yidx = idx['y']
        gphi[..., xidx[0], 0] = np.einsum('i, ...i->...i', xidx[1], phi) 
        gphi[..., yidx[0], 1] = np.einsum('i, ...i->...i', yidx[1], phi)

        if scaled:
            if point.shape[-2] == num:
                return gphi/h[index].reshape(-1, 1, 1)
            elif point.shape[0] == num:
                return gphi/h[index].reshape(-1, 1, 1, 1)
        else:
            return gphi

    @cartesian
    def laplace_basis(self, point, index=np.s_[:], p=None, scaled=True):
        p = self.p if p is None else p

        area = self.cellmeasure
        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        shape = point.shape[:-1]+(ldof,)
        lphi = np.zeros(shape, dtype=np.float)
        if p > 1:
            phi = self.basis(point, index=index, p=p-2)
            idx = self.diff_index_2(p=p)
            lphi[..., idx['xx'][0]] += np.einsum('i, ...i->...i', idx['xx'][1], phi)
            lphi[..., idx['yy'][0]] += np.einsum('i, ...i->...i', idx['yy'][1], phi)

        if scaled:
            return lphi/area[index].reshape(-1, 1)
        else:
            return lphi

    @cartesian
    def hessian_basis(self, point, index=np.s_[:], p=None, scaled=True):
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

        area = self.cellmeasure
        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        shape = point.shape[:-1]+(ldof, 2, 2)
        hphi = np.zeros(shape, dtype=np.float)
        if p > 1:
            phi = self.basis(point, index=index, p=p-2)
            idx = self.diff_index_2(p=p)
            hphi[..., idx['xx'][0], 0, 0] = np.einsum('i, ...i->...i', idx['xx'][1], phi)
            hphi[..., idx['xy'][0], 0, 1] = np.einsum('i, ...i->...i', idx['xy'][1], phi)
            hphi[..., idx['yy'][0], 1, 1] = np.einsum('i, ...i->...i', idx['yy'][1], phi)
            hphi[..., 1, 0] = hphi[..., 0, 1] 

        if scaled:
            return hphi/area[index].reshape(-1, 1, 1, 1)
        else:
            return hphi

    @cartesian
    def value(self, uh, point, index=np.s_[:]):
        phi = self.basis(point, index=index)
        cell2dof = self.dof.cell2dof[index]
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ij, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        return np.einsum(s1, phi, uh[cell2dof])

    @cartesian
    def grad_value(self, uh, point, index=np.s_[:]):
        gphi = self.grad_basis(point, index=index)
        cell2dof = self.dof.cell2dof
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

    @cartesian
    def laplace_value(self, uh, point, index=np.s_[:]):
        lphi = self.laplace_basis(point, index=index)
        cell2dof = self.dof.cell2dof
        return np.einsum('...ij, ij->...i', lphi, uh[cell2dof[index]])

    @cartesian
    def hessian_value(self, uh, point, index=np.s_[:]):
        #TODO:
        pass

    def function(self, dim=None, array=None):
        f = Function(self, dim=dim, array=array, coordtype='cartesian')
        return f

    def array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim in {None, 1}:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=np.float)

    def dof_array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim in {None, 1}:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return np.zeros(shape, dtype=np.float)

    def number_of_local_dofs(self, p=None, doftype='cell'):
        return self.dof.number_of_local_dofs(p=p, doftype=doftype)

    def number_of_global_dofs(self, p=None):
        return self.dof.number_of_global_dofs(p=p)


    def cell_mass_matrix(self, p=None):
        return self.matrix_H(p=p)

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

    def edge_cell_mass_matrix(self, p=None): 
        p = self.p if p is None else p
        mesh = self.mesh

        edge = mesh.entity('edge')
        measure = mesh.entity_measure('edge')

        edge2cell = mesh.ds.edge_to_cell()

        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights
        ps = self.mesh.edge_bc_to_point(bcs)

        phi0 = self.edge_basis(ps, p=p)
        phi1 = self.basis(ps, index=edge2cell[:, 0], p=p+1)
        phi2 = self.basis(ps, index=edge2cell[:, 1], p=p+1)
        LM = np.einsum('i, ijk, ijm, j->jkm', ws, phi0, phi1, measure, optimize=True)
        RM = np.einsum('i, ijk, ijm, j->jkm', ws, phi0, phi2, measure, optimize=True)
        return LM, RM 

    def stiff_matrix(self, p=None):
        """

        Note:
            这个程序仅用于多边形网格上的刚度矩阵组装
        """
        p = self.p if p is None else p

        @cartesian
        def f(x, index):
            gphi = self.grad_basis(x, index=index, p=p)
            return np.einsum('ijkm, ijpm->ijkp', gphi, gphi)

        A = self.integralalg.cell_integral(f, q=p+3)
        cell2dof = self.cell_to_dof(p=p)
        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
        J = I.swapaxes(-1, -2)
        gdof = self.number_of_global_dofs(p=p)


        # Construct the stiffness matrix
        A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return A

    def mass_matrix(self, p=None):
        M = self.cell_mass_matrix(p=p) # 单元质量矩阵
        cell2dof = self.cell_to_dof(p=p)
        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        I = np.einsum('k, ij->ijk', np.ones(ldof), cell2dof)
        J = I.swapaxes(-1, -2)
        gdof = self.number_of_global_dofs(p=p)
        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return M 

    def penalty_matrix(self, p=None):
        p = p or self.p
        mesh = self.mesh
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()
        isInEdge = edge2cell[:, 0] != edge2cell[:, 1]
        eh = mesh.entity_measure('edge')
        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights
        ps = self.mesh.edge_bc_to_point(bcs)

        ldof = self.number_of_local_dofs(doftype='cell')
        shape = ps.shape[:-1] + (2*ldof, )
        phi = np.zeros(shape, dtype=self.ftype)
        phi[:, :, :ldof] = self.basis(ps, index=edge2cell[:, 0]) # (NQ, NE, ldof)
        phi[:, isInEdge, ldof:] = -self.basis(ps, index=edge2cell[isInEdge, 1]) # (NQ, NE, ldof)
        H = np.einsum('i, ijk, ijm, j->jkm', ws, phi, phi, eh, optimize=True)
        cell2dof = self.cell_to_dof()

    def source_vector(self, f, celltype=False, q=None):
        """

        """

        cell2dof = self.cell_to_dof()
        gdof = self.number_of_global_dofs()
        b = (self.basis, cell2dof, gdof)
        F = self.integralalg.serial_construct_vector(f, b, celltype=celltype,
                q=q) 
        return F 

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

        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        H = np.zeros((NC, ldof, ldof), dtype=np.float)
        np.add.at(H, edge2cell[:, 0], H0)
        np.add.at(H, edge2cell[isInEdge, 1], H1)

        multiIndex = self.dof.multi_index_matrix(p=p)
        q = np.sum(multiIndex, axis=1)
        H /= q + q.reshape(-1, 1) + 2
        return H

    def local_projection(self, f, q=None):
        """

        Notes
        -----

        结定一个函数 f， 把它投影到缩放单项式空间
        """

        def u(x, index):
            return np.einsum('ij, ijm->ijm', f(x), self.basis(x, index=index))
        b = self.integralalg.integral(u, celltype=True)
        M = self.cell_mass_matrix()
        F = inv(M)@b[:, :, None]
        F = self.function(array=F.reshape(-1))
        return F

    def projection(self, f):
        """

        Notes
        -----
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
        ldofs = self.number_of_local_dofs(doftype='cell')
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

    @cartesian
    def to_cspace_function(self, uh):
        """
        Notes
        -----
        把分片的 p 次多项式空间的函数  uh， 恢复到分片连续的函数空间。这里假定网
        格是三角形网格。

        TODO
        ----
        1. 实现多个函数同时恢复的情形 
        """

        # number of function in uh

        p = self.p
        mesh  = self.mesh
        bcs = multi_index_matrix2d(p)
        ps = mesh.bc_to_point(bcs)
        val = self.value(uh, ps) # （NQ, NC, ...)

        space = LagrangeFiniteElementSpace(mesh, p=p)
        gdof = space.number_of_global_dofs()
        cell2dof = space.cell_to_dof()
        deg = np.zeros(gdof, dtype=space.itype)
        np.add.at(deg, cell2dof, 1)
        ruh = space.function()
        np.add.at(ruh, cell2dof, val.T)
        ruh /= deg
        return ruh

