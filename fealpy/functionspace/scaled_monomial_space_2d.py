import matplotlib.pyplot as plt

#from .femdof import multi_index_matrix2d, multi_index_matrix1d
#from .lagrange_fe_space import LagrangeFESpace

from typing import Optional, TypeVar, Union, Generic, Callable
from ..typing import TensorLike, Index, _S, Threshold

from ..backend import TensorLike
from ..backend import backend_manager as bm
from ..mesh.mesh_base import Mesh
from .space import FunctionSpace
from .dofs import LinearMeshCFEDof, LinearMeshDFEDof
from .function import Function
from fealpy.decorator import barycentric, cartesian


_MT = TypeVar('_MT', bound=Mesh)

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
        idx = bm.arange(0, ldof)
        idx0 = bm.floor((-1 + bm.sqrt(1 + 8*idx))/2)
        multiIndex = bm.zeros((ldof, 2), dtype=bm.int32)
        multiIndex[:, 1] = idx - idx0*(idx0 + 1)/2
        multiIndex[:, 0] = idx0 - multiIndex[:, 1]
        return multiIndex

    def cell_to_dof(self, p=None):
        mesh = self.mesh
        NC = mesh.number_of_cells()
        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        cell2dof = bm.arange(NC*ldof).reshape(NC, ldof)
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


class ScaledMonomialSpace2d(FunctionSpace, Generic[_MT]):
    def __init__(self, mesh, p, q=None, bc=None):
        """
        The Scaled Momomial Space in R^2
        """

        self.mesh = mesh
        self.cellbarycenter = mesh.entity_barycenter('cell') if bc is None else bc
        self.p = p
        self.cellmeasure = mesh.entity_measure('cell')

        self.cellsize = bm.sqrt(self.cellmeasure)
        self.dof = SMDof2d(mesh, p)
        self.GD = 2

        q = q if q is not None else p+3

        mtype = mesh.meshtype

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

        x, = bm.nonzero(index[:, 1] > 0) # 关于 x 求导非零的缩放单项式编号
        y, = bm.nonzero(index[:, 2] > 0) # 关于 y 求导非零的缩放单项式编号

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

        xx, = bm.nonzero(index[:, 1] > 1)
        yy, = bm.nonzero(index[:, 2] > 1)

        xy, = bm.nonzero((index[:, 1] > 0) & (index[:, 2] > 0))

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
        x, = bm.nonzero(index[:, 0] > 0)
        y, = bm.nonzero(index[:, 1] > 0)
        return {'x': x, 'y':y}

    def edge_index_1(self, p=None):
        """
        Parameters
        ----------
        p : >= 1
        """
        p = self.p if p is None else p
        index = multi_index_matrix1d(p)
        x, = bm.nonzero(index[:, 0] > 0)
        y, = bm.nonzero(index[:, 1] > 0)
        return {'x': x, 'y':y}

    def geo_dimension(self):
        return self.GD

    def cell_to_dof(self, p=None):
        return self.dof.cell_to_dof(p=p)

    @cartesian
    def edge_basis(self, point, index=_S, p=None):
        p = self.p if p is None else p
        if p == 0:
            shape = len(point.shape)*(1, )
            return bm.array([1.0], dtype=self.ftype).reshape(shape)

        ec = self.integralalg.edgebarycenter
        eh = self.integralalg.edgemeasure
        et = self.mesh.edge_unit_tangent()
        val = bm.sum((point - ec[index])*et[index], axis=-1)/eh[index]
        phi = bm.ones(val.shape + (p+1,), dtype=self.ftype)
        if p == 1:
            phi[..., 1] = val
        else:
            phi[..., 1:] = val[..., bm.newaxis]
            bm.multiply.accumulate(phi, axis=-1, out=phi)
        return phi

    @barycentric
    def edge_basis_with_barycentric(self, bcs, p=None):
        """!
        @brief 边上的重心坐标函数和缩放单项式函数有一定的关系
        @param bcs : (..., 2)
        @return phi : (..., p+1)
        """
        p = self.p if p is None else p
        if p == 0:
            shape = len(bcs.shape)*(1, )
            return bm.array([[1.0]], dtype=self.ftype).reshape(shape)
        else:
            shape = bcs.shape[:-1]+(p+1, )
            phi = bm.ones(shape, dtype=self.ftype)
            phi[..., 1:] = bcs[..., 1, None]-0.5
            bm.multiply.accumulate(phi, axis=-1, out=phi)
            return phi

    @cartesian
    def basis(self, point, index=_S, p=None):
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
            return bm.array([1.0], dtype=self.ftype).reshape(shape)

        shape = point.shape[:-1]+(ldof,)
        phi = bm.ones(shape, dtype=self.ftype)  # (..., M, ldof)

        phi[..., 1:3] = (point - self.cellbarycenter[index])/h[index].reshape(-1, 1)
        if p > 1:
            start = 3
            for i in range(2, p+1):
                phi[..., start:start+i] = phi[..., start-i:start]*phi[..., [1]]
                phi[..., start+i] = phi[..., start-1]*phi[..., 2]
                start += i+1
        return phi

    @cartesian
    def grad_basis(self, point, index=_S, p=None, scaled=True):
        """

        p >= 0
        """

        p = self.p if p is None else p
        h = self.cellsize

        num = len(h) if type(index) is slice else len(index)

        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        shape = point.shape[:-1]+(ldof, 2)
        gphi = bm.zeros(shape, dtype=self.ftype)

        if p == 0:
            return gphi

        phi = self.basis(point, index=index, p=p-1)
        idx = self.diff_index_1(p=p)
        xidx = idx['x']
        yidx = idx['y']
        gphi[..., xidx[0], 0] = bm.einsum('i, ...i->...i', xidx[1], phi)
        gphi[..., yidx[0], 1] = bm.einsum('i, ...i->...i', yidx[1], phi)

        if scaled:
            if point.shape[-2] == num:
                return gphi/h[index].reshape(-1, 1, 1)
            elif point.shape[0] == num:
                return gphi/h[index].reshape(-1, 1, 1, 1)
        else:
            return gphi

    @cartesian
    def laplace_basis(self, point, index=_S, p=None, scaled=True):
        p = self.p if p is None else p

        area = self.cellmeasure
        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        shape = point.shape[:-1]+(ldof,)
        lphi = bm.zeros(shape, dtype=self.ftype)
        if p > 1:
            phi = self.basis(point, index=index, p=p-2)
            idx = self.diff_index_2(p=p)
            lphi[..., idx['xx'][0]] += bm.einsum('i, ...i->...i', idx['xx'][1], phi)
            lphi[..., idx['yy'][0]] += bm.einsum('i, ...i->...i', idx['yy'][1], phi)

        if scaled:
            return lphi/area[index].reshape(-1, 1)
        else:
            return lphi

    @cartesian
    def hessian_basis(self, point, index=_S, p=None, scaled=True):
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
        hphi = bm.zeros(shape, dtype=self.ftype)
        if p > 1:
            phi = self.basis(point, index=index, p=p-2)
            idx = self.diff_index_2(p=p)
            hphi[..., idx['xx'][0], 0, 0] = bm.einsum('i, ...i->...i', idx['xx'][1], phi)
            hphi[..., idx['xy'][0], 0, 1] = bm.einsum('i, ...i->...i', idx['xy'][1], phi)
            hphi[..., idx['yy'][0], 1, 1] = bm.einsum('i, ...i->...i', idx['yy'][1], phi)
            hphi[..., 1, 0] = hphi[..., 0, 1]

        if scaled:
            return hphi/area[index].reshape(-1, 1, 1, 1)
        else:
            return hphi

    def grad_m_basis(self, m, point, index=_S, p=None, scaled=True):
        """!
        @brief m=3时导数排列顺序: [xxx, xxy, xyx, xyy, yxx, yxy, yyx, yyy]
        """
        #TODO test
        phi = self.basis(point, index=index, p=p)
        gmphi = bm.zeros(phi.shape+(2**m, ), dtype=self.ftype)
        P = self.partial_matrix(index=index)
        f = lambda x: bm.array([int(ss) for ss in bm.binary_repr(x, m)], dtype=bm.int32)
        idx = bm.array(list(map(f, bm.arange(2**m))))
        for i in range(2**m):
            M = P[idx[i, 0]].copy()
            for j in range(1, m):
                M = bm.einsum("cij, cjk->cik", M, P[idx[i, j]])
            gmphi[..., i] = bm.einsum('cli, ...cl->...ci', M, phi)
        return gmphi

    def partial_matrix(self, p=None, index=_S):
        p = p or self.p
        #mindex = multi_index_matrix2d(p)
        mindex = bm.multi_index_matrix(p, 2) # TODO
        N = len(mindex)
        cellarea = self.mesh.entity_measure("cell")
        NC = self.mesh.number_of_cells()
        h = bm.sqrt(cellarea)

        I, = bm.where(mindex[:, 1] > 0)
        Px = bm.zeros([NC, N, N], dtype=self.ftype)
        Px[:, bm.arange(len(I)), I] = mindex[None, I, 1]/h[:, None]

        I, = bm.where(mindex[:, 2] > 0)
        Py = bm.zeros([NC, N, N], dtype=self.ftype)
        Py[:, bm.arange(len(I)), I] = mindex[None, I, 2]/h[:, None]
        return Px[index], Py[index]

    def partial_matrix_on_edge(self, p=None):
        p = p or self.p
        I = bm.arange(p)

        h = self.mesh.entity_measure("edge")
        NE = self.mesh.number_of_edges()

        P = bm.zeros([NE, p+1, p+1], dtype=self.ftype)
        P[:, I, I+1] = bm.arange(1, p+1)[None, :]/h[:, None]
        return P

    @cartesian
    def value(self, uh, point, index=_S):
        phi = self.basis(point, index=index)
        cell2dof = self.dof.cell2dof[index]
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ij, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        return bm.einsum(s1, phi, uh[cell2dof])

    @cartesian
    def grad_value(self, uh, point, index=_S):
        gphi = self.grad_basis(point, index=index)
        cell2dof = self.dof.cell2dof
        if (type(index) is bm.ndarray) and (index.dtype.name == 'bool'):
            N = bm.sum(index)
        elif type(index) is slice:
            N = len(cell2dof)
        else:
            N = len(index)
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        if point.shape[-2] == N:
            s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
            return bm.einsum(s1, gphi, uh[cell2dof[index]])
        elif point.shape[0] == N:
            return bm.einsum('ikjm, ij->ikm', gphi, uh[cell2dof[index]])

    @cartesian
    def laplace_value(self, uh, point, index=_S):
        lphi = self.laplace_basis(point, index=index)
        cell2dof = self.dof.cell2dof
        return bm.einsum('...ij, ij->...i', lphi, uh[cell2dof[index]])

    @cartesian
    def hessian_value(self, uh, point, index=_S):
        hphi = self.hessian_basis(point, index=index) #(NQ, NC, ldof, 2, 2)
        cell2dof = self.dof.cell2dof
        return bm.einsum('...clij, cl->...cij', hphi, uh[cell2dof[index]])

    @cartesian
    def grad_3_value(self, uh, point, index=_S):
        #TODO
        gmphi = self.grad_m_basis(3, point, index=index) #(NQ, NC, ldof, 8)
        cell2dof = self.dof.cell2dof
        return bm.einsum('...cli, cl->...ci', gmphi, uh[cell2dof[index]])

    def function(self, dim=None, array=None, dtype=None):
        ftype = self.ftype if dtype is None else dtype
        f = Function(self, dim=dim, array=array, coordtype='cartesian',
                dtype=ftype)
        return f

    def array(self, dim=None, dtype=None):
        ftype = self.ftype if dtype is None else dtype
        gdof = self.number_of_global_dofs()
        if dim in {None, 1}:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return bm.zeros(shape, dtype=dtype)

    def dof_array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim in {None, 1}:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return bm.zeros(shape, dtype=self.ftype)

    def number_of_local_dofs(self, p=None, doftype='cell'):
        return self.dof.number_of_local_dofs(p=p, doftype=doftype)

    def number_of_global_dofs(self, p=None):
        return self.dof.number_of_global_dofs(p=p)
    def show_function_image(self, u, uh, t=None, plot_solution=True):
        mesh = uh.space.mesh
        fig = plt.figure()
        fig.set_facecolor('white')
        axes = plt.axes(projection='3d')

        NE = mesh.number_of_edges()
        mid = mesh.entity_barycenter("cell")
        node = mesh.entity("node")
        edge = mesh.entity("edge")
        edge2cell = mesh.ds.edge_to_cell()

        coor = bm.zeros([2*NE, 3, 2], dtype=bm.float64)
        coor[:NE, :2] = node[edge]
        coor[:NE, 2] = mid[edge2cell[:, 0]]
        coor[NE:, :2] = node[edge]
        coor[NE:, 2] = mid[edge2cell[:, 1]]

        val = bm.zeros([2*NE, 3])
        val[:NE] = uh(coor[:NE].swapaxes(0, 1), index=edge2cell[:, 0]).swapaxes(0 ,1)
        val[NE:] = uh(coor[NE:].swapaxes(0, 1), index=edge2cell[:, 1]).swapaxes(0 ,1)

        for ii in range(2*NE):
            axes.plot_trisurf(coor[ii, :, 0], coor[ii, :, 1], val[ii], color = 'r', lw=0.0)#数值解图像
        if plot_solution:
            if t is not None:
                fval = u(coor.swapaxes(0, 1), t).swapaxes(0, 1)
            else:
                fval = u(coor.swapaxes(0, 1)).swapaxes(0, 1)
            for ii in range(2*NE):
                axes.plot_trisurf(coor[ii, :, 0], coor[ii, :, 1], fval[ii], color = 'b', lw=0.0)#真解图像
        plt.show()
        return

    def cell_mass_matrix(self, p=None):
        return self.matrix_H(p=p)

    def edge_mass_matrix(self, p=None):
        p = self.p if p is None else p
        mesh = self.mesh
        NE = mesh.number_of_edges()
        edge = mesh.entity('edge')
        eh = mesh.entity_measure('edge')
        ldof = p + 1
        q = bm.arange(p+1)
        Q = q.reshape(-1, 1) + q + 1
        flag = Q%2==1
        H = bm.zeros((NE, ldof, ldof), dtype=self.ftype)
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
        H = bm.einsum('i, ijk, ijm, j->jkm', ws, phi, phi, measure, optimize=True)
        return H

    def edge_cell_mass_matrix(self, p=None, cp=None):
        p = self.p if p is None else p
        cp = p+1 if cp is None else cp

        mesh = self.mesh

        edge = mesh.entity('edge')
        measure = mesh.entity_measure('edge')

        edge2cell = mesh.ds.edge_to_cell()

        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights
        ps = self.mesh.edge_bc_to_point(bcs)

        phi0 = self.edge_basis(ps, p=p)
        phi1 = self.basis(ps, index=edge2cell[:, 0], p=cp)
        phi2 = self.basis(ps, index=edge2cell[:, 1], p=cp)
        LM = bm.einsum('i, ijk, ijm, j->jkm', ws, phi0, phi1, measure, optimize=True)
        RM = bm.einsum('i, ijk, ijm, j->jkm', ws, phi0, phi2, measure, optimize=True)
        return LM, RM

    def cell_hessian_matrix(self, p=None):
        """

        Note:
            这个程序仅用于多边形网格上 (\nabla^2 u, \nabla^2 v)
        """
        p = self.p if p is None else p
        @cartesian
        def f(x, index):
            hphi = self.hessian_basis(x, index=index, p=p)
            return bm.einsum('qclij, qcmij->qclm', hphi, hphi)

        A = self.integralalg.cell_integral(f, q=p+3)
        if 0:
            M = self.cell_mass_matrix()
            Px, Py = self.partial_matrix()
            Pxx = bm.einsum("cij, cjk->cik", Px, Px)
            Pxy = bm.einsum("cij, cjk->cik", Px, Py)
            Pyy = bm.einsum("cij, cjk->cik", Py, Py)
            v0 = bm.einsum('cji, cjk, ckl->cil', Pxx, M, Pxx)
            v1 = bm.einsum('cji, cjk, ckl->cil', Pxy, M, Pxy)
            v2 = bm.einsum('cji, cjk, ckl->cil', Pyy, M, Pyy)
        return A

    def cell_grad_m_matrix(self, m, p=None):
        """

        Note:
            这个程序仅用于多边形网格上 (\nabla^2 u, \nabla^2 v)
        """
        p = self.p if p is None else p
        @cartesian
        def f(x, index):
            gmphi = self.grad_m_basis(m, x, index=index, p=p)
            return bm.einsum('qcli, qcmi->qclm', gmphi, gmphi)

        A = self.integralalg.cell_integral(f, q=p+3)
        return A

    def stiff_matrix(self, p=None):
        """

        Note:
            这个程序仅用于多边形网格上的刚度矩阵组装
        """
        p = self.p if p is None else p

        @cartesian
        def f(x, index):
            gphi = self.grad_basis(x, index=index, p=p)
            return bm.einsum('ijkm, ijpm->ijkp', gphi, gphi)

        A = self.integralalg.cell_integral(f, q=p+3)
        cell2dof = self.cell_to_dof(p=p)
        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        I = bm.einsum('k, ij->ijk', bm.ones(ldof), cell2dof)
        J = I.swapaxes(-1, -2)
        gdof = self.number_of_global_dofs(p=p)


        # Construct the stiffness matrix
        A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return A

    def mass_matrix(self, p=None):
        M = self.cell_mass_matrix(p=p) # 单元质量矩阵
        cell2dof = self.cell_to_dof(p=p)
        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        I = bm.einsum('k, ij->ijk', bm.ones(ldof), cell2dof)
        J = I.swapaxes(-1, -2)
        gdof = self.number_of_global_dofs(p=p)
        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return M

    def penalty_matrix(self, p=None, index=_S):
        """
        Notes
        -----

        h_e^{-1}<[u], [v]>_e

        """
        p = p or self.p
        mesh = self.mesh
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()[index]
        isInEdge = edge2cell[:, 0] != edge2cell[:, 1]

        eh = mesh.entity_measure('edge')
        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights
        ps = self.mesh.edge_bc_to_point(bcs, index=index)

        gdof = self.number_of_global_dofs(p=p)
        ldof = self.number_of_local_dofs(doftype='cell')

        shape = ps.shape[:-1] + (2*ldof, )
        phi = bm.zeros(shape, dtype=self.ftype)

        phi[:, :, :ldof] = self.basis(ps, index=edge2cell[:, 0]) # (NQ, NE, ldof)
        phi[:, isInEdge, ldof:] = -self.basis(ps[:, isInEdge], index=edge2cell[isInEdge, 1]) # (NQ, NE, ldof)

        A = bm.einsum('q, qei, qej->eij', ws, phi, phi, optimize=True)

        cell2dof = self.cell_to_dof()
        edge2dof = bm.block([cell2dof[edge2cell[:, 0]], cell2dof[edge2cell[:, 1]]])
        I = bm.broadcast_to(edge2dof[:, :, None], shape=A.shape)
        J = bm.broadcast_to(edge2dof[:, None, :], shape=A.shape)
        A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return A


    def flux_matrix(self, p=None, index=_S):
        '''
        Notes:
        ------
        <{u_n}, [v]>_e

        '''
        p = p or self.p
        mesh = self.mesh
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()[index]
        isInEdge = edge2cell[:, 0] != edge2cell[:, 1]

        ldof = self.number_of_local_dofs(doftype='cell')
        gdof = self.number_of_global_dofs(p=p)
        GD = mesh.geo_dimension()

        eh = mesh.entity_measure('edge', index=index)
        en = mesh.edge_normal(index=index)
        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights
        ps = self.mesh.edge_bc_to_point(bcs, index=index)

        shape = ps.shape[:-1] + (2*ldof, )
        phi = bm.zeros(shape, dtype=self.ftype)
        gphi = bm.zeros(shape, dtype=self.ftype)

        phi[:, :, :ldof] = self.basis(ps, index=edge2cell[:, 0]) # (NQ, NE, ldof)
        phi[:, isInEdge, ldof:] = -self.basis(ps[:, isInEdge], index=edge2cell[isInEdge, 1]) # (NQ, NE, ldof)

        gphi[:, ~isInEdge, :ldof]= bm.sum(self.grad_basis(ps[: ,~isInEdge], index=edge2cell[~isInEdge, 0])*en[~isInEdge, None, :], axis=-1) # (NQ, NE, ldof)
        gphi[:, isInEdge, :ldof]= 0.5*bm.sum(self.grad_basis(ps[:, isInEdge], index=edge2cell[isInEdge, 0])*en[isInEdge, None, :], axis=-1) # (NQ, NE, ldof)
        gphi[:, isInEdge, ldof:] = 0.5*bm.sum(self.grad_basis(ps[:, isInEdge], index=edge2cell[isInEdge, 1])*en[isInEdge, None, :], axis=-1) # (NQ, NE, ldof)

        A = bm.einsum('q, qei, qej->eij', ws, phi, gphi, optimize=True)

        cell2dof = self.cell_to_dof()
        edge2dof = bm.block([cell2dof[edge2cell[:, 0]], cell2dof[edge2cell[:, 1]]])
        I = bm.broadcast_to(edge2dof[:, :, None], shape=A.shape)
        J = bm.broadcast_to(edge2dof[:, None, :], shape=A.shape)
        A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return A

    def normal_grad_penalty_matrix(self, p=None, index=_S):
        """
        Notes
        -----

        \\beta h_e < [u_n], [v_n]>_e
        """
        p = p or self.p
        mesh = self.mesh
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()[index]
        isInEdge = edge2cell[:, 0] != edge2cell[:, 1]

        ldof = self.number_of_local_dofs(doftype='cell')
        gdof = self.number_of_global_dofs(p=p)
        GD = mesh.geo_dimension()

        eh = mesh.entity_measure('edge', index=index)
        en = mesh.edge_normal(index=index)
        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights
        ps = self.mesh.edge_bc_to_point(bcs, index=index)

        shape = ps.shape[:-1] + (2*ldof, )
        gphi = bm.zeros(shape, dtype=self.ftype)

        gphi[:, :, :ldof]= bm.sum(self.grad_basis(ps, index=edge2cell[:, 0])*en[:, None, :], axis=-1) # (NQ, NE, ldof)
        gphi[:, isInEdge, ldof:] = -bm.sum(self.grad_basis(ps[:, isInEdge], index=edge2cell[isInEdge, 1])*en[isInEdge, None, :], axis=-1) # (NQ, NE, ldof)

        A = bm.einsum('q, qei, qej->eij', ws, gphi, gphi, optimize=True)

        cell2dof = self.cell_to_dof()
        edge2dof = bm.block([cell2dof[edge2cell[:, 0]], cell2dof[edge2cell[:, 1]]])
        I = bm.broadcast_to(edge2dof[:, :, None], shape=A.shape)
        J = bm.broadcast_to(edge2dof[:, None, :], shape=A.shape)
        A = csr_matrix((A.flat, (I.flat, J.flat)), shape=(gdof, gdof))
        return A

    def edge_normal_source_vector(self, g, p = None, index=_S):

        """
        Notes
        -----

        h_e^{-1}<g, [v_n]>_e
        """

        p = p or self.p
        mesh = self.mesh
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()[index]
        isInEdge = edge2cell[:, 0] != edge2cell[:, 1]

        eh = mesh.entity_measure('edge', index=index)
        en = mesh.edge_normal(index=index)

        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights
        ps = self.mesh.edge_bc_to_point(bcs, index=index) #(NQ, NE, 2)

        gval = g(ps) #(NQ, NE)

        ldof = self.number_of_local_dofs(doftype='cell')
        gdof = self.number_of_global_dofs(p=p)

        shape = ps.shape[:-1] + (2*ldof, )
        gphi = bm.zeros(shape, dtype=self.ftype)

        gphi[:, :, :ldof]= bm.sum(self.grad_basis(ps, index=edge2cell[:, 0])*en[:, None, :], axis=-1) # (NQ, NE, ldof)
        gphi[:, isInEdge, ldof:] = -bm.sum(self.grad_basis(ps[:, isInEdge], index=edge2cell[isInEdge, 1])*en[isInEdge, None, :], axis=-1) # (NQ, NE, ldof)

        A = bm.einsum('q, qe, qej->ej', ws, gval, gphi, optimize=True)

        F = bm.zeros(gdof, dtype=self.ftype)
        cell2dof = self.cell_to_dof()
        bm.add.at(F, cell2dof[edge2cell[:, 0]], A[:, :ldof])
        bm.add.at(F, cell2dof[edge2cell[isInEdge, 1]], A[isInEdge, ldof:])
        return F

    def edge_source_vector(self, g, p = None, index=_S, hpower=-1):

        """
        Notes
        -----

        其中 g 可以是标量函数也可以是向量函数, 当是标量函数的时候计算的是:

        h_e^{hpower}<g, [v]>_e

        当是向量函数的时候计算的是:

        h_e^{hpower}<g_n, [v]>_e


        """
        p = p or self.p
        mesh = self.mesh
        edge = mesh.entity('edge')
        edge2cell = mesh.ds.edge_to_cell()[index]
        isInEdge = edge2cell[:, 0] != edge2cell[:, 1]

        eh = mesh.entity_measure('edge', index=index)
        en = mesh.edge_unit_normal(index=index)

        qf = GaussLegendreQuadrature(p + 3)
        bcs, ws = qf.quadpts, qf.weights
        ps = self.mesh.edge_bc_to_point(bcs, index=index)

        gval = g(ps)
        if len(gval.shape)==3:
            gval = bm.einsum('qei, ei->qe', gval, en)

        ldof = self.number_of_local_dofs(doftype='cell')
        gdof = self.number_of_global_dofs(p=p)

        shape = ps.shape[:-1] + (2*ldof, )
        phi = bm.zeros(shape, dtype=self.ftype)

        phi[:, :, :ldof] = self.basis(ps, index=edge2cell[:, 0]) # (NQ, NE, ldof)
        phi[:, isInEdge, ldof:] = -self.basis(ps[:, isInEdge], index=edge2cell[isInEdge, 1]) # (NQ, NE, ldof)

        S = bm.einsum('q, qe, qei->ei', ws, gval, phi, optimize=True)
        if hpower!=-1:
            S = S*(eh.reshape(-1, 1)**(hpower+1))
        F = bm.zeros(gdof, dtype=self.ftype)
        cell2dof = self.cell_to_dof()
        bm.add.at(F, cell2dof[edge2cell[:, 0]], S[:, :ldof])
        bm.add.at(F, cell2dof[edge2cell[isInEdge, 1]], S[isInEdge, ldof:])
        return F

    def source_vector0(self, f, p = None, celltype=False, q=None):
        """
        @brief (f, v)_T
        @param f : (NQ, NC, 2) -> (NQ, NC) or (NQ, NC, l)
        """

        p = p if p is not None else self.p
        mesh = self.mesh
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        gdof = self.number_of_global_dofs(p=p)

        bc = mesh.entity_barycenter('cell')
        cell2dof = self.cell_to_dof(p=p)
        edge2cell = mesh.ds.edge_to_cell()

        qf = mesh.integrator(p+4)
        bcs, ws = qf.quadpts, qf.weights
        tri_0 = [bc[edge2cell[:, 0]], node[edge[:, 0]], node[edge[:, 1]]]
        a_0 = self.triangle_measure(tri_0)#NE
        pp_0 = bm.einsum('qj, jem->qem', bcs, tri_0)#每个三角形中高斯积分点对应的笛卡尔坐标点
        fval_0 = f(pp_0)

        if len(fval_0.shape)>2:
            shape = (gdof, fval_0.shape[-1])
        else:
            shape = (gdof, )

        phi_0 = self.basis(pp_0, edge2cell[:, 0], p=p)

        F = bm.zeros(shape, dtype=self.ftype)
        bb_0 = bm.einsum('q, qe..., qel,e->el...', ws, fval_0, phi_0, a_0)

        bm.add.at(F, cell2dof[edge2cell[:, 0]], bb_0)
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        if bm.sum(isInEdge) > 0:
            tri_1 = [
                    bc[edge2cell[isInEdge, 1]],
                    node[edge[isInEdge, 1]],
                    node[edge[isInEdge, 0]]
                    ]
            a_1 = self.triangle_measure(tri_1)
            pp_1 = bm.einsum('ij, jkm->ikm', bcs, tri_1)
            fval_1 = f(pp_1)
            phi_1 = self.basis(pp_1, edge2cell[isInEdge, 1], p=p)
            bb_1 = bm.einsum('q, qe..., qel,e->el...', ws, fval_1, phi_1, a_1)
            bm.add.at(F, cell2dof[edge2cell[isInEdge, 1]], bb_1)
        return F

    def source_vector1(self, f, celltype=False, q=None):
        """
        @brief (f, v)_T
        @param f : (NQ, NC, 2) -> (NQ, NC) or (NQ, NC, l), 即 f 可以是一个高维函数
        """

        p = self.p
        if q is None: q = p
        mesh = self.mesh
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        gdof = self.number_of_global_dofs(p=p)

        bc = mesh.entity_barycenter('cell')
        cell2dof = self.cell_to_dof()
        edge2cell = mesh.ds.edge_to_cell()

        qf = mesh.integrator(q)
        bcs, ws = qf.quadpts, qf.weights
        tri_0 = [bc[edge2cell[:, 0]], node[edge[:, 0]], node[edge[:, 1]]]
        a_0 = self.triangle_measure(tri_0)#NE
        pp_0 = bm.einsum('qj, jem->qem', bcs, tri_0)#每个三角形中高斯积分点对应的笛卡尔坐标点
        fval_0 = f(pp_0, edge2cell[:, 0])

        if len(fval_0.shape)>2:
            shape = (gdof, fval_0.shape[-1])
        else:
            shape = (gdof, )

        phi_0 = self.basis(pp_0, edge2cell[:, 0])

        F = bm.zeros(shape, dtype=self.ftype)
        bb_0 = bm.einsum('q, qe..., qel,e->el...', ws, fval_0, phi_0, a_0)

        bm.add.at(F, cell2dof[edge2cell[:, 0]], bb_0)
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        if bm.sum(isInEdge) > 0:
            tri_1 = [
                    bc[edge2cell[isInEdge, 1]],
                    node[edge[isInEdge, 1]],
                    node[edge[isInEdge, 0]]
                    ]
            a_1 = self.triangle_measure(tri_1)
            pp_1 = bm.einsum('ij, jkm->ikm', bcs, tri_1)
            fval_1 = f(pp_1, edge2cell[isInEdge, 1])
            phi_1 = self.basis(pp_1, edge2cell[isInEdge, 1])
            bb_1 = bm.einsum('q, qe..., qel,e->el...', ws, fval_1, phi_1, a_1)
            bm.add.at(F, cell2dof[edge2cell[isInEdge, 1]], bb_1)
        return F

    def coefficient_of_cell_basis_under_edge_basis(slef, p=None):
        """!
        @brief 计算单元上基函数在边界基函数上的系数
        """
        p = p or self.p
        mesh = self.mesh
        NC = self.mesh.number_of_cells()
        cell2edge, cell2edgeloc = mesh.ds.cell_to_edge()
        ldof = self.dof.number_of_local_dofs()

        if p == 0:
            bcs = bm.array([[0.5, 0.5]])
        else:
            bcs = bm.zeros((p+1, 2), dtype=self.ftype)
            bcs[:, 0] = bm.arange(p)/p
            bcs[:, 1] = 1-bcs[:, 1]

        #M (NE, p+1, p+1) 是每个边上的 p+1 个基函数在 p+1 个点处的值组成矩阵的逆
        M = bm.linalg.inv(self.edge_basis_with_barycentric(bcs, p))
        #C (N, ldof, p+1) 是每个单元基函数在每条边上基函数的系数
        C = bm.zeros((cell2edgeloc[-1], ldof, p+1), dtype=self.ftype)

        points = self.mesh.bc_to_point(bcs) #(p+1, NE, 2)
        isNotOK = bm.ones(NC, dtype=bm.bool_)
        start = cell2edgeloc[:-1].copy()
        while bm.any(isNotOK):
            index = start[isNotOK]
            eidx = cell2edge[index]
            phi = self.basis(points[:, eidx], index=isNotOK, p=p) #(p+1, NC, ldof)
            C[index] = bm.einsum("cij, jcl->cli", M[eidx], phi)
            start[isNotOK] = start[isNotOK]+1
            isNotOK = start<cell2edgeloc[1:]
        return C

    def triangle_measure(self, tri):
        v1 = tri[1] - tri[0]
        v2 = tri[2] - tri[0]
        area = bm.cross(v1, v2)/2
        return area

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
        ps = bm.einsum('ij, kjm->ikm', bcs, node[edge])
        phi0 = self.basis(ps, index=edge2cell[:, 0], p=p)
        phi1 = self.basis(ps[:, isInEdge, :], index=edge2cell[isInEdge, 1], p=p)
        H0 = bm.einsum('i, ijk, ijm->jkm', ws, phi0, phi0)
        H1 = bm.einsum('i, ijk, ijm->jkm', ws, phi1, phi1)

        nm = mesh.edge_normal()
        b = node[edge[:, 0]] - self.cellbarycenter[edge2cell[:, 0]]
        H0 = bm.einsum('ij, ij, ikm->ikm', b, nm, H0)
        b = node[edge[isInEdge, 0]] - self.cellbarycenter[edge2cell[isInEdge, 1]]
        H1 = bm.einsum('ij, ij, ikm->ikm', b, -nm[isInEdge], H1)

        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        H = bm.zeros((NC, ldof, ldof), dtype=self.ftype)
        bm.add.at(H, edge2cell[:, 0], H0)
        bm.add.at(H, edge2cell[isInEdge, 1], H1)

        multiIndex = self.dof.multi_index_matrix(p=p)
        q = bm.sum(multiIndex, axis=1)
        H /= q + q.reshape(-1, 1) + 2
        return H

    def local_projection(self, f, q=None):
        """

        @brief 给定一个函数 f， 把它投影到缩放单项式空间
        @param[in] f 关于（x, y) 的函数，注意输入是笛卡尔坐标

        """

        @cartesian
        def u(x, *args):
            if len(args) == 0:
                return bm.einsum('ij, ijm->ijm', f(x), self.basis(x))
            elif len(args) == 1:
                index, = args
                return bm.einsum('ij, ijm->ijm', f(x), self.basis(x, index=index))

        b = self.integralalg.cell_integral(u)
        M = self.cell_mass_matrix()
        F = inv(M)@b[:, :, None]
        F = self.function(array=F.reshape(-1))
        return F

    def projection(self, F):
        """

        Notes
        -----
        """
        mspace = F.space
        C = self.matrix_C(mspace)
        H = self.matrix_H()
        PI0 = inv(H)@C
        SS = self.function()
        SS[:] = bm.einsum('ikj, ij->ik', PI0, F[self.cell_to_dof()]).reshape(-1)
        return SS

    def matrix_C(self, mspace):
        def f(x, index):
            return bm.einsum(
                    '...im, ...in->...imn',
                    self.basis(x, index),
                    mspace.basis(x, index)
                    )
        C = self.integralalg.integral(f, celltype=True)
        return C

    def matrix_H_in(self):
        def f(x, index):
            phi = self.basis(x, index)
            return bm.einsum(
                    '...im, ...in->...imn',
                    phi, phi
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

        num = bm.zeros(NC, dtype=self.itype)
        bm.add.at(num, HB[:, 0], 1)

        m = HB.shape[0]
        td = bm.zeros((m, ldofs), dtype=self.ftype)

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

        bm.add.at(d, (HB[:, 0], _S), td)
        d /= num.reshape(-1, 1)
        return sh1

    def error(self, u, uh):
        """!
        @brief 求 H1 误差
        """

        def f(p, index):
            val = (u(p) - uh.value(p, index))**2
            return val
        err = bm.sqrt(self.integralalg.integral(f))
        return err
    def H1_error(self, u, uh):
        """!
        @brief 求 H1 误差
        """

        def f(p, index):
            val = bm.sum((u(p) - uh.grad_value(p, index))**2, axis=-1)
            return val
        err = bm.sqrt(self.integralalg.integral(f))
        return err

    def H2_error(self, u, uh):
        """!
        @brief 求 H2 误差
        """

        def f(p, index):
            val = bm.sum(bm.sum((u(p) - uh.hessian_value(p, index))**2, axis=-1),
                    axis=-1)
            return val
        err = bm.sqrt(self.integralalg.integral(f))
        return err

    def H3_error(self, u, uh):
        """!
        @brief 求 H3 误差
        """

        def f(p, index):
            val = bm.sum((u(p) - uh.grad_3_value(p, index))**2, axis=-1)
            return val
        err = bm.sqrt(self.integralalg.integral(f))
        return err

