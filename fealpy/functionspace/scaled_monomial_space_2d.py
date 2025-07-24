import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from itertools import combinations_with_replacement

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

class ScaledMonomialSpace2d(FunctionSpace, Generic[_MT]):
    def __init__(self, mesh, p, q=None, bc=None):
        """
        The Scaled Momomial Space in R^2
        """

        self.mesh = mesh
        self.device = mesh.device
        self.ikwargs = bm.context(mesh.cell[0]) if mesh.meshtype =='polygon' else bm.context(mesh.cell)
        self.fkwargs = bm.context(mesh.node)
        self.cellbarycenter = mesh.entity_barycenter('cell') if bc is None else bc
        self.p = p
        self.cellmeasure = mesh.entity_measure('cell')

        self.cellsize = bm.sqrt(self.cellmeasure)
        self.GD = 2

        q = q if q is not None else p+3

        mtype = mesh.meshtype

        self.itype = self.mesh.itype
        self.ftype = self.mesh.ftype

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

    def edge_to_dof(self, p=None):
        mesh = self.mesh
        return mesh.face_to_cell()[:,:2]

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



    def diff_index_1(self, p=None):
        """

        Notes
        -----
        对基函数求一阶导后非零项的编号，及系数
        """
        p = self.p if p is None else p
        #index = multi_index_matrix2d(p)
        index = self.mesh.multi_index_matrix(p, 2)

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
        #index = multi_index_matrix2d(p)
        index = self.mesh.multi_index_matrix(p, 2)

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
        #index = multi_index_matrix1d(p)
        index = self.mesh.multi_index_matrix(p, 1)
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
        index = self.mesh.multi_index_matrix(p, 1)
        #index = multi_index_matrix1d(p)
        x, = bm.nonzero(index[:, 0] > 0)
        y, = bm.nonzero(index[:, 1] > 0)
        return {'x': x, 'y':y}

    def geo_dimension(self):
        return self.GD

    @cartesian
    def edge_basis(self, point, index=_S, p=None):
        p = self.p if p is None else p
        if p == 0:
            shape = len(point.shape)*(1, )
            return bm.array([1.0], **self.fkwargs).reshape(shape)

        #ec = self.integralalg.edgebarycenter
        #eh = self.integralalg.edgemeasure
        ec = self.mesh.entity_barycenter('edge') #(NE,2)
        eh = self.mesh.entity_measure('edge') # NE
        et = self.mesh.edge_tangent(unit=True) 
        val = bm.sum((point - ec[:,None,:][index])*et[:,None,:][index],
                     axis=-1)/eh[:, None][index] #(NE, NQ, GD)
        phi = bm.ones(val.shape + (p+1,), **self.fkwargs) #(NE, NQ, GD, p+1)
        if p == 1:
            phi[..., 1] = val
        else:
            phi[..., 1:] = val[..., bm.newaxis]
            #bm.multiply.accumulate(phi, axis=-1, out=phi)
            bm.cumprod(phi, axis=-1, out=phi)
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
            return bm.array([[1.0]], **self.fkwargs).reshape(shape)
        else:
            shape = bcs.shape[:-1]+(p+1, )
            phi = bm.ones(shape, **self.fkwargs)
            phi[..., 1:] = bcs[..., 1, None]-0.5
            #bm.multiply.accumulate(phi, axis=-1, out=phi)
            bm.cumprod(phi, axis=-1, out=phi)
            return phi

    @cartesian
    def basis(self, point, index=_S, p=None):
        """
        Compute the basis values at point

        Parameters
        ----------
        point : ndarray
            The shape of point is (M, ..., 2), M is the number of cells

        Returns
        -------
        phi : ndarray
            The shape of `phi` is (M, ..., ldof)

        """
        p = self.p if p is None else p
        h = self.cellsize
        NC = self.mesh.number_of_cells()
        if isinstance(point, tuple):
            point = point[0]    
        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        if p == 0:
            shape = len(point.shape)*(1, )
            return bm.array([1.0], **self.fkwargs).reshape(shape)

        shape = point.shape[:-1]+(ldof,)
        phi = bm.ones(shape, **self.fkwargs)  # (..., M, ldof)

        phi[..., 1:3] = (point -
                         self.cellbarycenter.reshape((NC,)+(1,)*int(point.ndim-2)+(2,))[index])/h[index].reshape((-1,)+(1,)*int(point.ndim-1))
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
        gphi = bm.zeros(shape, **self.fkwargs)

        if p == 0:
            return gphi

        phi = self.basis(point, index=index, p=p-1)
        idx = self.diff_index_1(p=p)
        xidx = idx['x']
        yidx = idx['y']
        gphi[..., xidx[0], 0] = bm.einsum('i, ...i->...i', xidx[1], phi)
        gphi[..., yidx[0], 1] = bm.einsum('i, ...i->...i', yidx[1], phi)
        if scaled:
            return gphi/h[index].reshape((-1,)+(1,)*int(gphi.ndim-1))
        else:
            return gphi


        #if scaled:
        #    if point.shape[-2] == num:
        #        return gphi/h[index].reshape(-1, 1, 1)
        #    elif point.shape[0] == num:
        #        return gphi/h[index].reshape(-1, 1, 1, 1)
        #else:
        #    return gphi

    @cartesian
    def laplace_basis(self, point, index=_S, p=None, scaled=True):
        p = self.p if p is None else p

        area = self.cellmeasure
        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        shape = point.shape[:-1]+(ldof,)
        lphi = bm.zeros(shape, **self.fkwargs)
        if p > 1:
            phi = self.basis(point, index=index, p=p-2)
            idx = self.diff_index_2(p=p)
            lphi[..., idx['xx'][0]] += bm.einsum('i, ...i->...i', idx['xx'][1], phi)
            lphi[..., idx['yy'][0]] += bm.einsum('i, ...i->...i', idx['yy'][1], phi)

        if scaled:
            return lphi/area[index].reshape((-1,)+(1,)*(point.ndim-1))
        else:
            return lphi

    @cartesian
    def hessian_basis(self, point, index=_S, p=None, scaled=True):
        """
        Compute the value of the hessian of the basis at a set of 'point'

        Parameters
        ----------
        point : numpy array
            The shape of point is (NC, ..., 2)

        Returns
        -------
        hphi : numpy array
            the shape of hphi is (NC, ldof, 2, 2)
        """
        p = self.p if p is None else p

        area = self.cellmeasure
        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        shape = point.shape[:-1]+(ldof, 2, 2)
        hphi = bm.zeros(shape, **self.fkwargs)
        if p > 1:
            phi = self.basis(point, index=index, p=p-2)
            idx = self.diff_index_2(p=p)
            hphi[..., idx['xx'][0], 0, 0] = bm.einsum('i, ...i->...i', idx['xx'][1], phi)
            hphi[..., idx['xy'][0], 0, 1] = bm.einsum('i, ...i->...i', idx['xy'][1], phi)
            hphi[..., idx['yy'][0], 1, 1] = bm.einsum('i, ...i->...i', idx['yy'][1], phi)
            hphi[..., 1, 0] = hphi[..., 0, 1]

        if scaled:
            return hphi/area[index].reshape((-1,)+(1,)*int(hphi.ndim-1))
        else:
            return hphi

    def grad_m_basis(self, m, point, index=_S, p=None, scaled=True):
        """!
        @brief m=3时导数排列顺序: [xxx, xxy, xyx, xyy, yxx, yxy, yyx, yyy]
        """
        #TODO test
        phi = self.basis(point, index=index, p=p)
        gmphi = bm.zeros(phi.shape+(2**m, ), **self.fkwargs)
        P = self.partial_matrix(index=index)

        #f = lambda x: bm.array([int(ss) for ss in bm.binary_repr(x, m)], dtype=bm.int32)
        #idx = bm.array(list(map(f, bm.arange(2**m))))
        def to_binary_array(x, m):
            # 获取 x 的二进制位，确保长度为 m
            return bm.tensor([((x >> (m - i - 1)) & 1) for i in bm.arange(m,dtype=bm.int32)], dtype=bm.int32)
        idx = bm.stack([to_binary_array(x, m) for x in bm.arange(2**m,dtype=bm.int32)])
        for i in range(2**m):
            M = bm.copy(P[idx[i, 0]])
            for j in range(1, m):
                M = bm.einsum("cij, cjk->cik", M, P[idx[i, j]])
            gmphi[..., i] = bm.einsum('cli, c...l->c...i', M, phi)
        return gmphi

    def partial_matrix(self, p=None, index=_S):
        """
        \partial m = mP
        """
        p = p or self.p
        #mindex = multi_index_matrix2d(p)
        mindex = bm.multi_index_matrix(p, 2) 
        N = len(mindex)
        cellarea = self.mesh.entity_measure("cell")
        NC = self.mesh.number_of_cells()
        h = bm.sqrt(cellarea)

        I, = bm.where(mindex[:, 1] > 0)
        Px = bm.zeros([NC, N, N], **self.fkwargs)
        Px[:, bm.arange(len(I)), I] = mindex[None, I, 1]/h[:, None]

        I, = bm.where(mindex[:, 2] > 0)
        Py = bm.zeros([NC, N, N], **self.fkwargs)
        Py[:, bm.arange(len(I)), I] = mindex[None, I, 2]/h[:, None]
        return Px[index], Py[index]

    def cell_mass_matrix(self, p=None):
        """
        Cell mass matrix, shape:(NC, ldof, ldof)
        """
        #M = self.matrix_H(p=p)
        p = self.p if p is None else p
        def f(x, index):
            phi = self.basis(x, index=index, p=p)
            return bm.einsum('eqi, eqj -> eqij', phi, phi)
        return self.integral(f) # 积分

    def cell_stiff_matrix(self, p=None):
        p = self.p if p is None else p
        M = self.cell_mass_matrix()
        Px, Py = self.partial_matrix()
        S1 = bm.einsum("cji, cjk, ckl -> cil", Px, M, Px)
        S2 = bm.einsum("cji, cjk, ckl -> cil", Py, M, Py)
        return S1 + S2


    def edge_integral(self, f):
        mesh = self.mesh
        p = self.p
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        edge2cell = mesh.edge_to_cell()

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        NC = mesh.number_of_cells()
        qf = mesh.quadrature_formula(p+3, etype='edge', qtype='legendre') # NQ
        bcs, ws = qf.quadpts, qf.weights # (NQ, 2)  (NQ,)
        ps = bm.einsum('ij, kjm->kim', bcs, node[edge]) # (NQ, 2) (NE, 2, 2)
        f1 = f(ps, index=edge2cell[:, 0]) # (NE, NQ, ldof)
        measure = mesh.entity_measure('edge')
        H0 = bm.einsum('eq..., q, e-> e...', f1, ws, measure) # (NC, 2, 2)
        f2 = f(ps, index=edge2cell[:, 1])
        H1 = bm.einsum('eq..., q, e-> e...', f2[isInEdge], ws, measure[isInEdge]) # (NC, 2, 2)
        H = bm.zeros((NC,)+ f1.shape[2:], **mesh.fkwargs)
        bm.index_add(H, edge2cell[:, 0], H0)
        bm.index_add(H, edge2cell[isInEdge, 1], H1)
        return H





    def integral(self, f):
        """
        homogenous function integral, applicable to arbitrary polygonal meshes
        """
        mesh = self.mesh
        p = self.p
        node = mesh.entity('node')
        edge = mesh.entity('edge')
        edge2cell = mesh.edge_to_cell()
        edgebarycenter = mesh.entity_barycenter('edge')
        cellbarycenter = mesh.entity_barycenter('cell')
        #edgebarycenter = node[edge[:, 0]] - cellbarycenter[edge2cell[:, 0]] # (NE, 2)

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        NC = mesh.number_of_cells()
        qf = mesh.quadrature_formula(p+3, etype='edge', qtype='legendre') # NQ
        bcs, ws = qf.quadpts, qf.weights # (NQ, 2)  (NQ,)
        ps = bm.einsum('ij, kjm->kim', bcs, node[edge]) # (NQ, 2) (NE, 2, 2)
        f1 = f(ps, index=edge2cell[:, 0]) # (NE, NQ, ldof)
        nm = mesh.edge_normal()
        b = node[edge[:, 0]] - cellbarycenter[edge2cell[:, 0]]
        H0 = bm.einsum('eq..., q, ed, ed-> e...', f1, ws, b, nm) # (NC, 2, 2)
        f2 = f(ps, index=edge2cell[:, 1])
        b = node[edge[isInEdge, 0]] - cellbarycenter[edge2cell[isInEdge, 1]]
        H1 = bm.einsum('eq..., q, ed, ed-> e...', f2[isInEdge], ws, b, -nm[isInEdge]) # (NC, 2, 2)
        H = bm.zeros((NC,)+ f1.shape[2:], **mesh.fkwargs)
        bm.index_add(H, edge2cell[:, 0], H0)
        bm.index_add(H, edge2cell[isInEdge, 1], H1)
        multiIndex = self.multi_index_matrix(p=p)
        q = bm.sum(multiIndex, axis=1)
        if H.ndim == 2:
            H /= q+2
        else:
            H /= q + q.reshape(-1, 1) + 2
        return H

    def partial_matrix_on_edge(self, p=None):
        p = p or self.p
        I = bm.arange(p)

        h = self.mesh.entity_measure("edge")
        NE = self.mesh.number_of_edges()

        P = bm.zeros([NE, p+1, p+1], **self.fkwargs)
        P[:, I, I+1] = bm.arange(1, p+1)[None, :]/h[:, None]
        return P

    @cartesian
    def value(self, uh, point, index=_S):
        phi = self.basis(point, index=index)
        cell2dof = self.cell_to_dof()[index]
        #cell2dof = self.cell2dof[index]
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = 'i...j, ij{}->i...{}'.format(s0[:dim], s0[:dim])
        return bm.einsum(s1, phi, uh[cell2dof])

    @cartesian
    def grad_value(self, uh, point, index=_S):
        gphi = self.grad_basis(point, index=index)
        cell2dof = self.cell_to_dof()
        if (type(index) is TensorLike) and (index.dtype.name == 'bool'):
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
        cell2dof = self.cell_to_dof()
        #cell2dof = self.cell2dof
        return bm.einsum('...ij, ij->...i', lphi, uh[cell2dof[index]])

    @cartesian
    def hessian_value(self, uh, point, index=_S):
        hphi = self.hessian_basis(point, index=index) #(NQ, NC, ldof, 2, 2)
        #cell2dof = self.cell2dof
        cell2dof = self.cell_to_dof()
        return bm.einsum('...clij, cl->...cij', hphi, uh[cell2dof[index]])

    @cartesian
    def grad_3_value(self, uh, point, index=_S):
        #TODO
        gmphi = self.grad_m_basis(3, point, index=index) #(NQ, NC, ldof, 8)
        #cell2dof = self.cell2dof
        cell2dof = self.cell_to_dof()
        return bm.einsum('...cli, cl->...ci', gmphi, uh[cell2dof[index]])

    #def function(self, dim=None, array=None, dtype=None):
    #    ftype = self.ftype if dtype is None else dtype
    #    f = Function(self, dim=dim, array=array, coordtype='cartesian',
    #            dtype=ftype)
    #    return f

    #def array(self, dim=None, dtype=None):
    #    ftype = self.ftype if dtype is None else dtype
    #    gdof = self.number_of_global_dofs()
    #    if dim in {None, 1}:
    #        shape = gdof
    #    elif type(dim) is int:
    #        shape = (gdof, dim)
    #    elif type(dim) is tuple:
    #        shape = (gdof, ) + dim
    #    return bm.zeros(shape, dtype=dtype)

    def dof_array(self, dim=None):
        gdof = self.number_of_global_dofs()
        if dim in {None, 1}:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim
        return bm.zeros(shape, **self.fkwargs)


    def show_function_image(self, u, uh, t=None, plot_solution=True):
        mesh = uh.space.mesh
        fig = plt.figure()
        fig.set_facecolor('white')
        axes = plt.axes(projection='3d')

        NE = mesh.number_of_edges()
        mid = mesh.entity_barycenter("cell")
        node = mesh.entity("node")
        edge = mesh.entity("edge")
        edge2cell = mesh.edge_to_cell()

        coor = bm.zeros([2*NE, 3, 2], **self.fkwargs)
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
        H = bm.zeros((NE, ldof, ldof), **self.fkwargs)
        H[:, flag] = eh.reshape(-1, 1)
        H[:, flag] /= Q[flag]
        H[:, flag] /=2**(Q[flag]-1)
        return H

    def edge_mass_matrix_1(self, p=None):
        p = self.p if p is None else p
        mesh = self.mesh
        edge = mesh.entity('edge')
        measure = mesh.entity_measure('edge')
        #qf = GaussLegendreQuadrature(p + 3)
        qf = mesh.quadrature_formula(p+3, etype=1, qtype='legendre') # NQ
        bcs, ws = qf.quadpts, qf.weights # NQ
        ps = self.mesh.edge_bc_to_point(bcs) # (NE, NQ, 2)
        phi = self.edge_basis(ps, p=p) #(NE,NQ,2,ldof=p+1)
        #H = bm.einsum('i, ijk, ijm, j->jkm', ws, phi, phi, measure, optimize=True)
        H = bm.einsum('q, eqk, eqm, e->ekm', ws, phi, phi, measure)
        return H

    def edge_cell_mass_matrix(self, p=None, cp=None):
        p = self.p if p is None else p
        cp = p+1 if cp is None else cp

        mesh = self.mesh

        edge = mesh.entity('edge')
        measure = mesh.entity_measure('edge')

        edge2cell = mesh.edge_to_cell()
        qf = mesh.quadrature_formula(p+3, etype='edge', qtype='legendre') # NQ
        bcs, ws = qf.quadpts, qf.weights # (NQ, 2)  (NQ,)
        #qf = GaussLegendreQuadrature(p + 3)
        #bcs, ws = qf.quadpts, qf.weights
        ps = self.mesh.edge_bc_to_point(bcs) # (NE, NQ, 2)

        phi0 = self.edge_basis(ps, p=p) # (NE, NQ, ldof=p+1)
        phi1 = self.basis(ps, index=edge2cell[:, 0], p=cp) # (NE, NQ, cldof)
        phi2 = self.basis(ps, index=edge2cell[:, 1], p=cp) # (NE, NQ, cldof)
        LM = bm.einsum('j, ijk, ijm, i->ikm', ws, phi0, phi1, measure)
        RM = bm.einsum('j, ijk, ijm, i->ikm', ws, phi0, phi2, measure)
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
            return bm.einsum('cqlij, cqmij->cqlm', hphi, hphi)
        A = self.mesh.integral(f, q=p+3, celltype=True) # (NC, ldof, ldof)
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

        A = self.mesh.integral(f, q=p+3, celltype=True) # (NC, ldof, ldof)
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

        A = self.mesh.integral(f, q=p+3, celltype=True) # (NC, ldof, ldof)
        cell2dof = self.cell_to_dof(p=p)
        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        I = bm.einsum('k, ij->ijk', bm.ones(ldof), cell2dof) # (NC, ldof, ldof)
        J = I.swapaxes(-1, -2)
        gdof = self.number_of_global_dofs(p=p)


        # Construct the stiffness matrix
        A = csr_matrix((A.flatten(), (I.flatten(), J.flatten())), shape=(gdof, gdof))
        return A

    def mass_matrix(self, p=None):
        M = self.cell_mass_matrix(p=p) # 单元质量矩阵
        cell2dof = self.cell_to_dof(p=p)
        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        I = bm.einsum('k, ij->ijk', bm.ones(ldof), cell2dof)
        J = I.swapaxes(-1, -2)
        gdof = self.number_of_global_dofs(p=p)
        M = csr_matrix((M.flatten(), (I.flatten(), J.flatten())), shape=(gdof, gdof))
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
        edge2cell = mesh.edge_to_cell()[index]
        isInEdge = edge2cell[:, 0] != edge2cell[:, 1]

        eh = mesh.entity_measure('edge')
        #qf = GaussLegendreQuadrature(p + 3)
        qf = mesh.quadrature_formula(p+3, etype=1, qtype='legendre') # NQ
        bcs, ws = qf.quadpts, qf.weights
        ps = self.mesh.edge_bc_to_point(bcs, index=index) # (NE, NQ, 2)

        gdof = self.number_of_global_dofs(p=p)
        ldof = self.number_of_local_dofs(doftype='cell')

        shape = ps.shape[:-1] + (2*ldof, ) # (NE, NQ, 2*ldof)
        phi = bm.zeros(shape, **self.fkwargs)

        phi[:, :, :ldof] = self.basis(ps, index=edge2cell[:, 0]) # (NE, NQ, ldof)
        phi[isInEdge, :, ldof:] = -self.basis(ps[isInEdge], index=edge2cell[isInEdge, 1]) # (NE, NQ, ldof)

        A = bm.einsum('q, eqi, eqj->eij', ws, phi, phi)

        cell2dof = self.cell_to_dof()
        edge2dof = bm.stack([cell2dof[edge2cell[:, 0]], cell2dof[edge2cell[:, 1]]],axis=1).reshape(edge2cell.shape[0],-1)
        I = bm.broadcast_to(edge2dof[:, :, None], shape=A.shape)
        J = bm.broadcast_to(edge2dof[:, None, :], shape=A.shape)
        A = csr_matrix((A.flatten(), (I.flatten(), J.flatten())), shape=(gdof, gdof))
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
        edge2cell = mesh.edge_to_cell()[index]
        isInEdge = edge2cell[:, 0] != edge2cell[:, 1]

        ldof = self.number_of_local_dofs(doftype='cell')
        gdof = self.number_of_global_dofs(p=p)
        GD = mesh.geo_dimension()

        eh = mesh.entity_measure('edge', index=index)
        en = mesh.edge_normal(index=index)

        qf = mesh.quadrature_formula(p+3, etype=1, qtype='legendre') # NQ
        bcs, ws = qf.quadpts, qf.weights
        ps = self.mesh.edge_bc_to_point(bcs, index=index)

        shape = ps.shape[:-1] + (2*ldof, )
        phi = bm.zeros(shape, **self.fkwargs)
        gphi = bm.zeros(shape, **self.fkwargs) # (NE,NQ,2*ldof)

        phi[:, :, :ldof] = self.basis(ps, index=edge2cell[:, 0]) # (NE, NQ, ldof)
        phi[isInEdge, :,  ldof:] = -self.basis(ps[isInEdge], index=edge2cell[isInEdge, 1]) # (NE, NQ, ldof)
        #phi[:, isInEdge, ldof:] = -self.basis(ps[:, isInEdge], index=edge2cell[isInEdge, 1]) # (NE, NQ, ldof)
        gphi[~isInEdge,:, :ldof]= bm.sum(self.grad_basis(ps[~isInEdge], index=edge2cell[~isInEdge, 0])*en[~isInEdge, None, None, :], axis=-1) # (NE, NQ, ldof, 2)*(NE,1,1,2) -> (NE, NQ, ldof)
        gphi[isInEdge,:, :ldof]= 0.5*bm.sum(self.grad_basis(ps[isInEdge], index=edge2cell[isInEdge, 0])*en[isInEdge, None, None, :], axis=-1) # (NE, NQ, ldof)
        gphi[isInEdge,:, ldof:] = 0.5*bm.sum(self.grad_basis(ps[isInEdge], index=edge2cell[isInEdge, 1])*en[isInEdge, None, None, :], axis=-1) # (NE, NQ, ldof)
        #gphi[:, ~isInEdge, :ldof]= bm.sum(self.grad_basis(ps[: ,~isInEdge], index=edge2cell[~isInEdge, 0])*en[~isInEdge, None, :], axis=-1) # (NQ, NE, ldof)
        #gphi[:, isInEdge, :ldof]= 0.5*bm.sum(self.grad_basis(ps[:, isInEdge], index=edge2cell[isInEdge, 0])*en[isInEdge, None, :], axis=-1) # (NQ, NE, ldof)
        #gphi[:, isInEdge, ldof:] = 0.5*bm.sum(self.grad_basis(ps[:, isInEdge], index=edge2cell[isInEdge, 1])*en[isInEdge, None, :], axis=-1) # (NQ, NE, ldof)

        A = bm.einsum('q, eqi, eqj->eij', ws, phi, gphi)

        cell2dof = self.cell_to_dof()
        edge2dof = bm.stack([cell2dof[edge2cell[:, 0]], cell2dof[edge2cell[:, 1]]],axis=1).reshape(edge2cell.shape[0],-1)
        I = bm.broadcast_to(edge2dof[:, :, None], shape=A.shape)
        J = bm.broadcast_to(edge2dof[:, None, :], shape=A.shape)
        A = csr_matrix((A.flatten(), (I.flatten(), J.flatten())), shape=(gdof, gdof))
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
        edge2cell = mesh.edge_to_cell()[index]
        isInEdge = edge2cell[:, 0] != edge2cell[:, 1]

        ldof = self.number_of_local_dofs(doftype='cell')
        gdof = self.number_of_global_dofs(p=p)
        GD = mesh.geo_dimension()

        eh = mesh.entity_measure('edge', index=index)
        en = mesh.edge_normal(index=index)

        qf = mesh.quadrature_formula(p+3, etype=1, qtype='legendre') # NQ
        bcs, ws = qf.quadpts, qf.weights
        ps = self.mesh.edge_bc_to_point(bcs, index=index) #(NE, NQ, 2)

        shape = ps.shape[:-1] + (2*ldof, ) # (NE, NQ, 2*ldof)
        gphi = bm.zeros(shape, **self.fkwargs)

        gphi[:, :, :ldof]= bm.sum(self.grad_basis(ps, index=edge2cell[:, 0])*en[:, None, None, :], axis=-1) # (NE, NQ, ldof)
        gphi[isInEdge, :, ldof:] = -bm.sum(self.grad_basis(ps[isInEdge], index=edge2cell[isInEdge, 1])*en[isInEdge, None, None, :], axis=-1) # (NE, NQ, ldof)

        A = bm.einsum('q, eqi, eqj->eij', ws, gphi, gphi)

        cell2dof = self.cell_to_dof()
        edge2dof = bm.stack([cell2dof[edge2cell[:, 0]], cell2dof[edge2cell[:, 1]]], axis=1).reshape(edge2cell.shape[0], -1)
        I = bm.broadcast_to(edge2dof[:, :, None], shape=A.shape)
        J = bm.broadcast_to(edge2dof[:, None, :], shape=A.shape)
        A = csr_matrix((A.flatten(), (I.flatten(), J.flatten())), shape=(gdof, gdof))
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
        edge2cell = mesh.edge_to_cell()[index]
        isInEdge = edge2cell[:, 0] != edge2cell[:, 1]

        eh = mesh.entity_measure('edge', index=index)
        en = mesh.edge_normal(index=index)

        #qf = GaussLegendreQuadrature(p + 3)
        qf = mesh.quadrature_formula(p+3, etype=1, qtype='legendre') # NQ
        bcs, ws = qf.quadpts, qf.weights
        ps = self.mesh.edge_bc_to_point(bcs, index=index) #(NE, NQ, 2)

        gval = g(ps) #(NE, NQ)

        ldof = self.number_of_local_dofs(doftype='cell')
        gdof = self.number_of_global_dofs(p=p)

        shape = ps.shape[:-1] + (2*ldof, )
        gphi = bm.zeros(shape, **self.fkwargs)

        gphi[:, :, :ldof]= bm.sum(self.grad_basis(ps, index=edge2cell[:, 0])*en[:, None, :], axis=-1) # (NE, NQ, ldof)
        gphi[:, isInEdge, ldof:] = -bm.sum(self.grad_basis(ps[isInEdge], index=edge2cell[isInEdge, 1])*en[isInEdge, None, :], axis=-1) # (NE, NQ, ldof)

        A = bm.einsum('q, eq, eqj->ej', ws, gval, gphi)

        F = bm.zeros(gdof, **self.fkwargs)
        cell2dof = self.cell_to_dof()
        bm.index_add(F, cell2dof[edge2cell[:, 0]], A[:, :ldof])
        bm.index_add(F, cell2dof[edge2cell[isInEdge, 1]], A[isInEdge, ldof:])
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
        edge2cell = mesh.edge_to_cell()[index]
        isInEdge = edge2cell[:, 0] != edge2cell[:, 1]

        eh = mesh.entity_measure('edge', index=index)
        en = mesh.edge_unit_normal(index=index)

        #qf = GaussLegendreQuadrature(p + 3)
        qf = mesh.quadrature_formula(p+3, etype=1, qtype='legendre') # NQ
        bcs, ws = qf.quadpts, qf.weights
        ps = self.mesh.edge_bc_to_point(bcs, index=index)

        gval = g(ps)
        if len(gval.shape)==3:
            gval = bm.einsum('qei, ei->qe', gval, en)

        ldof = self.number_of_local_dofs(doftype='cell')
        gdof = self.number_of_global_dofs(p=p)

        shape = ps.shape[:-1] + (2*ldof, )
        phi = bm.zeros(shape, **self.fkwargs)

        phi[:, :, :ldof] = self.basis(ps, index=edge2cell[:, 0]) # (NE, NQ, ldof)
        phi[isInEdge, ldof:] = -self.basis(ps[isInEdge], index=edge2cell[isInEdge, 1]) # (NE, NQ, ldof)
        S = bm.einsum('q, eq, eqi->ei', ws, gval, phi)
        #S = bm.einsum('q, qe, qei->ei', ws, gval, phi)
        if hpower!=-1:
            S = S*(eh.reshape(-1, 1)**(hpower+1))
        F = bm.zeros(gdof, **self.fkwargs)
        cell2dof = self.cell_to_dof()
        bm.index_add(F, cell2dof[edge2cell[:, 0]], S[:, :ldof])
        bm.index_add(F, cell2dof[edge2cell[isInEdge, 1]], S[isInEdge, ldof:])
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
        edge2cell = mesh.edge_to_cell()

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

        F = bm.zeros(shape, **self.fkwargs)
        #bb_0 = bm.einsum('q, qe..., qel,e->el...', ws, fval_0, phi_0, a_0)
        bb_0 = bm.einsum('q, eq..., eql,e->el...', ws, fval_0, phi_0, a_0)

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
            bb_1 = bm.einsum('q, eq..., eql,e->el...', ws, fval_1, phi_1, a_1)
            #bb_1 = bm.einsum('q, qe..., qel,e->el...', ws, fval_1, phi_1, a_1)
            bm.index_add(F, cell2dof[edge2cell[isInEdge, 1]], bb_1)
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
        edge2cell = mesh.edge_to_cell()

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

        F = bm.zeros(shape, **self.fkwargs)
        bb_0 = bm.einsum('q, eq..., eql,e->el...', ws, fval_0, phi_0, a_0)
        #bb_0 = bm.einsum('q, qe..., qel,e->el...', ws, fval_0, phi_0, a_0)

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
            bb_1 = bm.einsum('q, eq..., eql,e->el...', ws, fval_1, phi_1, a_1)
            bm.index_add(F, cell2dof[edge2cell[isInEdge, 1]], bb_1)
        return F

    def coefficient_of_cell_basis_under_edge_basis(self, p=None):
        """!
        @brief 计算单元上基函数在边界基函数上的系数
        """
        #p = p or self.p
        p = p if p is not None else self.p
        mesh = self.mesh
        NC = self.mesh.number_of_cells()
        cell2edge, cell2edgeloc = mesh.cell_to_edge()
        ldof = self.number_of_local_dofs()

        if p == 0:
            bcs = bm.array([[0.5, 0.5]])
        else:
            bcs = bm.zeros((p+1, 2), **self.fkwargs)
            bcs[:, 0] = bm.arange(p)/p
            bcs[:, 1] = 1-bcs[:, 1]

        #M (NE, p+1, p+1) 是每个边上的 p+1 个基函数在 p+1 个点处的值组成矩阵的逆
        M = bm.linalg.inv(self.edge_basis_with_barycentric(bcs, p))
        #C (N, ldof, p+1) 是每个单元基函数在每条边上基函数的系数
        C = bm.zeros((cell2edgeloc[-1], ldof, p+1), **self.fkwargs)

        points = self.mesh.bc_to_point(bcs) #(p+1, NE, 2)
        isNotOK = bm.ones(NC, dtype=bm.bool)
        start = bm.copy(cell2edgeloc[:-1])
        while bm.any(isNotOK):
            index = start[isNotOK]
            eidx = cell2edge[index]
            phi = self.basis(points[:, eidx], index=isNotOK, p=p) #(p+1, NC, ldof)
            C[index] = bm.einsum("cij, cjl->cli", M[eidx], phi)
            start[isNotOK] = start[isNotOK]+1
            isNotOK = start<cell2edgeloc[1:]
        return C

    def triangle_measure(self, tri):
        v1 = tri[1] - tri[0]
        v2 = tri[2] - tri[0]
        area = bm.cross(v1, v2)/2
        return area

    #def source_vector(self, f, celltype=False, q=None):
    #    """
    #    """
    #    cell2dof = self.cell_to_dof()
    #    gdof = self.number_of_global_dofs()
    #    b = (self.basis, cell2dof, gdof)
    #    F = self.integralalg.serial_construct_vector(f, b, celltype=celltype,
    #            q=q)
    #    return F

    def matrix_H(self, p=None):
        p = self.p if p is None else p
        mesh = self.mesh
        node = mesh.entity('node')

        edge = mesh.entity('edge')
        edge2cell = mesh.edge_to_cell()

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])

        NC = mesh.number_of_cells()
        qf = mesh.quadrature_formula(p+1, etype='edge', qtype='legendre') # NQ
        bcs, ws = qf.quadpts, qf.weights # (NQ, 2)  (NQ,)
        #qf = GaussLegendreQuadrature(p + 1)
        #bcs, ws = qf.quadpts, qf.weights
        ps = bm.einsum('ij, kjm->kim', bcs, node[edge]) # (NQ,2),(NE,2,2)->(NE, NQ, 2)
        phi0 = self.basis(ps, index=edge2cell[:, 0], p=p) #(NE, NQ, cldof)
        phi1 = self.basis(ps[isInEdge, :, :], index=edge2cell[isInEdge, 1], p=p) #(InEdge, NQ,cldof)
        H0 = bm.einsum('j, ijk, ijm->ikm', ws, phi0, phi0) # (NQ,), (NE, NQ, cldof), (NE, NQ, cldof) -> (NE, ldof, ldof)
        H1 = bm.einsum('j, ijk, ijm->ikm', ws, phi1, phi1) #(InEdge, cldof, cldof)

        nm = mesh.edge_normal() # (NE, 2)
        b = node[edge[:, 0]] - self.cellbarycenter[edge2cell[:, 0]] # (NE, 2)
        H0 = bm.einsum('ij, ij, ikm->ikm', b, nm, H0) # (NE, 2), (NE,2),(NE, ldof, ldof) -> (NE, ldof, ldof)
        b = node[edge[isInEdge, 0]] - self.cellbarycenter[edge2cell[isInEdge, 1]]

        H1 = bm.einsum('ij, ij, ikm->ikm', b, -nm[isInEdge], H1)

        ldof = self.number_of_local_dofs(p=p, doftype='cell')
        H = bm.zeros((NC, ldof, ldof), **self.fkwargs)
        bm.index_add(H, edge2cell[:, 0], H0)
        bm.index_add(H, edge2cell[isInEdge, 1], H1)

        multiIndex = self.multi_index_matrix(p=p)
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

        b = self.mesh.integral(u, q=q)
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

        num = bm.zeros(NC, **ikwargs)
        bm.add.at(num, HB[:, 0], 1)

        m = HB.shape[0]
        td = bm.zeros((m, ldofs), **self.fkwargs)

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

                
