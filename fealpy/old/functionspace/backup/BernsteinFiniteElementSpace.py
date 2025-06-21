import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, spdiags, bmat
from scipy.sparse.linalg import spsolve

from fealpy.decorator import barycentric

from fealpy.functionspace.Function import Function

from fealpy.functionspace.femdof import multi_index_matrix1d
from fealpy.functionspace.femdof import multi_index_matrix2d
from fealpy.functionspace.femdof import multi_index_matrix3d

from fealpy.functionspace.femdof import multi_index_matrix

from fealpy.functionspace.femdof import CPLFEMDof1d, CPLFEMDof2d, CPLFEMDof3d
from fealpy.functionspace.femdof import DPLFEMDof1d, DPLFEMDof2d, DPLFEMDof3d

from fealpy.quadrature import FEMeshIntegralAlg
from fealpy.decorator import timer


class BernsteinFiniteElementSpace():
    def __init__(self, mesh, p=1, spacetype='C', q=None, dof=None):
        self.mesh = mesh
        self.cellmeasure = mesh.entity_measure('cell')
        self.p = p
        if dof is None:
            if spacetype == 'C':
                if mesh.meshtype == 'interval':
                    self.dof = CPLFEMDof1d(mesh, p)
                    self.TD = 1
                elif mesh.meshtype == 'tri':
                    self.dof = CPLFEMDof2d(mesh, p)
                    self.TD = 2
                elif mesh.meshtype == 'halfedge2d':
                    assert mesh.ds.NV == 3
                    self.dof = CPLFEMDof2d(mesh, p)
                    self.TD = 2
                elif mesh.meshtype == 'stri':
                    self.dof = CPLFEMDof2d(mesh, p)
                    self.TD = 2
                elif mesh.meshtype == 'tet':
                    self.dof = CPLFEMDof3d(mesh, p)
                    self.TD = 3
            elif spacetype == 'D':
                if mesh.meshtype == 'interval':
                    self.dof = DPLFEMDof1d(mesh, p)
                    self.TD = 1
                elif mesh.meshtype == 'tri':
                    self.dof = DPLFEMDof2d(mesh, p)
                    self.TD = 2
                elif mesh.meshtype == 'tet':
                    self.dof = DPLFEMDof3d(mesh, p)
                    self.TD = 3
        else:
            self.dof = dof
            self.TD = mesh.top_dimension() 

        if len(mesh.node.shape) == 1:
            self.GD = 1
        else:
            self.GD = mesh.node.shape[1]

        self.spacetype = spacetype
        self.itype = mesh.itype
        self.ftype = mesh.ftype

        q = q if q is not None else p+3 
        self.integralalg = FEMeshIntegralAlg(
                self.mesh, q,
                cellmeasure=self.cellmeasure)
        self.integrator = self.integralalg.integrator

        self.multi_index_matrix = multi_index_matrix 
        self.stype = 'lagrange'

    def __str__(self):
        return "Bernstein finite element space!"

    def number_of_global_dofs(self):
        return self.dof.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='cell'):
        if self.spacetype == 'C':
            return self.dof.number_of_local_dofs(doftype=doftype)
        elif self.spacetype == 'D':
            return self.dof.number_of_local_dofs()


    def interpolation_points(self):
        return self.dof.interpolation_points()

    def cell_to_dof(self, index=np.s_[:]):
        return self.dof.cell2dof[index]

    def face_to_dof(self, index=np.s_[:]):
        return self.dof.face_to_dof()

    def edge_to_dof(self, index=np.s_[:]):
        return self.dof.edge_to_dof()

    def boundary_dof(self, threshold=None):
        if self.spacetype == 'C':
            return self.dof.boundary_dof(threshold=threshold)
        else:
            raise ValueError('This space is a discontinuous space!')

    def is_boundary_dof(self, threshold=None):
        if self.spacetype == 'C':
            return self.dof.is_boundary_dof(threshold=threshold)
        else:
            raise ValueError('This space is a discontinuous space!')

    def geo_dimension(self):
        return self.GD

    def top_dimension(self):
        return self.TD

    def residual_estimate(self, uh, f=None, c=None):
        """

        Parameters
        ----------
        uh: lagrange finite element solution
        f: the source, default None
        c: diffusion coefficient, default None

        Notes
        -----
            uh 是一个 线性有限元解，该函数计算 uh 对应的残量型后验误差估计。

        TODO
        ----
        1. 任意的 p 次元 
        """
        mesh = self.mesh
        GD = mesh.geo_dimension()
        TD = mesh.top_dimension()
        NC = mesh.number_of_cells()

        # 计算重心处的梯度值, p=1 时每个单元上的梯度是常向量
        bc = np.array([1/(TD+1)]*(TD+1), dtype=self.ftype)
        grad = self.grad_value(uh, bc)

        if callable(c): # 考虑存在扩散系数的情形
            if hasattr(c, 'coordtype'):
                if c.coordtype == 'barycentric':
                    c = c(bc)
                elif c.coordtype == 'cartesian':
                    ps = mesh.bc_to_point(bc)
                    c = c(ps)
                else:
                    raise ValueError('''
                    The coordtype must be `cartesian` or `barycentric`!
                    ''')
            else: 
                raise ValueError('''
                You should add decorator "cartesian" or "barycentric" on
                function "c".

                from fealpy.decorator import cartesian, barycentric

                @cartesian
                def c(p):
                    ...

                @barycentric
                def c(p):
                    ...

                ''')

        # A\nabla u_h
        if c is not None:
            if isinstance(c, {int, float}):
                grad *= c 
            elif isinstance(c, np.ndarray):
                if c.shape == (GD, GD):
                    grad = np.einsum('mn, in->im', c, grad)
                elif c.shape == (GD, ): # 系数为常数的对角阵
                    grad = np.einsum('m, im->im', c, grad)
                elif len(c.shape) == 1: # (NC, )
                    grad = np.einsum('i, im->im', c, grad)
                elif len(d.shape) == 2: # (NC, GD)
                    grad = np.einsum('im, im->im', c, grad)
                elif len(d.shape) == 3: # (NC, GD, GD)
                    grad = np.einsum('imn, in->im', c, grad)

        cellmeasure = mesh.entity_measure('cell')
        ch = cellmeasure**(1.0/TD)
        facemeasure = mesh.entity_measure('face')

        face2cell = mesh.ds.face_to_cell()
        n = mesh.face_unit_normal() # 单位法向
        J = facemeasure*np.sum((grad[face2cell[:, 0]] - grad[face2cell[:, 1]])*n, axis=-1)**2
        
        eta = np.zeros(NC, dtype=self.ftype)
        np.add.at(eta, face2cell[:, 0], J)
        np.add.at(eta, face2cell[:, 1], J)
        eta *= ch 
        eta *= 0.25 # 2D: 1/8, 3D:   

        if f is not None:
            # 计算  f**2 在每个单元上的积分
            eta += cellmeasure*self.integralalg.cell_integral(f, power=2) # \int_\tau f**2 dx

        return np.sqrt(eta)

    def recovery_estimate(self, uh, method='simple'):
        """
        """
        rguh = self.grad_recovery(uh, method=method)
        eta = self.integralalg.error(rguh.value, uh.grad_value, power=2,
                celltype=True) # 计算单元上的恢复型误差
        return eta

    def grad_recovery(self, uh, method='simple'):
        """

        Notes
        -----

        uh 是线性有限元函数，该程序把 uh 的梯度(分片常数）恢复到分片线性连续空间
        中。

        """
        GD = self.GD
        cell2dof = self.cell_to_dof()
        gdof = self.number_of_global_dofs()
        ldof = self.number_of_local_dofs()
        p = self.p
        bc = self.dof.multiIndex/p
        guh = uh.grad_value(bc)
        guh = guh.swapaxes(0, 1)
        rguh = self.function(dim=GD)

        if method == 'simple':
            deg = np.bincount(cell2dof.flat, minlength = gdof)
            if GD > 1:
                np.add.at(rguh, (cell2dof, np.s_[:]), guh)
            else:
                np.add.at(rguh, cell2dof, guh)

        elif method == 'area':
            measure = self.mesh.entity_measure('cell')
            ws = np.einsum('i, j->ij', measure,np.ones(ldof))
            deg = np.bincount(cell2dof.flat,weights = ws.flat, minlength = gdof)
            guh = np.einsum('ij..., i->ij...', guh, measure)
            if GD > 1:
                np.add.at(rguh, (cell2dof, np.s_[:]), guh)
            else:
                np.add.at(rguh, cell2dof, guh)

        elif method == 'distance':
            ipoints = self.interpolation_points()
            bp = self.mesh.entity_barycenter('cell')
            v = bp[:, np.newaxis, :] - ipoints[cell2dof, :]
            d = np.sqrt(np.sum(v**2, axis=-1))
            deg = np.bincount(cell2dof.flat,weights = d.flat, minlength = gdof)
            guh = np.einsum('ij..., ij->ij...', guh, d)
            if GD > 1:
                np.add.at(rguh, (cell2dof, np.s_[:]), guh)
            else:
                np.add.at(rguh, cell2dof, guh)

        elif method == 'area_harmonic':
            measure = 1/self.mesh.entity_measure('cell')
            ws = np.einsum('i, j->ij', measure,np.ones(ldof))
            deg = np.bincount(cell2dof.flat,weights = ws.flat, minlength = gdof)
            guh = np.einsum('ij..., i->ij...', guh, measure)
            if GD > 1:
                np.add.at(rguh, (cell2dof, np.s_[:]), guh)
            else:
                np.add.at(rguh, cell2dof, guh)

        elif method == 'distance_harmonic':
            ipoints = self.interpolation_points()
            bp = self.mesh.entity_barycenter('cell')
            v = bp[:, np.newaxis, :] - ipoints[cell2dof, :]
            d = 1/np.sqrt(np.sum(v**2, axis=-1))
            deg = np.bincount(cell2dof.flat,weights = d.flat, minlength = gdof)
            guh = np.einsum('ij..., ij->ij...',guh,d)
            if GD > 1:
                np.add.at(rguh, (cell2dof, np.s_[:]), guh)
            else:
                np.add.at(rguh, cell2dof, guh)
        rguh /= deg.reshape(-1, 1)
        return rguh

    @barycentric
    def basis(self, bc, index=np.s_[:], p=None):
        """
        compute the basis function values at barycentric point bc

        Parameters
        ----------
        bc : numpy.ndarray
            the shape of `bc` can be `(TD+1,)` or `(NQ, TD+1)`
        Returns
        -------
        phi : numpy.ndarray
            the shape of 'phi' can be `(1, ldof)` or `(NQ, 1, ldof)`

        See Also
        --------

        Notes
        -----

        """
        if p==None:
            p = self.p

        NQ = bc.shape[0]
        TD = bc.shape[1]-1
        multiIndex = self.multi_index_matrix[TD](p)
        ldof = multiIndex.shape[0]

        B = bc
        B = np.ones((NQ, p+1, TD+1), dtype=np.float_)
        B[:, 1:] = bc[:, None, :]
        B = np.cumprod(B, axis=1)

        P = np.arange(p+1)
        P[0] = 1
        P = np.cumprod(P).reshape(1, -1, 1)
        B = B/P

        # B : (NQ, p+1, TD+1) 
        # B[:, multiIndex, np.arange(TD+1).reshape(1, -1)]: (NQ, ldof, TD+1)
        phi = P[0, -1, 0]*np.prod(B[:, multiIndex, np.arange(TD+1).reshape(1, -1)], 
                axis=-1)
        return phi[..., None, :] 

    @barycentric
    def grad_basis(self, bc, index=np.s_[:], p=None):
        """
        compute the basis function values at barycentric point bc

        Parameters
        ----------
        bc : numpy.ndarray
            the shape of `bc` can be `(TD+1,)` or `(NQ, TD+1)`

        Returns
        -------
        gphi : numpy.ndarray
            the shape of `gphi` can b `(NC, ldof, GD)' or
            `(NQ, NC, ldof, GD)'

        See also
        --------

        Notes
        -----

        """
        if p==None:
            p = self.p

        NQ = bc.shape[0]
        TD = bc.shape[1]-1
        multiIndex = self.multi_index_matrix[TD](p)
        ldof = multiIndex.shape[0]

        B = bc
        B = np.ones((NQ, p+1, TD+1), dtype=np.float_)
        B[:, 1:] = bc[:, None, :]
        B = np.cumprod(B, axis=1)

        P = np.arange(p+1)
        P[0] = 1
        P = np.cumprod(P).reshape(1, -1, 1)
        B = B/P

        F = np.zeros(B.shape, dtype=np.float_)
        F[:, 1:] = B[:, :-1]

        shape = bc.shape[:-1]+(ldof, TD+1)
        R = np.zeros(shape, dtype=self.ftype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            idx = np.array(idx, dtype=np.int_)
            R[..., i] = np.prod(B[..., multiIndex[:, idx], idx.reshape(1, -1)],
                    axis=-1)*F[..., multiIndex[:, i], [i]]

        Dlambda = self.mesh.grad_lambda()
        gphi = P[0, -1, 0]*np.einsum("qlm, cmd->qcld", R, Dlambda)
        return gphi

    @barycentric
    def value(self, uh, bc, index=np.s_[:]):
        TD = bc.shape[-1] - 1
        phi = self.basis(bc)
        e2d = self.dof.entity_to_dof(etype=TD, index=index)

        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ij, ij{}->...i{}'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, phi, uh[e2d])
        return val

    @barycentric
    def grad_value(self, uh, bc, index=np.s_[:]):
        """
        Notes
        -----
        不同维度的实体
        """
        gphi = self.grad_basis(bc, index=index)
        cell2dof = self.dof.cell2dof
        dim = len(uh.shape) - 1
        s0 = 'abcdefg'
        s1 = '...ijm, ij{}->...i{}m'.format(s0[:dim], s0[:dim])
        val = np.einsum(s1, gphi, uh[cell2dof[index]])
        return val

    @barycentric
    def div_value(self, uh, bc, index=np.s_[:]):
        dim = len(uh.shape)
        GD = self.geo_dimension()
        if (dim == 2) & (uh.shape[1] == GD):
            val = self.grad_value(uh, bc, index=index)
            return val.trace(axis1=-2, axis2=-1)
        else:
            raise ValueError("The shape of uh should be (gdof, gdim)!")

    def linear_interpolation_matrix(self):
        """

        Notes
        -----
        把线性元基函数插值到 p 次元空间

        插值矩阵 I 的形状为 (gdof, NN)
        """
        TD = self.TD
        p = self.p

        gdof = self.number_of_global_dofs()
        NN = self.mesh.number_of_nodes()


        if self.spacetype == 'C':
            # 网格节点处的值
            val = np.broadcast_to(np.ones(1), shape=(NN, ))
            P = coo_matrix((val, (range(NN), range(NN))), shape=(gdof, NN),
                    dtype=self.ftype)

            for d in range(1, TD+1):
                if p > d:
                    entity = self.mesh.entity(etype=d)
                    e2d = self.dof.entity_to_dof(etype=d)
                    N = len(entity)
                    index = multi_index_matrix[d](p)
                    n = len(index)
                    flag = np.ones(n, dtype=np.bool_)
                    for i in range(d+1):
                        flag = flag & (index[:, i] != 0)
                    s = flag.sum()
                    bc = index[flag]/p
                    shape = (N, ) + bc.shape
                    val = np.broadcast_to(bc, shape=shape) 
                    I = np.broadcast_to(e2d[:, flag, None], shape=shape)
                    J = np.broadcast_to(entity[:, None, :], shape=shape)
                    P += coo_matrix((val.flat, (I.flat, J.flat)), shape=(gdof, NN), dtype=self.ftype)
            return P.tocsr()

        elif self.spacetype == 'D':
            NC = self.mesh.number_of_cells()
            c2d0 = self.cell_to_dof()
            c2d1 = np.arange((TD+1)*NC).reshape(NC, TD+1)
            bc = multi_index_matrix[TD](p)/p

            shape = (NC, ) + bc.shape
            val = np.broadcast_to(bc, shape=shape)
            I = np.broadcast_to(c2d0[:, :, None], shape=shape)
            J = np.broadcast_to(c2d1[:, None, :], shape=shape)
            P = csr_matrix((val.flat, (I.flat, J.flat)), shape=(dof, NN),
                    dtype=self.ftype)
            return P


    def projection(self, u, ptype='L2'):
        """
        """
        if ptype == 'L2':
            M= self.mass_matrix()
            F = self.source_vector(u)
            uh = self.function()
            uh[:] = spsolve(M, F).reshape(-1)
        elif ptype == 'H1':
            pass
        return uh

    def function(self, dim=None, array=None, dtype=np.float64):
        return Function(self, dim=dim, array=array, 
                coordtype='barycentric', dtype=dtype)

    def array(self, dim=None, dtype=np.float64):
        gdof = self.number_of_global_dofs()
        if dim in {None, 1}:
            shape = gdof
        elif type(dim) is int:
            shape = (gdof, dim)
        elif type(dim) is tuple:
            shape = (gdof, ) + dim

        return np.zeros(shape, dtype=dtype)


    def integral_basis(self):
        """
        """
        cell2dof = self.cell_to_dof()
        bcs, ws = self.integrator.get_quadrature_points_and_weights()
        phi = self.basis(bcs)
        cc = np.einsum('m, mik, i->ik', ws, phi, self.cellmeasure)
        gdof = self.number_of_global_dofs()
        c = np.zeros(gdof, dtype=self.ftype)
        np.add.at(c, cell2dof, cc)
        return c

    def revcovery_matrix(self, rtype='simple'):
        """

        Notes
        -----
        构造梯度恢复算子矩阵

        """
        NC = self.mesh.number_of_cells()
        NN = self.mesh.number_of_nodes()
        cell = self.mesh.entity('cell')
        GD = self.GD
        cellmeasure = self.cellmeasure
        gphi = self.mesh.grad_lambda() # (NC, GD+1, GD)
        G = []
        if rtype == 'simple':
            D = spdiags(1.0/np.bincount(cell.flat), 0, NN, NN)
        elif rtype == 'harmonic':
            gphi = gphi/cellmeasure.reshape(-1, 1, 1)
            d = np.zeros(NN, dtype=np.float64)
            np.add.at(d, cell, 1/cellmeasure.reshape(-1, 1))
            D = spdiags(1/d, 0, NN, NN)

        I = np.broadcast_to(cell[:, :, None], shape=(NC, GD+1, GD+1))
        J = np.broadcast_to(cell[:, None, :], shape=(NC, GD+1, GD+1))

        #I = np.einsum('k, ij->ijk', np.ones(GD+1), cell)
        #J = I.swapaxes(-1, -2)
        for i in range(GD):
            #val = np.einsum('k, ij->ikj', np.ones(GD+1), gphi[:, :, i])
            val = np.broadcast_to(gphi[:, :, i][:, None, :], shape=(NC, GD+1, GD+1))
            G.append(D@csc_matrix((val.flat, (I.flat, J.flat)), shape=(NN, NN)))
        return G

    def rigid_motion_matrix(self):
        GD = self.GD
        p = self.p
        if p == 1:
            NN = self.mesh.number_of_nodes()
            node = self.mesh.entity('node')
            if GD == 2:
                P = np.zeros((2*NN, 3), dtype=self.ftype)
                P[0*NN:1*NN, 0] = 1.0
                P[1*NN:2*NN, 1] = 1.0
                P[0*NN:1*NN, 2] = node[:, 1]
                P[1*NN:2*NN, 2] = -node[:, 0]
            elif GD == 3:
                P = np.zeros((3*NN, 6), dtype=self.ftype)
                P[0*NN:1*NN, 0] = 1.0
                P[1*NN:2*NN, 1] = 1.0
                P[2*NN:3*NN, 2] = 1.0

                P[0*NN:1*NN, 3] =  node[:, 1]
                P[1*NN:2*NN, 3] = -node[:, 0]

                P[1*NN:2*NN, 4] =  node[:, 2]
                P[2*NN:3*NN, 4] = -node[:, 1]

                P[0*NN:1*NN, 5] =  node[:, 2]
                P[2*NN:3*NN, 4] = -node[:, 0]
            return P
                
        else:
            return None

    def linear_elasticity_matrix(self, lam, mu, format='csr', q=None):
        """
        construct the linear elasticity fem matrix
        """

        GD = self.GD
        if GD == 2:
            idx = [(0, 0), (0, 1),  (1, 1)]
            imap = {(0, 0):0, (0, 1):1, (1, 1):2}
        elif GD == 3:
            idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
            imap = {(0, 0):0, (0, 1):1, (0, 2):2, (1, 1):3, (1, 2):4, (2, 2):5}
        A = []

        qf = self.integrator if q is None else self.mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        grad = self.grad_basis(bcs) # (NQ, NC, ldof, GD)

        cell2dof = self.cell_to_dof() # (NC, ldof)
        ldof = self.number_of_local_dofs()
        NC = self.mesh.number_of_cells()
        shape = (NC, ldof, ldof)
        I = np.broadcast_to(cell2dof[:, :, None], shape=shape)
        J = np.broadcast_to(cell2dof[:, None, :], shape=shape)

        # 分块组装矩阵
        gdof = self.number_of_global_dofs()
        cellmeasure = self.cellmeasure
        for k, (i, j) in enumerate(idx):
            Aij = np.einsum('i, ijm, ijn, j->jmn', ws, grad[..., i], grad[..., j], cellmeasure)
            A.append(csr_matrix((Aij.flat, (I.flat, J.flat)), shape=(gdof, gdof)))

        T = csr_matrix((gdof, gdof), dtype=self.ftype)
        D = csr_matrix((gdof, gdof), dtype=self.ftype)
        C = []
        for i in range(GD):
            D += A[imap[(i, i)]]
            C.append([T]*GD)
        D *= mu

        for i in range(GD):
            for j in range(i, GD):
                if i == j:
                    C[i][j] = D + (mu+lam)*A[imap[(i, i)]]
                else:
                    C[i][j] = lam*A[imap[(i, j)]] + mu*A[imap[(i, j)]].T
                    C[j][i] = C[i][j].T
        if format == 'csr':
            return bmat(C, format='csr') # format = bsr ??
        elif format == 'bsr':
            return bmat(C, format='bsr')
        elif format == 'list':
            return C

    def recovery_linear_elasticity_matrix(self, lam, mu, format='csr', q=None):
        """
        construct the recovery linear elasticity fem matrix
        """
        gdof = self.number_of_global_dofs()

        M = self.mass_matrix()
        G = self.revcovery_matrix()

        if format is None:
            return M, G

        cellmeasure = self.cellmeasure
        cell2dof = self.cell_to_dof()
        GD = self.GD

        qf = self.integrator if q is None else self.mesh.integrator(q, 'cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        grad = self.grad_basis(bcs)

        if GD == 2:
            idx = [(0, 0), (0, 1),  (1, 1)]
            imap = {(0, 0):0, (0, 1):1, (1, 1):2}
        elif GD == 3:
            idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
            imap = {(0, 0):0, (0, 1):1, (0, 2):2, (1, 1):3, (1, 2):4, (2, 2):5}
        A = []
        for k, (i, j) in enumerate(idx):
            A.append(G[i].T@M@G[j])

        T = csr_matrix((gdof, gdof), dtype=self.ftype)
        D = csr_matrix((gdof, gdof), dtype=self.ftype)
        C = []
        for i in range(GD):
            D += A[imap[(i, i)]]
            C.append([T]*GD)
        D *= mu
        for i in range(GD):
            C[i][i] = D + (mu+lam)*A[imap[(i, i)]]
            for j in range(i+1, GD):
                C[i][j] = lam*A[imap[(i, j)]] + mu*A[imap[(i, j)]].T
                C[j][i] = C[i][j].T
        if format == 'csr':
            return bmat(C, format='csr') # format = bsr ??
        elif format == 'bsr':
            return bmat(C, format='bsr')
        elif format == 'list':
            return C

    def parallel_stiff_matrix(self, c=None, q=None):
        """

        Notes
        -----
        并行组装刚度矩阵

        """
        gdof = self.number_of_global_dofs()
        cell2dof = self.cell_to_dof()
        b0 = (self.grad_basis, cell2dof, gdof)
        M = self.integralalg.parallel_construct_matrix(b0, c=c, q=q)
        return M

    def parallel_mass_matrix(self, c=None, q=None):
        """

        Notes
        -----
        并行组装质量矩阵 

        TODO:
        1. parallel_construct_matrix just work for stiff matrix
        """
        gdof = self.number_of_global_dofs()
        cell2dof = self.cell_to_dof()
        b0 = (self.basis, cell2dof, gdof)
        M = self.integralalg.parallel_construct_matrix(b0, cfun=cfun, q=q)
        return M

    def parallel_source_vector(self, f, dim=None):
        """

        Notes
        -----
        
        TODO
        ----
        1. 组装载荷向量时，用到的 einsum, 它不支持多线程， 下一步把它并行化

        """
        cell2dof = self.smspace.cell_to_dof()
        gdof = self.smspace.number_of_global_dofs()
        b = self.integralalg.construct_vector_s_s(f, self.basis, cell2dof, gdof=gdof) 
        return b

    def stiff_matrix(self, c=None, q=None, isDDof=None):
        gdof = self.number_of_global_dofs()
        cell2dof = self.cell_to_dof()
        b0 = (self.grad_basis, cell2dof, gdof)
        A = self.integralalg.serial_construct_matrix(b0, c=c, q=q)

        if isDDof is not None: # 处理 D 氏边界条件
            bdIdx = np.zeros(A.shape[0], dtype=np.int_)
            bdIdx[isDDof] = 1
            Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
            T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
            A = T@A@T + Tbd
        return A 

    def mass_matrix(self, c=None, q=None):
        gdof = self.number_of_global_dofs()
        cell2dof = self.cell_to_dof()
        b0 = (self.basis, cell2dof, gdof)
        A = self.integralalg.serial_construct_matrix(b0, c=c, q=q)
        return A 

    def div_matrix(self, pspace, q=None):
        """

        Notes
        -----
        (div v, p) 

        GD == 2
        [[phi, 0], [0, phi]]

        [[B0], [B1]]

        GD == 3
        [[phi, 0, 0], [0, phi, 0], [0, 0, phi]]

        [[B0], [B1], [B2]]
        """
        mesh = self.mesh
        GD = mesh.geo_dimension()
        qf = self.integrator if q is None else mesh.integrator(q, etype='cell')
        # bcs.shape == (NQ, TD+1)
        # ws.shape == (NQ, )
        bcs, ws = qf.get_quadrature_points_and_weights()
        cellmeasure = self.cellmeasure

        # gphi.shape == (NQ, NC, ldof, GD)
        gphi = self.grad_basis(bcs)
        # pphi.shape == (NQ, 1, ldof) 
        pphi = pspace.basis(bcs) # TODO: consider the scale polynomial case

        c2d0 = self.cell_to_dof() # (NC, ldof0)
        c2d1 = pspace.cell_to_dof() # (NC, ldof1)
        gdof0 = self.number_of_global_dofs()
        gdof1 = pspace.number_of_global_dofs()

        # I.shape == (NC, ldof0, ldof1)
        shape = c2d0.shape + c2d1.shape[1:]
        I = np.broadcast_to(c2d0[:, :, None], shape=shape)
        J = np.broadcast_to(c2d1[:, None, :], shape=shape)

        B = []
        for i in range(GD):
            D = np.einsum('q, qci, qcj, c->cij', ws, gphi[..., i], pphi, cellmeasure)
            D = csr_matrix(
                    (D.flat, (I.flat, J.flat)), shape=(gdof0, gdof1)
                    )
            B += [D]

        return B

    def velocity_matrix(self, u):

        """
        Notes
        ----
        ((u\\cdot \\nabla) u, v)

        GD == 2
        [[phi, 0], [0, phi]]

        GD == 3
        [[phi, 0, 0], [0, phi, 0], [0, 0, phi]]

        varphi = (u0 \\partial_x + u1 \\partial_y + u2 \\partial_z) phi

        [[varphi, 0, 0], [0, varphi, 0], [0, 0, varphi]]

        """
        mesh = self.mesh
        GD = mesh.geo_dimension()
        qf = self.integrator if q is None else mesh.integrator(q, etype='cell')
        # bcs.shape == (NQ, TD+1)
        # ws.shape == (NQ, )
        bcs, ws = qf.get_quadrature_points_and_weights()
        cellmeasure = self.cellmeasure

        val = u.value(bcs) # (NQ, NC, GD) 
        gphi = self.grad_basis(bcs) # (NQ, NC, ldof, GD)

        val = np.einsum('qcm, qcim->qci', val, gphi) # (NQ, NC, ldof)
        phi = self.basis(bcs) # (NQ, NC, ldof)

        gdof = self.number_of_global_dofs()
        c2d = self.cell_to_dof()

        shape = c2d.shape + c2d.shape[1:]
        I = np.broadcast_to(c2d[:, :, None], shape=shape)
        J = np.broadcast_to(c2d[:, None, :], shape=shape)

        val = np.einsum('q, qci, qcj, c->cij', ws, val, phi, cellmeasure)
        M = csr_matrix(
                (val.flat, (I.flat, J.flat)),
                shape=(gdof, gdof)
                )
        return M



    def convection_matrix(self, c=None, q=None):
        gdof = self.number_of_global_dofs()
        cell2dof = self.cell_to_dof()
        b0 = (self.grad_basis, cell2dof, gdof)
        b1 = (self.basis, cell2dof, gdof)
        A = self.integralalg.serial_construct_matrix(b0, b1=b1, c=c, q=q)
        return A 

    def source_vector(self, f, dim=None, q=None):
        p = self.p
        cellmeasure = self.cellmeasure
        bcs, ws = self.integrator.get_quadrature_points_and_weights()

        if hasattr(f, 'coordtype'):
            if f.coordtype == 'cartesian':
                pp = self.mesh.bc_to_point(bcs)
                fval = f(pp)
            elif f.coordtype == 'barycentric':
                fval = f(bcs)
            else:
                raise ValueError('''
                The coordtype must be `cartesian` or `barycentric`!

                from fealpy.decorator import cartesian, barycentric

                ''')
        else: 
            raise ValueError('''
            You should add decorator "cartesian" or "barycentric" on
            function "c".

            from fealpy.decorator import cartesian, barycentric

            @cartesian
            def c(p):
                ...

            @barycentric
            def c(p):
                ...

            ''')

        gdof = self.number_of_global_dofs()
        shape = gdof if dim is None else (gdof, dim)
        b = np.zeros(shape, dtype=self.ftype)

        if p > 0:
            if type(fval) in {float, int}:
                if fval == 0.0:
                    return b
                else:
                    phi = self.basis(bcs)
                    bb = np.einsum('m, mik, i->ik...', 
                            ws, phi, self.cellmeasure)
                    bb *= fval
            else:
                phi = self.basis(bcs)
                bb = np.einsum('m, mi..., mik, i->ik...',
                        ws, fval, phi, self.cellmeasure)
            cell2dof = self.cell_to_dof() #(NC, ldof)
            if dim is None:
                np.add.at(b, cell2dof, bb)
            else:
                np.add.at(b, (cell2dof, np.s_[:]), bb)
        else:
            b = np.einsum('i, ik..., k->k...', ws, fval, cellmeasure)

        return b

    def interpolation(self, f, dim=3, index=np.s_[:]):
        """
        Bernstein 有限元插值
        """
        p = self.p
        e2d = self.dof.entity_to_dof(dim, index = index)

        multiIndex = self.multi_index_matrix[dim](p)
        x = multiIndex.astype(np.float_)/p
        A = self.basis(x).reshape(multiIndex.shape[0], -1)

        ipoints = self.mesh.bc_to_point(x, index = index) # (NEntity, ldof, GD)
        fval = f(ipoints) #(NEntity, ldof)
        uval = np.linalg.solve(A, fval)
        u = self.function()
        u[e2d] = uval.T
        return u

    def set_dirichlet_bc(self, gD, uh, threshold=None, q=None):
        """
        初始化解 uh  的第一类边界条件。
        """

        p = self.p
        TD = self.mesh.top_dimension()
        isbdFace = self.mesh.ds.boundary_face_flag()
        uh[:] = self.interpolation(gD, TD-1, index=isbdFace)[:]
        isDDof = self.boundary_dof(threshold=threshold)
        return isDDof

    def set_neumann_bc(self, gN, F=None, threshold=None, q=None):

        """

        Notes
        -----
        设置 Neumann 边界条件到载荷向量 F 中

        TODO: 考虑更多 gN 的情况, 比如 gN 可以是一个数组
        """
        mesh = self.mesh
        gdof = self.number_of_global_dofs()
       
        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_face_index()
            if threshold is not None:
                bc = self.mesh.entity_barycenter('face', index=index)
                flag = threshold(bc)
                index = index[flag]

        face2dof = self.face_to_dof()[index]
        n = mesh.face_unit_normal(index=index)
        measure = mesh.entity_measure('face', index=index)

        qf = self.integralalg.faceintegrator if q is None else mesh.integrator(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()
        phi = self.face_basis(bcs)

        pp = mesh.bc_to_point(bcs, index=index)
        val = gN(pp, n) # (NQ, NF, ...), 这里假设 gN 是一个函数

        if len(val.shape) == 2:
            dim = 1
            if F is None:
                F = np.zeros((gdof, ), dtype=self.ftype)
        else:
            dim = val.shape[-1]
            if F is None:
                F = np.zeros((gdof, dim), dtype=self.ftype)


        bb = np.einsum('m, mi..., mik, i->ik...', ws, val, phi, measure)
        if dim == 1:
            np.add.at(F, face2dof, bb)
        else:
            np.add.at(F, (face2dof, np.s_[:]), bb)

        return F

    def set_robin_bc(self, gR, F=None, threshold=None, q=None):
        """

        Notes
        -----

        设置 Robin 边界条件

        TODO: 考虑更多的 gR 的情况

        """
        p = self.p
        mesh = self.mesh
        gdof = self.number_of_global_dofs()

        if type(threshold) is np.ndarray:
            index = threshold
        else:
            index = self.mesh.ds.boundary_face_index()
            if threshold is not None:
                bc = self.mesh.entity_barycenter('face', index=index)
                flag = threshold(bc)
                index = index[flag]

        face2dof = self.face_to_dof()[index]

        qf = self.integralalg.faceintegrator if q is None else mesh.integrator(q, 'face')
        bcs, ws = qf.get_quadrature_points_and_weights()

        measure = mesh.entity_measure('face', index=index)

        phi = self.face_basis(bcs)
        pp = mesh.bc_to_point(bcs, index=index)
        n = mesh.face_unit_normal(index=index)

        val, kappa = gR(pp, n) # (NQ, NF, ...)

        bb = np.einsum('m, mi..., mik, i->ik...', ws, val, phi, measure)

        if len(val.shape) == 2:
            dim = 1
            if F is None:
                F = np.zeros((gdof, ), dtype=self.ftype)
        else:
            dim = val.shape[-1]
            if F is None:
                F = np.zeros((gdof, dim), dtype=self.ftype)

        if dim == 1:
            np.add.at(F, face2dof, bb)
        else:
            np.add.at(F, (face2dof, np.s_[:]), bb)

        FM = np.einsum('m, mi, mij, mik, i->ijk', ws, kappa, phi, phi, measure)
        I = np.broadcast_to(face2dof[:, :, None], shape=FM.shape)
        J = np.broadcast_to(face2dof[:, None, :], shape=FM.shape)
        R = csr_matrix((FM.flat, (I.flat, J.flat)), shape=(gdof, gdof))

        return R, F


    def to_function(self, data):
        p = self.p
        if p == 1:
            uh = self.function(array=data)
            return uh
        elif p == 2:
            cell2dof = self.cell_to_dof()
            uh = self.function()
            uh[cell2dof] = data[:, [0, 5, 4, 1, 3, 2]]
            return uh


