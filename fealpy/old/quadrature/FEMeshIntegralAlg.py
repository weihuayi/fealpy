import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import multiprocessing as mp
from multiprocessing.pool import ThreadPool as Pool
from ..decorator import timer


class FEMeshIntegralAlg():
    def __init__(self, mesh, q, cellmeasure=None):
        """
        Parameters
        ----------
            mesh: mesh object, which can be Triangle, Quadrangle,
            Tetrahedron, Hexadron mesh
            q: int, the index of the quadrature formula
        """
        self.mesh = mesh
        self.integrator = mesh.integrator(q, etype='cell')

        self.cellintegrator = self.integrator
        self.cellbarycenter =  mesh.entity_barycenter('cell')
        self.cellmeasure = cellmeasure if cellmeasure is not None else mesh.entity_measure('cell')

        self.edgemeasure = mesh.entity_measure('edge')
        self.edgebarycenter = mesh.entity_barycenter('edge')
        self.edgeintegrator = mesh.integrator(q, 'edge')

        GD = mesh.geo_dimension()
        if GD == 3:
            self.facemeasure = mesh.entity_measure('face')
            self.facebarycenter = mesh.entity_measure('face')
            self.faceintegrator = mesh.integrator(q, 'face') 
        else:
            self.facemeasure = self.edgemeasure
            self.facebarycenter = self.edgebarycenter
            self.faceintegrator = self.edgeintegrator

    def error(self, u, v, power=2, celltype=False, q=None):
        """

        @brief 给定两个函数，计算两个函数的之间的差，默认计算 L2 差（power=2)
               power 的取值可以是任意的 p

        TODO
        ----
        1. 考虑无穷范数的情形
        """
        mesh = self.mesh
        GD = mesh.geo_dimension()

        qf = self.integrator if q is None else mesh.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = mesh.bc_to_point(bcs)
        if callable(u):
            if not hasattr(u, 'coordtype'): 
                u = u(ps)
            else:
                if u.coordtype == 'cartesian':
                    u = u(ps)
                elif u.coordtype == 'barycentric':
                    u = u(bcs)

        if callable(v):
            if not hasattr(v, 'coordtype'):
                v = v(ps)
            else:
                if v.coordtype == 'cartesian':
                    v = v(ps)
                elif v.coordtype == 'barycentric':
                    v = v(bcs)

        if u.shape[-1] == 1:
            u = u[..., 0]

        if v.shape[-1] == 1:
            v = v[..., 0]

        f = np.power(np.abs(u - v), power) 
        if isinstance(f, (int, float)): # f为标量常函数
            e = f*self.cellmeasure
        elif isinstance(f, np.ndarray):
            if f.shape == (GD, ): # 常向量函数
                e = self.cellmeasure[:, None]*f
            elif f.shape == (GD, GD):
                e = self.cellmeasure[:, None, None]*f
            else:
                e = np.einsum('q, qc..., c->c...', ws, f, self.cellmeasure)

        if celltype == False:
            e = np.power(np.sum(e), 1/power)
        else:
            e = np.power(np.sum(e, axis=tuple(range(1, len(e.shape)))), 1/power)
        return e # float or (NC, )

    def mesh_integral(self, u, etype='cell', q=None, power=None):
        """
        @brief 计算函数 u 在指定网格实体上的整体积分。
        """
        e = self.entity_integral(u, etype=etype, q=q, power=power)
        if power:
            e = np.power(np.sum(e), 1/power)
        else:
            e = np.sum(e)
        return e

    def entity_integral(self, f, etype='cell', q=None, power=None):
        """
        @brief 在网格的每个实体上积分函数 f 
        """
        mesh = self.mesh
        measure = self.mesh.entity_measure(etype)
        GD = mesh.geo_dimension()

        qf = self.integrator if q is None else mesh.integrator(q, etype=etype)
        bcs, ws = qf.get_quadrature_points_and_weights()

        if callable(f):
            if not hasattr(f, 'coordtype'):
                ps = mesh.bc_to_point(bcs) 
                f = f(ps) 
            else:
                if f.coordtype == 'cartesian':
                    ps = mesh.bc_to_point(bcs) 
                    f = f(ps)
                elif f.coordtype == 'barycentric':
                    f = f(bcs)

        if power is not None:
            f = np.power(f, power) 

        if isinstance(f, (int, float)): # f为标量常函数
            e = f*self.cellmeasure
        elif isinstance(f, np.ndarray):
            if f.shape == (GD, ): # 常向量函数
                e = self.cellmeasure[:, None]*f
            elif f.shape == (GD, GD): # 常矩阵函数
                e = self.cellmeasure[:, None, None]*f
            else:
                e = np.einsum('q, qi..., i->i...', ws, f, measure)
        return e

    def edge_integral(self, f, q=None):
        """
        @brief 在网格的每条边上积分函数 f 
        """

        mesh = self.mesh
        GD = mesh.geo_dimension()
        qf = self.edgeintegrator if q is None else mesh.integrator(q, etype='edge')
        bcs, ws = qf.get_quadrature_points_and_weights()

        if callable(f):
            if f.coordtype == 'cartesian':
                ps = mesh.bc_to_point(bcs) # (NQ, NE, 2)
                f = f(ps) # (NQ, NE, ...)
            elif f.coordtype == 'barycentric':
                f = f(bcs)

        if isinstance(f, (int, float)): # f为标量常函数
            e = f*self.edgemeasure
        elif isinstance(f, np.ndarray):
            if f.shape == (GD, ): # 常向量函数
                e = self.edgemeasure[:, None]*f
            elif f.shape == (GD, GD):
                e = self.edgemeasure[:, None, None]*f
            else:
                e = np.einsum('q, qe..., e->e...', ws, f, self.edgemeasure)
        return e

    def face_integral(self, f, q=None):
        """
        @brief 在网格的每个面上积分函数 f 
        """

        mesh = self.mesh
        GD = mesh.geo_dimension()

        qf = self.faceintegrator if q is None else mesh.integrator(q, etype='face')
        bcs, ws = qf.get_quadrature_points_and_weights()

        if callable(f):
            if f.coordtype == 'cartesian':
                ps = mesh.bc_to_point(bcs) # (NQ, NF, GD)
                f = f(ps) # (NQ, NF, ...)
            elif f.coordtype == 'barycentric':
                f = f(bcs)

        if isinstance(f, (int, float)): # f为标量常函数
            e = f*self.facemeasure
        elif isinstance(f, np.ndarray):
            if f.shape == (GD, ): # 常向量函数
                e = self.facemeasure[:, None]*f
            elif f.shape == (GD, GD): # 常矩阵函数
                e = self.facemeasure[:, None, None]*f
            else:
                e = np.einsum('q, qf..., f->f...', ws, f, self.facemeasure)
        return e

    def cell_integral(self, f, q=None, power=None):
        """
        @brief 在网格的每个单元上积分函数 f 
        """
        mesh = self.mesh
        GD = mesh.geo_dimension()

        qf = self.integrator if q is None else mesh.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        if callable(f):
            if f.coordtype == 'cartesian':
                ps = mesh.bc_to_point(bcs) # (NQ, NC, GD)
                f = f(ps) # (NQ, NC, ...k)
            elif f.coordtype == 'barycentric':
                f = f(bcs)

        if power is not None:
            f = np.power(f, power) 

        if isinstance(f, (int, float)): # f为标量常函数
            e = f*self.cellmeasure
        elif isinstance(f, np.ndarray):
            if f.shape == (GD, ): # 常向量函数
                e = self.cellmeasure[:, None]*f
            elif f.shape == (GD, GD): # 常矩阵函数
                e = self.cellmeasure[:, None, None]*f
            else:
                e = np.einsum('q, qc..., c->c...', ws, f, self.cellmeasure)
        return e




    @timer
    def parallel_construct_matrix(self, b0, 
            b1=None, c=None, q=None):
        """

        Parameters
        ----------
        b0: tuple, 
            b0[0]: basis function
            b0[1]: cell2dof
            b0[2]: number of global dofs
        b1: default is None, just like b0

        Notes
        -----
        
        把网格中的单元分组，再分组组装相应的矩阵。对于三维大规模问题，如果同时计
        算所有单元的矩阵，占用内存会过多，效率过低。


        TODO
        -----
            1. 给定一个计算机内存的大小和 cpu 的个数，动态决定合理的问题分割策略
            2. 考虑存在系数的情况
        """

        mesh = self.mesh
        NC = mesh.number_of_cells()
        qf = self.integrator if q is None else mesh.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        basis0 = b0[0]
        cell2dof0 = b0[1]
        gdof0 = b0[2]

        # 对问题进行分割
        nc = mp.cpu_count()-2

        block = NC//nc
        r = NC%nc
        index = np.full(nc+1, block)
        index[0] = 0
        index[1:r+1] += 1
        np.cumsum(index, out=index)

        if b1 is None:
            gdof1 = gdof0
        else:
            gdof1 = b1[2]

        A = csr_matrix((gdof0, gdof1))
        def f(i):
            s = slice(index[i], index[i+1])
            measure = self.cellmeasure[s]
            c2d0 = cell2dof0[s]
            if b1 is None:
                c2d1 = c2d0
            else:
                c2d1 = b1[1][s]

            shape = (len(measure), c2d0.shape[1], c2d1.shape[1])
            M = np.zeros(shape, measure.dtype)
            for bc, w in zip(bcs, ws): # 对所有积分点进行循环
                phi0 = basis0(bc, index=s)
                if b1 is None:
                    phi1 = phi0
                else:
                    phi1 = b1[0](bc, index=s)
                M += np.einsum('jkd, jmd, j->jkm', phi0, phi1, w*measure)

            I = np.broadcast_to(c2d0[:, :, None], shape=M.shape)
            J = np.broadcast_to(c2d1[:, None, :], shape=M.shape)

            Bi = csr_matrix((M.flat, (I.flat, J.flat)), shape=(gdof0, gdof1))
            return Bi 

        # 并行组装总矩阵
        with Pool(nc) as p:
            B = p.map(f, range(nc))

        for val in B:
            A += val

        return A

    @timer
    def serial_construct_matrix(self, b0, 
            b1=None, c=None, q=None):
        """

        Parameters
        ----------
        b0: tuple, 
            b0[0]: basis function
            b0[1]: cell2dof
            b0[2]: number of global dofs
        b1: default is None, just like b0

        Notes
        -----
        """

        basis0 = b0[0]
        cell2dof0 = b0[1]
        gdof0 = b0[2]

        mesh = self.mesh
        qf = self.integrator if q is None else mesh.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        ps = mesh.bc_to_point(bcs)

        if hasattr(basis0, 'coordtype'):
            if basis0.coordtype == 'barycentric':
                phi0 = basis0(bcs) # (NQ, NC, ldof, ...)
            elif basis0.coordtype == 'cartesian':
                phi0 = basis0(ps)
            else:
                raise ValueError('''
                The coordtype must be `cartesian` or `barycentric`!

                from fealpy.decorator import cartesian, barycentric

                ''')
        else: 
            raise ValueError('''
            You should add decorator "cartesian" or "barycentric" on
            function `basis0`

            from fealpy.decorator import cartesian, barycentric

            @cartesian
            def basis0(p):
                ...

            @barycentric
            def basis0(p):
                ...

            ''')

        if len(phi0.shape) == 3:
            GD = 1
        else:
            GD = phi0.shape[3]

        if b1 is not None:
            if b1[0].coordtype == 'barycentric':
                phi1 = b1[0](bcs) # (NQ, NC, ldof, ...)
            elif b1[0].coordtype == 'cartesian':
                phi1 = b1[0](ps)
        else:
            phi1 = phi0

        if c is None:
            M = np.einsum('i, ijk..., ijm..., j->jkm', ws, phi0, phi1,
                    self.cellmeasure, optimize=True)
        else: 
            if callable(c):
                if c.coordtype == 'barycentric':
                    c = c(bcs)
                elif c.coordtype == 'cartesian':
                    c = c(ps)

            if isinstance(c, (int, float)):
                M = np.einsum('i, ijk..., ijm..., j->jkm', c*ws, phi0, phi1,
                        self.cellmeasure, optimize=True)
            elif isinstance(c, np.ndarray): 
                if c.shape == (GD, GD): # constant diffusion coefficient
                    phi0 = np.einsum('mn, ijkn->ijkm', c, phi0)
                    M = np.einsum('i, ijkl, ijml, j->jkm', ws, phi0, phi1,
                            self.cellmeasure, optimize=True)
                elif c.shape == (GD, ): # constant convection coefficient
                    phi0 = np.einsum('m, ijkm->ijk', c, phi0)
                    M = np.einsum('i, ijk, ijm, j->jkm', ws, phi0, phi1,
                            self.cellmeasure, optimize=True)
                elif len(c.shape) == 2: # (NQ, NC)
                    M = np.einsum('i, ij, ijk..., ijm..., j->jkm', ws, c, phi0, phi1,
                            self.cellmeasure, optimize=True)
                elif len(c.shape) == 3: # (NQ, NC, GD)
                    phi0 = np.einsum('ijm, ijkm->ijk', c, phi0)
                    M = np.einsum('i, ijk, ijm, j->jkm', ws, phi0, phi1,
                            self.cellmeasure, optimize=True)
                elif len(c.shape) == 4: # (NQ, NC, GD, GD)
                    phi0 = np.einsum('ijmn, ijkn->ijkm', c, phi0)
                    M = np.einsum('i, ijkl, ijml, j->jkm', ws, phi0, phi1,
                            self.cellmeasure, optimize=True)

        if cell2dof0 is None: # 仅组装单元矩阵 
            return M

        if b1 is None:
            gdof1 = gdof0
            cell2dof1 = cell2dof0
        else:
            cell2dof1 = b1[1]
            gdof1 = b1[2]

        I = np.broadcast_to(cell2dof0[:, :, None], shape=M.shape)
        J = np.broadcast_to(cell2dof1[:, None, :], shape=M.shape)

        M = csr_matrix((M.flat, (I.flat, J.flat)), 
                shape=(gdof0, gdof1))
        return M

    @timer
    def serial_construct_vector(self, f, b, celltype=False, q=None):
        """

        Notes
        -----
        组装向量， 这里要考虑 f 是标量还是向量函数， 也要考虑基函数是向量还是标
        量函数
        """
        basis = b[0]
        cell2dof = b[1]
        gdof = b[2]

        mesh = self.mesh
        GD = mesh.geo_dimension()
        qf = self.integrator if q is None else mesh.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = mesh.bc_to_point(bcs)


        if basis.coordtype == 'barycentric':
            phi = basis(bcs)
        elif basis.coordtype == 'cartesian':
            phi = basis(ps)

        if callable(f):
            if f.coordtype == 'barycentric':
                val = f(bcs)
            elif f.coordtype == 'cartesian':
                val = f(ps)
        else:
            #TODO: f 可以是多种形式的, 函数, 数组 
            val = f

        if isinstance(val, (int, float)):
            val = np.array([[val]], dtype=mesh.ftype)
        elif isinstance(val, np.ndarray) and (val.shape == (GD, )): 
            val = val.reshape(-1, -1, GD)

        if (len(phi.shape) - len(val.shape)) == 1:
            # f 是标量函数 (NQ, NC)，基是标量函数 (NQ, NC, ldof)
            # f 是向量函数 (NQ, NC, GD)， 基是向量函数 (NQ, NC, ldof, GD)
            if len(val.shape) == 2: #TODO: einsum have bug for ...?
                bb = np.einsum('i, ij, ijk, j->jk', ws, val, phi, self.cellmeasure)
            else:
                bb = np.einsum('i, ijn, ijkn, j->jk', ws, val, phi, self.cellmeasure)

            if celltype:
                return bb
            shape = (gdof, )
            F = np.zeros(shape, dtype=mesh.ftype)
            np.add.at(F, cell2dof, bb)
            return F 
        elif len(val.shape) == len(phi.shape): 
            # f 是向量函数 (NQ, NC, GD)， 基是标量函数 (NQ, NC, ldof)
            bb = np.einsum('i, ijn, ijk, j->jkn', ws, val, phi, self.cellmeasure)
            if celltype:
                return bb
            shape = (gdof, GD)
            F = np.zeros(shape, dtype=mesh.ftype)
            np.add.at(F, (cell2dof, np.s_[:]), bb)
            return F
        else:
            print('Warning!, we can not deal with this f function!')


    @timer
    def construct_matrix(self, basis0, 
            basis1=None,  c=None, 
            cell2dof0=None, gdof0=None, 
            cell2dof1=None, gdof1=None, 
            q=None):
        """

        Parameters
        ---------

        c: 

        Notes
        -----

        给定两个空间的基函数, 组装对应的离散算子. 
        """

        mesh = self.mesh
        qf = self.integrator if q is None else mesh.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()

        ps = mesh.bc_to_point(bcs)
        if basis0.coordtype == 'barycentric':
            phi0 = basis0(bcs) # (NQ, NC, ldof, ...)
        elif basis0.coordtype == 'cartesian':
            phi0 = basis0(ps)

        if basis1 is not None:
            if basis1.coordtype == 'barycentric':
                phi1 = basis1(bcs) # (NQ, NC, ldof, ...)
            elif basis1.coordtype == 'cartesian':
                phi1 = basis1(ps)
        else:
            phi1 = phi0

        if c is None:
            M = np.einsum('i, ijk..., ijm..., j->jkm', ws, phi0, phi1,
                    self.cellmeasure, optimize=True)
        else: # TODO: make here work
            if isinstance(c, (int, float)):
                M = np.einsum('i, ijk..., ijm..., j->jkm', c*ws, phi0, phi1,
                        self.cellmeasure, optimize=True)
            elif callable(c):
                if c.coordtype == 'barycentric':
                    c = c(bcs)
                elif c.coordtype == 'cartesian':
                    c = c(ps)

                if isinstance(c, (int, float)):
                    M = np.einsum('i, ijk..., ijm..., j->jkm', c*ws, phi0, phi1,
                            self.cellmeasure, optimize=True)
                elif isinstance(c, np.ndarray):
                    # user should make `c` have the correct shape
                    if len(c.shape) == 2:
                        M = np.einsum('i, ij, ijk..., ijm..., j->jkm', ws, c, phi0, phi1,
                                self.cellmeasure, optimize=True)
                    elif len(c.shape) == 3:
                        M = np.einsum('i, ijk..., ijk..., ijm..., j->jkm', ws, c[:, :, None, :], phi0, phi1,
                                self.cellmeasure, optimize=True)
                    elif len(c.shape) == 4:
                        M = np.einsum('i, ijkab, ijkb, ijma, j->jkm', ws, c[:, :, None, :, :], phi0, phi1,
                                self.cellmeasure, optimize=True)

        if cell2dof0 is None: # just construct cell matrix
            return M

        gdof0 = gdof0 or cell2dof0.max()
        if cell2dof1 is None:
            gdof1 = gdof0
            cell2dof1 = cell2dof0
        else:
            gdof1 = gdof1 or cell2dof1.max()

        I = np.broadcast_to(cell2dof0[:, :, None], shape=M.shape)
        J = np.broadcast_to(cell2dof1[:, None, :], shape=M.shape)

        M = csr_matrix((M.flat, (I.flat, J.flat)), shape=(gdof0, gdof1))

        return M


    def construct_vector_s_s(self, f, basis, cell2dof, gdof=None, q=None):
        """
        Notes
        -----
        f 是标量函数
        basis 是标量函数
        """
        mesh = self.mesh
        qf = self.integrator if q is None else mesh.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = mesh.bc_to_point(bcs)

        if basis.coordtype == 'barycentric':
            phi = basis(bcs)
        elif basis.coordtype == 'cartesian':
            phi = basis(ps)

        if callable(f):
            if f.coordtype == 'barycentric':
                val = f(bcs)
            elif f.coordtype == 'cartesian':
                val = f(ps)
        else:
            #TODO: f 可以是多种形式的, 函数, 数组 
            val = f

        #TODO: consider more case
        bb = np.einsum('i, ij, ijk, j->jk', ws, val, phi, self.cellmeasure)

        gdof = gdof or cell2dof.max()
        shape = (gdof, )
        b = np.zeros(shape, dtype=phi.dtype)
        np.add.at(b, cell2dof, bb)
        return b

    def construct_vector_v_v(self, f, basis, cell2dof, gdof=None, q=None, dtype=None):
        """
        Notes
        -----
        f 是向量函数
        basis 是向量函数
        """
        mesh = self.mesh
        qf = self.integrator if q is None else mesh.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = mesh.bc_to_point(bcs)

        if basis.coordtype == 'barycentric':
            phi = basis(bcs)
        elif c.coordtype == 'cartesian':
            phi = basis(ps)

        if callable(f):
            if f.coordtype == 'barycentric':
                val = f(bcs)
            elif f.coordtype == 'cartesian':
                val = f(ps)

            if len(val.shape) == 1: # (GD, )
                val = val[None, None, :]
        elif isinstance(f, np.ndarray):
            val = f[None, None, :]

        bb = np.einsum('i, ijm, ijkm, j->jk', ws, val, phi, self.cellmeasure)

        gdof = gdof or cell2dof.max()
        dtype = phi.dtype if dtype is None else dtype
        b = np.zeros(gdof, dtype=dtype)
        np.add.at(b, cell2dof, bb)
        return b

    def construct_vector_v_s(self, f, basis, cell2dof, gdof=None, q=None):
        """

        Notes
        -----
        f 是向量函数
        basis 是标量函数
        """
        mesh = self.mesh
        qf = self.integrator if q is None else mesh.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = mesh.bc_to_point(bcs)

        if basis.coordtype == 'barycentric':
            phi = basis(bcs)
        elif c.coordtype == 'cartesian':
            phi = basis(ps)

        if callable(f):
            if f.coordtype == 'barycentric':
                val = f(bcs)
            elif f.coordtype == 'cartesian':
                val = f(ps)
            bb = np.einsum('i, ij, ijk, j->jk',
                    ws, val, phi, self.cellmeasure)
        elif isinstance(f, (int, float)):
            bb = np.einsum('m, mik, i->ik',
                    f*ws, phi, self.cellmeasure)

        gdof = gdof or cell2dof.max()
        shape = (gdof, val.shape[-1])
        b = np.zeros(shape, dtype=phi.dtype)
        np.add.at(b, (cell2dof, np.s_[:]), bb)

        return b



# old api 
    def integral(self, u, celltype=False, barycenter=True):
        """
            """
        qf = self.integrator
        bcs = qf.quadpts  # 积分点 (NQ, 3)
        ws = qf.weights  # 积分点对应的权重 (NQ, )
        if barycenter:
            val = u(bcs)
        else:
            ps = self.mesh.bc_to_point(bcs)  # (NQ, NC, 2)
            val = u(ps)
        dim = len(ws.shape)
        s0 = 'abcde'
        s1 = '{}, {}j..., j->j...'.format(s0[0:dim], s0[0:dim])
        e = np.einsum(s1, ws, val, self.cellmeasure)
        if celltype is True:
            return e
        else:
            return e.sum()

    def L2_norm(self, uh, celltype=False):
        def f(x):
            return uh(x)**2
        e = self.integral(f, celltype=celltype)
        if celltype is False:
            return np.sqrt(e.sum())
        else:
            return np.sqrt(e)

    def L2_norm_1(self, uh, celltype=False):
        def f(x):
            return np.sum(uh**2, axis=-1)*self.cellmeasure

        e = self.integral(f, celltype=celltype)
        if celltype is False:
            return np.sqrt(e.sum())
        else:
            return np.sqrt(e)

    def L1_error(self, u, uh, celltype=False):
        def f(x):
            xx = self.mesh.bc_to_point(x)
            return np.abs(u(xx) - uh(x))
        e = self.integral(f, celltype=celltype)
        if celltype is False:
            return e.sum()
        else:
            return e
        return

    def L2_error(self, u, uh, celltype=False):
        def f(bc):
            xx = self.mesh.bc_to_point(bc)
            return (u(xx) - uh(bc))**2
        e = self.integral(f, celltype=celltype)
        if celltype is False:
            return np.sqrt(e.sum())
        else:
            return np.sqrt(e)
        return 

    def L2_error_uI_uh(self, uI, uh, celltype=False):
        def f(x):
            return (uI(x) - uh(x))**2
        e = self.integral(f, celltype=celltype)
        if celltype is False:
            return np.sqrt(e.sum())
        else:
            return np.sqrt(e)
        return 

    def Lp_error(self, u, uh, p, celltype=False):
        def f(x):
            xx = self.mesh.bc_to_point(x)
            return np.abs(u(xx) - uh(x))**p
        e = self.integral(f, celltype=celltype)
        if celltype is False:
            return e.sum()**(1/p)
        else:
            return e**(1/p)
        return 
