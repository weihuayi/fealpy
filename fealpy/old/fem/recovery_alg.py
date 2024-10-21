import numpy as np

class LinearRecoveryAlg():

    def recovery_estimate(self, uh, method='simple'):
        """
        """
        mesh = uh.space.mesh
        rguh = self.grad_recovery(uh, method=method)
        eta = mesh.error(rguh.value, uh.grad_value, power=2, celltype=True) # 计算单元上的恢复型误差
        return eta

    def grad_recovery(self, uh, method='simple'):
        """
        @brief 输入一个线性有限元函数，把其梯度恢复到分片线性有限元空间
        @todo 检查拉格朗日有限元空间中自由度排序的问题
        """
        space = uh.space
        TD = space.top_dimension()
        GD = space.geo_dimension()
        gdof = space.number_of_global_dofs()
        cell2dof = space.cell_to_dof()

        bc = np.array([1/3]*(TD+1), dtype=np.float64)
        guh = uh.grad_value(bc) # (NC, GD)

        # 'sdofs': 标量自由度优先排序，例如 x_0, x_1, ..., y_0, y_1, ..., z_0, z_1, ...
        # 'vdims': 向量分量优先排序，例如 x_0, y_0, z_0, x_1, y_1, z_1, ...

        rguh = space.function(dim=GD) # 默认是 vdims, (gdof, GD)
        deg = np.zeros(gdof, dtype=np.float64)

        if method == 'simple':
            np.add.at(deg, cell2dof, 1)
            np.add.at(rguh, (cell2dof, np.s_[:]), guh[:, None, :])
        elif method == 'harmonic':
            val = 1.0/space.mesh.entity_measure('cell')
            np.add.at(deg, cell2dof, val[:, None])
            guh *= val[:, None] 
            np.add.at(rguh, (cell2dof, np.s_[:]), guh[:, None, :])

        rguh /= deg[:, None]
        return rguh


class recovery_alg:
    def __init__(self, space, q=None, cellmeasure=None):
        self.space = space
        self.p = space.p
        q = q if q is not None else self.p+3
        self.q = q
        self.mesh = space.mesh
        self.integrator = self.mesh.integrator(q, etype='cell')
        self.cellmeasure = cellmeasure if cellmeasure is not None else self.mesh.entity_measure('cell')

    def recovery_estimate(self, uh, method='simple'):
        """
        """
        space = self.space
        rguh = self.grad_recovery(uh, method=method)
        guh = uh.grad_value
        eta = self.error(rguh.value, uh.grad_value, power=2, celltype=True) # 计算单元上的恢复型误差
        return eta

    def grad_recovery(self, uh, method='simple'):
        """

        Notes
        -----

        uh 是线性有限元函数，该程序把 uh 的梯度(分片常数）恢复到分片线性连续空间
        中。

        """
        space = self.space
        GD = space.GD
        cell2dof = space.cell_to_dof()
        gdof = space.number_of_global_dofs()
        ldof = space.number_of_local_dofs()
        p = self.p
        bc = space.dof.multiIndex/p
        guh = uh.grad_value(bc)
        guh = guh.swapaxes(0, 1)
        if space.doforder == 'sdofs':
            rguh0 = space.function(dim=GD)
            rguh = rguh0.T[:]
        else:
            rguh = space.function(dim=GD)

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
            ipoints = space.interpolation_points()
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
        if space.doforder == 'sdofs':
            rguh0[:] = rguh.T
            return rguh0
        else:
            return rguh
    
    def error(self, u, v, power=2, celltype=False, q=None):
        """

        @brief 给定两个函数，计算两个函数的之间的差，默认计算 L2 差（power=2)
               power 的取值可以是任意的 p

        TODO
        ----
        1. 考虑无穷范数的情形
        """
        mesh = self.mesh
        space = self.space
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
        
        if space.doforder == 'sdofs':
            u = np.transpose(u, (0, 2, 1))
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

