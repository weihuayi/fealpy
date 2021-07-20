import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

from fealpy.functionspace import ParametricLagrangeFiniteElementSpace

class LagrangianHydrodynamicsSimulator():

    def __init__(self, model, p, NS=0, NT=100):

        self.model = model # 物理模型
        self.mesh0 = model.space_mesh(NS=NS, p=p) # 初始网格
        self.mesh1 = model.space_mesh(NS=NS, p=p) # 计算网格

        self.timeline = model.time_mesh(NT=NT) # 时间离散网格

        # 运动学空间
        self.cspace = ParametricLagrangeFiniteElementSpace(self.mesh1, p=p, spacetype='C')
        # 热力学空间
        self.dspace = ParametricLagrangeFiniteElementSpace(self.mesh1, p=p-1, spacetype='D')
        
        self.integrator = self.cspace.integralalg.integrator

        bc = self.mesh0.entity_barycenter('cell')
        # 物质密度
        self.rho = model.init_rho(bc) # (NC, )
        # 绝热指数 
        self.gamma = model.adiabatic_index(bc) #(NC, )

        self.MV = self.cspace.mass_matrix(c=self.rho) # 运动空间的质量矩阵
        self.ME = self.dspace.mass_matrix(c=self.rho) # 热力学空间的质量矩阵

        GD = self.mesh1.geo_dimension()
        self.x = self.cspace.function(dim=GD) # 位置
        self.v = self.cspace.function(dim=GD) # 速度
        self.e = self.dspace.function() # 能量

        self.x[:] = self.mesh1.entity('node') # 初始化自由度位置
        model.init_velocity(self.v) # 初始化速度
        model.init_energe(self.e) # 初始化能量

        self.cx = self.cspace.function(dim=GD) # 保存临时的解
        self.cv = self.cspace.function(dim=GD)
        self.ce = self.dspace.function()

        self.cx[:] = self.mesh1.entity('node')
        model.init_velocity(self.cv)
        model.init_energe(self.ce)

        self.mesh1.celldata['rho'] = self.rho
        self.mesh1.celldata['gamma'] = self.gamma
        self.mesh1.nodedata['velocity'] = self.cv


    def get_force_matrix(self, q=None):

        # 积分公式
        qf = self.integrator if q is None else self.mesh1.integrator(q, etype='cell')
        # bcs : (NQ, n)
        # ws : (NQ, )
        bcs, ws = qf.get_quadrature_points_and_weights() 

        rm = self.mesh1.reference_cell_measure() # 参考单元测度
        d = self.mesh1.first_fundamental_form(bcs)
        d = np.sqrt(np.linalg.det(d))

        d *= self.model.stress(bcs, self.ce, self.rho, self.gamma, self.mesh0, self.mesh1)

        gphi = self.cspace.grad_basis(bcs) # (NQ, NC, ldof, GD)
        phi = self.dspace.basis(bcs)

        M0 = np.einsum('q, qci, qcj, qc->cij', ws*rm, gphi[..., 0], phi, d) # (NC, ldof0, ldof1)
        M1 = np.einsum('q, qci, qcj, qc->cij', ws*rm, gphi[..., 1], phi, d) # (NC, ldof0, ldof1)

        c2d0 = self.cspace.cell_to_dof()
        c2d1 = self.dspace.cell_to_dof()

        I = np.broadcast_to(c2d0[:, :, None], shape=M0.shape)
        J = np.broadcast_to(c2d1[:, None, :], shape=M0.shape)

        gdof0 = self.cspace.number_of_global_dofs()
        gdof1 = self.dspace.number_of_global_dofs()
        M0 = csr_matrix(
                (M0.flat, (I.flat, J.flat)), 
                shape=(gdof0, gdof1)
                )

        M1 = csr_matrix(
                (M1.flat, (I.flat, J.flat)), 
                shape=(gdof0, gdof1)
                )

        return M0, M1


    def solve_one_step(self):

        dt = self.timeline.current_time_step_length()
        
        M0, M1 = self.get_force_matrix()

        one = np.ones(M0.shape[1])
        F0 = spsolve(self.MV, M0@one)
        F1 = spsolve(self.MV, M1@one)

        self.cv[:, 0] = self.v[:, 0] - dt/2*F0 
        self.cv[:, 1] = self.v[:, 1] - dt/2*F1

        # 网格节点自由度
        # dof == 0: 表示固定点
        # dof == 1: 表示边界上的点
        # dof == 2: 区域内部点
        dof = self.mesh0.nodedata['dof'] 

        # 边界条件处理
        self.cv[dof==0] = 0.0
        en = self.mesh0.meshdata['bd_normal']
        vv = self.cv[dof==1]
        l = np.sum(vv*en, axis=-1) # (NE, )
        self.cv[dof==1] -= l[:, None]*en


        F = spsolve(self.ME, self.cv[:, 0]@M0 + self.cv[:, 1]@M1)
        self.ce[:] = self.e + dt/2*F

        self.cx[:] = self.x + dt/2*self.cv

        self.mesh1.node[:] = self.cx
        M0, M1 = self.get_force_matrix() 

        F0 = spsolve(self.MV, M0@one)
        F1 = spsolve(self.MV, M1@one)
        self.cv[:, 0] = self.v[:, 0] - dt*F0 
        self.cv[:, 1] = self.v[:, 1] - dt*F1

        # 边界条件处理
        self.cv[dof==0] = 0.0
        vv = self.cv[dof==1]
        l = np.sum(vv*en, axis=-1) # (NE, )
        self.cv[dof==1] -= l[:, None]*en


        v = (self.cv + self.v)/2
        F = spsolve(self.ME, v[:, 0]@M0 + v[:, 1]@M1)
        self.ce[:] = self.e + dt*F
        self.cx[:] = self.x + dt*v

        self.x[:] = self.cx
        self.v[:] = self.cv
        self.e[:] = self.ce

        self.mesh1.node[:] = self.x
        self.mesh1.nodedata['velocity'] = self.v

    def solve(self, step=1):
        """

        Notes
        -----

        计算所有的时间层。
        """

        timeline = self.timeline
        dt = timeline.current_time_step_length()
        timeline.reset() # 时间置零

        n = timeline.current
        fname = 'test_'+ str(n).zfill(10) + '.vtu'
        self.mesh1.to_vtk(fname=fname)
        while not timeline.stop():
            self.solve_one_step()
            timeline.current += 1
            if timeline.current%step == 0:
                n = timeline.current
                fname = 'test_'+ str(n).zfill(10) + '.vtu'
                self.mesh1.to_vtk(fname=fname)
        timeline.reset()
