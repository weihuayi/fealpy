import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

from fealpy.functionspace import ParametricLagrangeFiniteElementSpace



class TriplePointShockInteractionModel:

    def __init__(self):
        self.domain = [0, 7, 0, 3]

    def space_mesh(self, p, NS=0):
        from fealpy.mesh import MeshFactory
        mf = MeshFactory()
        mesh = mf.boxmesh2d(self.domain, nx=70, ny=30, p=p, meshtype='quad') 
        mesh.uniform_refine(NS)
        return mesh

    def time_mesh(self, NT=100):
        from fealpy.timeintegratoralg.timeline import UniformTimeLine
        timeline = UniformTimeLine(0, 1, NT)
        return timeline

    def subdomain(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = np.zeros(p.shape[:-1], dtype=p.dtype)
        flag[x < 1] = 1  
        flag[(x > 1) & (y < 1.5)] = 2
        flag[(x > 1) & (y > 1.5)] = 3 
        return flag

    def init_rho(self, p):
        x = p[..., 0]
        y = p[..., 1]

        val = np.zeros(p.shape[:-1], dtype=p.dtype)
        val[x < 1] = 1.0
        val[(x > 1) & (y < 1.5)] = 1.0
        val[(x > 1) & (y > 1.5)] = 0.125
        return val

    def init_velocity(self, v):
        v[:] = 0.0

    def init_energe(self, e):
        p = e.space.interpolation_points()
        x = p[..., 0]
        y = p[..., 1]
        e[x < 1] = 1.0/(1.5 - 1)/1.0
        e[(x > 1) & (y < 1.5)] = 0.1/(1.4-1)/1.0
        e[(x > 1) & (y > 1.5)] = 0.1/(1.5-1)/0.125 

    def adiabatic_index(self, p):
        """
        Notes
        -----
        绝热指数
        """
        x = p[..., 0]
        y = p[..., 1]

        val = np.zeros(p.shape[:-1], dtype=p.dtype)
        val[x < 1] = 1.5
        val[(x > 1) & (y < 1.5)] = 1.4 
        val[(x > 1) & (y > 1.5)] = 1.5 
        return val

    def stress(self, bc, e, rho, gamma, mesh0, mesh1):
        """

        Notes

        rho: (NC, ) 每个单元上的初始密度
        gamma: (NC, ) 每个单元上的绝热指数
        """

        J0 = mesh0.jacobi_matrix(bc) # (NQ, NC, GD, GD)
        J1 = mesh1.jacobi_matrix(bc) # (NQ, NC, GD, GD)
        J0 = np.linalg.det(J0) # (NQ, NC)
        J1 = np.linalg.det(J1) # (NQ, NC)

        val = J0
        val /= J1 
        val *= e.value(bc)
        val *= (1 - gamma)*rho
        return val 


class ModelSover():

    def __init__(self, model, p, NS=0, NT=100):
        self.model = model

        self.mesh0 = model.space_mesh(NS=NS, p=p) # 初始网格
        self.mesh1 = model.space_mesh(NS=NS, p=p) # 计算网格

        self.timeline = model.time_mesh(NT=NT)

        self.cspace = ParametricLagrangeFiniteElementSpace(self.mesh1, p=p, spacetype='C')
        self.dspace = ParametricLagrangeFiniteElementSpace(self.mesh1, p=p-1, spacetype='D')
        
        self.integrator = self.cspace.integralalg.integrator

        bc = self.mesh0.entity_barycenter('cell')
        self.rho = model.init_rho(bc) # (NC, )
        self.gamma = model.adiabatic_index(bc) #(NC, )

        self.MV = self.cspace.mass_matrix(c=self.rho)
        self.ME = self.dspace.mass_matrix(c=self.rho)

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

        # 边界条件处理

        edge2cell = self.mesh1.ds.edge_to_cell()
        isBdEdge = edge2cell[:, 0] == edge2cell[:, 1]
        edge2dof = self.cspace.edge_to_dof()[isBdEdge]
        en = self.mesh1.edge_unit_normal(index=isBdEdge) # (NE, GD)

        # (NE, ldof, GD) * (NE, 1, GD) = (NE, ldof, GD) --> (NE, ldof)
        l = np.sum(self.cv[edge2dof, :]*en[:, None, :], axis=-1)
        val = l[..., None]*en[:, None, :] # (NE, ldof, 1) * (NE, 1, GD)--> 
        np.subtract.at(self.cv, (edge2dof, np.s_[:]), val)

        F = spsolve(self.ME, self.cv[:, 0]@M0 + self.cv[:, 1]@M1)
        self.ce[:] = self.e + dt/2*F

        self.cx[:] = self.x + dt/2*self.cv

        self.mesh1.node[:] = self.cx
        M0, M1 = self.get_force_matrix() 

        F0 = spsolve(self.MV, M0@one)
        F1 = spsolve(self.MV, M1@one)
        self.cv[:, 0] = self.v[:, 0] - dt*F0 
        self.cv[:, 1] = self.v[:, 1] - dt*F1

        l = np.sum(self.cv[edge2dof, :]*en[:, None, :], axis=-1)  # (NE, ldof)
        val = l[..., None]*en[:, None, :] # (NE, ldof, GD)
        np.subtract.at(self.cv, (edge2dof, np.s_[:]), val)

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


if __name__ == '__main__':
    model = TriplePointShockInteractionModel()

    p = 2
    solver = ModelSover(model, p, NS=0, NT=1000) 
    solver.solve(step=1)

