import sys
import numpy as np
from scipy.sparse import bmat, spdiags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.mesh import MeshFactory
from fealpy.decorator import cartesian
from fealpy.timeintegratoralg.timeline import UniformTimeLine
from fealpy.functionspace import RaviartThomasFiniteElementSpace2d

class PDE():
    def __init__(self,  T):
        self.T = T

    def domain(self):
        return np.array([0, 1, 0, 1])

    @cartesian
    def y_solution(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.sin(pi*x)*np.sin(pi*y)*np.exp(2*t)
        return val # val.shape == x.shape

    @cartesian
    def yd(self, p, t):
        T = self.T
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = (np.exp(2*t) + pi*np.cos(pi*t))*np.sin(pi*x)*np.sin(pi*y) \
                + 2*pi*(np.cos(pi*t) - np.cos(pi*T) - pi*np.sin(pi*t)) \
                *np.sin(pi*x)*np.sin(pi*y) \
                + (1 - T + t)*pi**2*np.sin(pi*x)*np.sin(pi*y)
        return val # val.shape == x.shape

    @cartesian
    def p_solution(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = pi*np.cos(pi*x)*np.sin(pi*y)*np.exp(2*t)
        val[..., 1] = pi*np.sin(pi*x)*np.cos(pi*y)*np.exp(2*t)
        return val # val.shape == x.shape


    @cartesian
    def tp_solution(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = pi*np.cos(pi*x)*np.sin(pi*y)*(np.exp(2*t)+1)/2
        val[..., 1] = pi*np.sin(pi*x)*np.cos(pi*y)*(np.exp(2*t)+1)/2
        return val # val.shape == x.shape

    @cartesian
    def z_solution(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.sin(pi*t)*np.sin(pi*x)*np.sin(pi*y)
        return val

    @cartesian
    def u_solution(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = (4/pi**2 - np.sin(pi*x)*np.sin(pi*y))*np.sin(pi*t)
        return val


    @cartesian
    def tpd(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = pi*np.cos(pi*x)*np.sin(pi*y)*np.exp(2*t)/2
        val[..., 1] = pi*np.sin(pi*x)*np.cos(pi*y)*np.exp(2*t)/2
        return val # val.shape == x.shape

    @cartesian
    def q_solution(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi*np.sin(pi*t)*np.cos(pi*x)*np.sin(pi*y) \
                + pi*np.cos(pi*x)*np.sin(pi*y)/2
        val[..., 1] = -pi*np.sin(pi*t)*np.sin(pi*x)*np.cos(pi*y) \
                + pi*np.sin(pi*x)*np.cos(pi*y)/2
        return val # val.shape == x.shape

    @cartesian
    def tq(self, p, t, T):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = (np.cos(pi*t) - np.cos(pi*T) - pi*np.sin(pi*t)) \
                *np.cos(pi*x)*np.sin(pi*y) \
                + (1 - T + t)*pi*np.cos(pi*x)*np.sin(pi*y)/2
        val[..., 1] = (np.cos(pi*t) - np.cos(pi*T) - pi*np.sin(pi*t)) \
                *np.sin(pi*x)*np.cos(pi*y) \
                + (1 - T + t)*pi*np.sin(pi*x)*np.cos(pi*y)/2
        return val # val.shape == x.shape

    @cartesian
    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = (2*np.exp(2*t) + pi**2*np.exp(2*t) \
                + pi**2)*np.sin(pi*x)*np.sin(pi*y) \
                - (4/(pi**2) - np.sin(pi*x)*np.sin(pi*y))*np.sin(pi*t)
        return val


class Model():
    def __init__(self, pde, mesh, timeline):
        self.pde = pde
        self.mesh = mesh

        self.uspace = RaviartThomasFiniteElementSpace2d(mesh, p=0)
        self.pspace = self.uspace.smspace 

        self.timeline = timeline
        NL = timeline.number_of_time_levels()

        # state variable
        self.yh = self.pspace.function(dim=NL)
        self.uh = self.pspace.function(dim=NL)
        self.tph = self.uspace.function(dim=NL)
        self.ph = self.uspace.function()

        f = cartesian(lambda p: pde.y_solution(p, 0))
        self.yh[:, 0] = self.pspace.local_projection(f) 

        # costate variable
        self.zh = self.pspace.function(dim=NL)
        self.tqh = self.uspace.function(dim=NL)
        self.qh = self.uspace.function()

        self.A = self.uspace.stiff_matrix() # RT 质量矩阵
        self.D = self.uspace.div_matrix() # (p, \div v) 
        self.M = self.pspace.mass_matrix()

    def get_state_current_right_vector(self, sp, nt):
        '''
        sp: (NE, 1), 前 n-1 层的累加
        nt: 表示 n-1 时间层,要求的是 n 个时间层的方程
        '''
        dt = self.dt
        NC = self.mesh.number_of_cells()
        u = self.uh[:, nt+1]
        NE = self.mesh.number_of_edges()
       # f1 = -\sum_{i=1}^n-1\Delta t (p^i, v^i)
        f1 = -dt*sp

        # f2 = 0
        f2 = np.zeros(NE, dtype=mesh.ftype)

        # f3 = \Delta t(f^n + u^n, w_h) + (y^{n-1}, w_h)
        cell2dof = self.pspace.cell_to_dof()
        gdof = self.pspace.number_of_global_dofs()
        qf = self.integrator
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = mesh.bc_to_point(bcs, etype='cell')
        phi = self.pspace.basis(bcs)
        val = self.pde.source(ps, (nt+1)*dt)
        bb = np.einsum('i, ij, jk, j->jk', ws, val, phi, self.cellmeasure)

        gdof = gdof or cell2dof.max()
        shape = (gdof, )
        b = np.zeros(shape, dtype=phi.dtype)
        np.add.at(b, cell2dof, bb)

        f3 = dt*b + dt*self.M@u + self.M@self.yh[:, nt]

        return np.r_[np.r_[f1, f2], f3]

    def get_costate_current_right_vector(self, sq, nt):
        '''
        sq: (NE, 1) n 到 N 时间层的累加
        nt: 正序的 n-1 个时间层的值, 则逆序所对应的时间层为 NL - nt -1
        NL: 时间方向剖分的总的层数
        '''
        dt = self.dt
        NL = timeline.number_of_time_levels()
        NC = self.mesh.number_of_cells()
        tpd = self.uspace.function()
 
        #  f1 = -\sum_{i=1}^n-1\Delta t (p^i, v)
        f1 = -dt*sq

        # f2 = (\tilde p^n_h - \tilde p^n_d, v)
        edge2dof = self.uspace.dof.edge_to_dof()
        en = self.mesh.edge_unit_normal()
        def f0(bc):
            ps = mesh.bc_to_point(bc, etype='edge')
            return np.einsum('ijk, jk, ijm->ijm', self.pde.tpd(ps, NL-nt-1), en, self.pspace.edge_basis(ps))
        tpd[edge2dof] = self.pspace.integralalg.edge_integral(f0, edgetype=True)
        f2 = self.A@self.tph[:, NL-nt-1] - self.A@tpd

        # f3 = \Delta t(y^n_h - y^n_d, w) + (z^n_h, w)
        bc = np.array([1/3, 1/3, 1/3])
        ps = mesh.bc_to_point(bc, etype='cell')
        bb1 = self.M@self.zh[:, NL-nt-1]
        val = self.pde.yd(ps, (NL-nt-1)*dt, 1)
        bb2 = dt*self.M@(self.yh[:, NL-nt-1] - val)

        f3 = bb1 + bb2

        return np.r_[np.r_[f1, f2], f3]

    def state_one_step_solve(self, t, sp):
        dt = self.dt
        F = self.get_state_current_right_vector(sp, t)
        A = bmat([[self.A, (dt-1)*self.A, None],[None, self.A, self.D], \
                [-self.dt*self.D.T, None, self.M]], format='csr')
        PU = spsolve(A, F)
        return PU

    def costate_one_step_solve(self, t, sq):
        dt = self.dt
        F = self.get_costate_current_right_vector(sq, t)
        A = bmat([[self.A, (dt-1)*self.A, None],[None, self.A, -self.D], \
                [self.dt*self.D.T, None, self.M]], format='csr')
        QZ = spsolve(A, F)
        return QZ

    def state_solve(self):
        timeline = self.timeline
        timeline.reset()
        NC = self.mesh.number_of_cells()
        NE = self.mesh.number_of_edges()
        sp = np.zeros(NE, dtype=mesh.ftype)
        while not timeline.stop():
            PU = self.state_one_step_solve(timeline.current, sp)
            self.tph[:, timeline.current+1] = PU[:NE]
            self.ph[:] = PU[NE:2*NE]
            self.yh[:, timeline.current+1] = PU[2*NE:]
            timeline.current += 1
            sp = sp + self.A@self.ph[:] 
        timeline.reset()
        return sp

    def costate_solve(self):
        timeline = self.timeline
        NL = timeline.number_of_time_levels()
        NE = self.mesh.number_of_edges()
        qf = self.integrator
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = mesh.bc_to_point(bcs, etype='cell')
        sq = np.zeros(NE, dtype=mesh.ftype)
        
        timeline.reset()
        while not timeline.stop():
            QZ = self.costate_one_step_solve(timeline.current, sq)
            self.tqh[:, NL-timeline.current-2] = QZ[:NE]
            self.qh[:] = QZ[NE:2*NE]
            self.zh[:, NL-timeline.current-2] = QZ[2*NE:]
            zh1 = self.pspace.function(array=self.zh[:, NL - timeline.current-2])
            val = zh1(ps)
            e = np.einsum('i, ij, j->j', ws, val, self.cellmeasure)
            self.uh[:, NL-timeline.current-1] = max(0, np.sum(e)/np.sum(self.cellmeasure)) \
                            - self.zh[:, NL - timeline.current-2]

#            self.uh[:, NL-timeline.current-1] = self.pspace.integralalg.cell_integral \
#                    (zh1, celltype=True)/np.sum(self.cellmeasure)
            timeline.current += 1
            sq = sq + self.qh
        timeline.reset()
        return sq

    def nonlinear_solve(self, maxit=20):

        eu = 1
        k = 1
        dt = self.dt
        timeline = self.timeline
        NL = timeline.number_of_time_levels()
        timeline.reset()
        uI = self.pspace.function(NL)
        while not timeline.stop():
            uI[:, timeline.current+1] = self.pde.u_solution(self.bc, (timeline.current+1)*dt)
            timeline.current += 1
        timeline.reset()
        while eu > 1e-4 and k < 20:
            print('maxit', maxit)
            print('ui', uI)
            uh1 = self.uh.copy()
#            print('uh1', uh1)
            sp = self.state_solve()
            sq = self.costate_solve()
            print('uh', self.uh)
            eu = abs(sum(sum(uh1 - self.uh)))
            k = k + 1
            print('eu', eu)
            print('k', k)

        return sp, sq

    def L2error(self):
        dt = self.dt
        NL = timeline.number_of_time_levels()
        T = dt*(NL-1)

#        def f(bc, t):
#            xx = mesh.bc_to_point(bc)
#            return (self.pde.p_solution(xx, t) - self.ph(xx))**2
#        pL2error = self.uspace.integralalg.integral(lambda x: f(x, T))
        pL2error = self.uspace.integralalg.L2_error(lambda x: \
                self.pde.p_solution(x, T), self.ph)
        print(pL2error)
        tpL2error = self.uspace.integralalg.L2_error(lambda x: \
                self.pde.tp_solution(x, T), \
                self.uspace.function(array=self.tph[:, NL-1])) 
        print(tpL2error)


        return pL2error


        
pde = PDE()
mesh = pde.init_mesh(n=1, meshtype='tri')
space = RaviartThomasFiniteElementSpace2d(mesh, p=0)
timeline = UniformTimeLine(0, 1, 10)
MFEMModel = Model(pde, mesh, timeline)
MFEMModel.nonlinear_solve()
pL2error = MFEMModel.L2error()
#state = StateModel(pde, mesh, timeline)




