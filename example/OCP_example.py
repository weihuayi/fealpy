import sys
import numpy as np
from scipy.sparse import bmat, spdiags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh
from fealpy.decorator import cartesian
from fealpy.timeintegratoralg.timeline import UniformTimeLine
from fealpy.functionspace import RaviartThomasFiniteElementSpace2d
from fealpy.functionspace.femdof import multi_index_matrix2d

class PDE():

    def domain(self):
        return np.array([0, 1, 0, 1])

    def init_mesh(self, n=1, meshtype='tri'):
        """ generate the initial mesh
        """
        node = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float64)

        if meshtype == 'tri':
            cell = np.array([(1, 2, 0), (3, 0, 2)], dtype=np.int_)
            mesh = TriangleMesh(node, cell)
            mesh.uniform_refine(n)
            return mesh
        else:
            raise ValueError("".format)


    @cartesian
    def y(self, point, t):
        x1 = point[..., 0]
        x2 = point[..., 1]
        pi = np.pi
        val = np.sin(pi*x1)*np.sin(pi*x2)*np.exp(2*t)
        return val # val.shape == x.shape

    @cartesian
    def p(self, point, t):
        x1 = point[..., 0]
        x2 = point[..., 1]
        pi = np.pi
        val = np.zeros(point.shape, dtype=np.float64)
        val[..., 0] = pi*np.cos(pi*x1)*np.sin(pi*x2)*np.exp(2*t)
        val[..., 1] = pi*np.sin(pi*x1)*np.cos(pi*x2)*np.exp(2*t)
        return val # val.shape == x.shape


    @cartesian
    def tp(self, point, t):
        x1 = point[..., 0]
        x2 = point[..., 1]
        pi = np.pi
        val = np.zeros(point.shape, dtype=np.float64)
        val[..., 0] = pi*np.cos(pi*x1)*np.sin(pi*x2)*(np.exp(2*t)+1)/2
        val[..., 1] = pi*np.sin(pi*x1)*np.cos(pi*x2)*(np.exp(2*t)+1)/2
        return val # val.shape == x.shape

    @cartesian
    def z(self, point, t):
        x1 = point[..., 0]
        x2 = point[..., 1]
        pi = np.pi
        val = np.sin(pi*t)*np.sin(pi*x1)*np.sin(pi*x2)
        return val

    @cartesian
    def u(self, point, t):
        x1 = point[..., 0]
        x2 = point[..., 1]
        pi = np.pi
        val = 4*np.sin(pi*t)*np.sin(pi*x1)*np.sin(pi*x2)/pi/pi
        return val


    @cartesian
    def tpd(self, point, t):
        x1 = point[..., 0]
        x2 = point[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = pi*np.cos(pi*x1)*np.sin(pi*x2)*np.exp(2*t)/2
        val[..., 1] = pi*np.sin(pi*x1)*np.cos(pi*x2)*np.exp(2*t)/2
        return val # val.shape == x.shape

    @cartesian
    def q(self, point, t):
        x1 = point[..., 0]
        x2 = point[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi*np.sin(pi*t)*np.cos(pi*x1)*np.sin(pi*x2) \
                + pi*np.cos(pi*x1)*np.sin(pi*x2)/2
        val[..., 1] = -pi*np.sin(pi*t)*np.sin(pi*x1)*np.cos(pi*x2) \
                + pi*np.sin(pi*x1)*np.cos(pi*x2)/2
        return val # val.shape == x.shape

    @cartesian
    def tq(self, point, t, T):
        x1 = point[..., 0]
        x2 = point[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = (np.cos(pi*t) - np.cos(pi*T) - pi*np.sin(pi*t)) \
                *np.cos(pi*x1)*np.sin(pi*x2) \
                + (1 - T - t)*pi*np.cos(pi*x1)*np.sin(pi*x2)/2
        val[..., 1] = (np.cos(pi*t) - np.cos(pi*T) - pi*np.sin(pi*t)) \
                *np.sin(pi*x1)*np.cos(pi*x2) \
                + (1 - T - t)*pi*np.sin(pi*x1)*np.cos(pi*x2)/2
        return val # val.shape == x.shape


    @cartesian
    def source(self, p, t):
        x1 = p[..., 0]
        x2 = p[..., 1]
        pi = np.pi
        val = (2*np.exp(2*t) + pi**2*np.exp(2*t) \
                + pi**2)*np.cos(pi*x1)*np.cos(pi*x2) \
                - (4/(pi**2) - np.sin(pi*x1)*np.sin(pi*x2))*np.sin(pi*t)
        return val

class Model():
    def __init__(self, pde, mesh, timeline):
        self.pde = pde
        self.mesh = mesh
        NC = mesh.number_of_cells()
        self.integrator = mesh.integrator(3, 'cell')
        self.bc = mesh.entity_barycenter('cell')
        self.ec = mesh.entity_barycenter('edge')
        self.cellmeasure = mesh.entity_measure('cell')
        self.uspace = RaviartThomasFiniteElementSpace2d(mesh, p=0)
        self.pspace = self.uspace.smspace 

        self.timeline = timeline
        NL = timeline.number_of_time_levels()
        self.dt = timeline.current_time_step_length()

        # state variable
        self.yh = self.pspace.function(dim=NL)
        self.uh = self.pspace.function(dim=NL)
        self.tph = self.uspace.function(dim=NL)
        self.ph = self.uspace.function()
        self.yh[:, 0] = pde.y(self.bc, 0)

        # costate variable
        self.zh = self.pspace.function(dim=NL)
        self.tqh = self.uspace.function(dim=NL)
        self.qh = self.uspace.function()


        self.A = self.uspace.stiff_matrix() # RT 质量矩阵
        self.D = self.uspace.div_matrix() # TODO: 确定符号
        data = self.cellmeasure*np.ones(NC, )
        self.M = spdiags(data, 0, NC, NC)

#        self.M = self.pspace.cell_mass_matrix() # 每个单元上是 1 x 1 的矩阵i

    def get_state_current_right_vector(self, sp, t):
        dt = self.dt
        f1 = dt*sp
        NC = self.mesh.number_of_cells()
        u = self.uh[:, t].reshape(NC,1)
        NE = self.mesh.number_of_edges()
        f2 = np.zeros(NE, dtype=mesh.ftype)

        cell2dof = self.pspace.cell_to_dof()
        gdof = self.pspace.number_of_global_dofs()
        qf = self.integrator
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = mesh.bc_to_point(bcs, etype='cell')
        phi = self.pspace.basis(bcs)
        val = pde.source(ps, t)
        bb1 = np.einsum('i, ij, jk, j->jk', ws, val, phi, self.cellmeasure)
        bb2 = self.M@u
        bb = bb1 + bb2

        gdof = gdof or cell2dof.max()
        shape = (gdof, )
        b = np.zeros(shape, dtype=phi.dtype)
        np.add.at(b, cell2dof, bb)
        print('t', t)

        f3 = dt*b + self.yh[:, t]

        return np.r_[np.r_[f1, f2], f3]

    def get_costate_current_right_vector(self):
        pass

    def state_one_step_solve(self, t, sp):

        F = self.get_state_current_right_vector(sp, t)
        M = self.M
        A = bmat([[self.A, None, None],[None, self.A, self.D], \
                [-self.dt*self.D.T, None, self.M]], format='csr')
        PU = spsolve(A, F)
        return PU

    def costate_one_step_solve(self):
        pass

    def state_solve(self):
        timeline = self.timeline
        timeline.reset()
        NC = self.mesh.number_of_cells()
        NE = self.mesh.number_of_edges()
        no = self.mesh.node_to_cell()
        print(no)
        sp = np.zeros(NE, dtype=mesh.ftype)
        while not timeline.stop():
#            self.state_solve()
            print('time', timeline.current)
            PU = self.state_one_step_solve(timeline.current, sp)
            self.tph[:, timeline.current+1] = PU[:NE]
            self.ph[:] = PU[NE:2*NE]
            self.yh[:, timeline.current+1] = PU[2*NE:]
            timeline.current += 1
            print('time', timeline.current)
            sp = sp + self.A@self.ph[:] 
        timeline.reset()

    def costate_solve(self):
        timeline = self.timeline
        timeline.reset()
        while not timeline.stop():
            self.state_solve()
            timeline.current += 1
        timeline.reset()
        pass

    def nonlinear_solve(self, maxit=1000):
        pass



pde = PDE()
mesh = pde.init_mesh(n=1, meshtype='tri')
space = RaviartThomasFiniteElementSpace2d(mesh, p=0)
timeline = UniformTimeLine(0, 1, 2)
MFEMModel = Model(pde, mesh, timeline)
M = MFEMModel.state_solve()
#state = StateModel(pde, mesh, timeline)




