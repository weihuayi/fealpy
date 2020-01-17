import numpy as np
import matplotlib.pyplot as plt
from fealpy.functionspace import PrismFiniteElementSpace

from fealpy.timeintegratoralg.timeline_new import ChebyshevTimeLine, UniformTimeLine
from fealpy.solver import MatlabSolver
from scipy.sparse.linalg import spsolve

def pscftmodel_options(
        nspecies = 2,
        nblend = 1,
        nblock = 2,
        ndeg = 100,
        fA = 0.2,
        chiAB = 0.25,
        dim = 3,
        T0 = 4,
        T1 = 16,
        nupdate = 1,
        order = 1):
    """
    Get the options used in model.
    """

    # the parameter for scft model
    options = {
            'nspecies'    :nspecies,
            'nblend'      :nblend,
            'nblock'      :nblock,
            'ndeg'        :ndeg,
            'fA'          :fA,
            'chiAB'       :chiAB,
            'dim'         :dim,
            'T0'          :T0,
            'T1'          :T1,
            'nupdate'     :nupdate,
            'order'       :order,
            'integrator'  :integrator
            }

    options['chiN'] = options['chiAB']*options['ndeg']
    return options

class PDEModel():
    def __init__(self, A, M):
        self.A = A
        self.M = M

    def init_solution(self, timeline):
        NL = timeline.number_of_time_levels()
        gdof = self.space.number_of_global_dofs()
        uh = np.zeros((gdof, NL), dtype=self.mesh.ftype)
        uh[:, 0] = 1
        return uh

    def init_delta(self, timeline):
        NL = timeline.number_of_time_levels()
        gdof = self.space.number_of_global_dofs()
        delta = np.zeros((gdof, NL), dtype=self.mesh.ftype)
        return delta

    def get_current_left_matrix(self, dt, F):
        ##　F的加入
        dt = timeline.current_time_step_length()
        return self.M + 0.5*dt*(self.A + F)

    def get_current_right_vector(self, uh, F, timeline):
        dt = timeline.current_time_step_length()
        i = timeline.current
        return self.M@uh[:, i] - 0.5*dt*(self.A@uh[:, i] + F@uh[:,i])


    def solve(self, uh, A, b, solver, timeline):
        i = timeline.current
        uh[:,i+1] = solver(A,b)

    def apply_boundary_condition(self, A, b):
        ##TODO
        return A, b

    def residual_integration(self, uh, F,timeline):
        ##残差的积分项
        A = self.space.stiff_matrix()
        q = -A@uh - F@uh
        return timeline.dct_time_integral(q, return_all=True)

    def error_integration(self, data, timeline):
        ##残差的导数
        uh = data[0]
        intq = data[1]
        M = self.space.mass_matrix()
        r = uh[:, [0]] + spsolve(M, intq) - uh
        return timeline.diff(r)

    def get_error_right_vector(self, data, timeline):
        uh = data[0]
        d = data[2]
        delta = data[-1]
        M = self.space.mass_matrix()
        i = timeline.current
        dt = timeline.current_time_step_length()
        return self.get_current_right_vector(delta, timeline, sdc=True) + dt*M@d[:, i+1]


class PRISMSCFTFEMModel():
    def __init__(self, mesh, options=None):
        if options == None:
            options = pscftmodel_options()
        self.options = options

        self.space = PrismFiniteElementSpace(mesh, p=options['order'])
        self.mesh = self.space.mesh
        self.totalArea = np.sum(self.mesh.cell_volume)
        self.count = 0

        fA = options['fA']
        T0 = options['T0']
        T1 = options['T1']
        self.timeline0 = ChebyshevTimeLine(0, fA, T0)
        self.timeline1 = ChebyshevTimeLine(fA, 1, T1)
        N = T0 + T1 + 1
        gdof = self.space.number_of_global_dofs()
        self.gof = gdof

        self.q0 = np.zeros((gdof, N), dtype=self.mesh.ftype)
        self.q1 = np.zeros((gdof, N), dtype=self.mesh.ftype)

        self.rho = np.zeros((gdof, 2), dtype=self.mesh.ftype)
        self.grad = np.zeros((gdof, 2), dtype=self.mesh.ftype)
        self.w = np.zeros((gdof, 2), dtype=self.mesh.ftype)
        self.sQ1 = np.zeros((N, 1), dtype=self.mesh.ftype)

        self.sQ = 0.0
        nupdate = options['nupdate']
        self.A = self.space.stiff_matrix()
        self.M = self.space.mass_matrix()
        self.dmodel = PDEModel(self.A, self.M)

        self.solver = MatlabSolver()

    def reinit(self, mesh):
        options = self.options
        self.space = PrismFiniteElementSpace(mesh, p=options['order'])
        self.mesh = self.space.mesh
        self.totalArea = np.sum(self.mesh.cell_volume)
        self.count = 0

        fA = options['fA']
        T0 = options['T0']
        T1 = options['T1']
        self.timeline0 = ChebyshevTimeLine(0, fA, T0)
        self.timeline1 = ChebyshevTimeLine(fA, 1, T1)
        N = T0 + T1 + 1
        gdof = self.space.number_of_global_dofs()
        self.gof = gdof

        self.q0 = np.zeros((gdof, N), dtype=self.mesh.ftype)
        self.q1 = np.zeros((gdof, N), dtype=self.mesh.ftype)

        self.rho = np.zeros((gdof, 2), dtype=self.mesh.ftype)
        self.grad = np.zeros((gdof, 2), dtype=self.mesh.ftype)
        self.w = np.zeros((gdof, 2), dtype=self.mesh.ftype)
        self.sQ1 = np.zeros((N, 1), dtype=self.mesh.ftype)

        self.sQ = 0.0
        nupdate = options['nupdate']
        self.A = self.space.stiff_matrix()
        self.M = self.space.mass_matrix()
        self.dmodel = PDEModel(self.A, self.M)

        self.solver = MatlabSolver()

   def init_value(self, fieldstype = 1):
        gdof = self.space.number_of_global_dofs()
        mesh = self.space.mesh
        node = mesh.node
        chiN = self.options['chiN']
        fields = np.zeros((gdof, 2), dtype = mesh.ftype)
        mu = np.zeros((gdof, 2), dtype = mesh.ftype)
        w = np.zeros((gdof, 2), dtype = mesh.ftype)

        if fieldstype == 1:
            fields[:, 0] = chiN*(-1 + 2*np.random.rand(1, gdof))
            fields[:, 1] = chiN*(-1 + 2*np.random.rand(1, gdof))

        w[:, 0] = fields[:, 0] - fields[:, 1]
        w[:, 1] = fields[:, 0] + fields[:, 1]

        mu[:, 0] = 0.5*(w[:, 0] + w[:, 1])
        mu[:, 1] = 0.5*(w[:, 1] - w[:, 0])

        return mu

    def __call__(self, mu, returngrad=True):
        """
        目标函数，给定外场，计算哈密尔顿量及其梯度
        """
        self.w[:, 0] = mu[:, 0] - mu[:, 1]
        self.w[:, 1] = mu[:, 0] + mu[:, 1]

        start = timer()
        self.compute_propagator()
        print('Times for PDE solving:', timer() - start)

        self.compute_singleQ()
        self.compute_density_new()


        u0 = self.space.function(array = mu[:, 0]).value
        mu1_int = self.integral_space(u0)

        u1 = self.space.function(array = mu[:, 1]).value
        u = lambda x : u1(x) **2

        mu2_int = self.integral_space(u)

        chiN = self.options['chiN']
        self.H = -mu1_int + mu2_int/chiN
        self.H = self.H/self.totalArea - np.log(self.sQ)

        self.save_data(fname='./data/test'+str(self.count)+'.mat')
        self.count +=1
        self.grad[:, 0] = self.rho[:, 0]  + self.rho[:, 1] - 1.0
        self.grad[:, 1] = 2.0*mu[:, 1]/chiN - self.rho[:, 0] + self.rho[:, 1]

        if returngrad:
            return self.H, self.grad
        else:
            return self.H

    def compute_propagator(self):
        ###TODO
        n0 = self.timeline0.NT
        n1 = self.timeline1.NT

#        w = self.space.function(array = self.w[:, 0]).value
#        F0 = self.space.mass_matrix(cfun=w)
#
#        w = self.space.function(array = self.w[:, 1]).value
#        F1 = self.space.mass_matrix(cfun=w)
        self.q0[:,0:n0] = self.dmodel.init_solution(self.timeline0)
        self.timeline0.time_integration(self.q0[:,0:n0], self.dmodel, self.solver.divide,
                self.options['nupdate'])
        self.t imeline1.time_integration(self.q0[:,n0-1:], self.dmodel, self.solver.divide,
                self.options['nupdate'])

        self.q1[:,0:n1] = self.dmodel.init_solution(self.timeline1)
        self.t imeline1.time_integration(self.q1[:,0:n1], self.dmodel, self.solver.divide,
                self.options['nupdate'])
        self.timeline0.time_integration(self.q1[:,n1-1:], self.dmodel, self.solver.divide,
                self.options['nupdate'])

    def compute_singleQ(self):
        q = self.q0*self.q1[:, -1::-1]
        self.sQ = self.integral_space(self.q0[:, -1])/self.totalArea

    def compute_density(self):
        q = self.q0*self.q1[:, -1::-1]
        n0 = self.timeline0.NL
        self.rho[:, 0] = self.integral_time(q[:, 0:n0], self.timeline0.dt)/self.sQ
        self.rho[:, 1] = self.integral_time(q[:, n0-1:], self.timeline1.dt)/self.sQ

    def compute_density_new(self):
        q = self.q0*self.q1[:, -1::-1]
        n0 = self.timeline0.NL
        self.rho[:, 0] = self.timeline0.dct_time_integral(q[:, 0:n0],
                return_all = False)/self.sQ
        self.rho[:, 1] = self.timeline1.new_time_integral(q[:,
            n0-1:], return_all = False)/self.sQ

    def integral_time(self, q, dt):
        f = -0.625*(q[:, 0] + q[:, -1]) + 1/6*(q[:, 1] + q[:, -2]) - 1/24*(q[:, 2] + q[:, -3])
        f += np.sum(q, axis=1)
        f *= dt
        return f

    def integral_space(self, u):
        Q = self.space.integralalg.integral(u)
        return Q

    def output(self, tag, queue=None, stop=False):
        if queue is not None:
            if not stop:
                queue.put({tag:self.rho[:, 0]})
            else:
                queue.put(-1)


    def save_data(self, fname='test.mat'):
        import scipy.io as sio

        mesh = self.mesh
        node = mesh.node
        cell = mesh.ds.cell
        cellLocation = mesh.ds.cellLocation
        Q = self.sQ1
        H = self.H
        q = self.q0
        q1 = self.q1

        mu = self.w.copy()
        mu[:, 0] = 0.5*(self.w[:, 0] + self.w[:, 1])
        mu[:, 1] = 0.5*(self.w[:, 1] - self.w[:, 0])

        eta = self.eta

        data = {
                'node':node,
                'cell':cell,
                'cellLocation':cellLocation,
                'rho':self.rho,
                'Q':Q,
                'H':H,
                'mu':mu,
                'q0':q,
                'q1':q1,
               }
        sio.savemat(fname, data)
