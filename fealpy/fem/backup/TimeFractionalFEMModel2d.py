import numpy as np

from ..functionspace.lagrange_fem_space import LagrangeFiniteElementSpace
from ..boundarycondition import DirichletBC
from scipy.sparse.linalg import spsolve
from scipy.special import gamma, beta

class TimeFractionalFEMModel2d():

    def __init__(self, pde, mesh, p=1, q=3):
        self.space = LagrangeFiniteElementSpace(mesh, p, q=q)
        self.mesh = self.space.mesh
        self.pde = pde

    def solve(self, t0, t1, NT):
        """
        Parameters
        ----------
        t0 : the start time
        t1 : the stop time
        NT : the number of segments on [t0, t1]
        """
        A = self.space.stiff_matrix()
        M = self.space.mass_matrix()
        timeline = self.pde.time_mesh(t0, t1, NT)
        dt = timeline.get_current_time_step_length()

    def fast_solve(self, t0, t1, NT):
        """
        Parameters
        ----------
        t0 : the start time
        t1 : the stop time
        NT : the number of time segments on [t0, t1]
        """
        NL = NT + 1 # the number of time levels
        timeline = self.pde.time_mesh(t0, t1, NT)
        dt = timeline.get_current_time_step_length()

        space = self.space
        gdof = space.number_of_global_dofs()

        A = self.space.stiff_matrix()
        M = self.space.mass_matrix()

        self.uI = self.space.function(dim=NL)
        self.uI[:, 0] = self.space.interpolation(self.pde.init_value)


    def sum_of_exp_approximation(self, b, dt, t1, eps=1e-10):
        """
        Parameters
        ----------
        b:
        dt : the time step length
        t1 : the stop time level
        reps :
        """
        pi = np.pi
        d = dt/t1
        h = 2*pi/(np.log(3) + b*np.log(1/np.cos(1)) + np.log(1/eps))
        tlower = np.log(eps*gamma(1 + b))
        tupper = np.log(1/d) + np.log(np.log(1/eps)) + np.log(b) + 1/2

        M = np.floor(tlower/h) - 30;
        N = np.ceil(tupper/h)
        n1 = np.arange(M, 0)

        xs1 = -np.exp(h*n1)
        ws1 = h/gamma(b)*np.exp(b*h*n1)
        xs1new, ws1new = prony(xs1, ws1)

        n2 = np.arange(0, N+1)
        xs2 = -np.exp(h*n2)
        ws2 = h/gamma(b)*np.exp(b*h*n2)

        wn2, pn2, bound = squareroot(xs2, ws2, eps)
        pass

    def prony(self, xs, ws):
        pass

    def square_root(xs, ws, eps):
        pass

    def error(self):
        pass
