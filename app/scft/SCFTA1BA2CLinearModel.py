
import numpy as np
from fealpy.functionspace import FourierSpace
from fealpy.timeintegratoralg.timeline_new import UniformTimeLine
from ParabolicFourierSolver import ParabolicFourieeSolver

def model_options(
        nspecies = 3,
        nblend = 1,
        nblock = 4,
        ndeg = 100,
        fA1 = 0.25,
        fA2 = 0.25,
        fB = 0.25,
        fC = 0.25,
        maxdt = 0.01,
        chiAB = 0.25,
        chiAC = 0.25,
        chiBC = 0.25,
        dim = 2,
        NS = 16,
        maxdt = 0.01,
        bA = 1,
        bB = 1,
        bC = 1):
        # the parameter for scft model
        options = {
                'nspecies': nspecies,
                'nblend': nblend,
                'nblock': nblock,
                'ndeg': ndeg,
                'fA1': fA1,
                'fA2': fA2,
                'fB': fB,
                'fC': fC,
                'maxit': maxdt,
                'chiAB': chiAB,
                'chiAC': chiAC,
                'chiBC': chiBC,
                'dim': dim,
                'NS' : NS,
                'maxdt' : = maxdt,
                'bA': bA,
                'bB': bB,
                'bC': bC
                }
        return options


class SCFTA1BA2CLinearModel():
    def __init__(self, options=None):
        if options == None:
            options = pscftmodel_options()
        self.options = options
        dim = options['dim']
        box = np.diag(dim*[2*np.pi])
        self.space = FourierSpace(box,  options['NS'])

        fA1 = options['fA1']
        fB  = options['fB']
        fA2 = options['fA2']
        fC  = options['fC']
        maxdt = options['maxdt']

        self.timelines = []
        self.timelines.append(UniformTimeLine(0, fA1, int(fA1//maxdt)))
        self.timelines.append(UniformTimeLine(0, fB, int(fB//maxdt)))
        self.timelines.append(UniformTimeLine(0, fA2, int(fA2//maxdt)))
        self.timelines.append(UniformTimeLine(0, fC, int(fC//maxdt)))

        self.pdesolvers = []
        for i in range(4):
            self.pdesolvers.append(
                    ParabolicFourierSolver(self.space, self.timelines[i])
                    )

        TNL = 0 # total number of time levels
        for i in range(4):
            TNL += self.timelines[i].number_of_time_levels()
        TNL -= options['nblock'] - 1

        self.qf = self.space.function(dim=TNL) # forward  propagator 
        self.qb = self.space.function(dim=TNL) # backward propagator

        self.qf[0] = 1
        self.qb[0] = 1

        self.rho = self.space.function(dim=options['nspecies'])
        self.grad = self.space.function(dim=options['nspecies']+1)
        self.Q = np.zeros(options['nblend'], dtype=np.float)
        self.w = self.space.function(dim=options['nspecies']+1)

    def __call__(self, w, returngrad=True):
        """
        目标函数，给定外场，计算哈密尔顿量及其梯度
        """

        # solver the forward and backward equation
        self.compute_propagator(w)
        # compute single chain partition function Q
        self.compute_single_Q()
        # compute density
        self.compute_density()
        # compute energy function and its gradient
        self.compute_energe()


    def compute_energe(self, w):
        chiABN = options['chiAB']*options['ndeg']
        chiBCN = options['chiBC']*options['ndeg']
        chiACN = options['chiAC']*options['ndeg']
        rho = self.rho

        E = chiABN*rho[0]*rho[1] 
        E += chiBCN*rho[1]*rho[2]
        E += chiACN*rho[0]*rho[2]
        E -= w[1]*rho[0]
        E -= w[2]*rho[1]
        E -= w[3]*rho[2]
        E -= w[0]*(1 - rho.sum(axis=0))
        E = np.fft.fftn(E)
        self.H = np.real(E.flat[0])
        self.H -= np.log(self.Q[0])


    def compute_propagator(self, w):

        qf = self.qf
        qb = self.qb

        start = 0
        F = [w[1], w[2], w[1], w[3]]
        for i in range(options['nblock']):
            NL = self.timelines[i].number_of_time_levels()
            self.pdesolvers[i].initialize(self.qf[start:start + NL], F[i])
            start += NL - 1

        start = 0
        F = [w[3], w[1], w[2], w[1]]
        for i in range(options['nblock']):
            NL = self.timelines[i].number_of_time_levels()
            self.pdesolvers[i].initialize(self.qb[start:start + NL], F[i])
            start += NL - 1


    def compute_single_Q(self):
        q = self.qf[-1]
        q = np.fft.fftn(q)
        self.Q[0] = np.real(q.flat[0])


    def compute_density(self):
        q = self.qf*self.qb[-1::-1]

        start = 0
        rho = []
        for i in range(options['nblock']):
            NL = self.timelines[i].number_of_time_levels()
            dt = self.timelines[i].current_time_step_length()
            rho.append(self.integral_time(q[start:start+NL], dt))
            start += NL - 1

        self.rho[0] = rho[0] + rho[2]
        self.rho[1] = rho[1]
        self.rho[2] = rho[3]
        self.rho /= self.Q[0]

    def integral_time(self, q, dt):
        f = -0.625*(q[0] + q[-1]) + 1/6*(q[1] + q[-2]) - 1/24*(q[2] + q[-3])
        f += np.sum(q, axis=0)
        f *= dt
        return f
