
import numpy as np
import matplotlib.pyplot as plt

from fealpy.functionspace import FourierSpace
from fealpy.timeintegratoralg.timeline_new import UniformTimeLine

class ParabolicFourierSolver():
    def __init__(self, space, timeline):
        self.space = space 
        self.k, self.k2 = self.space.reciprocal_lattice(return_square=True)
        self.timeline = timeline 
        dt = self.timeline.current_time_step_length()
        self.E1 = np.exp(-dt*self.k2)
        self.E3 = np.exp(-dt/2*self.k2)

    def initialize(self, q, w):
        """
        Parameters
        ----------

        q : 
        w :

        Note
        ----
        """

        self.E0 = np.exp(-dt/2*w)
        self.E2 = np.exp(-dt/4*w)

        NL = self.timeline.number_of_time_levels()

        E0 = self.E0
        E1 = self.E1

        E2 = self.E2
        E3 = self.E3 

        for i in range(1, 4):
            q0 = q[i-1]
            q1 = np.fft.fftn(E0*q0)
            q1 *= E1
            q[i] = np.fft.ifftn(q1).real
            q[i] *= E0

        q0 = q[0]
        for i in range(1, 4):
            q1 = np.fft.fftn(E2*q0)
            q1 *= E3
            q1 = np.fft.ifftn(q1).real
            q1 *= E2

            q1 = np.fft.fftn(E2*q1)
            q1 *= E3
            q1 = np.fft.ifftn(q1).real
            q1 *= E2
            q[i] *= -1/3
            q[i] += 4*q1/3

    def solve(self, q): 
        NL = self.timeline.number_of_time_levels()
        dt = self.timeline.current_time_step_length()
        E0 = self.E0
        E1 = self.E1
        w = self.w
        k2 = self.k2

        for i in range(4, NL):
            q0 = 4*q[i-1] - 3*q[i-2] + 4*q[i-3]/3 - q[i-4]/4
            q1 = 4*q[i-1] - 6*q[i-2] + 4*q[i-3] - q[i-4]
            q1 *= w
            q1 *= dt
            q0 -= q1
            q1 = np.fft.fftn(q0)
            q1 /= 25/12 + dt*k2
            q[i] = np.fft.ifftn(q1).real

init_value = {
        "bcc":
        np.array([
            [ 1,	 1,	 0,     0.3],
            [-1,	 1,	 0,	0.3],
            [-1,	-1,	 0,	0.3],
            [ 1,	-1,	 0,	0.3],
            [ 0,	 1,	 1,	0.3],
            [ 0,	-1,	 1,	0.3],
            [ 0,	-1,	-1,	0.3],
            [ 0,	 1,	-1,	0.3],
            [ 1,	 0,	 1,	0.3],
            [-1,	 0,	 1,	0.3],
            [-1,	 0,	-1,	0.3],
            [ 1,	 0,	-1,	0.3]], dtype=np.float),
        "cam6fold":
        np.array([
               [ 30,     0,	0.3],
               [ 15,    26,	0.3],
               [-15,    26,	0.3],
               [-30,     0,	0.3],
               [-15,   -26,	0.3],
               [ 15,   -26,	0.3]], dtype=np.float)
        }

def model_options(
        nspecies = 3,
        nblend = 1,
        nblock = 4,
        ndeg = 100,
        fA1 = 0.25,
        fA2 = 0.25,
        fB = 0.25,
        fC = 0.25,
        chiAB = 0.25,
        chiAC = 0.25,
        chiBC = 0.25,
        box = np.diag(2*[2*np.pi]),
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
                'chiAB': chiAB,
                'chiAC': chiAC,
                'chiBC': chiBC,
                'box': box,
                'dim': len(box),
                'NS' : NS,
                'maxdt': maxdt,
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
        self.w = self.space.function(dim=options['nspecies']+1)
        self.Q = np.zeros(options['nblend'], dtype=np.float)

    def init_field(self):
        pass

    def __call__(self, w):
        """
        目标函数，给定外场，计算哈密尔顿量及其梯度
        """
        self.compute_wplus(w)
        # solver the forward and backward equation
        self.compute_propagator(w)
        # compute single chain partition function Q
        self.compute_single_Q()
        # compute density
        self.compute_density()
        # compute energy function and its gradient
        self.compute_energe()

    def update_field(self, w):
        alpha = 0.01
        chiABN = options['chiAB']*options['ndeg']
        chiBCN = options['chiBC']*options['ndeg']
        chiACN = options['chiAC']*options['ndeg']
        
        w[1] *= 1+alpha
        w[1] -= alpha*chiABN*self.rho[1]
        w[1] -= alpha*chiACN*self.rho[2]
        w[1] -= alpha*w[0]

        w[2] *= 1+alpha
        w[2] -= alpha*chiABN*self.rho[0]
        w[2] -= alpha*chiBCN*self.rho[2]
        w[2] -= alpha*w[0]

        w[3] *= 1+alpha
        w[3] -= alpha*chiACN*self.rho[0]
        w[3] -= alpha*chiBCN*self.rho[1]
        w[3] -= alpha*w[0]
        return w
        

    def compute_wplus(self, w):
        chiAB = options['chiAB']
        chiBC = options['chiBC']
        chiAC = options['chiAC']

        XA = chiBC*(chiAB + chiAC - chiBC)
        XB = chiAC*(chiBC + chiAB - chiAC)
        XC = chiAB*(chiAC + chiBC - chiAB)

        w[0] = XA*w[1] + XB*w[2] + XC*w[3]
        w[0]/= XA + XB + XC


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

    def compute_gradient(self, w):
        chiABN = options['chiAB']*options['ndeg']
        chiBCN = options['chiBC']*options['ndeg']
        chiACN = options['chiAC']*options['ndeg']
        rho = self.rho
        self.grad[0] = rho[0] + rho[1] + rho[2] - 1
        self.grad[1] = w[1] - chiABN*rho[1] - chiACN*rho[2] - w[0]
        self.grad[2] = w[2] - chiABN*rho[0] - chiBCN*rho[2] - w[0]
        self.grad[3] = w[3] - chiACN*rho[0] - chiBCN*rho[1] - w[0]

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


if __name__ == "__main__":

    rho = init_value['cam6fold']
    h = 30 
    box = np.array([[h, 0], [0, h]], dtype=np.float)
    options = model_options(box=box, NS=256)
    model = SCFTA1BA2CLinearModel(options=options)
    rho = model.space.fourier_interpolation(rho)

    fig = plt.figure()
    axes = fig.gca()
    axes.imshow(rho)
    plt.show()

