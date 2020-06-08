import numpy as np

class ParabolicFourierSolver():
    def __init__(self, space, timeline, plan):
        self.space = space 
        self.plan = plan
        self.k, self.k2 = self.space.reciprocal_lattice(return_square=True)
        self.timeline = timeline 
        dt = self.timeline.current_time_step_length()
        self.E1 = np.exp(-dt*self.k2)
        self.E3 = np.exp(-dt/2*self.k2)

        print("E1", self.E1.dtype)
        print("E3", self.E3.dtype)

    def initialize(self, q, w):
        """
        Parameters
        ----------

        q : 
        w :

        Note
        ----
        """

        space = self.space
        self.w = w
        dt = self.timeline.current_time_step_length()
        self.E0 = np.exp(-dt/2*w)
        self.E2 = np.exp(-dt/4*w)

        NL = self.timeline.number_of_time_levels()

        E0 = self.E0
        E1 = self.E1

        E2 = self.E2
        E3 = self.E3 

        for i in range(1, 4):
            q0 = q[i-1]
            q1 = space.ifftn(E0*q0)
            q1 *= E1
            q[i] = space.fftn(q1).real
            q[i] *= E0

        for i in range(1, 4):
            q0 = q[i-1]
            q1 = space.ifftn(E2*q0)
            q1 *= E3
            q1 = space.fftn(q1).real
            q1 *= E2

            q1 = space.ifftn(E2*q1)
            q1 *= E3
            q1 = space.fftn(q1).real
            q1 *= E2
            q[i] *= -1/3
            q[i] += 4*q1/3

    def solve(self, q): 
        space = self.space
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
            q1 = space.ifftn(q0)
            q1 /= 25/12 + dt*k2
            q[i] = space.fftn(q1).real


if __name__ == "__main__":
    from fealpy.functionspace import FourierSpace
    from fealpy.timeintegratoralg.timeline_new import UniformTimeLine
    import pyfftw

    dim = 2
    NS = 16
    a = pyfftw.empty_aligned((NS, NS), dtype='complex128')
    b = pyfftw.empty_aligned((NS, NS), dtype='complex128')
    axes = tuple(range(2))
    plan = pyfftw.FFTW(a, b, axes=axes)

    box = np.diag(2*[6*np.pi])
    timeline = UniformTimeLine(0, 0.3, 0.01)
    space = FourierSpace(box, 16)
    solver = ParabolicFourierSolver(space, timeline, plan)
    NL = timelines.number_of_time_levels()
    q = space.function(dim=NL) 
    w = 0
    solver.initialize(q, w)
    solver.solve(q)

