
import numpy as np


class ParabolicFourierModel():
    def __init__(self, space, timeline, q0, w):
        self.space = space 
        self.k, self.k2 = self.space.reciprocal_lattice(return_square=True)

        self.timeline = timeline 

        dt = self.timeline.current_time_step_length()
        self.q0 = q0 # initial value
        self.w = w # field

        # 
        self.E0 = np.exp(-dt/2*w)
        self.E1 = np.exp(-dt*self.k2)
        # half time step size   
        self.E2 = np.exp(-dt/4*w)
        self.E3 = np.exp(-dt/2*self.k2)

    def init_solution(self):
        NL = self.timeline.number_of_time_levels()

        E0 = self.E0
        E1 = self.E1

        E2 = self.E2
        E3 = self.E3 

        q = self.space.function(dim=NL)
        q[0] = self.q0 

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
        return q

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
            



if __name__ == "__main__":

    box = np.array([
        [2*np.pi, 0],
        [0, 2*np.pi]])

    T = [0, 1]
    NS = 4
    NT = 10 
    w = 0
    q0 = 1

    model = ParabolicFourierModel(box, T, NS, NT, q0, w)
    q = model.init_solution()
    model.solve(q)
    print(q)


