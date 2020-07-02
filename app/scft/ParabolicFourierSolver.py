import numpy as np

class ParabolicFourierSolver():
    def __init__(self, space, timeline, plan=None):
        self.space = space 
        self.plan = plan
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
            
    def operator_split_2(self, q, w, dt):
        """

        Parameters
        ----------

        References
        ----------

        Notes
        -----

        """
        space = self.space
        self.w = w
        self.E1 = np.exp(-dt*self.k2)
        self.E0 = np.exp(-dt/2*w)

        E0 = self.E0
        E1 = self.E1

        q1 = space.ifftn(E0*q)
        q1 *= E1
        q = space.fftn(q1).real
        q *= E0
        return q
    

    def solve(self, q, w, method='BDF4'):
        if method == 'BDF4':
            self.BDF4(q, w)
    
    # BDF4 sover 模块
    def BDF4(self, q, w):
        space = self.space
        NL = self.timeline.number_of_time_levels()
        dt = self.timeline.current_time_step_length()
        k2 = self.k2
        for i in range(1,4):
            q0 = q[i-1]
            q1 = self.operator_split_2(q0, w, dt)
            qhalf = self.operator_split_2(q0, w, 0.5*dt)
            qhalf = self.operator_split_2(qhalf, w, 0.5*dt)
            q[i] = -1/3*q1 + 4/3*qhalf

        for i in range(4, NL):
            q0 = 4*q[i-1] - 3*q[i-2] + 4*q[i-3]/3 - q[i-4]/4
            q1 = 4*q[i-1] - 6*q[i-2] + 4*q[i-3] - q[i-4]
            q1 *= w
            q1 *= dt
            q0 -= q1

            q1 = space.ifftn(q0)
            q1 /= 25/12 + dt*k2
            q[i] = space.fftn(q1).real
