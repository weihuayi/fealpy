from fealpy.backend import backend_manager as bm

class UniformTimeLine():
    def __init__(self, T0, T1, NT, options={'Output':False}):
        """
        Parameter
        ---------
        T0: the initial time
        T1: the end time
        NT: the number of time segments
        """
        self.T0 = T0
        self.T1 = T1
        self.NL = NT + 1 # the number of time levels
        self.dt = (self.T1 - self.T0)/NT
        self.current = int(0)

    def add_time(self, n):
        """

        Notes
        -----
        增加计算时间， n 表示增加的步数
        """

        self.T1 = self.T1 + n*self.dt
        self.NL += n

    def uniform_refine(self, n=1):
        for i in range(n):
            self.NL = 2*(self.NL - 1) + 1
            self.dt = (self.T1 - self.T0)/(self.NL - 1)
        self.current = int(0)

    def number_of_time_levels(self):
        return self.NL

    def all_time_levels(self):
        return bm.linspace(self.T0, self.T1, num=self.NL)

    def current_time_level_index(self):
        return self.current

    def current_time_level(self):
        return self.T0 + self.current*self.dt

    def next_time_level(self):
        return self.T0 + (self.current + 1)*self.dt

    def prev_time_level(self):
        return self.T0 + (self.current - 1)*self.dt

    def current_time_step_length(self):
        return self.dt

    def stop(self, order='forward'):
        if order == 'forward':
            return self.current >= self.NL - 1
        elif order == 'backward':
            return self.current <= 0

    def advance(self):
        self.current += 1

    def forward(self):
        self.current += 1

    def backward(self):
        self.current -= 1

    def reset(self, order='forward'):
        if order == 'forward':
            self.current = 0 
        elif order == 'backward':
            self.current = self.NL - 1

