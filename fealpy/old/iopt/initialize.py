import numpy as np

class initialize:
    def __init__ (self, N, dim, ub, lb, MaxIT, fobj):
        self.N = N
        self.dim = dim
        self.ub = ub
        self.lb = lb
        self.MaxIT = MaxIT
        self.fobj = fobj


    
    def initialize(self):
        #种群
        if isinstance(self.ub, int) :
            X = np.random.rand(self.N, self.dim) * (self.ub - self.lb) + self.lb
            fit=np.zeros((self.N,1))
            for i in range(0, self.N):
                fit[i,0]=self.fobj(X[i,:].reshape((1, self.dim)))
        else:
            
          
            X = np.zeros((self.N, self.dim))
            for i in range(self.dim):
                ub_i = self.ub[i]
                lb_i = self.lb[i]
                x_i = np.random.rand(self.N, 1) * (ub_i-lb_i) + lb_i
                x_i = x_i.flatten()
                for j in range(self.N):
                    X[j][i] = x_i[j]

                fit=np.zeros((self.N,1))
                for i in range(0, self.N):
                    fit[i,0]=self.fobj(X[i,:].reshape((1, self.dim)))
                    
                
        best = np.zeros(self.MaxIT)
        gbest = np.zeros((1,self.dim))
        gbest_f = 0

        #全局最优
        gbest_idx = np.argmin(fit)
        gbest_f = fit[gbest_idx]
        gbest = X[gbest_idx]       
        return fit, X, best, gbest, gbest_f