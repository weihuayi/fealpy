import numpy as np

class initialize_2:
    def __init__ (self, N, dim, ub, lb, MaxIT, fobj, init_x):
        self.N = N
        self.dim = dim
        self.ub = ub
        self.lb = lb
        self.MaxIT = MaxIT
        self.fobj = fobj 
        self.init_x = init_x
    
    def initialize_2(self):
        #种群
        if isinstance(self.ub, (float, int)):           
            X = np.random.rand(self.N, self.dim) * (self.ub - self.lb) + self.lb
        else:   
            X = np.zeros((self.N - 1, self.dim))  
            for i in range(self.dim):  
                ub_i = self.ub[i]  
                lb_i = self.lb[i]  
                X[:, i] = np.random.rand(self.N - 1) * (ub_i - lb_i) + lb_i  
            X = np.concatenate((X, self.init_x.reshape(1, self.dim)), axis=0) 
          
        # 计算所有个体的适应度  
        fit = np.apply_along_axis(self.fobj, 1, X)  # 假设fobj接受一维数组  
       

        #全局最优
        gbest_idx = np.argmin(fit)
        gbest_f = fit[gbest_idx]
        gbest = X[gbest_idx]       
        return fit, X, gbest, gbest_f
