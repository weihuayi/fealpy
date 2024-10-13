from fealpy.backend import backend_manager as bm

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
        if isinstance(self.ub, (float, int)):           
            X = bm.random.rand(self.N, self.dim) * (self.ub - self.lb) + self.lb
        else:   
            X = bm.zeros((self.N, self.dim))  
            for i in range(self.dim):  
                ub_i = self.ub[i]  
                lb_i = self.lb[i]  
                X[:, i] = bm.random.rand(self.N) * (ub_i - lb_i) + lb_i  

        # 计算所有个体的适应度  
        fit = bm.apply_along_axis(self.fobj, 1, X)  # 假设fobj接受一维数组  

        #全局最优
        gbest_idx = bm.argmin(fit)
        gbest_f = fit[gbest_idx]
        gbest = X[gbest_idx]       
        return fit, X, gbest, gbest_f
