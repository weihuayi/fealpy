#import numpy as np 
import matplotlib.pyplot as plt
from fealpy.experimental.iopt import initialize
from ..backend import backend_manager as bm
#bm.set_backend('pytorch')
import math

class QPSO:
    def __init__ (self, N, dim, ub, lb, Max_iter, fobj):
        self.N = N
        self.dim = dim
        self.ub = ub
        self.lb = lb
        self.Max_iter = Max_iter
        self.fobj = fobj  

    #QPSO
    def cal(self):
        C=2
        beta=6
        H = initialize(self.N, self.dim, self.ub, self.lb, self.Max_iter, self.fobj)
        fit, x, gbest, gbest_f = H.initialize() 
        fit = bm.zeros([self.N,1])
        for i in range(0, self.N):
            fit[i,0] = self.fobj(x[i,:])

        gbest_idx = bm.argmin(fit)
        f_prey = fit[gbest_idx,0]
        #x_prey = x[gbest_idx,:].copy()  
        x_prey = bm.copy(x[gbest_idx,:]) 
        CNVG = bm.zeros(self.Max_iter)
        eps=2.2204e-16

        for t in range(0,self.Max_iter):
            alpha = C*math.exp(-t/self.Max_iter)
            for i in range(0,self.N):
                if i == self.N-1:
                    S = bm.linalg.norm(x[i, :] - x[0, :] + eps) ** 2  
                else:
                    S = bm.linalg.norm(x[i, :] - x[i + 1, :] + eps) ** 2  
                di = bm.linalg.norm(x[i,:] - x_prey + eps)**2
                r2 = bm.random.rand(1)  
                I = r2 * S / (4 * bm.pi * di)
                rr = bm.random.rand(1) 
                if rr < 0.5:
                    F = 1
                else:
                    F = -1
                r = bm.random.rand(1) 
                di = x_prey-x[i,:]
                if r < 0.5:
                    r3 = bm.random.rand(1,self.dim) 
                    r4 = bm.random.rand(1,self.dim)
                    r5 = bm.random.rand(1,self.dim)
                    Xnew = x_prey+F*beta*I*x_prey+F*alpha*r3*di*bm.abs(bm.cos(2*bm.pi*r4)*bm.cos(2*bm.pi*r5))
                else:
                    r7 = bm.random.rand(1,self.dim)
                    Xnew = x_prey + F* alpha * r7 * di
                Xnew = bm.clip(Xnew, self.lb, self.ub)
                Xnew = Xnew.reshape(-1)
                fnew = self.fobj(Xnew)
                
                if fnew < fit[i,0]:
                
                    fit[i,0] = fnew
                    #x[i,:] = Xnew.copy()
                    x[i,:] = bm.copy(Xnew)
            gbest_idx = bm.argmin(fit)
            if fit[gbest_idx,0] < f_prey:
                f_prey = fit[gbest_idx,0]
                #x_prey = x[gbest_idx,:].copy()
                x_prey = bm.copy(x[gbest_idx,:])
            CNVG[t] = f_prey
        return x_prey,f_prey,CNVG
        # I = initialize(self.N, self.dim, self.ub, self.lb, self.Max_iter, self.fobj)
        # _, X, _, _ = I.initialize()
        
        # Objective_values = bm.zeros((len(X)))
        # for i in range(0, self.N):
        #     Objective_values[i] = self.fobj(X[i,:])
        
        
        # #个体最优
        # pbest=X.copy()
        # pbest_f=Objective_values.copy()

        # #全局最优
        # gbest_idx = bm.argmin(pbest_f)
        # gbest_f = pbest_f[gbest_idx]
        # gbest = pbest[gbest_idx]   

        # #空列表
        # Convergence_curve = []
        # Convergence_curve.append(gbest_f)
        
        # # 更新迭代
        # for t in range(self.Max_iter):
        #     alpha=1-(t+1)/(2*self.Max_iter)
        #     mbest=sum(pbest)/self.N
        #     for i in range(0, self.N):
        #         phi=bm.random.rand(1, self.dim)
        #         p=phi*pbest[i,:]+(1-phi)* gbest
        #         u=bm.random.rand(1, self.dim)
        #         rand=bm.random.rand()
        #         X[i, :] = p + alpha * bm.abs(mbest - X[i, :]) * bm.log(1. / u) * (1 - 2 * (rand>= 0.5))
        #         X[i,:] = bm.clip(X[i,:], self.lb, self.ub)
        #         Objective_values[i] = self.fobj(X[i,:])
        #         #更新个体最优
        #         (pbest_f[i], pbest[i, :]) = (Objective_values[i], X[i, :]) if Objective_values[i] < pbest_f[i] else (pbest_f[i], pbest[i, :])
        #         # 更新全局最优
        #         (gbest_f, gbest) = (pbest_f[i], pbest[i, :]) if pbest_f[i] < gbest_f else (gbest_f, gbest)
            
        #     Convergence_curve.append(gbest_f)
        #     # print('QPSO: The optimum at iteration', t + 1, 'is', Convergence_curve[t])

        # return gbest, gbest_f, Convergence_curve

if __name__ == "__main__":  
    print('--------QPSO----------------')

    F = 'F2'
    f = Function(F)
    fobj, lb, ub, dim = f.Functions()

    N = 100
    Max_iter  = 1000

    s = QPSO(N, dim, ub, lb, Max_iter, fobj)
    Best_pos, Best_score, Convergence_curve = s.QPSO()

    print(s)
    print('The best solution obtained is:', Best_pos)
    print('The best optimal value of the objective function found by SAO is:', Best_score)


