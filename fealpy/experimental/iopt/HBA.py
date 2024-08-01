import time 
import random
import math
import matplotlib.pyplot as plt  
from fealpy.experimental.iopt import initialize
from fealpy.experimental.backend import backend_manager as bm

#bm.set_backend('pytorch')
#bm.set_backend('jax')

class HBA():
    def __init__(self,N, dim, UB, LB, T, fobj):
        self.N = N
        self.T = T
        self.dim = dim
        self.UB = UB
        self.LB = LB
        self.fobj = fobj


    def cal(self):
        C=2
        beta=6
        H = initialize(self.N, self.dim, self.UB, self.LB, self.T, self.fobj)
        fit, x, gbest, gbest_f = H.initialize() 
        fit = bm.zeros([self.N,1])
        for i in range(0, self.N):
            fit[i,0] = self.fobj(x[i,:])

        gbest_idx = bm.argmin(fit)
        f_prey = fit[gbest_idx,0]
        #x_prey = x[gbest_idx,:].copy()  
        x_prey = bm.copy(x[gbest_idx,:]) 
        CNVG = bm.zeros(self.T)
        eps=2.2204e-16

        for t in range(0,self.T):
            alpha = C*math.exp(-t/self.T)
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
                Xnew = bm.clip(Xnew, self.LB, self.UB)
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
    

# F_name = 'F24'
# f = Function(F_name)
# fobj, LB, UB, dim = f.Functions()
# N = 100
# T = 1000
# H = HBA(N, T, dim, UB, LB, fobj)


# xmin, fmin, CNVG = H.HBA1() 
# # print(CNVG)
# # print('循环用的时间:', time1)  
# # 绘制收敛曲线  
# plt.figure()  
# plt.semilogy(CNVG, 'r')  
# plt.xlim([0, T])  
# plt.title('Convergence curve')  
# plt.xlabel('Iteration')  
# plt.ylabel('Best fitness obtained so far')  
# plt.legend(['HBA'])  
# plt.show()  
  
# # 显示最佳位置和适应度得分  
# print(f"The best location= {xmin}")  
# print(f"The best fitness score = {fmin}")
