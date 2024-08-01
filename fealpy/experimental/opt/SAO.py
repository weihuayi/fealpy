import matplotlib.pyplot as plt
from fealpy.experimental.iopt import initialize
from ..backend import backend_manager as bm
#bm.set_backend('pytorch')
import math


class SAO:
    def __init__ (self, N, dim, ub, lb, Max_iter, fobj):

        self.N = N
        self.dim = dim
        self.ub = ub
        self.lb = lb
        self.Max_iter = Max_iter
        self.fobj = fobj 

    #SAO
    def cal(self):

        I = initialize(self.N, self.dim, self.ub, self.lb, self.Max_iter, self.fobj)
        fit, X, gbest, gbest_f = I.initialize()  

        Objective_values = bm.zeros(len(X))
        for i in range(len(X)):
            Objective_values[i] = self.fobj(X[i, :])

        Convergence_curve = []
        
        N1 = int(math.floor(self.N * 0.5))
        Elite_pool = []

        #升序 的索引数组
        idx1 = bm.argsort(Objective_values)
        
        #存储迭代优解原数组、最优解
        Best_pos = bm.copy(X[idx1[0], :])
        Best_score = bm.copy(Objective_values[idx1[0]])

        second_best = bm.copy(X[idx1[1], :])
        third_best = bm.copy(X[idx1[2], :])

        sum1 = bm.sum(X[idx1[:N1], :], axis=0)#精英池原数组均值
        half_best_mean = sum1 / N1

        Elite_pool.append(Best_pos)
        Elite_pool.append(second_best)
        Elite_pool.append(third_best)
        Elite_pool.append(half_best_mean)

        Convergence_curve.append(Best_score)
        
        #分割
        index = bm.arange(self.N)
        Na = int(self.N / 2)
        Nb = int(self.N / 2)

        # 更新迭代
        for t in range(self.Max_iter):
            RB = bm.random.randn(self.N, self.dim)
            T = math.exp(-t / self.Max_iter)
            # eq.(9)
            DDF = 0.35 + 0.25 * (math.exp(t / self.Max_iter) - 1) / (math.exp(1) - 1)
            # eq.(10)
            #融雪速率
            M = DDF * T

            #种群质心位置
            X_centroid = bm.mean(X, axis=0)

            # 随机分配索引号
            index1 = bm.random.choice(self.N, Na, replace=False)
            index2 = bm.setdiff1d(index, index1)

            #位置矩阵更新
            for i in range(Na):

                r1 = bm.random.rand(1)#控制个体位置的更新
                k1 = bm.random.randint(4)#选择精英池中的一个个体。
                X[index1[i], :] = Elite_pool[k1] + RB[index1[i], :] * (r1 * (Best_pos - X[index1[i], :]) + (1 - r1) * (X_centroid - X[index1[i], :]))

            if Na < self.N:
                Na += 1
                Nb -= 1

            if Nb >= 1:
                for i in range(Nb):
                    r2 = 2 * bm.random.rand(1) - 1
                    X[index2[i], :] = M * Best_pos + RB[index2[i], :] * (
                            r2 * (Best_pos - X[index2[i], :]) + (1 - r2) * (X_centroid - X[index2[i], :]))

            #检查是否超出搜索空间
            for i in range(len(X)):
                X[i, :] = X[i, :] + (self.lb - X[i, :]) * (X[i, :] < self.lb) + (self.ub - X[i, :]) * (X[i, :] > self.ub)
                #更新函数值
                Objective_values[i] = self.fobj(X[i, :])

                # 更新最优函数值以及原数组
                if Objective_values[i] < Best_score:
                    Best_pos = bm.copy(X[i, :])
                    Best_score = Objective_values[i]

            # 更新精英池
            idx1 = bm.argsort(Objective_values)
            second_best = bm.copy(X[idx1[1], :])
            third_best = bm.copy(X[idx1[2], :])
            sum1 = bm.sum(X[idx1[:N1], :], axis=0)
            half_best_mean = sum1 / N1
            Elite_pool[0] = Best_pos
            Elite_pool[1] = second_best
            Elite_pool[2] = third_best
            Elite_pool[3] = half_best_mean
            Convergence_curve.append(Best_score)
            #print('The optimum at iteration', t + 1, 'is', Convergence_curve[t])

        return Best_pos, Best_score, Convergence_curve

'''
if __name__ == "__main__":  
    print('--------SAO----------------')

    F = 'F2'
    f = Function(F)
    fobj, lb, ub, dim = f.Functions()

    N = 100
    Max_iter  = 1000

    s = SAO(N, dim, ub, lb, Max_iter, fobj)
    Best_pos, Best_score, Convergence_curve = s.SAO()

    print(s)
    print('The best solution obtained is:', Best_pos)
    print('The best optimal value of the objective function found by SAO is:', Best_score)

'''
