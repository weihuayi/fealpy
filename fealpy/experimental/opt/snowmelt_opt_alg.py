
from ..backend import backend_manager as bm 
from ..typing import TensorLike, Index, _S
from .. import logger
from .optimizer_base import Optimizer

import random


class SnowmeltOptAlg(Optimizer):

    def __init__(self, option) -> None:
        super().__init__(option)
    
    def run(self):
        option = self.options
        X = option["x0"]
        lb, ub = option["domain"]
        dim = option["ndim"]
        Objective_values = self.fun(X)
        N = option["NP"]
        Max_iter = option["MaxIters"]

        if not isinstance(lb, list):  
            lb = bm.array([lb] * dim)  
        if not isinstance(ub, list):
            ub = bm.array([ub] * dim)
        
        #空列表
        Convergence_curve = []

        N1 = int(bm.floor(bm.array(N * 0.5)))

        #升序 的索引数组
        idx1 = bm.argsort(Objective_values)
        
        #存储迭代优解原数组、最优解
        Best_pos = bm.copy(X[idx1[0], :])
        Best_score = bm.copy(Objective_values[idx1[0]])

        second_best = bm.copy(X[idx1[1], :])
        third_best = bm.copy(X[idx1[2], :])

        sum1 = bm.sum(X[idx1[:N1], :], axis=0)#精英池原数组均值
        half_best_mean = sum1 / N1
        Elite_pool = bm.concatenate((Best_pos.reshape(dim, 1), second_best.reshape(dim, 1), third_best.reshape(dim, 1), half_best_mean.reshape(dim, 1)),axis=1)
        Elite_pool = Elite_pool.reshape(4, dim)

        Convergence_curve.append(Best_score)
        
        #分割
        index = bm.arange(N)
        Na = int(N / 2)
        Nb = int(N / 2)

        # 更新迭代
        for t in range(Max_iter):
            RB = bm.random.randn(N, dim)
            T = bm.exp(bm.array(-t / Max_iter))
            # eq.(9)
            DDF = 0.35 + 0.25 * (bm.exp(bm.array(t / Max_iter)) - 1) / (bm.exp(bm.array((1)) - 1))
            # eq.(10)
            #融雪速率
            M = DDF * T

            #种群质心位置
            X_centroid = bm.mean(X, axis=0)

            # 随机分配索引号
            index1 = list(set([random.randint(0, N - 1) for _ in range(Na)]))
            index2 = list(set(index).difference(index1))
            
            r1 = bm.random.rand(len(index1), 1)
            k1 = [random.randint(0,3) for _ in range(len(index1))]
                  
            X[index1] = Elite_pool[k1] + RB[index1] * (r1 * (Best_pos - X[index1]) + (1 - r1) * (X_centroid - X[index1]))
                
            Na, Nb = (Na + 1, Nb - 1) if Na < N else (Na, Nb)
            
            if Nb >=1:
                r2 = 2 * bm.random.rand(len(index2), 1) - 1
                X[index2] = M * Best_pos + RB[index2] * (r2 * (Best_pos - X[index2]) + (1 - r2) * (X_centroid - X[index2])) 

            #检查是否超出搜索空间
            X = X + (lb - X) * (X < lb) + (ub - X) * (X > ub)
            Objective_values = self.fun(X)
            
            best_idx = bm.argmin(Objective_values)
            (Best_pos, Best_score) = (bm.copy(X[best_idx]), Objective_values[best_idx]) if Objective_values[best_idx] < Best_score else (Best_pos, Best_score)
            
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
        return Best_pos, Best_score


     

