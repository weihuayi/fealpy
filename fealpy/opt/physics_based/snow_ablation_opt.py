
from ...backend import backend_manager as bm 
from ...typing import TensorLike, Index, _S
from ... import logger
from ..optimizer_base import Optimizer

"""
Snow Ablation Optimizer

Reference
~~~~~~~~~
Lingyun Deng, Sanyang Liu.
Snow ablation optimizer: A novel metaheuristic technique for numerical optimization and engineering design.
Expert Systems With Applications, 2023, 225: 120069
"""
class SnowAblationOpt(Optimizer):

    def __init__(self, option) -> None:
        super().__init__(option)
    
    def run(self):
        Objective_values = self.fun(self.x)

        if not isinstance(self.lb, list):  
            self.lb = bm.array([self.lb] * self.dim)  
        if not isinstance(self.ub, list):
            self.ub = bm.array([self.ub] * self.dim)

        N1 = bm.array(int(bm.floor(bm.array(self.N * 0.5))))

        #升序 的索引数组
        idx1 = bm.argsort(Objective_values)
        
        #存储迭代优解原数组、最优解
        self.gbest = bm.copy(self.x[idx1[0], :])
        self.gbest_f = bm.copy(Objective_values[idx1[0]])

        second_best = bm.copy(self.x[idx1[1], :])
        third_best = bm.copy(self.x[idx1[2], :])

        # sum1 = #精英池原数组均值
        half_best_mean = bm.sum(self.x[idx1[:N1], :], axis=0) / N1
        Elite_pool = bm.concatenate((self.gbest.reshape(self.dim, 1), second_best.reshape(self.dim, 1), third_best.reshape(self.dim, 1), half_best_mean.reshape(self.dim, 1)),axis=1)
        Elite_pool = Elite_pool.reshape(4, self.dim)
        
        #分割
        index = bm.arange(self.N)
        Na = bm.array(int(self.N / 2))
        Nb = bm.array(int(self.N / 2))

        # 更新迭代
        for it in range(self.MaxIT):
            self.D_pl_pt(it)

            RB = bm.random.randn(self.N, self.dim)
            
            # eq.(9)
            DDF = 0.35 + 0.25 * (bm.exp(bm.array(it / self.MaxIT)) - 1) / (bm.exp(bm.array((1)) - 1))
            
            # eq.(10)
            #融雪速率
            M = DDF * bm.exp(bm.array(-it / self.MaxIT))

            # 随机分配索引号
            index1 = bm.unique(bm.random.randint(0, self.N - 1, (Na,)))
            index2 = bm.array(list(set(index.tolist()).difference(index1.tolist())))

            r1 = bm.random.rand(len(index1), 1)
            k1 = bm.random.randint(0, 3, (len(index1),))
            
            self.x[index1] = Elite_pool[k1] + RB[index1] * (r1 * (self.gbest - self.x[index1]) + (1 - r1) * (bm.mean(self.x, axis=0) - self.x[index1]))
                
            Na, Nb = (Na + 1, Nb - 1) if Na < self.N else (Na, Nb)
            
            if Nb >=1:
                r2 = 2 * bm.random.rand(len(index2), 1) - 1
                self.x[index2] = M * self.gbest + RB[index2] * (r2 * (self.gbest - self.x[index2]) + (1 - r2) * (bm.mean(self.x, axis=0) - self.x[index2])) 

            #检查是否超出搜索空间
            self.x = self.x + (self.lb - self.x) * (self.x < self.lb) + (self.ub - self.x) * (self.x > self.ub)
            Objective_values = self.fun(self.x)
            
            self.update_gbest(self.x, Objective_values)
            
            # 更新精英池
            idx1 = bm.argsort(Objective_values)
            second_best = bm.copy(self.x[idx1[1], :])
            third_best = bm.copy(self.x[idx1[2], :])
            half_best_mean = bm.sum(self.x[idx1[:N1], :], axis=0) / N1
            
            Elite_pool[0] = self.gbest
            Elite_pool[1] = second_best
            Elite_pool[2] = third_best
            Elite_pool[3] = half_best_mean
            self.curve[it] = self.gbest_f