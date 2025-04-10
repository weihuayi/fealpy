from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class PlantRhizomeGrowthBasedOpt(Optimizer):
    """
    Plant Rhizome Growth-Based Optimization (PRGBO) algorithm, subclass of Optimizer.

    This class implements the Plant Rhizome Growth-Based Optimization (PRGBO) algorithm, a population-based optimization
    method inspired by the growth process of plant rhizomes. The algorithm simulates the growth of plant rhizomes by 
    iterating through different growth stages and adjusting the solutions based on fitness, diversity, and other dynamic factors.

    Parameters:
        option (dict): Configuration options for the optimizer, such as initial solution, population size, 
                       maximum iterations, dimensionality, bounds, and objective function.

    Reference:
    ~~~~~~~~~~
    Fu Yan, Jianqiang Yang, Jin Zhang. 
    Plant Rhizome Growth-based Optimization Algorithm. 
    Systems Engineering-Theory & Practice, 2025.
    """
    def __init__(self, option) -> None:
        """
        Initializes the Plant Rhizome Growth-Based Optimization (PRGBO) optimizer by calling the parent class constructor.

        Parameters:
            option (dict): Configuration options for the optimizer.
        """
        super().__init__(option)


    def run(self):
        """
        Runs the Plant Rhizome Growth-Based Optimization (PRGBO) algorithm.

        This method performs the main optimization loop, adjusting the population based on fitness, diversity,
        and movement strategies, and tracking the global best solution over iterations.
        """
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        seed1 = self.x + ((bm.random.rand(self.N, self.dim) * 2 - 0.5) * 
                          (self.gbest - self.x[bm.random.randint(0, self.N, (self.N,))]))
        seed1 = seed1 + (self.lb - seed1) * (seed1 < self.lb) + (self.ub - seed1) * (seed1 > self.ub)
        seed2 = (bm.mean(self.x, axis=1)[:, None] + 
                 (bm.random.rand(self.N, self.dim) * 2 - 1) * 
                 (bm.mean(self.x, axis=1)[:, None] - self.x[bm.random.randint(0, self.N, (self.N,))]))
        seed2 = seed2 + (self.lb - seed2) * (seed2 < self.lb) + (self.ub - seed2) * (seed2 > self.ub)
        epsilon = bm.random.randint(0, 2, (self.N, 1))
        seed3 = (seed2 + (seed1 - (self.ub - self.lb)) * 
                 (epsilon * bm.random.rand(self.N, self.dim) + 1 - epsilon))
        seed3 = seed3 + (self.lb - seed3) * (seed3 < self.lb) + (self.ub - seed3) * (seed3 > self.ub)
        seed4 = ((bm.random.randint(0, 2, (self.N, 1)) * 
                  (self.x[bm.random.randint(0, self.N, (self.N,))] + 
                   self.x[bm.random.randint(0, self.N, (self.N,))] + self.x)) / 3 + 
                 (self.gbest + self.x[bm.random.randint(0, self.N, (self.N,))] + 
                  self.x[bm.random.randint(0, self.N, (self.N,))] - self.x))
        seed4 = seed4 + (self.lb - seed4) * (seed4 < self.lb) + (self.ub - seed4) * (seed4 > self.ub)
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)
            p = bm.random.rand(self.N, 1)
            
            seed1 = ((p > 0.5) * (self.x + (bm.random.rand(self.N, self.dim) * 2 - 0.5) * 
                                  (self.gbest - self.x[bm.random.randint(0, self.N, (self.N,))])) + 
                     (p <= 0.5) * (seed1))
            seed1 = seed1 + (self.lb - seed1) * (seed1 < self.lb) + (self.ub - seed1) * (seed1 > self.ub)
            fit1 = self.fun(seed1)
            
            seed2 = ((p > 0.5) * (bm.mean(self.x, axis=1)[:, None] + 
                                  (bm.random.rand(self.N, self.dim) * 2 - 1) * 
                                  (bm.mean(self.x, axis=1)[:, None] - 
                                   self.x[bm.random.randint(0, self.N, (self.N,))])) + 
                     (p <= 0.5) * (seed2))
            seed2 = seed2 + (self.lb - seed2) * (seed2 < self.lb) + (self.ub - seed2) * (seed2 > self.ub)
            fit2 = self.fun(seed2)
            
            epsilon = bm.random.randint(0, 2, (self.N, 1))
            seed3 = ((p > 0.5) * (seed2 + (seed1 - (self.ub - self.lb)) * 
                                  (epsilon * bm.random.rand(self.N, self.dim) + 1 - epsilon)) + 
                     (p <= 0.5) * (seed3))
            seed3 = seed3 + (self.lb - seed3) * (seed3 < self.lb) + (self.ub - seed3) * (seed3 > self.ub)
            fit3 = self.fun(seed3)
            
            seed4 = ((p <= 0.5) * (((bm.random.randint(0, 2, (self.N, 1)) * 
                                     (self.x[bm.random.randint(0, self.N, (self.N,))] + 
                                      self.x[bm.random.randint(0, self.N, (self.N,))] + self.x)) / 3 + 
                                    (self.gbest + self.x[bm.random.randint(0, self.N, (self.N,))] + 
                                     self.x[bm.random.randint(0, self.N, (self.N,))] - self.x))) + 
                     (p > 0.5) * (seed4))
            seed4 = seed4 + (self.lb - seed4) * (seed4 < self.lb) + (self.ub - seed4) * (seed4 > self.ub)
            fit4 = self.fun(seed4)
            
            seeds = bm.stack([seed1, seed2, seed3, seed4], axis=2)
            fit_matrix = bm.stack((fit1, fit2, fit3, fit4), axis=1)
            
            x_new = seeds[bm.arange(seeds.shape[0]), :, bm.argmin(fit_matrix, axis=1)]
            fit_new = fit_matrix[bm.arange(seeds.shape[0]), bm.argmin(fit_matrix, axis=1)]
            mask = fit_new < fit
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f