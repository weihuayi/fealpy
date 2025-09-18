from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class HybridAdaptiveCrayfishOptAlg(Optimizer):
    """
    Hybrid Adaptive Crayfish Optimization Algorithm (HACOA) implementation.
    
    A novel bio-inspired optimization algorithm that simulates crayfish behaviors including:
    - Summer shelter seeking (temperature > 30°C)
    - Cave competition 
    - Adaptive foraging strategies (shredding vs direct feeding)
    Integrated with Differential Evolution for enhanced global search capability.

    Parameters:
        options (dict): Optimization configuration dictionary containing:
            - 'NP': Population size
            - 'dim': Problem dimension
            - 'MaxIT': Maximum iterations
            - 'MaxFes': Maximum function evaluations
            - 'lb': Lower bounds
            - 'ub': Upper bounds
            - 'fun': Objective function

    Attributes:
        Fes (int): Current number of function evaluations.
        gbest (array): Global best position found.
        gbest_f (float): Global best fitness value.

    Reference:
        Honghua Rao, Heming Jia, Xinyao Zhang and Laith Abualigah.
        Hybrid Adaptive Crayfish Optimization with Differential Evolution for Color Multi-Threshold Image Segmentation.
        Biomimetics, 2025, 10(4): 218.
    """

    def __init__(self, options):
        """Initialize the optimizer with given configuration."""
        super().__init__(options)
        self.Fes = 0   # current number of function evaluations
    
    def run(self):
        """
        Execute the hybrid optimization process combining Crayfish behavior and Differential Evolution.
        
        The algorithm features:
        1. Temperature-dependent behavior switching (summer shelter/competition/foraging)
        2. Adaptive foraging strategy with food quantity assessment
        3. Integrated Differential Evolution mutation and crossover
        4. Dynamic convergence factor for balanced exploration-exploitation
        """

        # Initialize fitness and global best
        fit = self.fun(self.x)
        gbest_idx = bm.argmin(fit)
        self.gbest_f = fit[gbest_idx]
        self.gbest = self.x[gbest_idx] 

        global_position = self.gbest  # Track global best position
        global_fitness = self.gbest_f  # Track global best fitness

        # Main loop until reaching maximum evaluations
        while self.Fes <= self.MaxFes:

            # ========== Crayfish Behavior Phase ==========
            
            # Convergence factor (linearly decreasing from 2 to 1)
            C = 2 - (self.Fes / self.MaxIT)
            
            # Environment temperature (20-35°C, controls behavior switching)
            temp = bm.random.rand(1) * 15 + 20
            
            # Cave position (midpoint between current global best and historical best)
            xf = (self.gbest + global_position) / 2
            
            # Adaptive Foraging Strategy: adaptive food quantity probability
            # Gaussian distribution centered at 25°C with std=3
            p = ((fit - self.gbest_f + 1) / (fit + 1) * 
                 (1 / (bm.sqrt(bm.array(2 * bm.pi) * 3))) * 
                 bm.exp(bm.array(- (temp - 25) ** 2 / (2 * 3 ** 2))))

            rand = bm.random.rand(self.N, 1)  # Random threshold for behavior selection

            # Reshape gbest for broadcasting operations
            self.gbest = self.gbest.reshape(1, self.dim)

            # Food factor Q: determines shredding or direct feeding behavior
            P = 3 * bm.random.rand(self.N) * fit / (self.gbest_f + 2.2204e-16)

            # Position update based on temperature and food conditions
            x_new = ((temp > 30) * ((rand < 0.5) * 
                                    # Summer shelter stage: move towards cave (temperature > 30°C)
                                    (self.x + C * bm.random.rand(self.N, self.dim) * (xf - self.x)) + 
                                    (rand > 0.5) * 
                                    # Competition stage: compete for caves
                                    (self.x - self.x[bm.random.randint(0, self.N, (self.N,))] + xf)) + 
                    (temp <= 30) * ((P[:, None] > 2) * 
                                    # Foraging stage: shred food first (abundant food)
                                    (self.x + (bm.cos(2 * bm.random.rand(self.N, self.dim) * bm.pi) - 
                                               bm.sin(2 * bm.pi * bm.random.rand(self.N, self.dim))) * 
                                               bm.exp(bm.array(-1 / P[:, None])) * self.gbest * p[:, None]) + 
                                    (P[:, None] <= 2) * 
                                    # Foraging stage: eat directly (scarce food)
                                    ((self.x - self.gbest) * p[:,None] + p[:,None] * bm.random.rand(self.N, self.dim) * self.x)))

            # Boundary constraint handling
            x_new = bm.clip(x_new, self.lb, self.ub)        
            
            # Evaluate new solutions and apply greedy selection
            fit_new = self.fun(x_new)
            self.Fes += self.N  # Update evaluation counter
            mask = fit_new < fit
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)
            
            # Update global best position
            newbest_id = bm.argmin(fit_new)
            if fit_new[newbest_id] < global_fitness:
                global_position = bm.copy(x_new[newbest_id])
                global_fitness = bm.copy(fit_new[newbest_id])
            self.update_gbest(self.x, fit)

            # ========== Differential Evolution (DE) Enhancement Phase ==========

            # Mutation: DE/rand/1 strategy - Vi = Xr1 + F * (Xr2 - Xr3)
            r1 = bm.random.randint(0, self.N, (self.N,))
            r2 = bm.random.randint(0, self.N, (self.N,))
            r3 = bm.random.randint(0, self.N, (self.N,))
            V = self.x[r1] + 0.85 * (self.x[r2] - self.x[r3])  # Scale factor F=0.85
            V = bm.clip(V, self.lb, self.ub)

            # Crossover: binomial crossover with probability Pc = 0.8
            mask = bm.random.rand(self.N, self.dim) < 0.8
            idx = bm.random.randint(0, self.dim, (self.N,))  # Ensure at least one dimension crosses
            mask[bm.arange(self.N), idx] = 1
            x_new = bm.where(mask, V, self.x)

            # Selection: greedy replacement strategy
            fit_new = self.fun(x_new)
            self.Fes += self.N  # Update evaluation counter
            mask = fit_new < fit
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)

            # Update global best position again
            newbest_id = bm.argmin(fit_new)
            if fit_new[newbest_id] < global_fitness:
                global_position = bm.copy(x_new[newbest_id])
                global_fitness = bm.copy(fit_new[newbest_id])
            self.update_gbest(self.x, fit)