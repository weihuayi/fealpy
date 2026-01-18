from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer
from ..opt_function import levy

class SecretaryBirdOpt(Optimizer):
    """
    A class representing the Secretary Bird Optimization (SBO) algorithm, inheriting from Optimizer.

    This meta-heuristic algorithm simulates the hunting and escaping behaviors of secretary birds. 
    The search process is divided into three stages: exploration, predation, and escape, 
    controlled by the iteration progress. It manages a population of potential solutions (x) 
    and tracks the global best position found so far.

    Parameters:
        options(dict): A dictionary containing configuration parameters such as population size (N), 
            dimension (dim), maximum iterations (MaxIT), and objective function (fun).

    Attributes:
        x(Tensor): A tensor of shape (N, dim) representing the current population of solutions.
        gbest(Tensor): A tensor representing the best solution found globally.
        gbest_f(float): The fitness value of the global best solution.
        curve(Tensor): A tensor tracking the best fitness value across iterations.
    
    Methods:
        run(): Executes the main loop of the Secretary Bird Optimization algorithm.
    """
    def __init__(self, options):
        super().__init__(options)

    def run(self):
        """
        Executes the optimization process by iterating through exploration, exploitation, and update phases.

        The method updates the population positions based on three temporal phases:
        1. it <= MaxIT/3: Random exploration using neighbor differences.
        2. MaxIT/3 < it <= 2*MaxIT/3: Predation phase focusing on the global best.
        3. it > 2*MaxIT/3: Escape phase utilizing Levy flights.
        It further refines positions with a secondary update logic and clips them within bounds [lb, ub].
        """
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        for it in range(self.MaxIT):

            self.D_pl_pt(it)

            if it <= self.MaxIT / 3:
                # Exploration: Random movement based on neighbor differences
                r1 = bm.random.randint(0, self.N, (self.N,))
                r2 = bm.random.randint(0, self.N, (self.N,))
                x_new = self.x + bm.random.rand(self.N, self.dim) * (self.x[r1] - self.x[r2])
            elif self.MaxIT / 3 < it and it <= 2 * self.MaxIT / 3:
                # Predation: Exploitation toward global best
                coef = bm.exp(bm.array((it / self.MaxIT) ** 4))
                x_new = self.gbest + coef * (bm.random.randn(self.N, self.dim) - 0.5) * (self.gbest - self.x)
            else:
                # Escape: Levy flight for global search jumping
                CF = (1 - it / self.MaxIT) ** (2 * it / self.MaxIT)
                x_new = self.gbest + CF * self.x * 0.5 * levy(self.N, self.dim, 1.5)

            # Boundary check and first greedy selection
            x_new = bm.clip(x_new, self.lb, self.ub)
            fit_new = self.fun(x_new)
            mask = fit_new < fit 
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)

            r = bm.random.rand(self.N, 1)
            mask_r = r < 0.5
            
            # Option 1: Local exploitation; Option 2: Random neighborhood search
            opt1 = self.gbest + self.x * (bm.random.randn(self.N, self.dim) * 2 - 1) * (1 - it / self.MaxIT) ** 2
            opt2 = self.x + bm.random.rand(self.N, self.dim) * (
                self.x[bm.random.randint(0, self.N, (self.N,))] - 
                self.x * bm.round(1 + bm.random.rand(self.N, 1))
            )
            x_new = bm.where(mask_r, opt1, opt2)

            # Final boundary check and update
            x_new = bm.clip(x_new, self.lb, self.ub)
            fit_new = self.fun(x_new)
            mask = fit_new < fit 
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)

            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f

class CrossoverSecretaryBirdOpt(SecretaryBirdOpt):
    """
    An enhanced SBO variant that incorporates a horizontal and vertical crossover strategy.

    This class inherits from SecretaryBirdOpt and modifies the optimization process by 
    introducing a crossover mechanism to increase population diversity and prevent premature convergence.

    Parameters:
        options(dict): Configuration parameters for the optimizer.
    
    Methods:
        horizontal_vertical_crossover_strategy(x, fit): Performs dual-stage crossover on the population.
        run(): Executes the optimization loop including the crossover steps.
    """
    def __init__(self, options):
        super().__init__(options)
    
    def horizontal_vertical_crossover_strategy(self, x, fit):
        """
        Applies horizontal and vertical crossover operators to the current population.

        The horizontal stage pairs odd and even indices for arithmetic crossover, 
        while the vertical stage performs dimension-wise feature exchange within individuals.

        Parameters:
            x(Tensor): The current population positions of shape (N, dim).
            fit(Tensor): The current fitness values of the population.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - x(Tensor): The updated population after crossover and selection.
                - fit(Tensor): The updated fitness values.
        """
        # --- Horizontal Crossover ---
        r1, r2 = bm.random.rand(self.N // 2, self.dim), bm.random.rand(self.N // 2, self.dim)
        c1 = -1 + 2 * bm.random.rand(self.N // 2, self.dim)
        c2 = -1 + 2 * bm.random.rand(self.N // 2, self.dim)
        
        x_odd, x_even = x[0::2], x[1::2] # Using slicing for readability
        x_new1 = r1 * x_odd + (1 - r1) * x_even + c1 * (x_odd - x_even)
        x_new2 = r2 * x_even + (1 - r2) * x_odd + c2 * (x_even - x_odd)
        
        x_new = bm.zeros((self.N, self.dim))
        x_new[0::2], x_new[1::2] = x_new1, x_new2
        
        x_new = bm.clip(x_new, self.lb, self.ub)
        fit_new = self.fun(x_new)
        mask = fit_new < fit
        x, fit = bm.where(mask[:, None], x_new, x), bm.where(mask, fit_new, fit)

        # --- Vertical Crossover ---
        q1 = bm.random.randint(0, self.dim, (self.N,))
        offset = bm.random.randint(1, self.dim, (self.N,))
        q2 = (q1 + offset) % self.dim
        r_v = bm.random.rand(self.N,)
        
        x_new[bm.arange(self.N), q1] = r_v * x[bm.arange(self.N), q1] + (1 - r_v) * x[bm.arange(self.N), q2]
        x_new = bm.clip(x_new, self.lb, self.ub)
        fit_new = self.fun(x_new)
        mask = fit_new < fit
        x, fit = bm.where(mask[:, None], x_new, x), bm.where(mask, fit_new, fit)
        
        return x, fit

    def run(self):
        """
        Executes the SBO optimization with the integrated crossover strategy.

        This method follows the standard SBO stages but inserts a 
        horizontal_vertical_crossover_strategy after each main position update to refine 
        the candidate solutions.
        """
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        for it in range(self.MaxIT):

            self.D_pl_pt(it)

            CF = (1 - it / self.MaxIT) ** (2 * it / self.MaxIT)
            
            if it <= self.MaxIT / 3:
                # Exploration: Random movement based on neighbor differences
                r1 = bm.random.randint(0, self.N, (self.N,))
                r2 = bm.random.randint(0, self.N, (self.N,))
                r3 = bm.random.randint(0, self.N, (self.N,))
                x_new = self.x + CF * (self.x[r1] - self.x[r2]) + CF * (self.x[r3] - self.x)
            elif self.MaxIT / 3 < it and it <= 2 * self.MaxIT / 3:
                # Predation: Exploitation toward global best
                coef = bm.exp(bm.array((it / self.MaxIT) ** 4))
                x_new = self.gbest + coef * (bm.random.randn(self.N, self.dim) - 0.5) * (self.gbest - self.x)
            else:
                # Escape: Levy flight for global search jumping
                x_new = self.gbest + CF * self.x * 0.5 * levy(self.N, self.dim, 1.5)

            x_new = bm.clip(x_new, self.lb, self.ub)
            fit_new = self.fun(x_new)
            mask = fit_new < fit 
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)

            r = bm.random.rand(self.N, 1)
            mask_r = r < 0.5
            opt1 = self.gbest + self.x * (bm.random.randn(self.N, self.dim) * 2 - 1) * (1 - it / self.MaxIT) ** 2
            opt2 = self.x + bm.random.rand(self.N, self.dim) * (
                self.x[bm.random.randint(0, self.N, (self.N,))] - 
                self.x * bm.round(1 + bm.random.rand(self.N, 1))
            )
            x_new = bm.where(mask_r, opt1, opt2)
            
            x_new = bm.clip(x_new, self.lb, self.ub)
            fit_new = self.fun(x_new)
            mask = fit_new < fit
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)

            self.x, fit = self.horizontal_vertical_crossover_strategy(self.x, fit)

            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f