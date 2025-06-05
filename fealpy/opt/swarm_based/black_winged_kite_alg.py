from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class BlackwingedKiteAlg(Optimizer):
    """
    Black-winged Kite Algorithm (BWKA) for optimization problems.

    This algorithm simulates the behavior of black-winged kites, which engage in attacking and migration behaviors.
    These behaviors are used to explore and exploit the search space during the optimization process.

    Parameters:
        option (dict): A dictionary containing algorithm parameters (population size, dimensions, etc.)

    Attributes:
        gbest (Tensor): The global best solution found during the optimization process.
        gbest_f (float): The fitness value corresponding to the global best solution.
        curve (Tensor): Tracks the global best fitness value at each iteration.
    
    Methods:
        run(): Executes the Black-winged Kite optimization algorithm for optimization.

    Reference
        Jun Wang, Wen-chuan Wang, Xiao-xue Hu, Lin Qiu, Hong-fei Zang.
        Black-winged kite algorithm: a nature-inspired meta-heuristic for solving benchmark functions and engineering problems.
        Artificial Intelligence Review, 2024, 2024: 57-98.
    """

    def __init__(self, option) -> None:
        """
        Initializes the Black-winged Kite optimization algorithm with the provided options.
        
        Parameters:
            option (dict): A dictionary containing optimizer settings.
        """
        super().__init__(option)

    def run(self, params={'p':0.9}):
        """
        Executes the Black-winged Kite optimization algorithm for optimization.

        The algorithm uses attacking behavior for exploration and migration behavior for exploitation.

        Parameters:
            p (float): Probability factor for controlling attacking behavior.
        """
        # Evaluate initial fitness for all solutions
        p = params.get('p')
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        # Main optimization loop
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)  # Update positions of individuals
            
            # Attacking behavior: Exploration using exponential decay and sinusoidal oscillations
            R = bm.random.rand(self.N, 1)
            n = 0.05 * bm.exp(bm.array(-2 * ((it / self.MaxIT) ** 2)))  # Eq.(6)
            x_new = ((p < R) * 
                     (self.x + n * (1 + bm.sin(R)) * self.x) +  # Global exploration Eq.(5)
                     (p >= R) * 
                     (self.x * (n * (2 * bm.random.rand(self.N, self.dim) - 1) + 1)))  # Local exploration Eq.(5)
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            fit_new = self.fun(x_new)
            mask = fit_new < fit
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)

            # Migration behavior: Exploitation based on fitness and migration mechanics
            m = 2 * bm.sin(R + bm.pi / 2)  # Eq.(8)
            s = bm.random.randint(0, int(0.3 * self.N), (self.N,))  # Random selection of individuals
            fit_r = fit[s]  # Fitness of selected individuals
            cauchy_num = 1 / (bm.pi * ((bm.random.rand(self.N, self.dim) * bm.pi - bm.pi / 2) ** 2 + 1))  # Cauchy distribution
            
            # Update solutions based on fitness comparison
            x_new = ((fit < fit_r)[:, None] * (self.x + cauchy_num * (self.x - self.gbest)) + 
                     (fit >= fit_r)[:, None] * (self.x + cauchy_num * (self.gbest - m * self.x)))  # Eq.(7)
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            fit_new = self.fun(x_new)
            mask = fit_new < fit
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)
            
            # Update the global best solution
            self.update_gbest(self.x, fit)
            
            # Track the global best fitness value for plotting
            self.curve[it] = self.gbest_f
