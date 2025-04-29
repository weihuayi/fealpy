from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class ButterflyOptAlg(Optimizer):
    """
    Butterfly Optimization Algorithm (BOA) for optimization problems.

    This algorithm simulates the foraging behavior of butterflies, balancing between global exploration
    and local exploitation using fragrance values to guide the search process.

    Parameters:
        option (dict): A dictionary containing algorithm parameters (population size, dimensions, etc.)

    Attributes:
        gbest (Tensor): The global best solution found during the optimization process.
        gbest_f (float): The fitness value corresponding to the global best solution.
        curve (Tensor): Tracks the global best fitness value at each iteration.
    
    Methods:
        run(): Executes the Butterfly optimization algorithm for optimization.
    """

    def __init__(self, option) -> None:
        """
        Initializes the Butterfly optimization algorithm with the provided options.
        
        Parameters:
            option (dict): A dictionary containing optimizer settings.
        """
        super().__init__(option)
    
    def run(self, params={'a':0.1, 'c':0.01, 'p':0.8}):
        """
        Executes the Butterfly optimization algorithm for optimization.

        The algorithm uses fragrance to guide the search process. It explores the search space globally using
        the global best solution and refines the search locally by updating solutions based on their proximity
        to each other.

        Parameters:
            a (float): Exponent used to control the fragrance effect.
            c (float): A scaling factor that influences the fragrance.
            p (float): Probability for choosing between global and local search strategies.
        """
        # Evaluate initial fitness for all solutions
        a = params.get('a')
        c = params.get('c')
        p =params.get('p')
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        # Main optimization loop
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)  # Update positions of individuals

            # Calculate fragrance based on fitness
            f = c * (fit ** a)  # Fragrance Eq.(1)

            # Random probability for deciding between global and local search
            rand = bm.random.rand(self.N, 1)

            # Global and local search update equations
            x_new = ((rand < p) * # Global search Eq.(2)
                     (self.x + 
                      ((bm.random.rand(self.N, self.dim) ** 2) * self.gbest - self.x) * f[:, None]) + 
                     (rand >= p) * # Local search Eq.(3)
                     (self.x + 
                      ((bm.random.rand(self.N, self.dim) ** 2) * 
                       self.x[bm.random.randint(0, self.N, (self.N,))] - 
                       self.x[bm.random.randint(0, self.N, (self.N,))]) * f[:, None])) 

            # Boundary handling: Ensure solutions stay within bounds
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)

            # Evaluate fitness for the new population
            fit_new = self.fun(x_new)
            mask = fit_new < fit

            # Update the population with better solutions
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)

            # Update the global best solution
            self.update_gbest(self.x, fit)

            # Track the global best fitness value for plotting
            self.curve[it] = self.gbest_f

