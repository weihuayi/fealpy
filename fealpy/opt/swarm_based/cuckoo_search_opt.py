from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer
from ..opt_function import levy

class CuckooSearchOpt(Optimizer):
    """
    Cuckoo Search Optimization Algorithm
    
    This algorithm mimics the egg-laying behavior of cuckoo birds.
    The cuckoo birds lay their eggs in random nests, and the best eggs are chosen based on the fitness of the solution.
    The algorithm uses two main mechanisms: Levy flight for exploration and nest updates for exploitation.

    Parameters:
        option (dict): A dictionary containing algorithm parameters like population size, dimensions, and max iterations.

    Attributes:
        gbest (Tensor): The global best solution found during the optimization process.
        gbest_f (float): The fitness value corresponding to the global best solution.
        curve (Tensor): Stores the progress of the best solution (fitness) over iterations.

    Methods:
        run(alpha=0.01, p=0.25): Executes the Cuckoo Search optimization algorithm with the given parameters.
    """

    def __init__(self, option) -> None:
        """
        Initializes the Cuckoo Search algorithm with the provided options.
        
        Parameters:
            option (dict): A dictionary containing optimizer settings (e.g., population size, dimensionality).
        """
        super().__init__(option)

    def run(self, params={'alpha':0.01, 'p':0.25}):
        """
        Executes the Cuckoo Search optimization algorithm to find the optimal solution.
        
        Parameters:
            alpha (float): The step size for the Levy flight, controlling the exploration of the search space.
            p (float): The probability of a nest being replaced by a new cuckoo egg (local search probability).
        """
        # Evaluate initial fitness for all solutions
        alpha = params.get('alpha')
        p = params.get('p')
        fit = self.fun(self.x)
        
        # Initialize the global best solution
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        # Main optimization loop
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)  # Update positions of individuals

            # Levy flight: Explore the search space based on the global best solution
            x_new = self.x + alpha * levy(self.N, self.dim, 1.5) * (self.x - self.gbest)
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            fit_new = self.fun(x_new)
            mask = fit_new < fit  # Select the best solutions
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)

            # Nest update: Perform a local search for solutions in the nests
            x_new = (self.x + bm.random.rand(self.N, self.dim) * 
                     bm.where((bm.random.rand(self.N, self.dim) - p) < 0, 0, 1) * 
                     (self.x[bm.random.randint(0, self.N, (self.N,))] - 
                      self.x[bm.random.randint(0, self.N, (self.N,))]))
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            fit_new = self.fun(x_new)
            mask = fit_new < fit  # Select the best solutions
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)

            # Update the global best solution
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f

