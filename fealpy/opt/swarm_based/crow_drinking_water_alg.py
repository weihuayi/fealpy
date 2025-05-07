from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class CrowDrinkingWaterAlg(Optimizer):
    """
    Crow Drinking Water Algorithm (CDWA) for optimization.

    This algorithm simulates the behavior of crows searching for drinking water.
    The crows exploit the area around the best solution (gbest) and also explore the search space.

    Parameters:
        option (dict): A dictionary containing algorithm parameters (population size, dimensions, etc.)

    Attributes:
        gbest (Tensor): The global best solution found during the optimization process.
        gbest_f (float): The fitness value corresponding to the global best solution.
        curve (Tensor): Tracks the global best fitness value at each iteration.
    
    Methods:
        run(p=0.9): Executes the Crow Drinking Water algorithm for optimization.
    """

    def __init__(self, option) -> None:
        """
        Initializes the Crow Drinking Water algorithm with the provided options.
        
        Parameters:
            option (dict): A dictionary containing optimizer settings.
        """
        super().__init__(option)

    def run(self, params={'p':0.9}):
        """
        Executes the Crow Drinking Water optimization algorithm.
        
        Parameters:
            p (float): Probability factor for exploitation vs exploration.
        """
        # Evaluate initial fitness for all solutions
        p = params.get('p')
        fit = self.fun(self.x)

        # Initialize the global best solution
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        # Main optimization loop
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)  # Update positions of individuals

            # Random factor to decide whether to exploit or explore
            r = bm.random.rand(self.N, 1)
            
            # Exploitation: Move closer to the best solution
            self.x = ((r < p) * 
                      (self.x + bm.random.rand(self.N, 1) * (self.ub - self.x) + 
                       bm.random.rand(self.N, 1) * self.lb) + 
                      # Exploration: Move randomly within the search space
                      (r >= p) * 
                      ((2 * bm.random.rand(self.N, self.dim) - 1) * (self.ub - self.lb) + self.lb))

            # Boundary check to ensure solutions stay within bounds
            self.x = self.x + (self.lb - self.x) * (self.x < self.lb) + (self.ub - self.x) * (self.x > self.ub)

            # Evaluate fitness of the new positions
            fit = self.fun(self.x)

            # Update the global best solution
            self.update_gbest(self.x, fit)

            # Track the best fitness value for plotting
            self.curve[it] = self.gbest_f
