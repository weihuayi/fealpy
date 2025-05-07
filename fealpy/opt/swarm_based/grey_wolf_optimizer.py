from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class GreyWolfOpt(Optimizer):
    """
    A class implementing the Grey Wolf Optimization (GWO) algorithm.
    
    Grey Wolf Optimization mimics the hunting and social hierarchy of grey wolves. 
    The wolves are divided into Alpha, Beta, and Delta wolves, where Alpha leads the pack, Beta assists, 
    and Delta follows the orders. The wolves hunt collectively to find the optimal solution.

    Parameters:
        option (dict): A dictionary containing algorithm parameters like population size, dimensions, and max iterations.

    Attributes:
        gbest (Tensor): The global best solution found during the optimization process.
        gbest_f (float): The fitness value corresponding to the global best solution.
        curve (Tensor): Stores the progress of the best solution (fitness) over iterations.

    Methods:
        run(): The method that runs the Grey Wolf Optimization (GWO) algorithm.
    """
    
    def __init__(self, option) -> None:
        """
        Initializes the Grey Wolf Optimization algorithm with the provided options.
        
        Parameters:
            option (dict): A dictionary containing optimizer settings (e.g., population size, dimensionality).
        """
        super().__init__(option)

    def run(self):
        """
        Executes the Grey Wolf Optimization (GWO) algorithm to find the optimal solution.
        
        The method follows the social hierarchy of grey wolves (Alpha, Beta, Delta) to guide the search:
        1. **Alpha wolf**: Represents the best solution found so far.
        2. **Beta wolf**: Assists the Alpha wolf.
        3. **Delta wolf**: Helps with hunting but is not as influential as Alpha or Beta wolves.
        
        The algorithm uses the positions of these wolves and their interactions to explore and exploit the search space.
        """
        # Evaluate initial fitness for all solutions
        fit = self.fun(self.x)

        # Sort the fitness values and find the best three solutions (Alpha, Beta, Delta)
        X_fit_sort = bm.argsort(fit, axis=0)

        # Alpha wolf (best solution)
        x_alpha_fit = fit[X_fit_sort[0]]
        x_alpha = self.x[X_fit_sort[0]]

        # Beta wolf (second best solution)
        x_beta_fit = fit[X_fit_sort[1]]
        x_beta = self.x[X_fit_sort[1]]

        # Delta wolf (third best solution)
        x_delta_fit = fit[X_fit_sort[2]]
        x_delta = self.x[X_fit_sort[2]]

        # Initialize the global best solution with Alpha wolf's solution
        self.gbest_f = x_alpha_fit
        self.gbest = x_alpha

        # Main optimization loop
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)  # Update positions of individuals

            # Calculate the parameter 'a' which decreases over iterations to control the exploration-exploitation balance
            a = 2 - 2 * it / self.MaxIT

            # Update the positions of the wolves (Alpha, Beta, and Delta)
            x1 = (x_alpha - (2 * a * bm.random.rand(self.N, self.dim) - a) * 
                  bm.abs(2 * bm.random.rand(self.N, self.dim) * x_alpha - self.x))
            x2 = (x_beta - (2 * a * bm.random.rand(self.N, self.dim) - a) * 
                  bm.abs(2 * bm.random.rand(self.N, self.dim) * x_beta - self.x))
            x3 = (x_delta - (2 * a * bm.random.rand(self.N, self.dim) - a) * 
                  bm.abs(2 * bm.random.rand(self.N, self.dim) * x_delta - self.x))

            # Calculate the new positions as the average of the three wolves
            self.x = (x1 + x2 + x3) / 3

            # Apply boundary conditions (ensure that the positions are within the bounds)
            self.x = self.x + (self.lb - self.x) * (self.x < self.lb) + (self.ub - self.x) * (self.x > self.ub)

            # Evaluate fitness for the new positions
            fit = self.fun(self.x)

            # Sort the fitness values to identify the best solutions
            sort_index = bm.argsort(fit)
            X_sort = self.x[sort_index[:3]]
            fit_sort = fit[sort_index[:3]]
            
            # Update Alpha, Beta, and Delta based on the new fitness values
            if fit_sort[0] < x_alpha_fit:
                x_alpha, x_alpha_fit = X_sort[0], fit_sort[0]

            if x_alpha_fit < fit_sort[1] < x_beta_fit:
                x_beta, x_beta_fit = X_sort[1], fit_sort[1]
                
            if x_beta_fit < fit_sort[2] < x_delta_fit:
                x_delta, x_delta_fit = X_sort[2], fit_sort[2]

            # Update the global best solution and store the fitness of the best solution
            self.gbest = x_alpha
            self.gbest_f = x_alpha_fit
            self.curve[it] = self.gbest_f

        