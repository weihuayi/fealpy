from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class ZebraOptAlg(Optimizer):
    """
    A class representing the Zebra Optimization Algorithm (ZebraOptAlg), inheriting from the Optimizer class.

    This optimization algorithm is inspired by the foraging behavior of zebras. The algorithm simulates how zebras search 
    for food (exploitation) and move to new areas (exploration) in order to find the best solution to the optimization 
    problem. It uses a combination of exploration and exploitation strategies to iteratively improve the candidate solutions.

    Parameters:
        option (dict): A dictionary containing configuration options for the algorithm, including the maximum number of 
                       iterations, lower and upper bounds, population size, and other optimization parameters.
    
    Attributes:
        gbest (Tensor): The global best solution found so far.
        gbest_f (float): The fitness value corresponding to the global best solution.
        MaxIT (int): The maximum number of iterations for the optimization.
        lb (Tensor): Lower bounds of the solution space.
        ub (Tensor): Upper bounds of the solution space.
        N (int): The size of the population.
        dim (int): The dimensionality of the search space.
        fun (callable): The objective (fitness) function to evaluate solutions.
        x (Tensor): The current population of candidate solutions.
        curve (dict): A dictionary to store the fitness values of the global best solution at each iteration.
    
    Methods:
        run():
            Executes the Zebra Optimization Algorithm. It iteratively updates the solutions based on foraging behavior (both 
            exploration and exploitation) of zebras. The algorithm uses specific equations for exploration and exploitation 
            to update the positions of candidate solutions and tracks the global best solution over multiple iterations.

    Reference:
        Eva Trojovska, Mohammad Dehghani, Pavel Trojovsky.
        Zebra Optimization Algorithm: A New Bio-Inspired Optimization Algorithm for Solving Optimization Algorithm.
        IEEE Access, 10: 49445-49473.
    """

    def __init__(self, option) -> None:
        """
        Initializes the ZebraOptAlg algorithm with the provided options.
        
        Parameters:
            option (dict): Configuration options for the optimization process, such as maximum iterations, bounds, etc.
        """
        super().__init__(option)


    def run(self):
        """
        Runs the Zebra Optimization Algorithm.

        This method performs the iterative optimization process where the positions of candidate solutions are updated 
        based on two main behaviors: exploration (searching new areas) and exploitation (refining the current best solutions). 
        The algorithm uses equations to adjust the candidate solutions, ensuring they stay within the search space. 

        Key steps in each iteration:
            1. Evaluate the fitness values of the current population.
            2. Update the positions based on foraging behavior (exploration and exploitation).
            3. Ensure that the updated positions remain within the search space boundaries.
            4. Evaluate the fitness of the new population.
            5. Update the global best solution if a better one is found.
            6. Track the fitness of the global best solution over iterations.
        """
        # Evaluate initial fitness values
        fit = self.fun(self.x)

        # Identify the index of the best solution
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]
        
        # Iterative optimization process
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)  # Update any auxiliary data structures (method not shown here)

            # Eq.(3) - Foraging behavior (exploration)
            x_new = self.x + (bm.random.rand(self.N, self.dim) * 
                              (self.gbest - (1 + bm.random.rand(self.N, self.dim)) * self.x))
            
            # Ensure the new positions stay within the bounds
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            
            # Evaluate fitness of new positions
            fit_new = self.fun(x_new)
            
            # Update population with better solutions
            mask = fit_new < fit 
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)

            # Eq.(5).S1 - Exploitation behavior (against) and Eq.(5).S2 - Attraction behavior
            r = bm.random.rand(self.N, 1)
            x_new = ((r < 0.5) * (self.x + 0.01 * self.x * 
                                  (2 * bm.random.rand(self.N, self.dim) - 1) * (1 - it / self.MaxIT)) + 
                     (r >= 0.5) * (self.x + bm.random.rand(self.N, self.dim) * 
                                   (self.x[bm.random.randint(0, self.N, (self.N,))] - 
                                    (1 + bm.random.rand(self.N, self.dim)) * self.x)))
            
            # Ensure the new positions stay within the bounds
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            
            # Evaluate fitness of new positions
            fit_new = self.fun(x_new)
            
            # Update population with better solutions
            mask = fit_new < fit 
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)

            # Update the global best solution
            self.update_gbest(self.x, fit)
            
            # Track the fitness of the global best solution
            self.curve[it] = self.gbest_f
