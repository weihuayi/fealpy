from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class CrayfishOptAlg(Optimizer):
    """
    Crayfish Optimization Algorithm (COA) for optimization problems.

    The algorithm simulates the foraging behavior of crayfish, dynamically adjusting the search strategy
    based on environmental temperature and individual exploration. It combines global and local search mechanisms.

    Parameters:
        option (dict): A dictionary containing algorithm parameters (population size, dimensions, etc.)

    Attributes:
        gbest (Tensor): The global best solution found during the optimization process.
        gbest_f (float): The fitness value corresponding to the global best solution.
        curve (Tensor): Tracks the global best fitness value at each iteration.
    
    Methods:
        run(): Executes the Crayfish optimization algorithm for optimization.

    Reference
        Heming Jia, Honghua Rao, Changsheng Wen, Seyedali Mirjalili. 
        Crayfish optimization algorithm. 
        Artificial Intelligence Review, 2023, 56: S1919-S1979.
    """

    def __init__(self, option) -> None:
        """
        Initializes the Crayfish optimization algorithm with the provided options.
        
        Parameters:
            option (dict): A dictionary containing optimizer settings.
        """
        super().__init__(option)
    
    def run(self):
        """
        Executes the Crayfish optimization algorithm for optimization.

        This method includes the main optimization loop where the positions of the solutions are updated
        based on different strategies and environmental factors.
        """
        # Evaluate initial fitness for all solutions
        fit = self.fun(self.x)
        gbest_idx = bm.argmin(fit)
        self.gbest_f = fit[gbest_idx]
        self.gbest = self.x[gbest_idx] 

        # Initialize other variables
        fit_new = bm.zeros((self.N,))
        global_position = self.gbest
        global_fitness = self.gbest_f

        # Main optimization loop
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)  # Update positions of individuals

            # Cooling factor controlling the balance of exploration and exploitation
            C = 2 - (it / self.MaxIT)
            
            # Temperature update
            temp = bm.random.rand(1) * 15 + 20
            
            # Calculating the foraging position (xf) as the midpoint between global best and individual position
            xf = (self.gbest + global_position) / 2
            
            # Probability distribution based on temperature
            p = (0.2 * (1 / (bm.sqrt(bm.array(2 * bm.pi) * 3))) * 
                 bm.exp(bm.array(- (temp - 25) ** 2 / (2 * 3 ** 2))))
            
            # Random probability for exploration and exploitation
            rand = bm.random.rand(self.N, 1)

            # Reshape gbest for matrix operations
            self.gbest = self.gbest.reshape(1, self.dim)

            # Calculate P as a probability factor based on fitness and gbest_f
            P = 3 * bm.random.rand(self.N) * fit / (self.gbest_f + 2.2204e-16)

            # Update positions using different movement strategies
            x_new = ((temp > 30) * ((rand < 0.5) * 
                                    (self.x + C * bm.random.rand(self.N, self.dim) * (xf - self.x)) + 
                                    (rand > 0.5) * 
                                    (self.x - self.x[bm.random.randint(0, self.N, (self.N,))] + xf)) + 
                    (temp <= 30) * ((P[:, None] > 2) * 
                                    (self.x + bm.cos(2 * bm.random.rand(self.N, self.dim) * bm.pi) * self.gbest * 
                                     p - bm.sin(2 * bm.pi * bm.random.rand(self.N, self.dim) * self.gbest * p)) + 
                                    (P[:, None] <= 2) * 
                                    ((self.x - self.gbest) * p + p * bm.random.rand(self.N, self.dim) * self.x)))

            # Boundary handling
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)          
            
            # Evaluate fitness for the new population
            fit_new = self.fun(x_new)
            mask = fit_new < fit

            # Update the population with better solutions
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)
            
            # Find the best solution in the new population
            newbest_id = bm.argmin(fit_new)

            # Update global best if a better solution is found
            if fit_new[newbest_id] < global_fitness:
                global_position = x_new[newbest_id]
                global_fitness = fit_new[newbest_id]

            # Track the global best solution
            self.update_gbest(self.x, fit)

            # Flatten the gbest for the next iteration
            self.gbest = self.gbest.flatten()
            
            # Track the best fitness value for plotting
            self.curve[it] = self.gbest_f
