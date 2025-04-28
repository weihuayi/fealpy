from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class ArtificialRabbitsOpt(Optimizer):
    """
    A class representing the Artificial Rabbits Optimization (ArtificialRabbitsOpt) algorithm, inheriting from the Optimizer class.

    This optimization algorithm is inspired by the behavior of rabbits in nature. The algorithm simulates the search and 
    exploration behavior of rabbits to find the optimal solution in the search space. The population of rabbits is iteratively 
    updated based on certain equations and rules that mimic the interaction and movement of rabbits in the environment.
    
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
            Executes the Artificial Rabbits Optimization algorithm. It iteratively improves the candidate solutions by adjusting 
            their positions based on the simulated rabbit movement, exploration, and interaction. The algorithm uses specific 
            equations to guide the exploration process and updates the best solution over multiple iterations.

    Reference:
    ~~~~~~~~~~
    Liying Wang, Qingjiao Cao, Zhenxing Zhang, Seyedali Mirjalili, Weiguo Zhao.
    Artificial rabbits optimization: A new bio-inspired meta-heuristic algorithm for solving engineering optimization problems.
    Engineering Applications of Artificial Intelligence, 2022, 114: 105082.
    """

    def __init__(self, option) -> None:
        """
        Initializes the ArtificialRabbitsOpt algorithm with the provided options.
        
        Parameters:
            option (dict): Configuration options for the optimization process, such as maximum iterations, bounds, etc.
        """
        super().__init__(option)
        
    def run(self):
        """
        Runs the Artificial Rabbits Optimization algorithm.

        This method performs the iterative optimization process where, in each iteration, the algorithm updates the 
        solutions based on a simulated rabbit exploration process. The position of each rabbit is adjusted by specific rules 
        and equations that guide their movement. The algorithm iterates until the maximum number of iterations (MaxIT) is 
        reached.

        Key steps in each iteration:
            1. Evaluate the fitness values of the current population.
            2. Compute the exploration factor (A) and the random movement (R) based on the iteration number.
            3. Update the positions of the rabbits based on the computed values and certain rules.
            4. Ensure that the updated positions stay within the search space boundaries.
            5. Evaluate the fitness of the updated population.
            6. Update the global best solution if a better one is found.
            7. Track the global best fitness value over iterations.
        """
        # Initial fitness values for the current population
        fit = self.fun(self.x)

        # Identify the index of the best solution
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]
        
        # Iterative optimization process
        for it in range(self.MaxIT):
            self.D_pl_pt(it)  # Update any auxiliary data structures (method not shown here)

            # Eq.(1) - Exploration factor (A)
            A = 4 * (1 - it / self.MaxIT) * bm.log(1 / bm.random.rand(1))
            
            # Eq.(2) - Random movement factor (R)
            R = ((bm.exp(bm.array(1)) - bm.exp(bm.array(((it - 1) / self.MaxIT) ** 2))) * 
                 bm.sin(2 * bm.pi * bm.random.rand(self.N, 1)) * 
                 bm.random.randint(0, 2, (self.N, self.dim)))

            # Randomly shuffle the population indices
            rand_index = bm.random.randint(0, self.N, (self.N,))

            # Eq.(3) - Calculate random movement scaling (H)
            r4 = bm.random.rand(self.N, 1)
            H = (self.MaxIT - it + 1) * r4 / self.MaxIT
            
            # Randomly choose a dimension to update for each rabbit
            k = bm.random.randint(0, self.dim, (self.N,))

            # Initialize a zero tensor for movement directions (g)
            g = bm.zeros((self.N, self.dim))
            g[bm.arange(self.N), k] = 1
            
            # Update position based on H and random scaling
            b = self.x + H * g * self.x
            
            # Apply the movement update based on the exploration factor A
            if A > 1:
                # Update with random exploration and position interaction
                x_new = (self.x[rand_index] + R * (self.x - self.x[rand_index]) + 
                         bm.round(0.5 * (0.05 + bm.random.rand(self.N, self.dim))) * bm.random.randn(self.N, self.dim))
            else:
                # Update with direct movement based on the random scaling
                x_new = self.x + R * (r4 * b - self.x)

            # Check and adjust the new positions to ensure they stay within the search space boundaries
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)

            # Evaluate the fitness of the new population
            fit_new = self.fun(x_new)
            
            # Update the population with the better solutions
            mask = fit_new < fit 
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)
            
            # Update the global best solution if necessary
            self.update_gbest(self.x, fit)
            
            # Track the fitness of the global best solution
            self.curve[it] = self.gbest_f
