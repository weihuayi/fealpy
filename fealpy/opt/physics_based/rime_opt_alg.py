from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class RimeOptAlg(Optimizer):
    """
    A class representing the Rime Optimization Algorithm (RimeOptAlg), inheriting from the Optimizer class.
    
    This algorithm is used for optimization problems and incorporates Rime factors to update the solution iteratively.
    It utilizes multiple equations to adjust the search space and improve the fitness of solutions over iterations.
    
    Parameters:
        option (dict): A dictionary containing configuration options for the algorithm, typically including 
                       the maximum number of iterations, search space boundaries, and other optimization parameters.
    
    Attributes:
        gbest (Tensor): The global best solution found so far.
        gbest_f (float): The fitness value corresponding to the global best solution.
        MaxIT (int): Maximum number of iterations to perform.
        lb (Tensor): Lower bounds for the optimization problem.
        ub (Tensor): Upper bounds for the optimization problem.
        N (int): Number of solutions (population size).
        dim (int): The dimensionality of the search space.
        fun (callable): The fitness function to evaluate solutions.
        x (Tensor): Current population of candidate solutions.
        curve (dict): A dictionary to store the fitness values of the global best solution at each iteration.
    
    Methods:
        run(w=5): 
            Runs the Rime Optimization algorithm, iteratively updating the population and tracking the best solution found.
            During each iteration, it applies specific equations (Eq. (3), (4), (5), (6), and (7)) to update the particles and
            calculate the fitness values. The algorithm stops after reaching the maximum number of iterations (MaxIT).

    Reference:
    ~~~~~~~~~~
    Hang Su, Dong Zhao, Ali Asghar Heidari, Lei Liu, Xiaoqin Zhang, Majdi Mafarja, Huiling Chen. 
    RIME: A physics-based optimization. 
    Neurocomputing, 2023, 532: 183-214.
    """
    
    def __init__(self, option) -> None:
        """
        Initializes the RimeOptAlg with the provided options.
        
        Parameters:
            option (dict): Configuration options for the optimization process, such as maximum iterations, bounds, etc.
        """
        super().__init__(option)

    def run(self, params={'w':5}):
        """
        Runs the Rime Optimization algorithm.

        This method performs the iterative optimization process where, at each iteration, the global best solution is 
        updated based on Rime factors and fitness evaluations. The solution is adjusted using various equations, and 
        the algorithm attempts to minimize the fitness function.

        Parameters:
            w (float): A weight factor to control the influence of the iteration number on the Rime factors. 
                       Defaults to 5.
        
        The following steps are performed in each iteration:
            1. Calculate the fitness values for the current population.
            2. Identify the global best solution based on the fitness values.
            3. Update the solution using Rime factors, influenced by the iteration number, and bounded by the 
               lower and upper bounds of the problem.
            4. Apply Eq.(3), (4), and (5) to compute the Rime factors and adjust the positions of the particles.
            5. Calculate Eq.(6) to update the exploration rate and apply Eq.(7) to normalize the Rime rates.
            6. If the fitness of a new solution is better, update the current population and global best solution.
            7. Track the fitness of the global best solution over iterations.
        """
        w = params.get('w')
        fit = self.fun(self.x)  # Evaluate fitness for the current population
        gbest_index = bm.argmin(fit)  # Find index of the best solution
        self.gbest = self.x[gbest_index]  # Set global best solution
        self.gbest_f = fit[gbest_index]  # Set fitness of the global best solution

        for it in range(0, self.MaxIT):  # Loop through each iteration
            self.D_pl_pt(it)  # Update any necessary auxiliary data structures (method not shown here)

            # Calculate Rime factors based on current iteration and other parameters (Eq.(3), (4), (5))
            RimeFactor = ((bm.random.rand(self.N, 1) - 0.5) * 2 * 
                          bm.cos(bm.array(bm.pi * it / (self.MaxIT / 10))) * 
                          (1 - bm.round(bm.array(it * w / self.MaxIT)) / w)) 
            E = ((it + 1) / self.MaxIT) ** 0.5  # Update exploration rate (Eq.(6))
            normalized_rime_rates = fit / (bm.linalg.norm(fit) + 1e-10)  # Normalize Rime rates (Eq.(7))

            # Generate new solution candidates based on Rime factors and the global best solution
            r1 = bm.random.rand(self.N, 1)
            x_new = ((r1 < E) * (self.gbest + RimeFactor * ((self.ub - self.lb) * bm.random.rand(self.N, 1) + self.lb)) + 
                     (r1 >= E) * self.x)

            r2 = bm.random.rand(self.N, self.dim)
            # Update new solutions based on normalized Rime rates
            x_new = ((r2 < normalized_rime_rates[:, None]) * (self.gbest) + 
                     (r2 >= normalized_rime_rates[:, None]) * x_new)
            
            # Bound new solutions within the given limits
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)

            # Evaluate fitness for the new solutions
            fit_new = self.fun(x_new)

            # Mask for better solutions and update the population
            mask = fit_new < fit
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)

            # Update the global best solution if necessary
            self.update_gbest(self.x, fit)

            # Store the fitness of the global best solution for tracking progress
            self.curve[it] = self.gbest_f
