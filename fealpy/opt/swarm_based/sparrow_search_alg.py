from ..opt_function import levy
from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

"""
Sparrow Search Algorithm  

Reference:
~~~~~~~~~~
Mohit Jain, Vijander Singh, Asha Rani.
A novel swarm intelligence optimization approach: sparrow search algorithm.
Systems Science & Control Engineering, 2020, 8: 22-34.
"""


class SparrowSearchAlg(Optimizer):
    """
    A class implementing the Sparrow Search Algorithm (SSA), inheriting from the Optimizer class.

    The Sparrow Search Algorithm is a population-based optimization method inspired by the foraging behavior of sparrows.
    Sparrows adjust their movements based on the best known solutions and interactions with other sparrows.

    Parameters:
        option (dict): Configuration options for the optimization process, such as maximum iterations, population size,
                       lower and upper bounds, etc.

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

    Reference:
        Mohit Jain, Vijander Singh, Asha Rani.
        A novel swarm intelligence optimization approach: sparrow search algorithm.
        Systems Science & Control Engineering, 2020, 8: 22-34.
    """
    
    def __init__(self, option) -> None:
        """
        Initializes the Sparrow Search Algorithm with the provided options.

        Parameters:
            option (dict): Configuration options for the optimization process, such as maximum iterations, bounds, etc.
        """
        super().__init__(option)

    def run(self, params={'st':0.6, 'pd':0.7, 'sd':0.2}):
        """
        Runs the Sparrow Search Algorithm.

        The method simulates the foraging behavior of sparrows. It divides the population into different subsets and 
        updates their positions based on various movement rules, which depend on the fitness values, distances, 
        and randomization factors.

        Key steps in each iteration:
            1. Evaluate the fitness values of the current population.
            2. Update the positions of the top-performing sparrows based on a time-varying factor (st).
            3. Update the remaining sparrows based on a combination of random movements and attraction to the best solutions.
            4. Update the positions of sparrows based on a fitness-dependent strategy (sd).
            5. Keep track of the global best solution and update it if a better solution is found.
        """
        # Initial fitness evaluation
        st = params.get('st')
        pd = params.get('pd')
        sd = params.get('sd')
        fit = self.fun(self.x)
        
        # Identify the global best solution and its fitness
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]
        best_f = fit[gbest_index]

        # Sort the population by fitness (ascending)
        index = bm.argsort(fit)
        self.x = self.x[index]
        fit = fit[index]
        
        # Define number of sparrows for each group (based on proportions pd and sd)
        pd_number = int(self.N * pd)  # Proportion for performing direct movement
        sd_number = int(self.N * sd)  # Proportion for performing a fitness-based movement

        for it in range(self.MaxIT):
            self.D_pl_pt(it)  # Update any auxiliary data structures (method not shown here)

            # Update positions of the first pd_number sparrows based on a time-varying factor
            if bm.random.rand(1) < st:
                self.x[0 : pd_number] = (self.x[0 : pd_number] * 
                                        bm.exp(-bm.arange(1, pd_number + 1)[:, None] / 
                                               (bm.random.rand(pd_number, 1) * self.MaxIT)))
            else:
                self.x[0 : pd_number] = (self.x[0 : pd_number] + 
                                        bm.random.randn(pd_number, 1) * bm.ones((pd_number, self.dim)))

            # Update positions of the remaining sparrows based on fitness-distance strategy
            self.x[pd_number : self.N] = ((bm.arange(pd_number, self.N)[:, None] > ((self.N - pd_number) / 2 + pd_number)) * 
                               (bm.random.randn(self.N - pd_number, 1) / 
                                bm.exp((self.x[self.N - 1] - self.x[pd_number : self.N]) / 
                                       (bm.arange(pd_number, self.N)[:, None] ** 2))) + 
                               (bm.arange(pd_number, self.N)[:, None] <= ((self.N - pd_number) / 2 + pd_number)) * 
                               (self.x[0] + 
                                (bm.where(bm.random.rand(self.N - pd_number, self.dim) < 0.5, -1, 1) / self.dim) * 
                                bm.abs(self.x[pd_number : self.N] - self.x[0]))) 
            
            # Update positions of sd_number sparrows based on fitness comparison with the global best
            sd_index = bm.random.randint(0, self.N, (sd_number,))
            self.x[sd_index] = ((fit[sd_index] > fit[0])[:, None] * 
                          (self.x[0] + bm.random.randn(sd_number, 1) * bm.abs(self.x[sd_index] - self.x[0])) + 
                          (fit[sd_index] == fit[0])[:, None] * 
                          (self.x[sd_index] + (2 * bm.random.rand(sd_number, 1) - 1) * 
                           (bm.abs(self.x[sd_index] - self.x[self.N - 1]) / 
                            (1e-8 + (fit[sd_index] - fit[self.N - 1])[:, None]))))
            
            # Ensure all positions stay within bounds
            self.x = self.x + (self.lb - self.x) * (self.x < self.lb) + (self.ub - self.x) * (self.x > self.ub)
            
            # Evaluate fitness of updated population
            fit = self.fun(self.x)

            # Sort the population by fitness again
            index = bm.argsort(fit)
            self.x = self.x[index]
            fit = fit[index]

            # Update the global best solution if a better one is found
            self.update_gbest(self.x, fit)

            # Track the fitness of the global best solution over iterations
            self.curve[it] = self.gbest_f
