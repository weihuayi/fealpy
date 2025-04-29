from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class ImprovedWhaleOptAlg(Optimizer):
    """
    Improved Whale Optimization Algorithm (IWOA) for optimization problems.
    
    This class implements the Improved Whale Optimization Algorithm (IWOA),
    an enhanced version of the standard Whale Optimization Algorithm (WOA), inspired by the
    bubble-net hunting strategy of humpback whales. The improvement introduces dynamic control
    of the exploration and exploitation phases, which enhances the algorithm's ability to find
    global optima in optimization problems.

    Attributes:
        gbest (bm.Tensor): The global best solution found so far.
        gbest_f (float): The fitness value of the global best solution.
        x (bm.Tensor): The population of candidate solutions.
        lb (bm.Tensor): The lower bounds for the variables.
        ub (bm.Tensor): The upper bounds for the variables.
        MaxIT (int): The maximum number of iterations for the algorithm.
        N (int): The size of the population.
        dim (int): The dimensionality of the problem.
        fun (function): The objective function to minimize.
        curve (bm.Tensor): The fitness values over iterations.

    Reference:
        Seyedali Mirjalili, Andrew Lewis.
        A novel improved whale optimization algorithm to solve numerical optimization and real-world applications.
        Artificial Intelligence Review, 2022, 55: 4605-4716.
    """

    def __init__(self, option) -> None:
        """
        Initializes the Improved Whale Optimization Algorithm with the given options.

        Args:
            option (dict): A dictionary containing parameters for the algorithm,
                           such as population size, bounds, and other configuration options.
        """
        super().__init__(option)

    def run(self):
        """
        Runs the Improved Whale Optimization Algorithm to optimize the objective function.
        
        The algorithm iterates through the population, updating the positions of the whales
        based on a combination of exploration and exploitation strategies. These strategies
        are controlled by the coefficients A, C, and p, which evolve over the iterations.
        
        Steps:
            1. Evaluate the fitness of the initial population.
            2. Perform the update of whale positions using dynamic control coefficients.
            3. Update the best solution found so far if a better solution is found.
            4. Track the fitness value of the best solution in each iteration.
            
        The process continues for the maximum number of iterations (MaxIT).
        """
        fit = self.fun(self.x)  # Evaluate the fitness of the current population.
    
        gbest_index = bm.argmin(fit)  # Find the index of the best solution.
        self.gbest = self.x[gbest_index]  # Set the global best solution.
        self.gbest_f = fit[gbest_index]  # Set the fitness of the global best solution.

        # Iterate through each iteration of the algorithm.
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)  # Update the position of the whales based on exploration/exploitation.

            # Coefficients for exploration and exploitation.
            A = 2 * (2 - it * 2 / self.MaxIT) * bm.random.rand(self.N, 1) - (2 - it * 2 / self.MaxIT)  # Eq.(3)
            C = 2 * bm.random.rand(self.N, 1)  # Eq.(4)
            p = bm.random.rand(self.N, 1)  # Probability for determining exploration or exploitation.
            l = (-2 - it / self.MaxIT) * bm.random.rand(self.N, 1) + 1  # Eq.(9)

            if it < int(self.MaxIT / 2):  # Exploration phase
                # Randomly select two whales from the population.
                rand1 = self.x[bm.random.randint(0, self.N, (self.N,))]
                rand2 = self.x[bm.random.randint(0, self.N, (self.N,))]
                rand_mean = (rand1 + rand2) / 2  # Compute the mean of the two random whales.
                
                # Select the whale based on the distance to rand1 and rand2.
                mask = bm.linalg.norm(self.x - rand1, axis=1)[:, None] < bm.linalg.norm(self.x - rand2, axis=1)[:, None]
                rand = bm.where(mask, rand2, rand1)
                
                # Update the position of the whale using Eq.(2).
                x_new = ((p < 0.5) * (rand - A * bm.abs(C * rand - self.x)) + 
                         (p >= 0.5) * (rand_mean - A * bm.abs(C * rand_mean - self.x)))
            else:  # Exploitation phase
                # Update the position of the whale using Eq.(6) and Eq.(8).
                x_new = ((p < 0.5) * (self.gbest - A * bm.abs(C * self.gbest - self.x)) +  # Eq.(6)
                         (p >= 0.5) * (bm.abs(self.gbest - self.x) * bm.exp(l) * bm.cos(l * 2 * bm.pi) + self.gbest))  # Eq.(8)
            
            # Apply boundary constraints to ensure the new position is within bounds.
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)
            
            # Evaluate the fitness of the new positions.
            fit_new = self.fun(x_new)
            
            # Update the population if the new solution has a better fitness.
            mask = (fit_new < fit)
            self.x = bm.where(mask[:, None], x_new, self.x)
            fit = bm.where(mask, fit_new, fit)
            
            # Update the global best solution if a better one is found.
            self.update_gbest(self.x, fit)
            
            # Record the best fitness value for this iteration.
            self.curve[it] = self.gbest_f
