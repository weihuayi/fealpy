from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class StarFishOptAlg(Optimizer):
    """
    A class implementing the Starfish Optimization Algorithm (SFOA), inheriting from the Optimizer class.

    The Starfish Optimization Algorithm is inspired by the movement patterns and behaviors of starfish. It includes 
    exploration and exploitation phases to optimize a given objective function. The algorithm employs unique strategies 
    such as position updates with sine and cosine functions, random selection of dimensions, and iterative refinement 
    of the global best solution.

    Parameters:
        option (dict): A dictionary containing configuration options for the optimization process, 
                       such as maximum iterations, population size, lower and upper bounds, etc.

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
        Changting Zhong, Gang Li, Zeng Meng, Haijiang Li, Ali Riza Yildiz, Seyedali Mirjalili.
        Starfish optimization algorithm (SFOA): a bio-inspired metaheuristic algorithm for global optimization compared with 100 optimizers.
        Neural Computing and Applications, 2024.
    """

    def __init__(self, option) -> None:
        """
        Initializes the Starfish Optimization Algorithm with the provided options.

        Parameters:
            option (dict): Configuration options for the optimization process, such as maximum iterations, bounds, etc.
        """
        super().__init__(option)
        
    def run(self):
        """
        Runs the Starfish Optimization Algorithm.

        This method performs the iterative optimization process, where the positions of candidate solutions are updated 
        based on exploration and exploitation behaviors. In each iteration, the algorithm explores new positions by using 
        sine/cosine functions and refines solutions toward the global best solution.

        Key steps in each iteration:
            1. Evaluate the fitness values of the current population.
            2. Update positions of solutions based on exploration and exploitation strategies:
                - Exploration: Use random movements and transformations to explore the search space.
                - Exploitation: Refine solutions by moving toward the global best or based on random selections of solution 
                  dimensions.
            3. Ensure the new positions stay within the bounds of the solution space.
            4. Evaluate the fitness of the new population.
            5. Update the global best solution if a better one is found.
            6. Track the fitness of the global best solution over iterations.
        """
        # Initial fitness evaluation
        fit = self.fun(self.x)

        # Identify the index of the best solution
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]
        
        # Initialize new solution matrix
        x_new = bm.zeros((self.N, self.dim))

        # Iterative optimization process
        for it in range(self.MaxIT):
            self.D_pl_pt(it)  # Update any auxiliary data structures (method not shown here)
            
            if bm.random.rand(1) > 0.5:
                if self.dim > 5:
                    # Exploration phase with high-dimensional solutions
                    j = bm.random.randint(0, self.dim, (self.N, self.dim))
                    r1 = bm.random.rand(self.N, 1)
                    
                    # Update positions with random dimensions based on sine/cosine transformations
                    x_new[bm.arange(self.N)[:, None], j] = (
                        (r1 > 0.5) * (self.x[bm.arange(self.N)[:, None], j] + 
                                      (2 * r1 - 1) * bm.pi  * 
                                      bm.cos(bm.array(bm.pi * it / (2 * self.MaxIT))) * 
                                      ((self.gbest[None, :] * bm.ones((self.N, self.dim)))[bm.arange(self.N)[:, None], j] - 
                                      self.x[bm.arange(self.N)[:, None], j])) + 
                        (r1 <= 0.5) * (self.x[bm.arange(self.N)[:, None], j] - 
                                       (2 * r1 - 1) * bm.pi * 
                                       bm.sin(bm.array(bm.pi * it / (2 * self.MaxIT))) * 
                                       ((self.gbest[None, :] * bm.ones((self.N, self.dim)))[bm.arange(self.N)[:, None], j] - 
                                       self.x[bm.arange(self.N)[:, None], j]))
                    )
                    
                    # Ensure the new positions stay within bounds
                    self.x[bm.arange(self.N)[:, None], j] = (
                        x_new[bm.arange(self.N)[:, None], j] + 
                        (x_new[bm.arange(self.N)[:, None], j] > self.ub) * 
                        (self.x[bm.arange(self.N)[:, None], j] - x_new[bm.arange(self.N)[:, None], j]) + 
                        (x_new[bm.arange(self.N)[:, None], j] < self.lb) * 
                        (self.x[bm.arange(self.N)[:, None], j] - x_new[bm.arange(self.N)[:, None], j])
                    )
                else:
                    # Update positions for lower-dimensional solutions
                    j = bm.random.randint(0, self.dim, (self.N, 1))
                    x_new[bm.arange(self.N)[:, None], j] = (
                        ((self.MaxIT - it) / self.MaxIT) * 
                        bm.cos(bm.array(bm.pi * it / (2 * self.MaxIT))) * 
                        self.x[bm.arange(self.N)[:, None], j] + 
                        (2 * bm.random.rand(self.N, 1) - 1) * 
                        (self.x[bm.random.randint(0, self.N, (self.N, 1)), j] - 
                         self.x[bm.arange(self.N)[:, None], j]) + 
                        (2 * bm.random.rand(self.N, 1) - 1) * 
                        (self.x[bm.random.randint(0, self.N, (self.N, 1)), j] - 
                         self.x[bm.arange(self.N)[:, None], j])
                    )

                    # Ensure the new positions stay within bounds
                    self.x[bm.arange(self.N)[:, None], j] = (
                        x_new[bm.arange(self.N)[:, None], j] + 
                        (x_new[bm.arange(self.N)[:, None], j] > self.ub) * 
                        (self.x[bm.arange(self.N)[:, None], j] - x_new[bm.arange(self.N)[:, None], j]) + 
                        (x_new[bm.arange(self.N)[:, None], j] < self.lb) * 
                        (self.x[bm.arange(self.N)[:, None], j] - x_new[bm.arange(self.N)[:, None], j])
                    )
            else:
                # Exploitation phase: refine the solutions toward the global best
                dm = self.gbest - self.x[bm.random.randint(0, self.N - 1, (5,))]
                x_new[0 : self.N - 1] = (
                    self.x[0 : self.N - 1] + 
                    bm.random.rand(self.N - 1, self.dim) * 
                    dm[bm.random.randint(0, 5, (self.N - 1,))] + 
                    bm.random.rand(self.N - 1, self.dim) * 
                    dm[bm.random.randint(0, 5, (self.N - 1,))])
                
                # Special update for the last element
                x_new[self.N - 1] = self.x[self.N - 1] * bm.exp(bm.array(-it * self.N / self.MaxIT))

                # Ensure the new positions stay within bounds
                self.x = x_new + (x_new > self.ub) * (self.ub - x_new) + (x_new < self.lb) * (self.lb - x_new)

            # Evaluate the fitness of the new population
            fit = self.fun(self.x)

            # Update the global best solution
            self.update_gbest(self.x, fit)

            # Track the fitness of the global best solution
            self.curve[it] = self.gbest_f
