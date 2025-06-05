from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class JellyfishSearchOpt(Optimizer):
    """
    A class implementing the Jellyfish Search Optimization Algorithm (JSO) for solving optimization problems.

    The Jellyfish Search Optimization algorithm is inspired by the hunting and searching behaviors of jellyfish.
    The algorithm involves two main strategies:
    - Exploration: Jellyfish search the space randomly or with guidance from the best-found solutions.
    - Exploitation: Jellyfish adjust their search based on the fitness values of the population.

    Parameters:
        option (dict): A dictionary containing parameters for the optimizer such as population size, dimensionality, and iteration limits.

    Attributes:
        gbest (Tensor): The global best solution found by the algorithm.
        gbest_f (float): The fitness value of the global best solution.
        curve (Tensor): A tensor to store the progress of the best solution over iterations.

    Methods:
        run(): Main method to run the Jellyfish Search Optimization algorithm and update the population based on fitness values.

    Reference:
        Jui-Sheng Chou, Dinh-Nhat Truong.
        A novel metaheuristic optimizer inspired by behavior of jellyfish in ocean.
        Applied Mathematics and Computation, 2021, 389: 125535.
    """

    def __init__(self, option) -> None:
        """
        Initializes the Jellyfish Search Optimization algorithm with the provided options.

        Parameters:
            option (dict): A dictionary containing optimizer settings.
        """
        super().__init__(option)
        
    def run(self):
        """
        Runs the Jellyfish Search Optimization algorithm to find the optimal solution.

        This method iteratively updates the population based on exploration and exploitation strategies.
        It adjusts the search space based on the fitness of solutions and updates the global best solution.

        The algorithm operates as follows:
        - Exploration: If a random coefficient `c` is greater than or equal to 0.5, the search moves towards the global best solution.
        - Exploitation: Otherwise, the search moves based on the relative fitness of solutions, balancing exploration and exploitation.

        The solutions are adjusted to stay within the bounds defined by the lower and upper bounds (self.lb, self.ub).
        """
        
        # Initial fitness and global best solution
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        # Iteration loop
        for it in range(self.MaxIT):
            self.D_pl_pt(it)  # Update jellyfish positions

            # Exploration-exploitation coefficient
            c = (1 - it / self.MaxIT) * (2 * bm.random.rand(1) - 1)
            
            # Exploration phase: jellyfish moves towards the global best solution with some randomness
            if c >= 0.5:
                x_new = self.x + (bm.random.rand(self.N, 1) * 
                                  (self.gbest - 3 * bm.random.rand(self.N, 1) * bm.mean(self.x, axis=0)))
            else:
                # Exploitation phase: jellyfish moves based on the relative fitness of solutions
                r = bm.random.rand(self.N, 1)

                # Direction of movement based on relative fitness of solutions
                rand_index = bm.random.randint(0, self.N, (self.N,))
                direction = (((fit[rand_index]) <= fit)[:, None] * (self.x[rand_index] - self.x) + 
                             ((fit[rand_index]) > fit)[:, None] * (self.x - self.x[rand_index]))

                # Adjust the movement based on the probability factor `r`
                x_new = ((r > (1 - c)) * (self.x + 0.1 * bm.random.rand(self.N, self.dim) * (self.ub - self.lb)) + 
                        (r <= (1 - c)) * (self.x + bm.random.rand(self.N, self.dim) * direction))
            
            # Apply boundary constraints
            x_new = x_new + (((self.ub + x_new - self.lb) - x_new) * (x_new < self.lb) + 
                             ((x_new - self.ub + self.lb) - x_new) * (x_new > self.ub))

            # Evaluate fitness of new solutions
            fit_new = self.fun(x_new)
            mask = fit_new < fit 
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)
            self.update_gbest(self.x, fit)

            # Store the best fitness value of this iteration
            self.curve[it] = self.gbest_f
