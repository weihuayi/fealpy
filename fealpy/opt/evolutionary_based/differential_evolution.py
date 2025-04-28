from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class DifferentialEvolution(Optimizer):
    """
    A differential evolution (DE) optimization algorithm, inheriting from the Optimizer class.

    This class implements the differential evolution algorithm, a population-based optimization method 
    commonly used for global optimization problems. It initializes with a set of options and iteratively 
    improves the solution through mutation, crossover, and selection operations.

    Parameters:
        option: Configuration options for the optimizer, typically including parameters like population size, 
                maximum iterations, and bounds for the search space.

    Attributes:
        gbest (array): The best solution found during the optimization process.
        gbest_f (float): The fitness value of the best solution.
        curve (array): An array storing the best fitness value at each iteration.

    Methods:
        run(F=0.2, CR=0.5): Executes the differential evolution algorithm.
            Parameters:
                F (float): The mutation factor, controlling the amplification of the differential variation.
                CR (float): The crossover probability, determining the likelihood of crossover between solutions.
    """
    def __init__(self, option) -> None:
        """
        Initializes the DifferentialEvolution optimizer with the given options.

        Parameters:
            option: Configuration options for the optimizer.
        """
        super().__init__(option)

    def run(self, params={'f':0.2, 'cr':0.5}):
        """
        Runs the differential evolution algorithm.

        Parameters:
            F (float): The mutation factor, default is 0.2.
            CR (float): The crossover probability, default is 0.5.
        """
        # Initialize fitness values and find the best solution in the initial population
        f = params.get('f')
        cr = params.get('cr')
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        # Iterate through the maximum number of iterations
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)

            # Mutation: Generate a mutant vector based on random individuals
            v = self.x[bm.random.randint(0, self.N, (self.N,))] + f * (
                self.x[bm.random.randint(0, self.N, (self.N,))] - 
                self.x[bm.random.randint(0, self.N, (self.N,))]
            )

            # Crossover: Combine the mutant vector with the current population
            mask = bm.random.rand(self.N, self.dim) < cr
            x_new = bm.where(mask, v, self.x)

            # Boundary handling: Ensure the new solutions stay within the bounds
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)

            # Evaluation: Calculate the fitness of the new solutions
            fit_new = self.fun(x_new)

            # Selection: Replace the current population with better solutions
            mask = fit_new < fit
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)
            self.update_gbest(self.x, fit)

            # Record the best fitness value at the current iteration
            self.curve[it] = self.gbest_f