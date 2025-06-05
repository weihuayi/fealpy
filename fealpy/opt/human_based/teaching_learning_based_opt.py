from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class TeachingLearningBasedAlg(Optimizer):
    """
    A Teaching-Learning-Based Optimization (TLBO) algorithm, inheriting from the Optimizer class.

    This class implements the Teaching-Learning-Based Optimization algorithm, which simulates the teaching and learning 
    process in a classroom. It initializes with a set of options and iteratively improves the solution through 
    teacher and learner phases.

    Parameters:
        option: Configuration options for the optimizer, typically including parameters like population size, 
                maximum iterations, and bounds for the search space.

    Attributes:
        gbest (array): The best solution found during the optimization process.
        gbest_f (float): The fitness value of the best solution.
        curve (array): An array storing the best fitness value at each iteration.

    Methods:
        run(): Executes the Teaching-Learning-Based Optimization algorithm.
    """
    def __init__(self, option) -> None:
        """
        Initializes the TeachingLearningBasedAlg optimizer with the given options.

        Parameters:
            option: Configuration options for the optimizer.
        """
        super().__init__(option)

    def run(self):
        """
        Runs the Teaching-Learning-Based Optimization algorithm.
        """
        # Initialize fitness values and find the best solution in the initial population
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        # Iterate through the maximum number of iterations
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)

            # Teacher phase: Update solutions based on the teacher (best solution)
            TF = bm.round(1 + bm.random.rand(self.N, self.dim))
            x_new = self.x + bm.random.rand(self.N, self.dim) * (self.gbest - TF * bm.mean(self.x, axis=0))
            fit_new = self.fun(x_new)
            mask = fit_new < fit
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)

            # Learner phase: Update solutions based on interactions between learners
            k = bm.random.randint(0, self.N, (self.N,))
            x_new = (
                (fit < fit[k])[:, None] * (self.x + bm.random.rand(self.N, self.dim) * (self.x - self.x[k])) +
                (fit >= fit[k])[:, None] * (self.x + bm.random.rand(self.N, self.dim) * (self.x[k] - self.x))
            )
            fit_new = self.fun(x_new)
            mask = fit_new < fit
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)

            # Update the best solution found so far
            self.update_gbest(self.x, fit)

            # Record the best fitness value at the current iteration
            self.curve[it] = self.gbest_f