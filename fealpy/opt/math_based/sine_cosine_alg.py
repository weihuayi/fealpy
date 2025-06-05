from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class SineCosineAlg(Optimizer):
    """
    A Sine Cosine Algorithm (SCA), inheriting from the Optimizer class.

    This class implements the Sine Cosine Algorithm, which uses sine and cosine functions to explore 
    and exploit the search space. It initializes with a set of options and iteratively improves the 
    solution through a combination of sine and cosine-based updates.

    Parameters:
        option: Configuration options for the optimizer, typically including parameters like population size, 
                maximum iterations, and bounds for the search space.

    Attributes:
        gbest (array): The best solution found during the optimization process.
        gbest_f (float): The fitness value of the best solution.
        curve (array): An array storing the best fitness value at each iteration.

    Methods:
        run(a=2): Executes the Sine Cosine Algorithm.
            Parameters:
                a (float): A control parameter that influences the amplitude of the sine and cosine functions, 
                           default is 2.
    """
    def __init__(self, option) -> None:
        """
        Initializes the SineCosineAlg optimizer with the given options.

        Parameters:
            option: Configuration options for the optimizer.
        """
        super().__init__(option)

    def run(self, params={'a':2}):
        """
        Runs the Sine Cosine Algorithm.

        Parameters:
            a (float): A control parameter that influences the amplitude of the sine and cosine functions, 
                       default is 2.
        """
        # Initialize fitness values and find the best solution in the initial population
        a = params.get('a')
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        # Iterate through the maximum number of iterations
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)

            # Calculate the dynamic parameter r1 for controlling the amplitude of sine and cosine functions
            r1 = a - it * a / self.MaxIT

            # Generate a random parameter r4 to decide between sine and cosine updates
            r4 = bm.random.rand(self.N, self.dim)

            # Update the population using sine and cosine functions
            self.x = (
                (r4 < 0.5) *
                (self.x + (r1 * bm.sin(2 * bm.pi * bm.random.rand(self.N, self.dim)) *
                           bm.abs(2 * bm.random.rand(self.N, self.dim) * self.gbest - self.x))) +
                (r4 >= 0.5) *
                (self.x + (r1 * bm.cos(2 * bm.pi * bm.random.rand(self.N, self.dim)) *
                           bm.abs(2 * bm.random.rand(self.N, self.dim) * self.gbest - self.x)))
            )

            # Ensure the new solutions stay within the bounds
            self.x = self.x + (self.lb - self.x) * (self.x < self.lb) + (self.ub - self.x) * (self.x > self.ub)

            # Evaluate the fitness of the new solutions
            fit = self.fun(self.x)

            # Update the best solution found so far
            self.update_gbest(self.x, fit)

            # Record the best fitness value at the current iteration
            self.curve[it] = self.gbest_f