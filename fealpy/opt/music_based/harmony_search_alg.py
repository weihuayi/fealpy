from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer
from ..opt_function import initialize

class HarmonySearchAlg(Optimizer):
    """
    A Harmony Search Algorithm (HSA), inheriting from the Optimizer class.

    This class implements the Harmony Search Algorithm, which is inspired by the process of musical improvisation.
    It initializes with a set of options and iteratively improves the solution through harmony memory consideration,
    pitch adjustment, and randomization.

    Parameters:
        option: Configuration options for the optimizer, typically including parameters like population size, 
                maximum iterations, and bounds for the search space.

    Attributes:
        gbest (array): The best solution found during the optimization process.
        gbest_f (float): The fitness value of the best solution.
        curve (array): An array storing the best fitness value at each iteration.

    Methods:
        run(): Executes the Harmony Search Algorithm.

    Reference:
        Zong Woo Geem, Joong Hoon Kim, G.V. Loganathan.
        A New Heuristic Optimization Algorithm: Harmony Search.
        Simulation, 2001, 76: 60-68.
    """
    def __init__(self, option) -> None:
        """
        Initializes the HarmonySearchAlg optimizer with the given options.

        Parameters:
            option: Configuration options for the optimizer.
        """
        super().__init__(option)

    def run(self):
        """
        Runs the Harmony Search Algorithm.
        """
        # Initialize fitness values and find the best solution in the initial population
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        # Initialize algorithm parameters
        FW = 0.02 * (self.ub - self.lb)  # Fret width (FW) for pitch adjustment
        nNew = int(0.8 * self.N)  # Number of new harmonies to generate
        HMCR = 0.9  # Harmony Memory Consideration Rate
        PAR = 0.1  # Pitch Adjustment Rate
        FW_damp = 0.995  # Damping factor for fret width

        # Sort the population based on fitness
        index = bm.argsort(fit)
        self.x = self.x[index]
        fit = fit[index]

        # Iterate through the maximum number of iterations
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)

            # Generate new harmonies
            x_new = initialize(nNew, self.dim, self.lb, self.ub)  # Initialize new harmonies
            mask = bm.random.rand(nNew, self.dim) <= HMCR  # Harmony Memory Consideration
            b = self.x[bm.random.randint(0, self.N, (nNew, self.dim)), bm.arange(self.dim)]  # Random selection from harmony memory
            x_new = mask * b + ~mask * x_new  # Combine new and existing harmonies

            # Pitch adjustment
            x_new = x_new + FW * bm.random.randn(nNew, self.dim) * (bm.random.rand(nNew, self.dim) <= PAR)

            # Ensure the new harmonies stay within the bounds
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)

            # Evaluate the fitness of the new harmonies
            fit_new = self.fun(x_new)

            # Combine the current population and the new harmonies
            self.x = bm.concatenate((self.x, x_new), axis=0)
            fit = bm.concatenate((fit, fit_new))

            # Select the best harmonies for the next generation
            index = bm.argsort(fit)
            self.x = self.x[index[0:self.N]]
            fit = fit[index[0:self.N]]

            # Update the best solution found so far
            self.update_gbest(self.x, fit)

            # Damp the fret width for the next iteration
            FW = FW * FW_damp

            # Record the best fitness value at the current iteration
            self.curve[it] = self.gbest_f