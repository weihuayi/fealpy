from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class HoneybadgerAlg(Optimizer):
    """
    A Honey Badger Algorithm (HBA) optimization class, inheriting from the Optimizer class.

    This class implements the Honey Badger Algorithm, a nature-inspired optimization method 
    that mimics the foraging behavior of honey badgers. It initializes with a set of options 
    and iteratively improves the solution through exploration and exploitation phases.

    Parameters:
        option: Configuration options for the optimizer, typically including parameters like 
                population size, maximum iterations, and bounds for the search space.

    Reference
        Fatma A. Hashim, Essam H. Houssein, Kashif Hussain, Mai S. Mabrouk, Walid Al-Atabany.
        Honey Badger Algorithm: New metaheuristic algorithm for solving optimization problems.
        Mathematics and Computers in Simulation, 2022, 192: 84-110. 
    """
    def __init__(self, option) -> None:
        """
        Initializes the HoneybadgerAlg optimizer with the given options.

        Parameters:
            option: Configuration options for the optimizer.
        """
        super().__init__(option)

    def run(self, params={'c':2, 'beta':6}):
        """
        Runs the Honey Badger Algorithm.

        Parameters:
            C (float): A control parameter for the exploration phase, default is 2.
            beta (float): A control parameter for the exploitation phase, default is 6.
        """
        # Initialize fitness values and find the best solution in the initial population
        c = params.get('c')
        beta = params.get('beta')
        fit = self.fun(self.x)
        gbest_idx = bm.argmin(fit)
        self.gbest_f = fit[gbest_idx]
        self.gbest = self.x[gbest_idx]

        # A small constant to avoid division by zero
        eps = 2.2204e-16

        # Iterate through the maximum number of iterations
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)

            # Calculate the dynamic parameter alpha for exploration and exploitation
            alpha = c * bm.exp(bm.array(-it / self.MaxIT))

            # Calculate the distance vector between the best solution and the current population
            di = self.gbest - self.x + eps

            # Calculate the intensity factor S based on the distance between neighboring solutions
            s = (self.x - bm.concatenate((self.x[1:], self.x[0:1]))) ** 2

            # Generate random values for intensity and direction
            r2 = bm.random.rand(self.N, 1)
            I = r2 * s / (4 * bm.pi * (di ** 2))
            F = bm.where(bm.random.rand(self.N, 1) < 0.5, bm.array(1), bm.array(-1))
            r = bm.random.rand(self.N, 1)

            # Update the population based on exploration and exploitation rules
            x_new = (
                (r < 0.5) * 
                (self.gbest + F * beta * I * self.gbest + F * bm.random.rand(self.N, self.dim) * alpha * di * 
                 bm.abs(bm.cos(2 * bm.pi * bm.random.rand(self.N, self.dim)) * 
                              (1 - bm.cos(2 * bm.pi * bm.random.rand(self.N, self.dim))))) + 
                (r >= 0.5) * 
                (self.gbest + F * bm.random.rand(self.N, self.dim) * alpha * di)
            )

            # Ensure the new solutions stay within the bounds
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)

            # Evaluate the fitness of the new solutions
            fit_new = self.fun(x_new)

            # Select better solutions to update the population
            mask = fit_new < fit
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)

            # Update the best solution found so far
            self.update_gbest(self.x, fit)

            # Record the best fitness value at the current iteration
            self.curve[it] = self.gbest_f