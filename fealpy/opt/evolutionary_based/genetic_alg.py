from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class GeneticAlg(Optimizer):
    """
    A Genetic Algorithm (GA) optimization class, inheriting from the Optimizer class.

    This class implements the Genetic Algorithm, a population-based optimization method inspired by natural selection.
    It initializes with a set of options and iteratively improves the solution through selection, crossover, and mutation operations.

    Parameters:
        option: Configuration options for the optimizer, typically including parameters like population size, 
                maximum iterations, and bounds for the search space.

    Attributes:
        gbest (array): The best solution found during the optimization process.
        gbest_f (float): The fitness value of the best solution.
        curve (array): An array storing the best fitness value at each iteration.

    Methods:
        run(pc=0.7, pm=0.01): Executes the Genetic Algorithm.
            Parameters:
                pc (float): The crossover probability, default is 0.7.
                pm (float): The mutation probability, default is 0.01.
    """
    def __init__(self, option) -> None:
        """
        Initializes the GeneticAlg optimizer with the given options.

        Parameters:
            option: Configuration options for the optimizer.
        """
        super().__init__(option)

    def run(self, params={'pc':0.7, 'pm':0.01}):
        """
        Runs the Genetic Algorithm.

        Parameters:
            pc (float): The crossover probability, default is 0.7.
            pm (float): The mutation probability, default is 0.01.
        """
        # Initialize fitness values and find the best solution in the initial population
        pc = params.get('pc')
        pm = params.get('pm')
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        # Sort the population based on fitness
        index = bm.argsort(fit)
        self.x = self.x[index]
        fit = fit[index]

        # Iterate through the maximum number of iterations
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)

            # Crossover: Generate new solutions based on crossover probability
            c_num = (bm.random.rand(self.N, self.dim) > pc)
            NF = bm.ones((self.N, self.dim)) * self.gbest
            new = c_num * NF + ~c_num * self.x

            # Mutation: Apply mutation to the new solutions based on mutation probability
            mum = bm.random.rand(self.N, self.dim) < pm
            new = mum * (self.lb + (self.ub - self.lb) * bm.random.rand(self.N, self.dim)) + ~mum * new

            # Evaluate the fitness of the new solutions
            new_f = self.fun(new)

            # Combine the current population and the new solutions
            all = bm.concatenate((self.x, new), axis=0)
            all_f = bm.concatenate((fit, new_f))

            # Select the best solutions for the next generation
            index = bm.argsort(all_f, axis=0)
            fit = all_f[index[0:self.N]]
            self.x = all[index[0:self.N]]

            # Update the best solution found so far
            self.update_gbest(self.x, fit)

            # Record the best fitness value at the current iteration
            self.curve[it] = self.gbest_f