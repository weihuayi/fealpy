from ..opt_function import levy
from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class SquirrelSearchAlg(Optimizer):
    """
    A class implementing the Squirrel Search Algorithm (SSA), inheriting from the Optimizer class.

    The Squirrel Search Algorithm is a population-based optimization method inspired by the behavior of squirrels in 
    searching for food. The algorithm updates positions based on the current best solutions, with random movements 
    influenced by the fitness landscape.

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
        A novel nature-inspired algorithm for optimization: Squirrel search algorithm.
        Swarm and Evolutionary Computation, 2019, 44: 148-175
    """
    
    def __init__(self, option) -> None:
        """
        Initializes the Squirrel Search Algorithm with the provided options.

        Parameters:
            option (dict): Configuration options for the optimization process, such as maximum iterations, bounds, etc.
        """
        super().__init__(option)

    def run(self, params={'g_c':1.9, 'p_dp':0.1}):
        """
        Runs the Squirrel Search Algorithm.

        This method performs the iterative optimization process, where the positions of candidate solutions are updated 
        based on random movements and interactions with the best-known solutions. The algorithm explores and exploits 
        the solution space by considering the global best solution and nearby candidates.

        Key steps in each iteration:
            1. Evaluate the fitness values of the current population.
            2. Update the positions of the top-performing solutions based on certain conditions.
            3. Use random movements to update other solutions, either exploring the space or refining their position.
            4. Update the global best solution if a better one is found.
            5. Track the fitness of the global best solution over iterations.
        """
        # Initial fitness evaluation
        g_c = params.get('g_c')
        p_dp = params.get('p_dp')
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest_f = fit[gbest_index]

        # Sort population by fitness to get the best solution
        index = bm.argsort(fit)
        self.gbest = self.x[index[0]]  # Global best solution
        FSa = self.x[index[1:4]]  # The second to fourth best solutions (for further refinement)
        FSn = self.x[index[4: self.N]]  # The remaining solutions for exploration

        # Iterative optimization process
        for it in range(self.MaxIT):
            self.D_pl_pt(it)  # Update any auxiliary data structures (method not shown here)

            # Random selection of a subset of solutions for exploration
            n2 = bm.random.randint(4, self.N, (1,))
            index2 = bm.unique(bm.random.randint(0, self.N - 4, (n2[0],)))
            index3 = bm.array(list(set(bm.arange(self.N - 4).tolist()).difference(index2.tolist())))

            # Length of the subsets
            n2 = len(index2)
            n3 = self.N - n2 - 4

            # Exploration phase for the second-to-fourth best solutions
            dg = 0.5 + (1.11 - 0.5) * bm.random.rand(3, 1)
            r1 = bm.random.rand(3, 1)
            FSa = ((r1 > p_dp) * (FSa + dg * g_c * (self.gbest - FSa)) + 
                   (r1 <= p_dp) * (self.lb + (self.ub - self.lb) * bm.random.rand(3, self.dim)))
            FSa = FSa + (self.lb - FSa) * (FSa < self.lb) + (self.ub - FSa) * (FSa > self.ub)

            # Exploration phase for other solutions
            r2 = bm.random.rand(n2, 1)
            dg = 0.5 + (1.11 - 0.5) * bm.random.rand(n2, 1)
            t = bm.random.randint(0, 3, (n2,))
            FSn[index2] = ((r2 > p_dp) * (FSn[index2] + dg * g_c * (FSa[t] - FSn[index2])) + 
                           (r2 <= p_dp) * (self.lb + (self.ub - self.lb) * bm.random.rand(n2, self.dim)))

            # Update for the remaining solutions (exploitation phase)
            r3 = bm.random.rand(n3, 1)
            dg = 0.5 + (1.11 - 0.5) * bm.random.rand(n3, 1)
            FSn[index3] = ((r3 > p_dp) * (FSn[index3] + dg * g_c * (self.gbest - FSn[index3])) + 
                           (r3 <= p_dp) * (self.lb + (self.ub - self.lb) * bm.random.rand(n3, self.dim)))

            # Ensure the new positions stay within bounds
            FSn = FSn + (self.lb - FSn) * (FSn < self.lb) + (self.ub - FSn) * (FSn > self.ub)

            # Special case where solutions are too close to the global best
            Sc = bm.sum((FSa - self.gbest) ** 2)
            Smin = 10 * bm.exp(bm.array(-6)) / 365 ** (it / (self.MaxIT / 2.5))
            if Sc < Smin:
                FSn = FSn + 0.01 * levy(self.N - 4, self.dim, 1.5) * (self.ub - self.lb)
                FSn = FSn + (self.lb - FSn) * (FSn < self.lb) + (self.ub - FSn) * (FSn > self.ub)

            # Concatenate the updated solutions
            self.x = bm.concatenate((self.gbest[None, :], FSa, FSn), axis=0)

            # Evaluate the fitness of the updated population
            fit = self.fun(self.x)

            # Sort the population by fitness
            index = bm.argsort(fit)
            gbest_mew = self.x[index[0]]
            gbest_f_mew = fit[index[0]]

            # Update the global best solution if a better one is found
            if gbest_f_mew < self.gbest_f:
                self.gbest = gbest_mew
                self.gbest_f = gbest_f_mew

            # Update the second-to-fourth best solutions and the remaining solutions
            FSa = self.x[index[1:4]]
            FSn = self.x[index[4: self.N]]

            # Track the fitness of the global best solution
            self.curve[it] = self.gbest_f
