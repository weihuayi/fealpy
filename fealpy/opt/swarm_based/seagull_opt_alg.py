from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class SeagullOptAlg(Optimizer):
    """
    A class implementing the Seagull Optimization Algorithm (SOA), inheriting from the Optimizer class.

    This algorithm simulates the foraging behavior of seagulls, where their movements are influenced by the global best solution
    and a combination of randomness and attraction. The seagulls explore the search space by adjusting their movement based on
    factors such as the global best solution and random perturbations.

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
        Gaurav Dhiman, Vijay Kumar.
        Seagull optimization algorithm: Theory and its applications for large-scale industrial engineering problems.
        Knowledge-Based Systems, 2019, 165: 169-196.
    """
    
    def __init__(self, option) -> None:
        """
        Initializes the Seagull Optimization Algorithm with the provided options.

        Parameters:
            option (dict): Configuration options for the optimization process, such as maximum iterations, bounds, etc.
        """
        super().__init__(option)

    def run(self, params={'fc':2, 'u':1, 'v':1}):
        """
        Runs the Seagull Optimization Algorithm.

        The method simulates the foraging behavior of seagulls. Each seagull adjusts its position based on the global best solution,
        as well as a series of random factors that encourage exploration. The movement formula involves attraction to the global
        best, along with random perturbations based on the parameters fc, u, and v.

        Key steps in each iteration:
            1. Evaluate the fitness of the current population.
            2. Update the positions of the seagulls based on the movement formula.
            3. Apply boundary conditions to ensure all positions stay within the feasible solution space.
            4. Track the global best solution and update it if a better solution is found.
        """
        # Initial fitness evaluation
        fc = params.get('fc')
        u = params.get('u')
        v = params.get('v')
        fit = self.fun(self.x)
        
        # Identify the global best solution and its fitness
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        for it in range(self.MaxIT):
            self.D_pl_pt(it)  # Update any auxiliary data structures (method not shown here)

            # Update positions based on a movement formula inspired by seagull foraging behavior
            self.x = (bm.abs((fc - (it * fc / self.MaxIT)) * self.x + 
                             2 * ((fc - (it * fc / self.MaxIT)) ** 2) * bm.random.rand(self.N, 1) * (self.gbest - self.x)) * 
                      u * bm.exp(v * bm.random.rand(self.N, 1) * 2 * bm.pi) * bm.cos(bm.random.rand(self.N, 1) * 2 * bm.pi) * 
                      u * bm.exp(v * bm.random.rand(self.N, 1) * 2 * bm.pi) * bm.sin(bm.random.rand(self.N, 1) * 2 * bm.pi) * 
                      u * bm.exp(v * bm.random.rand(self.N, 1) * 2 * bm.pi) * bm.random.rand(self.N, 1) * 2 * bm.pi + 
                      self.gbest)  # Eq.(14)

            # Apply boundary conditions to ensure positions are within the solution space
            self.x = self.x + (self.lb - self.x) * (self.x < self.lb) + (self.ub - self.x) * (self.x > self.ub)

            # Evaluate fitness of the updated population
            fit = self.fun(self.x)

            # Update the global best solution if a better one is found
            self.update_gbest(self.x, fit)

            # Track the fitness of the global best solution over iterations
            self.curve[it] = self.gbest_f
