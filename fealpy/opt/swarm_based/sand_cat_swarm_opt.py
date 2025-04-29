from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class SandCatSwarmOpt(Optimizer):
    """
    SandCat Swarm Optimization algorithm (SCSO), inspired by particle swarm optimization (PSO).
    
    This algorithm simulates particles moving in the search space, adjusting their positions 
    based on the global best solution and random factors that guide exploration. 
    
    Parameters:
        option (dict): Configuration options for the optimization process, such as maximum iterations, population size, bounds, etc.
    
    Attributes:
        gbest (Tensor): The global best solution found so far.
        gbest_f (float): The fitness value corresponding to the global best solution.
        MaxIT (int): The maximum number of iterations for the optimization process.
        lb (Tensor): Lower bounds of the solution space.
        ub (Tensor): Upper bounds of the solution space.
        N (int): The size of the population.
        dim (int): The dimensionality of the search space.
        fun (callable): The objective (fitness) function to evaluate solutions.
        x (Tensor): The current population of candidate solutions.
        curve (dict): A dictionary to store the fitness values of the global best solution at each iteration.

    Reference:
        Amir Seyyedabbasi, Farzad Kiani.
        Sand Cat swarm optimization: a nature-inspired algorithm to solve global optimization problems.
        Engineering with Computers, 2023, 39: 2627-2651.
    """
    
    def __init__(self, option) -> None:
        """
        Initializes the SandCat Swarm Optimization algorithm with the provided options.
        
        Parameters:
            option (dict): Configuration options for the optimization process.
        """
        super().__init__(option)
        
    def run(self):
        """
        Runs the SandCat Swarm Optimization algorithm. Particles adjust their positions based on attraction
        to the global best solution and random factors. 

        Key steps in each iteration:
            1. Evaluate the fitness of the current population.
            2. Update particle positions based on attraction to the global best solution and random perturbations.
            3. Apply boundary conditions to ensure all positions stay within the feasible solution space.
            4. Track the global best solution and update it if a better solution is found.
        """
        # Initial fitness evaluation
        fit = self.fun(self.x)
        
        # Identify the global best solution and its fitness
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        # Start optimization iterations
        for it in range(self.MaxIT):
            self.D_pl_pt(it)  # Update any auxiliary data structures (method not shown here)

            # Calculate movement parameters for the particles
            rg = 2 - 2 * it / self.MaxIT  # Global repulsion factor (Eq. 1)
            R = 2 * rg * bm.random.rand(self.N, 1) - rg  # Random component for movement (Eq. 2)
            r = rg * bm.random.rand(self.N, 1)  # Random scaling factor (Eq. 3)
            theta = 2 * bm.pi * bm.random.rand(self.N, 1)  # Random direction angle

            # Update positions based on the movement formulas
            self.x = ((bm.abs(R) <= 1) * (self.gbest - 
                                          r * bm.abs(bm.random.rand(self.N, self.dim) * self.gbest - self.x) * bm.cos(theta)) +  # Eq.(5)
                      (bm.abs(R) > 1) * (r * (self.gbest - bm.random.rand(self.N, self.dim) * self.x)))  # Eq.(4)
            
            # Apply boundary conditions: ensure positions are within the bounds
            self.x = self.x + (self.lb - self.x) * (self.x < self.lb) + (self.ub - self.x) * (self.x > self.ub)

            # Evaluate fitness of the updated population
            fit = self.fun(self.x)

            # Update the global best solution if a better one is found
            self.update_gbest(self.x, fit)

            # Track the fitness of the global best solution over iterations
            self.curve[it] = self.gbest_f
