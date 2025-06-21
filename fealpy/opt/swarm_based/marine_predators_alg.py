from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer
from ..opt_function import levy

class MarinePredatorsAlg(Optimizer):
    """
    A class implementing the Marine Predators Algorithm (MPA) for optimization tasks.
    
    This algorithm simulates the hunting behavior of marine predators to find optimal solutions.
    It includes a process where the predators search for prey in three phases:
    - Exploration Phase: Utilizing random strategies for exploration.
    - Exploitation Phase: Refining solutions based on the best-known predator.
    - Hybrid Phase: Combining exploration and exploitation techniques.

    Parameters:
        option (dict): A dictionary containing parameters for the optimizer, such as population size, dimensions, and iteration limits.
    
    Attributes:
        gbest (Tensor): The global best solution found by the algorithm.
        gbest_f (float): The fitness value of the global best solution.
        curve (Tensor): A tensor to store the progress of the best solution over iterations.
    
    Methods:
        run(P=0.5, FADs=0.2): Main method to run the Marine Predators Algorithm for optimization.

    Reference:
        Faramarzi A, Heidarinejad M, Mirjalili S, et al. 
        Marine Predators Algorithm: A nature-inspired metaheuristic. 
        Expert systems with applications, 2020, 152: 113377.
    """
    
    def __init__(self, option) -> None:
        """
        Initializes the Marine Predators Algorithm with the provided options.
        
        Parameters:
            option (dict): A dictionary containing optimizer settings.
        """
        super().__init__(option)

    def run(self, params={'P':0.5, 'FADs':0.2}):
        """
        Runs the Marine Predators Algorithm to find the optimal solution.

        This method goes through multiple iterations to update the population of candidates 
        and find the best solution based on the fitness function.
        
        Parameters:
            P (float): The probability factor influencing the exploration and exploitation phases. Default is 0.5.
            FADs (float): The FADs factor controlling the perturbation of the solution. Default is 0.2.
        """
        
        # Initial fitness and global best solution
        P = params.get('P')
        FADs = params.get('FADs')
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]
        NN = int(self.N / 2)

        # Iteration loop
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)  # Update predator positions
            
            # CF is a factor controlling the transition between exploration and exploitation
            CF = (1 - it / self.MaxIT) ** (2 * it / self.MaxIT)
            
            # Exploration phase: random strategies
            if it <= self.MaxIT / 3:
                RB = bm.random.randn(self.N, self.dim)
                x_new = self.x + P * bm.random.rand(self.N, self.dim) * RB * (self.gbest - RB * self.x)
            
            # Exploitation phase: refining solutions using the best-known predator
            elif it > self.MaxIT / 3 and it <= 2 * self.MaxIT / 3:
                RB = bm.random.randn(NN, self.dim)
                x_new[0:NN] = self.gbest + P * CF * RB * (RB * self.gbest - self.x[0:NN])
                RL = 0.05 * levy(NN, self.dim, 1.5)
                x_new[NN:self.N] = (self.x[NN:self.N] + 
                                    P * bm.random.rand(NN, self.dim) * RL * (self.gbest - RL * self.x[NN:self.N]))
            
            # Hybrid phase: combining exploration and exploitation
            else:
                RL = 0.05 * levy(self.N, self.dim, 1.5)
                x_new = self.gbest + P * CF * RL * (RL * self.gbest - self.x)

            # Apply boundary constraints
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)

            # Evaluate fitness of new solutions
            fit_new = self.fun(x_new)
            mask = fit_new < fit
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)
            self.update_gbest(self.x, fit)

            # FADs perturbation phase: modify the solution with a random adjustment
            if bm.random.rand(1) < FADs:
                self.x = self.x + CF * ((self.lb + bm.random.rand(self.N, self.dim) * (self.ub - self.lb)) * 
                                        (bm.random.rand(self.N, self.dim) < FADs))
                self.x = self.x + (self.lb - self.x) * (self.x < self.lb) + (self.ub - self.x) * (self.x > self.ub)
            else:
                self.x = self.x + ((FADs * (1 - bm.random.rand(1)) + bm.random.rand(1)) * 
                                   (self.x[bm.random.randint(0, self.N, (self.N,))] - 
                                    self.x[bm.random.randint(0, self.N, (self.N,))]))
                self.x = self.x + (self.lb - self.x) * (self.x < self.lb) + (self.ub - self.x) * (self.x > self.ub)
            
            # Recalculate fitness after perturbation
            fit = self.fun(self.x)
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f  # Store the best fitness value of this iteration
