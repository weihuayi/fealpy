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
            else:
                self.x = self.x + ((FADs * (1 - bm.random.rand(1)) + bm.random.rand(1)) * 
                                   (self.x[bm.random.randint(0, self.N, (self.N,))] - 
                                    self.x[bm.random.randint(0, self.N, (self.N,))]))
            self.x = bm.clip(self.x, self.lb, self.ub)
            
            # Recalculate fitness after perturbation
            fit = self.fun(self.x)
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f  # Store the best fitness value of this iteration

class TIS_MarinePredatorsAlg(Optimizer):
    """
    Thinking Innovation Strategy Marine Predators Algorithm (TIS-MPA) implementation.
    
    A hybrid optimization algorithm combining Marine Predators Algorithm (MPA) with Thinking-Inspired Strategy (TIS),
    simulating marine predator foraging behaviors with cognitive enhancement mechanisms.

    Parameters:
        options (dict): Optimization configuration dictionary containing:
            - 'NP': Population size
            - 'dim': Problem dimension
            - 'MaxIT': Maximum iterations
            - 'lb': Lower bounds
            - 'ub': Upper bounds
            - 'fun': Objective function

    Attributes:
        gbest (array): Global best position found.
        gbest_f (float): Global best fitness value.
        curve (array): Fitness value progression over iterations.
        person (array): Current best individual position.
        person_f (float): Current best individual fitness.
        fes (int): Function evaluation counter.
        max_fes (int): Maximum function evaluations.
    """
    def __init__(self, options):
        super().__init__(options)

    def run(self, params={'P':0.5, 'FADs':0.2}):
        # Initialize parameters
        P = params.get('P')  # Movement probability
        FADs = params.get('FADs')  # Fish Aggregating Devices effect

        # Initialize tracking variables
        self.person = self.x[0]  # Current best individual
        self.fes = 0  # Function evaluation counter
        self.max_fes = self.MaxIT * self.N  # Maximum evaluations
        fit = bm.full((self.N,), bm.inf)  # Current population fitness
        fit_new = bm.full((self.N,), bm.inf)  # New population fitness

        # Initial evaluation with Thinking Innovation Strategy
        self.person_f = fit[0]
        for i in range(self.N):
            self.x[i], fit[i] = self.thinking_innovation_strategy(self.x[i])
        
        # Identify initial best solution
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]
        NN = int(self.N / 2)  # Half population size
        
        # Main optimization loop
        for it in range(self.MaxIT):
            self.D_pl_pt(it)  # Dynamic parameter adjustment

            # Adaptive control parameter (balances exploration-exploitation)
            CF = (1 - it / self.MaxIT) ** (2 * it / self.MaxIT)
            
            # Phase 1: Exploration (first 1/3 iterations)
            if it <= self.MaxIT / 3:
                RB = bm.random.randn(self.N, self.dim)  # Brownian motion
                x_new = self.x + P * bm.random.rand(self.N, self.dim) * RB * (self.gbest - RB * self.x)
            
            # Phase 2: Exploitation (middle 1/3 iterations)
            elif it > self.MaxIT / 3 and it <= 2 * self.MaxIT / 3:
                # First half uses direct refinement
                RB = bm.random.randn(NN, self.dim)
                x_new[0:NN] = self.gbest + P * CF * RB * (RB * self.gbest - self.x[0:NN])
                
                # Second half uses Levy flight for local search
                RL = 0.05 * levy(NN, self.dim, 1.5)  # Levy flight with beta=1.5
                x_new[NN:self.N] = (self.x[NN:self.N] + 
                                    P * bm.random.rand(NN, self.dim) * RL * (self.gbest - RL * self.x[NN:self.N]))
            
            # Phase 3: Hybrid (last 1/3 iterations)
            else:
                RL = 0.05 * levy(self.N, self.dim, 1.5)
                x_new = self.gbest + P * CF * RL * (RL * self.gbest - self.x)

            # Boundary constraint handling
            x_new = x_new + (self.lb - x_new) * (x_new < self.lb) + (self.ub - x_new) * (x_new > self.ub)

            # Evaluate new solutions with Thinking Innovation Strategy
            for i in range(self.N):
                x_new[i], fit_new[i] = self.thinking_innovation_strategy(x_new[i])
            
            # Greedy selection
            mask = fit_new < fit
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)
            self.update_gbest(self.x, fit)
            
            # Termination check
            if self.fes >= self.max_fes:
                return
            
            # Fish Aggregating Devices effect (environmental changes)
            if bm.random.rand(1) < FADs:
                self.x = self.x + CF * ((self.lb + bm.random.rand(self.N, self.dim) * (self.ub - self.lb)) * 
                                        (bm.random.rand(self.N, self.dim) < FADs))
                self.x = self.x + (self.lb - self.x) * (self.x < self.lb) + (self.ub - self.x) * (self.x > self.ub)
            else:
                # Social learning component
                self.x = self.x + ((FADs * (1 - bm.random.rand(1)) + bm.random.rand(1)) * 
                                   (self.x[bm.random.randint(0, self.N, (self.N,))] - 
                                    self.x[bm.random.randint(0, self.N, (self.N,))]))
                self.x = self.x + (self.lb - self.x) * (self.x < self.lb) + (self.ub - self.x) * (self.x > self.ub)
            
            # Evaluation and tracking
            self.fun(self.x)
            self.fun(self.x)
            self.curve[it] = self.gbest_f