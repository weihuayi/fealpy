from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class CrayfishOptAlg(Optimizer):
    """
    Crayfish Optimization Algorithm (COA) for optimization problems.

    The algorithm simulates the foraging behavior of crayfish, dynamically adjusting the search strategy
    based on environmental temperature and individual exploration. It combines global and local search mechanisms.

    Parameters:
        option (dict): A dictionary containing algorithm parameters (population size, dimensions, etc.)

    Attributes:
        gbest (Tensor): The global best solution found during the optimization process.
        gbest_f (float): The fitness value corresponding to the global best solution.
        curve (Tensor): Tracks the global best fitness value at each iteration.
    
    Methods:
        run(): Executes the Crayfish optimization algorithm for optimization.

    Reference
        Heming Jia, Honghua Rao, Changsheng Wen, Seyedali Mirjalili. 
        Crayfish optimization algorithm. 
        Artificial Intelligence Review, 2023, 56: S1919-S1979.
    """

    def __init__(self, option) -> None:
        """
        Initializes the Crayfish optimization algorithm with the provided options.
        
        Parameters:
            option (dict): A dictionary containing optimizer settings.
        """
        super().__init__(option)
    
    def run(self):
        """
        Executes the Crayfish optimization algorithm for optimization.

        This method includes the main optimization loop where the positions of the solutions are updated
        based on different strategies and environmental factors.
        """
        # Evaluate initial fitness for all solutions
        fit = self.fun(self.x)
        gbest_idx = bm.argmin(fit)
        self.gbest_f = fit[gbest_idx]
        self.gbest = self.x[gbest_idx] 

        # Initialize other variables
        global_position = self.gbest
        global_fitness = self.gbest_f

        # Main optimization loop
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)  # Update positions of individuals

            # Cooling factor controlling the balance of exploration and exploitation
            C = 2 - (it / self.MaxIT)
            
            # Temperature update
            temp = bm.random.rand(1) * 15 + 20
            
            # Calculating the foraging position (xf) as the midpoint between global best and individual position
            xf = (self.gbest + global_position) / 2
            
            # Probability distribution based on temperature
            p = (0.2 * (1 / (bm.sqrt(bm.array(2 * bm.pi) * 3))) * 
                 bm.exp(bm.array(- (temp - 25) ** 2 / (2 * 3 ** 2))))
            
            # Random probability for exploration and exploitation
            rand = bm.random.rand(self.N, 1)

            # Reshape gbest for matrix operations
            self.gbest = self.gbest.reshape(1, self.dim)

            # Calculate P as a probability factor based on fitness and gbest_f
            P = 3 * bm.random.rand(self.N) * fit / (self.gbest_f + 2.2204e-16)

            # Update positions using different movement strategies
            x_new = ((temp > 30) * ((rand < 0.5) * 
                                    (self.x + C * bm.random.rand(self.N, self.dim) * (xf - self.x)) + 
                                    (rand > 0.5) * 
                                    (self.x - self.x[bm.random.randint(0, self.N, (self.N,))] + xf)) + 
                    (temp <= 30) * ((P[:, None] > 2) * 
                                    (self.x + (bm.cos(2 * bm.random.rand(self.N, self.dim) * bm.pi) - 
                                               bm.sin(2 * bm.pi * bm.random.rand(self.N, self.dim))) * 
                                               bm.exp(bm.array(-1 / P[:, None])) * self.gbest * p) + 
                                    (P[:, None] <= 2) * 
                                    ((self.x - self.gbest) * p + p * bm.random.rand(self.N, self.dim) * self.x)))

            # Boundary handling
            x_new = bm.clip(x_new, self.lb, self.ub)        
            
            # Evaluate fitness for the new population
            fit_new = self.fun(x_new)
            mask = fit_new < fit

            # Update the population with better solutions
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)
            
            # Find the best solution in the new population
            newbest_id = bm.argmin(fit_new)

            # Update global best if a better solution is found
            if fit_new[newbest_id] < global_fitness:
                global_position = bm.copy(x_new[newbest_id])
                global_fitness = bm.copy(fit_new[newbest_id])

            # Track the global best solution
            self.update_gbest(self.x, fit)

            # Flatten the gbest for the next iteration
            self.gbest = self.gbest.flatten()
            
            # Track the best fitness value for plotting
            self.curve[it] = self.gbest_f

class ModifiedCrayfishOptAlg(Optimizer):
    """
    Modified Crayfish Optimization Algorithm (MCOA).

    This class implements an improved version of the Crayfish Optimization Algorithm, 
    incorporating ghost opposition-based learning and roulette-wheel discretization strategies 
    for enhanced exploration and exploitation balance. It inherits from the `Optimizer` base class.

    Parameters:
        options(dict): Configuration dictionary containing algorithm-related parameters 
                       such as population size, dimensionality, fitness function, bounds, 
                       maximum function evaluations (MaxFes), etc.

    Attributes:
        x(Tensor): Current population of candidate solutions.
        gbest(Tensor): Global best solution found so far.
        gbest_f(float): Fitness value of the global best solution.
        lb(Tensor): Lower bound of the search space.
        ub(Tensor): Upper bound of the search space.
        Fes(int): Current number of function evaluations.
        MaxFes(int): Maximum number of function evaluations.
        N(int): Population size.
        dim(int): Problem dimensionality.

    Methods:
        environment_update_mechanism(xbest, v, c=2):
            Implements the environmental update rule (equations 14–16). 
            Simulates crayfish migration in response to water quality and flow direction.

        ghost_opposition_point(x):
            Generates ghost opposition-based solutions (equations 19–20), 
            improving population diversity and convergence.

        run():
            Main optimization loop. Iteratively updates the population via 
            environmental update, foraging, ghost opposition learning, and 
            roulette-wheel based water quality factor discretization.

        compute_s_from_fitness(f_vals, minimize=True, eps=1e-12):
            Normalizes raw fitness values to [0,1], serving as a measure of relative "badness" (s).

        combine_with_violation(s_fitness, violation=None):
            Integrates normalized fitness scores with constraint violation degree, 
            prioritizing feasibility in constrained problems.

        discretize_v_roulette(f_vals, violation=None, minimize=True, alpha=6.0):
            Discretizes the water quality factor V ∈ {0,...,5} using a roulette-wheel 
            selection scheme based on softmax probabilities derived from fitness and violation.
    
    Reference:
    Jia, H., Zhou, X., Zhang, J. et al. 
    Modified crayfish optimization algorithm for solving multiple engineering application problems. 
    Artif Intell Rev 57, 127 (2024). https://doi.org/10.1007/s10462-024-10738-x
    
    """
    def __init__(self, options):
        super().__init__(options)
    
    def environment_update_mechanism(self, xbest, v, c=2):
        """
        Environmental update mechanism (equations 14–16).

        Updates population positions based on environmental conditions, 
        simulating crayfish migration driven by water quality (v) and 
        hydrodynamic factors (B, θ).

        Parameters:
            xbest(Tensor): Current best solution in the population.
            v(Tensor): Water quality factor vector (N,1), discretized into [0..5].
            c(float, optional): Control coefficient for B. Defaults to 2.

        Returns:
            Tensor: Updated population positions (N, D).
        """
        B = c * bm.cos(bm.pi/2 * (1 - self.Fes / self.MaxFes))

        r1 = bm.random.rand(self.N, 1)
        r2 = bm.random.rand(self.N, 1)
        r3 = bm.random.rand(self.N, 1)

        x2 = (xbest - self.x) * r1
        x1 = self.x[bm.random.randint(0, self.N, (self.N,))]

        theta = 2 * bm.pi * r2

        x_new = (x2 + (x1 - self.x) * bm.cos(theta) * B * v * r3 
                    + x1 * bm.sin(theta) * B * r3)

        return x_new
    
    def ghost_opposition_point(self, x):
        """
        Ghost opposition-based learning mechanism (equations 19–20).

        Generates ghost opposition solutions by projecting current population 
        relative to the midpoint of the search space, controlled by dynamic parameter k.

        Parameters:
            x(Tensor): Current population matrix (N, D).

        Returns:
            Tensor: Ghost opposition-based solutions (N, D).
        """
        k = (1 + (self.Fes / self.MaxFes) ** 0.5) ** 10
        mid = (self.ub + self.lb) / 2
        x_star = mid + mid / k - x / k
        return x_star
    
    def run(self):
        """
        Executes the main optimization loop.

        Iteratively updates the population by:
            - Calculating environmental update (equations 14–16).
            - Foraging behavior adjustments based on temperature and probability (Section 3.1).
            - Discretizing water quality factor V via roulette-wheel selection.
            - Applying ghost opposition-based learning (equations 17–20).
            - Evaluating fitness and updating global best solution.

        Returns:
            None: Results are stored in class attributes (gbest, gbest_f).
        """
        fit = self.fun(self.x)
        gbest_idx = bm.argmin(fit)
        self.gbest_f = fit[gbest_idx]
        self.gbest = self.x[gbest_idx]
        self.Fes = self.N
        
        # Initialize other variables
        global_position = bm.copy(self.gbest)
        global_fitness = bm.copy(self.gbest_f)

        while self.Fes <= self.MaxFes:
            # Cooling factor controlling the balance of exploration and exploitation
            C = 2 - (self.Fes / self.MaxFes)
            
            # Temperature update
            temp = bm.random.rand(1) * 15 + 20
            
            # Calculating the foraging position (xf) as the midpoint between global best and individual position
            xf = (self.gbest + global_position) / 2
            
            # Probability distribution based on temperature
            p = (0.2 * (1 / (bm.sqrt(bm.array(2 * bm.pi) * 3))) * 
                 bm.exp(bm.array(- (temp - 25) ** 2 / (2 * 3 ** 2))))
            
            # Random probability for exploration and exploitation
            rand = bm.random.rand(self.N, 1)

            # Reshape gbest for matrix operations
            self.gbest = self.gbest.reshape(1, self.dim)
            
            # Calculate P as a probability factor based on fitness and gbest_f
            P = 3 * bm.random.rand(self.N) * fit / (self.gbest_f + 2.2204e-16)

            v = self.discretize_v_roulette(fit)

            x_new = (
                (v[:, None] > 3) * (self.environment_update_mechanism(global_position, v[:, None])) + 
                (v[:, None] <= 3) * ((temp > 30) * ((rand < 0.5) * 
                                                    (self.x + C * bm.random.rand(self.N, self.dim) * (xf - self.x)) + 
                                                    (rand > 0.5) * 
                                                    (self.x - self.x[bm.random.randint(0, self.N, (self.N,))] + xf)) + 
                                    (temp <= 30) * ((P[:, None] > 2) * 
                                                    (self.x + (bm.cos(2 * bm.random.rand(self.N, self.dim) * bm.pi) - 
                                                    bm.sin(2 * bm.pi * bm.random.rand(self.N, self.dim))) * 
                                                    bm.exp(bm.array(-1 / P[:, None])) * self.gbest * p) + 
                                                    (P[:, None] <= 2) * 
                                                    ((self.x - self.gbest) * p + p * bm.random.rand(self.N, self.dim) * self.x)))
            )
            x_new = bm.clip(x_new, self.lb, self.ub)
            fit_new = self.fun(x_new)
            self.Fes += self.N
            mask = fit_new < fit

            # Update the population with better solutions
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)
            
            # Find the best solution in the new population
            newbest_id = bm.argmin(fit_new)

            # Update global best if a better solution is found
            if fit_new[newbest_id] < global_fitness:
                global_position = bm.copy(x_new[newbest_id])
                global_fitness = bm.copy(fit_new[newbest_id])

            # Track the global best solution
            self.update_gbest(self.x, fit)

            x_new = self.ghost_opposition_point(self.x)
            x_new = bm.clip(x_new, self.lb, self.ub)
            # Evaluate fitness for the new population
            fit_new = self.fun(x_new)
            mask = fit_new < fit

            # Update the population with better solutions
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)
            
            # Find the best solution in the new population
            newbest_id = bm.argmin(fit_new)

            # Update global best if a better solution is found
            if fit_new[newbest_id] < global_fitness:
                global_position = bm.copy(x_new[newbest_id])
                global_fitness = bm.copy(fit_new[newbest_id])

            # Track the global best solution
            self.update_gbest(self.x, fit) 

    def discretize_v_roulette(self, f_vals):
        """
        Discretizes the water quality factor V using roulette-wheel selection.

        Constructs logits proportional to (s_i * alpha * k), applies softmax to 
        obtain probability distributions, and samples V ∈ {0,1,2,3,4,5} for each individual.

        Parameters:
            f_vals(array-like): Fitness values of the population.

        Returns:
            Tensor: Discretized water quality factors V for all individuals (N,).
        """
        gap = abs((f_vals - f_vals.min()) / (f_vals.max() - f_vals.min() + 1e-10))
        mark1 = gap < 0.2
        r = bm.random.rand(self.N,)
        mark2 = r < 0.5
        a = mark1 & mark2
        b = ~mark1 & mark2
        c = ~mark1 & ~mark2
        Vs = bm.zeros((self.N,))
        Vs[a] = bm.random.randint(0, 4, (bm.sum(a),))
        Vs[b] = bm.random.randint(4, 6, (bm.sum(b),))
        Vs[c] = bm.random.randint(2, 5, (bm.sum(c),))
        return Vs