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
                                    (self.x + bm.cos(2 * bm.random.rand(self.N, self.dim) * bm.pi) * self.gbest * 
                                     p - bm.sin(2 * bm.pi * bm.random.rand(self.N, self.dim) * self.gbest * p)) + 
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
                global_position = x_new[newbest_id]
                global_fitness = fit_new[newbest_id]

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

        x_new = x2 + (x1 - self.x) * bm.cos(theta) * B * v * r3 \
                    + x1 * bm.sin(theta) * B * r3

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
                                                    (self.x + bm.cos(2 * bm.random.rand(self.N, self.dim) * bm.pi) * self.gbest * 
                                                    p - bm.sin(2 * bm.pi * bm.random.rand(self.N, self.dim) * self.gbest * p)) + 
                                                    (P[:, None] <= 2) * 
                                                    ((self.x - self.gbest) * p + p * bm.random.rand(self.N, self.dim) * self.x)))
            )
            x_new = bm.clip(x_new, self.lb, self.ub)
            fit_new = self.fun(x_new)
            self.Fes += self.N
            # Evaluate fitness for the new population
            fit_new = self.fun(x_new)
            mask = fit_new < fit

            # Update the population with better solutions
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)
            
            # Find the best solution in the new population
            newbest_id = bm.argmin(fit_new)

            # Update global best if a better solution is found
            if fit_new[newbest_id] < global_fitness:
                global_position = x_new[newbest_id]
                global_fitness = fit_new[newbest_id]

            # Track the global best solution
            self.update_gbest(self.x, fit)

            x_new = self.ghost_opposition_point(self.x)
            x_new = bm.clip(x_new, self.lb, self.ub)
            fit_new = self.fun(x_new)
            self.Fes += self.N
            # Evaluate fitness for the new population
            fit_new = self.fun(x_new)
            mask = fit_new < fit

            # Update the population with better solutions
            self.x, fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, fit)
            
            # Find the best solution in the new population
            newbest_id = bm.argmin(fit_new)

            # Update global best if a better solution is found
            if fit_new[newbest_id] < global_fitness:
                global_position = x_new[newbest_id]
                global_fitness = fit_new[newbest_id]

            # Track the global best solution
            self.update_gbest(self.x, fit)
            pass
    
    def compute_s_from_fitness(self, f_vals, minimize=True, eps=1e-12):
        """
        Computes normalized fitness scores (s) in [0,1].

        Parameters:
            f_vals(array-like): Raw fitness values.
            minimize(bool, optional): Whether the problem is minimization. Defaults to True.
            eps(float, optional): Numerical safeguard against division by zero.

        Returns:
            Tensor: Normalized scores, higher values indicate worse solutions.
        """
        f = bm.array(f_vals, dtype=bm.float64)
        if not minimize:
            f = -f  # 最大化问题变为最小化处理
        worst = f.max()
        best = f.min()
        if worst - best < eps:
            return bm.zeros_like(f)  # 全部相同：没有劣度
        s = (f - best) / (worst - best)  # 归一化到 [0,1]
        return s

    def combine_with_violation(self, s_fitness, violation=None):
        """
        Combines normalized fitness scores with constraint violation degrees.

        Parameters:
            s_fitness(Tensor): Normalized fitness scores in [0,1].
            violation(array-like | None, optional): Constraint violation measures.

        Returns:
            Tensor: Combined normalized scores, prioritizing feasibility.
        """
        if violation is None:
            return s_fitness
        v = bm.array(violation, dtype=bm.float64)
        if v.max() - v.min() < 1e-12:
            s_v = bm.zeros_like(v)
        else:
            s_v = (v - v.min()) / (v.max() - v.min())
        return bm.maximum(s_fitness, s_v)  # 若违背大，则以违背为主

    def discretize_v_roulette(self, f_vals, violation=None, minimize=True, alpha=6.0):
        """
        Discretizes the water quality factor V using roulette-wheel selection.

        Constructs logits proportional to (s_i * alpha * k), applies softmax to 
        obtain probability distributions, and samples V ∈ {0,1,2,3,4,5} for each individual.

        Parameters:
            f_vals(array-like): Fitness values of the population.
            violation(array-like | None, optional): Constraint violation measures.
            minimize(bool, optional): Whether the problem is minimization. Defaults to True.
            alpha(float, optional): Scaling factor controlling selection pressure. Defaults to 6.0.

        Returns:
            Tensor: Discretized water quality factors V for all individuals (N,).
        """
        n = len(f_vals)
        s = self.compute_s_from_fitness(f_vals, minimize=minimize)  # in [0,1]
        s = self.combine_with_violation(s, violation)               # 合并违背度
        # 防止数值问题
        s = bm.clip(s, 0.0, 1.0)

        # 构造每个个体的 6 个档位 logits: logit_k = alpha * k * s_i
        ks = bm.arange(6)  # [0,1,2,3,4,5]
        # logits shape: (n,6)
        logits = (s * alpha)[:, None] * ks[None, :]   # 每行 i: alpha * s_i * [0..5]

        # softmax -> 概率分布 (n,6)
        # 为数值稳定性，减去每行最大值
        logits_max = logits.max(axis=1, keepdims=True)
        exps = bm.exp(logits - logits_max)
        probs = exps / (exps.sum(axis=1, keepdims=True) + 1e-16)

        # 对每行做一次按 probs 的抽样（轮盘赌）
        cumprobs = bm.cumsum(probs, axis=1)  # (n,6)
        us = bm.random.rand(n, 1)
        Vs = (us <= cumprobs).argmax(axis=1)  # 每行第一个满足的位置索引就是抽到的档位
        return Vs