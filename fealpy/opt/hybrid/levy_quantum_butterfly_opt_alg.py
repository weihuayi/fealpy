from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer
from ..opt_function import levy

class LevyQuantumButterflyOptAlg(Optimizer):
    """
    Levy Quantum Butterfly Optimization Algorithm (LQBOA) for optimization problems.

    This class implements the Levy Quantum Butterfly Optimization Algorithm, which is an
    enhancement of the standard Butterfly Optimization Algorithm. The algorithm is inspired
    by the flight behavior of butterflies and incorporates Levy flights and quantum-inspired
    mechanisms for improved exploration and exploitation during the optimization process.

    Attributes:
        gbest (bm.Tensor): The global best solution found so far.
        gbest_f (float): The fitness value of the global best solution.
        pbest (bm.Tensor): The personal best positions of the population.
        pbest_f (bm.Tensor): The fitness values of the personal best positions.
        curve (bm.Tensor): The fitness values of the best solution in each iteration.
        options (dict): A dictionary containing algorithm-specific options such as population
                        size, maximum iterations, and the domain bounds.

    Reference:
        Han-Bin Liu, Li-Bin Liu, Xiongfa Mai.
        A New Hybrid Levy Quantum-Behavior Butterfly Optimization Algorithm and its Application in NL5 Muskingum Model.
        Electronic Reasearch Archive, 2024, 32: 2380-2406.
    """

    def __init__(self, option) -> None:
        """
        Initializes the Levy Quantum Butterfly Optimization Algorithm with the given options.

        Args:
            option (dict): A dictionary containing parameters for the algorithm,
                           such as population size, bounds, and other configuration options.
        """
        super().__init__(option)

    def run(self):
        """
        Runs the Levy Quantum Butterfly Optimization Algorithm to optimize the objective function.
        
        The algorithm iterates through a population of candidate solutions, updating the positions
        of the particles using both global and local search strategies. The positions are updated
        using quantum-inspired techniques and Levy flights to achieve better exploration and exploitation.

        Steps:
            1. Initialize the population and evaluate the fitness of each individual.
            2. Update the positions based on global and local attractors.
            3. Perform a quantum-inspired update using Levy flights.
            4. Update personal best and global best solutions.
            5. Track the best fitness value in each iteration.
        
        The process continues for the maximum number of iterations (`MaxIters`).
        """
        options = self.options
        a = options["x0"]  # Initial population positions
        N = options["NP"]  # Population size
        fit = self.fun(a)[:, None]  # Fitness of the initial population
        MaxIT = options["MaxIters"]  # Maximum number of iterations
        dim = options["ndim"]  # Dimensionality of the problem
        lb, ub = options["domain"]  # Bounds for the variables

        # Initialize personal best positions and global best solution
        pbest = bm.copy(a)
        pbest_f = bm.copy(fit)
        gbest_index = bm.argmin(pbest_f)
        gbest = pbest[gbest_index]
        gbest_f = pbest_f[gbest_index]

        self.curve = bm.zeros((1, MaxIT))  # To store the best fitness value over iterations

        # Initialize random positions for x and z
        x = bm.random.rand(N, dim) * (ub - lb) + lb  # Candidate positions
        fit_x = self.fun(x)[:, None]  # Fitness of candidate positions
        z = bm.random.rand(N, dim) * (ub - lb) + lb  # Another set of candidate positions
        fit_z = self.fun(z)[:, None]  # Fitness of z positions

        NS = 0  # No Solution Counter
        best = bm.zeros((1, MaxIT))  # Store the best solution
        NN = int(N / 2)  # Half of the population
        c = 0.01  # A constant used in calculations

        # Main loop for optimization
        for it in range(0, MaxIT):
            # Rapid decreasing contraction-expansion coefficient
            alpha = 1 - 0.5 * bm.log(bm.array(it + 1)) / bm.log(bm.array(MaxIT))

            p = 0.1 + 0.8 * it / MaxIT  # Probability for determining global vs local search
            delta = bm.abs((fit - bm.max(fit)) / (bm.min(fit) - bm.max(fit) + 1e-10))
            mbest = bm.sum(delta * pbest, axis=0) / N  # Weighted mean of personal best positions
            a = bm.exp(bm.array(-it))  # Exponential decay for exploration/exploitation balance
            c = c + 0.5 / (c + MaxIT)  # Update constant c

            # Update the positions using global and local search strategies
            f = c * (fit_x ** a)
            rand = bm.random.rand(N, 1)
            x_new = ((rand < p) * 
                     (x + ((bm.random.rand(N, 1) ** 2) * gbest - x) * f) +  # Global search
                     (rand >= p) * 
                     (x + ((bm.random.rand(N, 1) ** 2) * x[bm.random.randint(0, N, (N,))] - 
                           x[bm.random.randint(0, N, (N,))]) * f))  # Local search
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)

            # Evaluate the fitness of the new positions
            fit_new = self.fun(x_new)[:, None]
            mask = fit_new < fit  # Mask for better fitness
            x, fit = bm.where(mask, x_new, x), bm.where(mask, fit_new, fit)
            
            # Update global best if a better solution is found
            gbest_idx = bm.argmin(fit)
            (gbest, gbest_f) = (x[gbest_idx], fit[gbest_idx]) if fit[gbest_idx] < gbest_f else (gbest, gbest)

            # Update personal best and global best using quantum-inspired update
            phi = bm.random.rand(N, 1)
            p = phi * pbest + (1 - phi) * gbest  # Local attractor
            u = bm.random.rand(N, 1)
            a = ((u > 0.5) * (levy(N, dim, 1.5) * bm.abs(a - p) + alpha * (mbest - a) * 
                              (1 - 2 * (bm.random.rand(N, 1) >= 0.5))) + 
                 (u <= 0.5) * (p + alpha * bm.abs(mbest - a) * 
                               bm.log(1 / bm.random.rand(N, 1)) * (1 - 2 * (bm.random.rand(N, 1) >= 0.5))))
            a = a + (lb - a) * (a < lb) + (ub - a) * (a > ub)

            # Update the fitness and personal best
            fit = self.fun(a)[:, None]
            mask = fit < pbest_f
            pbest, pbest_f = bm.where(mask, a, pbest), bm.where(mask, fit, pbest_f)
            
            # Update global best based on personal best fitness
            gbest_idx = bm.argmin(pbest_f)
            (gbest_f, gbest) = (pbest_f[gbest_idx], pbest[gbest_idx]) if pbest_f[gbest_idx] < gbest_f else (gbest_f, gbest)
            
            # Update `z` and perform the same fitness evaluations
            d_rand = bm.random.rand(N, 1)
            z_new = d_rand * a + (1 - d_rand) * x
            z_new = z_new + (lb - z_new) * (z_new < lb) + (ub - z_new) * (z_new > ub)
            fit_new = self.fun(z_new)[:, None]
            mask = fit_new < fit_z
            z, fit_z = bm.where(mask, z_new, z), bm.where(mask, fit_new, fit_z)

            # Check if `z` is better than `x` and update accordingly
            mask = fit_x < fit_z
            x, fit_x = bm.where(mask, z, x), bm.where(mask, fit_z, fit_x)
            index = bm.argmin(fit_x)
            (gbest_f, gbest) = (fit_x[index], x[index]) if fit_x[index] < gbest_f else (gbest_f, gbest)

            # Update `a` if necessary
            maks = fit < fit_z
            a, fit = bm.where(maks, z, a), bm.where(maks, fit_z, fit)

            # Update personal best and global best solutions
            mask = fit < pbest_f
            pbest, pbest_f = bm.where(mask, a, pbest), bm.where(fit < pbest_f, fit, pbest_f)
            gbest_idx = bm.argmin(pbest_f)    
            (gbest_f, gbest) = (pbest_f[gbest_idx], pbest[gbest_idx]) if pbest_f[gbest_idx] < gbest_f else (gbest_f, gbest)

            # Update global best solution with perturbation
            gbest_new = gbest + (ub - lb) * bm.random.randn(1, dim)
            gbest_new = gbest_new + (lb - gbest_new) * (gbest_new < lb) + (ub - gbest_new) * (gbest_new > ub)
            gbest_f_new = self.fun(gbest_new)
            if gbest_f_new < gbest_f:
                gbest, gbest_f = gbest_new, gbest_f_new
            
            # Store the best fitness value for this iteration
            best[0, it] = gbest_f[0]
            self.curve[0, it] = gbest_f[0]

        # Store the final global best solution and fitness
        self.gbest = gbest
        self.gbest_f = gbest_f
        self.curve = self.curve.flatten()
