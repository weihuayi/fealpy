from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer
from ..opt_function import levy

class AnimatedOatOpt(Optimizer):
    """
    Animated Oat Optimization Algorithm (AO-OA) implementation.
    
    A bio-inspired metaheuristic optimization algorithm inspired by oat plant growth behavior,
    incorporating Levy flight and trigonometric functions for exploration-exploitation balance.

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
    """

    def __init__(self, options):
        """Initialize the optimizer with given configuration."""
        super().__init__(options)

    def run(self):
        """
        Execute the optimization process.
        
        The algorithm consists of 5 main position update strategies combined with:
        - Levy flight for long-range exploration
        - Trigonometric functions for local exploitation
        - Adaptive parameters for dynamic search behavior
        """
        # Initial evaluation
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest = self.x[gbest_index]
        self.gbest_f = fit[gbest_index]

        # Initialize control parameters
        m = 0.5 * bm.random.rand(self.N, 1) / self.dim  # Mass coefficient
        x = 3 * bm.random.rand(self.N, 1) / self.dim    # Position coefficient
        L = self.N * bm.random.rand(self.N, 1) / self.dim  # Length coefficient
        e = 0.5 * bm.random.rand(self.N, 1) / self.dim  # Elasticity coefficient
        g = 9.8 / self.dim  # Gravity constant (normalized)

        for it in range(self.MaxIT):
            self.D_pl_pt(it)  # Dynamic parameter adjustment
            
            # Adaptive control parameter (decreases over iterations)
            c = 1 - (it / self.MaxIT) ** 3  # Eq.(3)

            # Generate Levy flight steps
            P = levy(self.N, self.dim, 1.5)  # Levy flight with beta=1.5
            
            # Random walk component
            W = c / bm.pi * (2 * bm.random.rand(self.N, self.dim) - 1) * self.ub  # Eq.(4)

            # Strategy 1: Combined position update (Eq.5)
            X1 = self.x + W  # Current position + random walk
            X2 = bm.mean(self.x, axis=0) + W  # Population mean + random walk
            X3 = self.gbest + W  # Global best + random walk
            i = bm.arange(self.N)
            index1 = (i + 1) % (self.N // 10) == 0  # 10% particles use mean position
            index2 = (i + 1) % (self.N // 10) == 1  # 10% particles use global best
            X1 = bm.where(index1[:, None], X2, X1)
            X1 = bm.where(index2[:, None], X3, X1)

            # Strategy 2: Dynamic radius search (Eq.6-10)
            A = self.ub - bm.abs((self.ub * it * bm.sin(2*bm.pi*bm.random.rand(self.N, 1))) / self.MaxIT)  # Eq.(6)
            R = (m * e + L**2) * (-A + bm.random.rand(self.N, self.dim) * 2 * A) / self.dim  # Eq.(7)
            X4 = self.gbest + R + c * P * self.gbest  # Eq.(10)

            # Strategy 3: Trigonometric oscillation search (Eq.11-14)
            B = self.ub - bm.abs((self.ub * it * bm.cos(2*bm.pi*bm.random.rand(self.N, 1))) / self.MaxIT)  # Eq.(11)
            
            # Oscillation parameters
            k = 0.5 * bm.random.rand(self.N, 1) + 0.5  # Spring constant
            theta = bm.pi * bm.random.rand(self.N, 1)  # Angle
            alpha = bm.exp(bm.random.randint(0, it+1, (self.N, 1)) / self.MaxIT) / bm.pi  # Damping factor

            J = 2 * k * x**2 * bm.sin(2*theta)/ m / g * (1 - alpha) / self.dim * (-B + bm.random.rand(self.N, self.dim) * 2 * B)  # Eq.(13)
            X5 = self.gbest + J + c * P * self.gbest  # Eq.(14)

            # Combine strategies probabilistically
            r1 = bm.random.rand(self.N, 1)
            r2 = bm.random.rand(self.N, 1)
            
            self.x = ((r1 > 0.5) * (X1) + 
                     (r1 <= 0.5) * ((r2 > 0.5) * (X4) + 
                                    (r2 <= 0.5) * (X5)))
            
            # Enforce bounds and evaluate
            self.x = bm.clip(self.x, self.lb, self.ub)
            fit = self.fun(self.x)
            self.update_gbest(self.x, fit)
            self.curve[it] = self.gbest_f