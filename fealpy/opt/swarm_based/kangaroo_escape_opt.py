from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class KangarooEscapeOpt(Optimizer):
    """
    Kangaroo Escape Optimization Algorithm (KEOA) implementation.
    
    A bio-inspired optimization algorithm that simulates kangaroo escape behaviors including:
    - Long jumps for global exploration
    - Zigzag movements for local exploitation  
    - Decoy dropping and safe zone strategies for predator avoidance
    - Energy-based strategy selection using chaotic maps

    Parameters:
        options (dict): Optimization configuration dictionary containing:
            - 'NP': Population size
            - 'dim': Problem dimension
            - 'MaxIT': Maximum iterations
            - 'lb': Lower bounds
            - 'ub': Upper bounds
            - 'fun': Objective function

    Attributes:
        group_size (int): Size of kangaroo groups for social learning (5% of population).
        decoy_drop (array): Pattern of decoy drops for predator distraction.
        energy (array): Energy levels of kangaroos for strategy selection.
        chaotic_val (float): Chaotic value for energy dynamics.
        fit (array): Current fitness values of population.
    """

    def __init__(self, options):
        """Initialize the optimizer with given configuration."""
        super().__init__(options)
        # Define group size as 5% of the population size for social behaviors
        self.group_size = bm.int64(bm.round(0.05 * self.N))

    def cal_decoy_drop(self):
        """
        Calculate decoy drop patterns for predator distraction.
        
        Decoy types:
        - Type 1: Uniform pattern (all ones)
        - Type 2: Binary random pattern  
        - Type 3: Multiplicative random pattern
        
        Each kangaroo randomly selects one of three decoy patterns with equal probability.
        """
        # Random probability values for decoy type selection
        r = bm.random.rand(self.N, 1)

        # Three cases of decoy drop mask with equal probability
        mask1 = r < 1/3    # Type 1: Uniform pattern
        mask2 = (1/3 <= r) & (r < 2/3)  # Type 2: Binary pattern
        mask3 = 2/3 <= r   # Type 3: Multiplicative pattern

        # Different decoy types
        d1 = bm.ones((self.N, self.dim))                           # All ones (uniform distraction)
        d2 = bm.round(bm.random.rand(self.N, self.dim))            # Binary mask (random on/off)
        d3 = bm.round(bm.random.rand(self.N, self.dim) * bm.random.rand(self.N, self.dim))  # Random product (complex pattern)

        # Final decoy drop pattern combining all types
        self.decoy_drop = (mask1 * d1 + mask2 * d2 + mask3 * d3)
    
    def strategy1(self):
        """
        Long jump strategy for global exploration.
        
        Simulates kangaroo's long-distance jumping behavior for escaping predators
        and exploring new areas of the search space.
        
        Returns:
            array: New positions after long jumps
        """
        # Long jump strategy (global exploration)
        jump = 2 * bm.random.randn(self.N, 1) * self.x * self.decoy_drop  # Gaussian-distributed jumps
        x_new = self.x + jump
        return x_new

    def strategy2(self, beta, theta_max=bm.pi/6, eps = 2.2204e-16):
        """
        Zigzag strategy with rotation for local exploitation.
        
        Simulates kangaroo's evasive zigzag movements when closely pursued.
        Uses vector rotation to create directional diversity.
        
        Params:
            beta (float): Step size control parameter
            theta_max (float): Maximum rotation angle (default: π/6 = 30°)
            eps (float): Small value to avoid division by zero
            
        Returns:
            array: New positions after zigzag movements
        """
        # Zigzag strategy with rotation
        theta = theta_max * bm.random.rand(self.N, 1)  # Random rotation angle

        # Direction vector to global best position
        v = self.gbest - self.x
        v_unit = v / (bm.linalg.norm(v, axis=1)[:, None] + eps)  # Unit direction vector

        if self.dim == 1:
            # Special case: 1D zigzag degenerates to simple sign flip
            u = 1
        else:
            # Generate a random vector and orthogonalize it w.r.t. v_unit
            rand_vec = bm.random.rand(self.N, self.dim)
            proj = bm.sum(rand_vec * v_unit, axis=1, keepdims=True) * v_unit  # Projection component
            rand_vec = rand_vec - proj  # Orthogonal component
            u = rand_vec / (bm.linalg.norm(rand_vec, axis=1, keepdims=True) + eps)  # Orthonormal basis

        # Rotate v by angle theta using orthogonal vector u (Rodrigues' rotation)
        v_rot = bm.cos(theta) * v + bm.sin(theta) * u * bm.linalg.norm(v, axis=1)[:, None]

        # Update position with zigzag motion (sign alternation for zigzag pattern)
        x_new = self.x + beta * bm.sign(theta) * bm.random.randn(self.N, self.dim) * v_rot
        return x_new

    def strategy3(self):
        """
        Safe zone search with decoy drop for predator avoidance.
        
        Simulates kangaroo seeking safe zones while dropping decoys to confuse predators.
        Uses social learning from group members in later stages.
        
        Returns:
            array: New positions in safe zones with decoy effects
        """
        # Safe zone search with decoy drop

        # Case 1: move toward a random kangaroo's position
        x_rand = self.x[bm.random.randint(0, self.N, (self.N,))]

        # Case 2: move toward the best kangaroo in a random group (social learning)
        group_idx = bm.random.randint(0, self.N, (self.N, self.group_size))
        group_fit = self.fit[group_idx]
        best_in_group = bm.argmin(group_fit, axis=1)  # Find best in each group
        best_idx = group_idx[bm.arange(self.N), best_in_group]
        x_group = self.x[best_idx]

        # Safe zone selection rule based on optimization phase
        if self.it < (3 * self.MaxIT // 4):
            # Early phase (first 75%): prefer random exploration
            x_safe = x_rand
        else:
            # Later phase (last 25%): 75% chance use group-best, 25% chance use global best
            mask2 = bm.random.rand(self.N, 1) < 0.75
            gbest_expand = bm.tile(self.gbest[None, :], (self.N, 1))  # Expand global best for broadcasting
            x_safe = bm.where(mask2, x_group, gbest_expand)

        # Update position according to safe zone + decoy drop (predator confusion)
        x_new = x_safe + bm.random.randn(self.N, 1) * self.decoy_drop * (self.x - x_safe)
        return x_new

    def run(self, params={'beta':1, 'energy_threshold':0.5, 'chaotic_val':0.7}):
        """
        Execute the kangaroo escape optimization process.
        
        params (dict): Algorithm control parameters with:
            - 'beta': Zigzag step size control (default: 1)
            - 'energy_threshold': Energy level for strategy switching (default: 0.5)
            - 'chaotic_val': Initial chaotic value for energy dynamics (default: 0.7)
        
        The algorithm features energy-based strategy selection using chaotic logistic maps
        and three distinct escape behaviors modeled after real kangaroo predator avoidance.
        """
        # Initialize parameters
        beta = params.get('beta')  # Zigzag step size control
        energy_threshold = params.get('energy_threshold')  # Strategy switching threshold
        self.chaotic_val = params.get('chaotic_val')  # Initial chaotic value

        # Evaluate initial population
        self.fit = self.fun(self.x)
        gbest_idx = bm.argmin(self.fit)
        self.gbest_f = self.fit[gbest_idx]
        self.gbest = self.x[gbest_idx]

        # Main optimization loop
        for self.it in range(self.MaxIT):
            # Apply boundary control (if any in parent class)
            self.D_pl_pt(self.it)

            # Generate decoy drop pattern for current iteration
            self.cal_decoy_drop()

            # Random selection between strategies
            r = bm.random.rand(self.N, 1)

            # Energy update using chaotic logistic map (nonlinear dynamics)
            self.energy = (1 - bm.random.rand(self.N, 1) * (self.it / self.MaxIT)) * (0.95 + 0.05 * self.chaotic_val)
            self.chaotic_val = 4 * self.chaotic_val * (1 - self.chaotic_val)  # Logistic map update

            # Select strategy according to probability and energy level
            mask1 = r < 0.5  # 50% chance for safe zone strategy
            mask2 = self.energy < energy_threshold  # Low energy → long jumps
            
            # Update positions using three strategies with priority:
            # 1. Safe zone strategy (50% chance)
            # 2. Long jump strategy (low energy)
            # 3. Zigzag strategy (high energy)
            x_new = bm.where(mask1, self.strategy3(), bm.where(mask2, self.strategy1(), self.strategy2(beta)))

            # Apply boundary constraints
            x_new = bm.clip(x_new, self.lb, self.ub)

            # Fitness evaluation of new positions
            fit_new = self.fun(x_new)

            # Greedy selection: replace if better fitness
            mask = fit_new < self.fit
            self.x, self.fit = bm.where(mask[:, None], x_new, self.x), bm.where(mask, fit_new, self.fit)

            # Update global best solution
            self.update_gbest(self.x, self.fit)
            self.curve[self.it] = self.gbest_f  # Record convergence curve