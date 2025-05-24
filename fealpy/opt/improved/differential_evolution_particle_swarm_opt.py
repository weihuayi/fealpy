from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer
from ..opt_function import initialize

class DifferentialEvolutionParticleSwarmOpt(Optimizer):
    """
    Hybrid Differential Evolution and Particle Swarm Optimization (DE-PSO) algorithm.

    This class implements a hybrid optimization algorithm that combines Differential Evolution (DE)
    and Particle Swarm Optimization (PSO) mechanisms to balance exploration and exploitation in 
    continuous optimization problems.

    Inherits from:
        Optimizer: Base class providing common optimizer functionality.

    Attributes:
        gbest (Tensor): Global best solution found so far.
        gbest_f (float): Fitness value of the global best.
        curve (Tensor): Convergence curve tracking the best fitness value at each iteration.
    """
    def __init__(self, options) -> None:
        """
        Initializes the DifferentialEvolution optimizer with the given options.

        Parameters:
            option: Configuration options for the optimizer.
        """
        super().__init__(options)

    def run(self, params={
        'f_l': 0.1, 'f_u': 0.8,
        'cr_l': 0.3, 'cr_u': 1.0,
        'c1': 2.0, 'c2': 2.0,
        'w_max': 0.9, 'w_min': 0.4,
        'SEP': 0.2
    }):
        """
        Execute the DE-PSO hybrid optimization process.

        Parameters:
            params (dict): A dictionary of control parameters:
                - f_l (float): Lower bound for differential weight factor F.
                - f_u (float): Upper bound for differential weight factor F.
                - cr_l (float): Lower bound for crossover probability CR.
                - cr_u (float): Upper bound for crossover probability CR.
                - c1 (float): PSO cognitive coefficient (individual learning factor).
                - c2 (float): PSO social coefficient (global learning factor).
                - w_max (float): Maximum inertia weight in PSO.
                - w_min (float): Minimum inertia weight in PSO.

        Behavior:
            - Initializes crossover rate and scaling factor for DE.
            - Partitions the population into elite (P) and rest (Q) groups.
            - With certain probability (`SP`), uses DE-style updates; otherwise, uses PSO update rules.
            - Applies selection based on fitness comparison.
            - Adapts CR and F for stagnant individuals.
            - Randomly reinitializes individuals with poor success rates using mutation.

        Updates:
            - Global best solution (`self.gbest`) and convergence curve (`self.curve`) are updated each iteration.
        """

        # Extract control parameters
        f_l, f_u = params.get('f_l'), params.get('f_u')
        cr_l, cr_u = params.get('cr_l'), params.get('cr_u')
        c1, c2 = params.get('c1'), params.get('c2')
        w_max, w_min = params.get('w_max'), params.get('w_min')
        SEP = params.get('SEP')
        NS_max = 5
        SEP = int(SEP * self.N)
        gamma = 0.001

        NS = bm.zeros((self.N, 1))
        cr = initialize(self.N, 1, cr_u, cr_l)
        f = initialize(self.N, 1, f_l, f_u)

        fit = self.fun(self.x)
        pbest = bm.copy(self.x)
        pbest_f = bm.copy(fit)

        gbest_index = bm.argmin(pbest_f)
        self.gbest = pbest[gbest_index]
        self.gbest_f = pbest_f[gbest_index]

        v = bm.zeros((self.N, self.dim))
        index = bm.argsort(fit)
        P = bm.copy(self.x[index[:SEP]])
        Q = bm.copy(self.x[index[SEP:]])
        
        for it in range(self.MaxIT):
            self.D_pl_pt(it)

            # Calculate switch probability (SP)
            tau = 1.5 + bm.random.rand(1) * (2.2 - 1.5)
            SP = 1 / (1 + bm.exp(bm.array(1 - (self.MaxIT / (1 + it)))) ** tau)

            # Inertia weight
            w = w_max - (it / self.MaxIT) * (w_max - w_min)

            # Generate new velocities using DE or PSO
            rr = bm.random.rand(self.N, 1)
            v = (
                (rr <= SP) * (
                    P[bm.random.randint(0, SEP, (self.N,))] +
                    f * (
                        P[bm.random.randint(0, SEP, (self.N,))] -
                        Q[bm.random.randint(0, self.N - SEP, (self.N,))]
                    )
                ) +
                (rr > SP) * (
                    w * self.x +
                    c1 * bm.random.rand(self.N, self.dim) * (pbest - self.x) +
                    c2 * bm.random.rand(self.N, self.dim) * (self.gbest - self.x)
                )
            )
            v = bm.clip(v, self.lb, self.ub)

            # Apply crossover
            r_zeros = bm.zeros((self.N, self.dim))
            r_zeros[bm.arange(self.N), bm.random.randint(0, self.dim, (self.N,))] = 1
            mask = (bm.random.rand(self.N, self.dim) < cr) + r_zeros
            mask = bm.clip(mask, 0, 1)
            u = mask * v + (1 - mask) * self.x

            # Evaluate offspring
            fit_u = self.fun(u)
            mask = fit_u < fit
            self.x = bm.where(mask[:, None], u, self.x)
            fit = bm.where(mask, fit_u, fit)
            NS = bm.where(mask[:, None], 0, NS + 1)

            # Update elite and rest populations
            index = bm.argsort(fit)
            P = bm.copy(self.x[index[:SEP]])
            Q = bm.copy(self.x[index[SEP:]])

            # Update personal and global bests
            mask = fit < pbest_f
            pbest = bm.where(mask[:, None], self.x, pbest)
            pbest_f = bm.where(mask, fit, pbest_f)
            self.update_gbest(pbest, pbest_f)

            # Parameter adaptation for stagnated individuals
            f = bm.where(NS > NS_max, initialize(self.N, 1, f_l, f_u), f)
            cr = bm.where(NS > NS_max, initialize(self.N, 1, cr_l, cr_u), cr)

            # Mutation-based reinitialization for long-stagnated individuals
            rr = bm.random.rand(self.N - SEP, self.dim)
            reinit = initialize(self.N - SEP, self.dim, self.ub, self.lb)
            mutated = (rr > gamma) * self.x[index[SEP:]] + (rr <= gamma) * reinit
            self.x[index[SEP:]] = bm.where(NS[index[SEP:]] > NS_max, mutated, self.x[index[SEP:]])

            self.curve[it] = self.gbest_f