from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer
from ..opt_function import levy, initialize

class LevyQuantumParticleSwarmOpt(Optimizer):
    """
    Levy Quantum Particle Swarm Optimization (LQPSO) algorithm class.

    This class implements the Levy Quantum Particle Swarm Optimization (LQPSO) algorithm, a variant of
    particle swarm optimization (PSO) that incorporates Levy flights to improve exploration and exploitation.

    Attributes:
        gbest (numpy.ndarray): The best global position found by the swarm.
        gbest_f (float): The fitness value of the global best position.
        D_pl (numpy.ndarray): Exploration percentage for each iteration.
        D_pt (numpy.ndarray): Exploitation percentage for each iteration.
        Div (numpy.ndarray): Population diversity for each iteration.
        curve (numpy.ndarray): The fitness value curve over iterations.

    Reference
        Xiao-li Lu, Guang He. 
        QPSO algorithm based on Lévy flight and its application in fuzzy portfolio.
        Applied Soft Computing Journal, 2021, 99: 106894.
    """

    def __init__(self, option) -> None:
        """
        Initializes the LQPSO optimizer with the given options.

        Args:
            option (dict): Dictionary containing the configuration options for the optimizer.
                Expected keys include:
                    - "x0": Initial position of the particles.
                    - "NP": Population size.
                    - "MaxIters": Maximum number of iterations.
                    - "ndim": Number of dimensions of the problem.
                    - "domain": Tuple specifying the lower and upper bounds for the domain.
        """
        super().__init__(option)

    def run(self, params={'delta':0.1, 'sigma':0.001}):
        """
        Runs the Levy Quantum Particle Swarm Optimization algorithm.

        This function performs the optimization process over a set number of iterations, adjusting the 
        particles' positions based on exploration, exploitation, and Levy flights. It tracks the best 
        solution found during the optimization process.

        The algorithm includes the following key steps:
            1. Initialization of particles and fitness.
            2. Exploration and exploitation based on the contraction-expansion coefficient.
            3. Update of personal best (pbest) and global best (gbest) positions.
            4. Diversity-based premature prevention mechanism.
            5. Updating the fitness curve for analysis.

        Updates the following class attributes:
            - `gbest`: Global best position.
            - `gbest_f`: Fitness of the global best position.
            - `curve`: Fitness values over all iterations.
        """
        delta = params.get('delta')
        sigma = params.get('sigma')
        options = self.options
        x = options["x0"]
        N = options["NP"]
        fit = self.fun(x)
        MaxIT = options["MaxIters"]
        dim = options["ndim"]
        lb, ub = options["domain"]
        pbest = bm.copy(x)
        pbest_f = bm.copy(fit)
        gbest_index = bm.argmin(pbest_f)
        self.gbest = pbest[gbest_index]
        self.gbest_f = pbest_f[gbest_index]
        self.D_pl = bm.zeros((MaxIT,))
        self.D_pt = bm.zeros((MaxIT,))
        self.Div = bm.zeros((1, MaxIT))
        self.curve = bm.zeros((MaxIT,))
        
        for it in range(0, MaxIT):
            # Update population diversity
            self.Div[0, it] = bm.sum(bm.sum(bm.abs(bm.mean(x, axis=0) - x)) / N)

            # Update exploration and exploitation percentages
            self.D_pl[it], self.D_pt[it] = self.D_pl_pt(self.Div[0, it])

            # Nonlinear contraction-expansion coefficient
            alpha = bm.array(0.5 + (1 - 0.5) * (1 - it / MaxIT) ** 2)

            # Update personal best and global best positions
            mbest = bm.sum(pbest, axis=0) / N
            phi = bm.random.rand(N, dim)
            p = phi * pbest + (1 - phi) * self.gbest
            u = bm.random.rand(N, dim)
            rand = bm.random.rand(N, 1)
            s = levy(N, dim, 1.5) * 0.05

            # Update particle positions using exploration and exploitation strategies
            x = ((u <= 0.5) * (p + alpha * bm.abs(mbest - x) * bm.log(1 / u) * (1 - 2 * (rand >= 0.5))) + 
                 (u > 0.5) * (s * bm.abs(x - p) + alpha * bm.abs(mbest - x) * bm.log(1 / u) * (1 - 2 * (rand >= 0.5))))

            # Boundary handling
            x = x + (lb - x) * (x < lb) + (ub - x) * (x > ub)

            # Recalculate fitness and update personal bests
            fit = self.fun(x)
            mask = fit < pbest_f
            pbest, pbest_f = bm.where(mask[:, None], x, pbest), bm.where(fit < pbest_f, fit, pbest_f)
            self.update_gbest(pbest, pbest_f)

            # Premature prevention mechanism based on population diversity
            Diversity = bm.sum(x - mbest) / (N * (ub - lb))  # Population diversity
            if Diversity < sigma:
                rand_individual = bm.random.randint(0, N, (int(delta * N),))
                x[rand_individual] = initialize(int(delta * N), dim, ub, lb)
                fit[rand_individual] = self.fun(x[rand_individual])
                mask = fit < pbest_f
                pbest, pbest_f = bm.where(mask[:, None], x, pbest), bm.where(fit < pbest_f, fit, pbest_f)
                self.update_gbest(pbest, pbest_f)

            # Update fitness curve for tracking progress
            self.curve[it] = self.gbest_f
