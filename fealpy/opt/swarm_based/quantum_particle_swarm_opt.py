from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class QuantumParticleSwarmOpt(Optimizer):
    """
    Quantum Particle Swarm Optimization (QPSO) Algorithm.

    QPSO is a quantum-inspired version of the traditional Particle Swarm Optimization (PSO). 
    It utilizes quantum principles to update particle positions in a more probabilistic manner, 
    enhancing the exploration capability in the solution space. The algorithm works by balancing 
    the global best and personal best positions through quantum-inspired operations, resulting 
    in a more efficient convergence.

    Methods:
    --------
    run(self, alpha_max=0.9, alpha_min=0.4):
        Executes the QPSO algorithm for a specified number of iterations, updating particle positions 
        based on quantum-inspired operations and local/global attraction, and tracking the global 
        best solution found.

    """
    def __init__(self, option) -> None:
        super().__init__(option)

    def run(self, params={'alpha_max':0.9, 'alpha_min':0.4}):
        """
        Run the Quantum Particle Swarm Optimization (QPSO) algorithm.

        Parameters:
            alpha_max (float): Maximum contraction-expansion coefficient.
            alpha_min (float): Minimum contraction-expansion coefficient.
        """
        # Compute the initial fitness of the population
        alpha_max = params.get('alpha_max')
        alpha_min = params.get('alpha_min')
        fit = self.fun(self.x)
        
        # Initialize the personal best positions (pbest) and their fitness values (pbest_f)
        pbest = bm.copy(self.x)
        pbest_f = bm.copy(fit)
        
        # Find the global best position (gbest), i.e., the position with the minimum fitness
        gbest_index = bm.argmin(pbest_f)
        self.gbest = pbest[gbest_index]
        self.gbest_f = pbest_f[gbest_index]
        
        for it in range(0, self.MaxIT):
            # Update auxiliary data structures (defined elsewhere)
            self.D_pl_pt(it)
            
            # Compute the contraction-expansion coefficient alpha
            alpha = bm.array(alpha_max - (alpha_max - alpha_min) * (it + 1) / self.MaxIT)
            
            # Compute the average of all particle best positions (mbest) for global search
            mbest = bm.sum(pbest, axis=0) / self.N
            
            # Generate random factors to simulate quantum behavior
            phi = bm.random.rand(self.N, self.dim)  # Random factor
            p = phi * pbest + (1 - phi) * self.gbest  # Local attractor: combination of personal best and global best
            u = bm.random.rand(self.N, self.dim)  # Random matrix for quantum update
            rand = bm.random.rand(self.N, 1)  # Random values for quantum update
            
            # Update the particle positions
            self.x = p + alpha * bm.abs(mbest - self.x) * bm.log(1 / u) * (1 - 2 * (rand >= 0.5))
            
            # Apply boundary conditions to ensure particles stay within the feasible solution space
            self.x = self.x + (self.lb - self.x) * (self.x < self.lb) + (self.ub - self.x) * (self.x > self.ub)
            
            # Recompute the fitness of the updated particles
            fit = self.fun(self.x)
            
            # Compare the new fitness with the personal best, and update pbest and pbest_f accordingly
            mask = fit < pbest_f
            pbest, pbest_f = bm.where(mask[:, None], self.x, pbest), bm.where(fit < pbest_f, fit, pbest_f)
            
            # If a better particle is found, update the global best position
            self.update_gbest(pbest, pbest_f)
            
            # Track the global best fitness for the current iteration
            self.curve[it] = self.gbest_f
