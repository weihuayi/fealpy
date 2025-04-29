from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class ParticleSwarmOpt(Optimizer):
    """
    Particle Swarm Optimization (PSO) algorithm implementation.

    This class implements the standard PSO algorithm, which is a population-based
    optimization technique inspired by the social behavior of bird flocking or fish schooling.
    It is designed to solve continuous optimization problems.

    """
    def __init__(self, option) -> None:
        super().__init__(option)


    def run(self, params={'c1':2, 'c2':2, 'w_max':0.9, 'w_min':0.4}):
        """
        Executes the PSO algorithm.

        Args:
            c1 (float, optional): Cognitive parameter (individual learning factor). Defaults to 2.
            c2 (float, optional): Social parameter (group learning factor). Defaults to 2.
            w_max (float, optional): Maximum inertia weight. Defaults to 0.9.
            w_min (float, optional): Minimum inertia weight. Defaults to 0.4.

        Returns:
            None

        Notes:
            - The algorithm updates particle positions and velocities iteratively.
            - The inertia weight `w` decreases linearly from `w_max` to `w_min` over iterations.
            - Particle velocities are clamped to a fraction of the search space bounds.
            - The best fitness value at each iteration is stored in `self.curve`.
            
        """
        c1 = params.get('c1')
        c2 = params.get('c2')
        w_max = params.get('w_max')
        w_min = params.get('w_min')
        fit = self.fun(self.x)
        pbest = bm.copy(self.x)
        pbest_f = bm.copy(fit)
        gbest_index = bm.argmin(pbest_f)
        self.gbest = pbest[gbest_index]
        self.gbest_f = pbest_f[gbest_index]
        v = bm.zeros((self.N, self.dim))
        vlb, vub = 0.2 * self.lb, 0.2 * self.ub
        for it in range(0, self.MaxIT):
            self.D_pl_pt(it)
            
            w = w_max - w_min * (it / self.MaxIT)
            
            v = w * v + c1 * bm.random.rand(self.N, self.dim) * (pbest - self.x) + c2 * bm.random.rand(self.N, self.dim) * (self.gbest - self.x)
            v = bm.clip(v, vlb, vub)

            self.x = self.x + v
            self.x = bm.clip(self.x, self.lb, self.ub)
            
            fit = self.fun(self.x)
            mask = fit < pbest_f
            pbest, pbest_f = bm.where(mask[:, None], self.x, pbest), bm.where(fit < pbest_f, fit, pbest_f)
            self.update_gbest(pbest, pbest_f)
            self.curve[it] = self.gbest_f