from ...backend import backend_manager as bm
from ...backend import TensorLike
from ..optimizer_base import Optimizer

class RemoraOptAlg(Optimizer):
    """
    Remora Optimization Algorithm.

    This class implements the Remora Optimization Algorithm, which simulates
    the behavior of remora fish attaching to hosts and using different feeding
    strategies. The algorithm combines experience-based attacks, host feeding,
    and multiple movement strategies for optimization.

    Attributes:
        pre (list): History of previous population positions for experience learning.
    """

    def __init__(self, options):
        """
        Initializes the Remora optimizer with configuration options.

        Parameters:
            options (dict): Configuration options for the optimizer.
        """
        super().__init__(options)
        self.pre = []
    
    def select_pre(self, it: int) -> TensorLike:
        """
        Select the previous population based on iteration count.

        Parameters:
            it (int): Current iteration number.

        Returns:
            TensorLike: The previous population position matrix.
        """
        if it == 0:
            R_pre = self.pre[0]
        else:
            R_pre = self.pre[it-1]
        return R_pre

    def experience_attack(self, R_pre: TensorLike) -> TensorLike:
        """
        Perform experience-based attack movement.

        Parameters:
            R_pre (TensorLike): Previous population positions.

        Returns:
            TensorLike: New positions after experience attack.
        """
        R_att = self.x + (self.x - R_pre) * bm.random.randn(self.N, self.dim)
        R_att = bm.clip(R_att, self.lb, self.ub)
        return R_att
    
    def host_feeding(self, it: int) -> TensorLike:
        """
        Perform host feeding strategy movement.

        Parameters:
            it (int): Current iteration number.

        Returns:
            TensorLike: New positions after host feeding.
        """
        V = 2 * (1 - it / self.MaxIT)
        B = 2 * V * bm.random.rand(self.N, self.dim) - V
        A = B * (self.x - self.c * self.gbest)
        R = self.x + A
        R = bm.clip(R, self.lb, self.ub)
        return R
    
    def woa_strategy(self, it: int) -> TensorLike:
        """
        Perform Whale Optimization Algorithm (WOA) inspired strategy.

        Parameters:
            it (int): Current iteration number.

        Returns:
            TensorLike: New positions after WOA strategy.
        """
        D = bm.abs(self.gbest - self.x)
        a = -(1 + it / self.MaxIT)
        alpha = bm.random.rand(self.N, 1) * (a - 1) + 1
        R = D * bm.exp(alpha) * bm.cos(2 * bm.pi * alpha) + self.x
        R = bm.clip(R, self.lb, self.ub)
        return R

    def sfo_strategy(self) -> TensorLike:
        """
        Perform Sunflower Optimization (SFO) inspired strategy.

        Returns:
            TensorLike: New positions after SFO strategy.
        """
        R_rand = self.x[bm.random.randint(0, self.N, (self.N,))]
        R = self.gbest - (bm.random.rand(self.N, self.dim) * (self.gbest - R_rand) / 2 - R_rand)
        R = bm.clip(R, self.lb, self.ub)
        return R
    
    def run(self, params={'c':0.1}) -> None:
        """
        Execute the Remora optimization process.

        Runs the main optimization loop with experience attacks, host feeding,
        and multiple movement strategies.

        Parameters:
            params (dict, optional): Algorithm parameters with keys:
                - c (float): Control parameter for host feeding strategy.

        Returns:
            None: The method updates the optimizer state internally.
        """
        fit = self.fun(self.x)
        gbest_index = bm.argmin(fit)
        self.gbest_f = fit[gbest_index]
        self.pre.append(bm.copy(self.x))
        
        # Sort population by fitness to get the best solution
        index = bm.argsort(fit)
        self.gbest = bm.copy(self.x[index[0]])  # Global best solution
        
        self.c = params.get('c')
        
        for it in range(self.MaxIT):
            self.D_pl_pt(it)  # Update any auxiliary data structures (method not shown here)
            R_pre = self.select_pre(it)
            R_att = self.experience_attack(R_pre)
            fit_att = self.fun(R_att)
            r = bm.random.rand(self.N,)
            mask1 = fit_att > fit
            mask2 = ~mask1 & (r > 0.5)
            R_host = self.host_feeding(it)
            R_woa = self.woa_strategy(it)
            R_sfo = self.sfo_strategy()
            self.x = bm.where(mask1[:, None], R_host,
                              bm.where(mask2[:, None], R_woa, R_sfo))
            fit = self.fun(self.x)
            # Update the global best solution
            self.update_gbest(self.x, fit)
            # Track the fitness of the global best solution
            self.curve[it] = self.gbest_f
            self.pre.append(bm.copy(self.x))