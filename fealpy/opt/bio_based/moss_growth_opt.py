from ...backend import backend_manager as bm
from ..optimizer_base import Optimizer

class MossGrowthOpt(Optimizer):
    """
    Moss Growth Optimization (MGO) algorithm, subclass of Optimizer.

    This class implements the Moss Growth Optimization (MGO) algorithm, a population-based optimization 
    method inspired by the growth process of moss. The algorithm simulates the spread of moss through 
    iterative processes, adjusting the solutions based on fitness and diversity.

    Parameters:
        option (dict): Configuration options for the optimizer, such as initial solution, population size, 
                       maximum iterations, dimensionality, bounds, and objective function.
    
    Reference:
    ~~~~~~~~~~
    Boli Zheng, Yi Chen, Chaofan Wang, Ali Asghar Heidari, Lei Liu, Huiling Chen. 
    The moss growth optimization (MGO): concepts and performance. 
    Journal of Computational Design and Engineering, 2024, 11, 184-221.
    """
    def __init__(self, option) -> None:
        """
        Initializes the Moss Growth Optimization (MGO) optimizer by calling the parent class constructor.

        Parameters:
            option (dict): Configuration options for the optimizer.
        """
        super().__init__(option)


    def run(self, params={'w':2, 'd1':0.2, 'rec_num':10}):
        """
        Runs the Moss Growth Optimization (MGO) algorithm.

        This method performs the main optimization loop, adjusting the population based on fitness, diversity,
        and movement strategies, and tracking the global best solution over iterations.
        """
        w = params.get('w')
        d1 = params.get('d1')
        rec_num = params.get('rec_num')
        options = self.options
        x = options["x0"]
        N = options["NP"]
        fit = self.fun(x)[:, None]
        MaxIT = options["MaxIters"]
        dim = options["ndim"]
        lb, ub = options["domain"]
        gbest_index = bm.argmin(fit)
        gbest = x[gbest_index]
        gbest_f = fit[gbest_index]
        divide_num = int(bm.ceil(bm.array(dim / 4)))
        divide_num = max((divide_num, 1))
        rec = 0
        rM = bm.zeros((N, dim, rec_num))
        rM_fit = bm.zeros((1, N, rec_num))
        rM[:, :, rec] = x
        rM_fit[0, :, rec] = bm.squeeze(fit)
        # curve = bm.zeros((1, MaxIT))
        for it in range(0, MaxIT):
            cal_zero = bm.zeros((N, dim))
            div_num = bm.random.randint(0, dim, (dim,))
            index = x[:, div_num] > gbest[div_num]
            sum_index = bm.sum(index, axis=0)
            for i in range(divide_num):
                if sum_index[i] < N / 2:
                    index[:, i] = ~(index[:, i])
                cal_zero = index[:,i][:, None] * x
            cal = cal_zero[cal_zero.any(axis=1)]
            D = gbest - cal
            D_wind = bm.sum(D, axis=0) / D.shape[0]

            beta = D.shape[0]/ N
            gama = 1 / (bm.sqrt(bm.array(1 - beta ** 2)) + 2.2204e-16)
            step = w * (bm.random.rand(1, dim) - 0.5) * (1 - it / MaxIT)
            step2 = 0.1 * w * (bm.random.rand(1, dim) - 0.5) * (1 - it / MaxIT) * (1 + 1 / 2 * (1 + bm.tanh(beta / gama)) * (1 - it / MaxIT))
            step3 = 0.1 * (bm.random.rand(1) - 0.5) * (1 - it / MaxIT)

            act = 1 / 1 + (0.5 - 10 * bm.random.rand(1, dim))
            act[act >= 0.5] = 1
            act[act < 0.5] = 0

            x_new = x
            r1 = bm.random.rand(N, 1)
            x_new = ((r1 > d1) * (x + step * D_wind) + 
                    (r1 <= d1) * (x + step2 * D_wind))

            r2 = bm.random.rand(N,)
            r3 = bm.random.rand(N,)
            x_new = ((r2[:, None] < 0.8) * ((r3[:, None] > 0.5) * (x) + 
                                  (r3[:, None] <= 0.5) * ((1 - act) * x + act * gbest)) + 
                    (r2[:, None] >= 0.8) * x)
            x_new[:, div_num[0]] = ((r2 < 0.8) * ((r3 > 0.5) * (gbest[div_num[0]] + step3 * D_wind[div_num[0]]) + (r3 <= 0.5) * x_new[:, div_num[0]]) + 
                                    (r2 >= 0.8) * x_new[:, div_num[0]])
            x_new = x_new + (lb - x_new) * (x_new < lb) + (ub - x_new) * (x_new > ub)
            fit_new = self.fun(x_new)[:, None]
            gbest_idx = bm.argmin(fit_new)
            (gbest, gbest_f) = (x_new[gbest_idx], fit_new[gbest_idx]) if fit_new[gbest_idx] < gbest_f else (gbest, gbest_f)

            rec = rec + 1
            if rec == rec_num:
                lcost, Iindex = bm.min(rM_fit, axis=2), bm.argmin(rM_fit, axis=2)
                x = rM[bm.arange(N), :, Iindex[0, :]]
                fit = lcost.reshape(-1, 1)
                rec = 0
        return gbest, gbest_f