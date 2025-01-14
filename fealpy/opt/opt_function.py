from fealpy.backend import backend_manager as bm
from scipy.special import gamma

import time
def levy(n, m, beta):
    """
    Levy flight
    """
    num = gamma(1 + beta) * bm.sin(bm.array(bm.pi * beta / 2))
    den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma_u = (num / den) ** (1 / beta)
    u = bm.random.randn(n, m) * sigma_u
    v = bm.random.randn(n, m)
    z = u / (bm.abs(v) ** (1 / beta))
    return z


def initialize(pop_size, dim, ub, lb, way: int = 0):
    """
    Initialization

    way = 1 -- 10 : chaos 
    """
    pop = bm.zeros((pop_size, dim))
    rand = bm.random.rand(pop_size, dim)

    if way == 0:
        rand = rand
    
    # Tent
    elif way == 1:
        tent = 1.1
        for j in range(1,dim):
            mask = rand[:, j - 1] < tent
            rand[:, j] = bm.where(mask, rand[:, j - 1] / tent, (1 - rand[:, j - 1]) / (1 - tent))
    
    # Logistic
    elif way == 2:
        miu = 4
        for i in range(1, dim):
            rand[:, i] = miu * rand[:, i - 1] * (1 - rand[:, i - 1])

    # Cubic
    elif way == 3:
        cubic = 1
        for j in range(1, dim):
            rand[:, j] = cubic * rand[:, j - 1] * (1 - rand[:, j - 1] ** 2)

    # Chebyshev
    elif way == 4:
        chebyshev = 4
        for j in range(1, dim):
            rand[:, j] = bm.cos(chebyshev * bm.arccos(rand[:, j - 1]))

    # Piecewise
    elif way == 5:
        p = 1
        for j in range(1, dim):
            prev_col = rand[:, j - 1]
            cond1 = (0 < prev_col) & (prev_col < p)
            cond2 = (p <= prev_col) & (prev_col < 0.5)
            cond3 = (0.5 <= prev_col) & (prev_col < 1 - p)
            cond4 = (1 - p <= prev_col) & (prev_col < 1)
            rand[:, j] = bm.where(cond1, prev_col / p,
                            bm.where(cond2, (prev_col - p) / (0.5 - p),
                            bm.where(cond3, (1 - p - prev_col) / (0.5 - p),
                            bm.where(cond4, (1 - prev_col) / p, rand[:, j]))))

    # Sinusoidal
    elif way == 6:
        sinusoidal = 2
        for j in range(1, dim):
            rand[:, j] = sinusoidal * rand[:, j - 1] ** 2 * bm.sin(bm.pi * rand[:, j - 1])

    # Sine
    elif way == 7:
        beta = 1
        alpha = 1
        for j in range(1, dim):
            rand[:, j] =alpha * bm.sin(beta * rand[:, j - 1])

    # Icmic
    elif way == 8:
        icmic = 2
        for j in range(1, dim):
            rand[:, j] = bm.sin(icmic / rand[:, j - 1])

    # Circle
    elif way == 9:
        a = 0.5
        b = 0.6
        for j in range(1, dim):
            rand[:, j] = (rand[:, j -1] + a - b / (2 * bm.pi) * bm.sin(2 * bm.pi * rand[:, j - 1])) % 1

    # Bernoulli
    elif way == 10:
        lammda = 0.4
        prev_col = rand[:, :-1]
        condition = prev_col < (1 - lammda)
        rand[:, 1:] = bm.where(
            condition,
            prev_col / (1 - lammda), 
            (prev_col - 1 + lammda) / lammda
        )

    if isinstance(ub, (float, int)):
        pop = lb + rand * (ub - lb)
    else:
        for i in range(dim):
            pop[:, i] = rand[:, i] * (ub[i] - lb[i]) + lb[i]

    
    return pop

def PathPlanning(MAP, dataS, dataE, method):
    start_time = time.perf_counter()
    from fealpy.opt.particle_swarm_opt_alg import PathPlanningProblem, PSO, QPSO
    from fealpy.opt import initialize, opt_alg_options
    if MAP[dataS[0]][dataS[1]] != 0 or MAP[dataE[0]][dataE[1]] != 0: 
        print("Error: Wrong start point or end point") # 判断起点终点坐标是否有效
    else:
        textMAP = PathPlanningProblem(MAP, dataS, dataE)
        textMAP.builddata() # 建立地图字典
        fobj = lambda x: textMAP.fitness(x)
        
        # 算法参数
        N = 20
        MaxIT = 50
        lb = 0
        ub = 1
        dim = textMAP.data["landmark"].shape[0]
        
        # 调用算法
        xo = initialize(N, dim, ub, lb)
        option = opt_alg_options(xo, fobj, (lb, ub), N, MaxIters=MaxIT)
        optimizer = method(option)
        optimizer.run()
        
        result = textMAP.calresult(optimizer.gbest)
        
        # 输出
        result["path"] = [x for x, y in zip(result["path"], result["path"][1:] + [None]) if x != y]
        print('The optimal path distance: ', optimizer.gbest_f)
        print("The optimal path: ", result["path"])
        end_time = time.perf_counter()
        running_time = end_time - start_time
        textMAP.printMAP(result, running_time)
