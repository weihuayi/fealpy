from fealpy.backend import backend_manager as bm

def Logistic(pop_size, dim, miu=4):
    rand = bm.random.rand(pop_size, dim)
    for i in range(1, dim):
        rand[:, i] = miu * rand[:, i - 1] * (1 - rand[:, i - 1])
    return rand

def Tent(pop_size, dim, tent=1.1):
    rand = bm.random.rand(pop_size, dim)
    for j in range(1, dim):
        mask = rand[:, j - 1] < tent
        rand[:, j] = bm.where(mask, rand[:, j - 1] / tent, (1 - rand[:, j - 1]) / (1 - tent))

def Cubic(pop_size, dim):
    rand = bm.random.rand(pop_size, dim)
    cubic = 1
    for j in range(1, dim):
        rand[:, j] = cubic * rand[:, j - 1] * (1 - rand[:, j - 1] ** 2)
    return rand 

def Chebyshev(pop_size, dim, chebyshev=4):
    rand = bm.random.rand(pop_size, dim)
    for j in range(1, dim):
        rand[:, j] = bm.cos(chebyshev * bm.arccos(rand[:, j - 1]))
    return rand

def Piecewise(pop_size, dim, p=1):
    rand = bm.random.rand(pop_size, dim)
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
    return rand

def Sinusoidal(pop_size, dim, sinusoidal=2):
    rand = bm.random.rand(pop_size, dim)
    for j in range(1, dim):
        rand[:, j] = sinusoidal * rand[:, j - 1] ** 2 * bm.sin(bm.pi * rand[:, j - 1])
    return rand

def Icmic(pop_size, dim ,icmic=2):
    rand = bm.random.rand(pop_size, dim)
    for j in range(1, dim):
            rand[:, j] = bm.sin(icmic / rand[:, j - 1])
    return rand

def Bernouli(pop_size, dim, lammda=0.4):
    rand = bm.random.rand(pop_size, dim)
    prev_col = rand[:, :-1]
    condition = prev_col < (1 - lammda)
    rand[:, 1:] = bm.where(
        condition,
        prev_col / (1 - lammda), 
        (prev_col - 1 + lammda) / lammda
    )
    return rand   

def Sine(pop_size, dim, alpha=1, beta=1):
    rand = bm.random.rand(pop_size, dim)
    for j in range(1, dim):
        rand[:, j] =alpha * bm.sin(beta * rand[:, j - 1])
    return rand

def Circle(pop_size, dim, a=0.5, b=0.6):
    rand = bm.random.rand(pop_size, dim)
    for j in range(1, dim):
            rand[:, j] = (rand[:, j -1] + a - b / (2 * bm.pi) * bm.sin(2 * bm.pi * rand[:, j - 1])) % 1
    return rand


def initialize(pop_size, dim, ub, lb, method=None):
    """
    Initialize a population with various method maps.

    Parameters:
    -----------
    pop_size : int
        Number of individuals in the population.
    dim : int
        Number of dimensions for each individual.
    ub : list
        Upper bounds for each dimension. Must be a list of length `dim`.
    lb : list
        Lower bounds for each dimension. Must be a list of length `dim`.
    method : callable, optional, default=None
        A function defining the chaotic map to generate the population.
        If None, a random distribution is used for initialization.

    Returns:
    --------
    pop : Tensor
        Initialized population of shape (pop_size, dim).
    """

    if isinstance(ub, (list, tuple)) and isinstance(lb, (list, tuple)):
        if len(ub) != dim or len(lb) != dim:
            raise ValueError(f"Lengths of 'ub' and 'lb' must match 'dim'. "
                             f"Received: len(ub)={len(ub)}, len(lb)={len(lb)}, dim={dim}")
    elif not isinstance(ub, (float, int)) or not isinstance(lb, (float, int)):
        raise TypeError("Both 'ub' and 'lb' must be either scalars or lists/tuples of length 'dim'.")
    

    pop = bm.zeros([pop_size, dim])
    if method == None:
        rand = bm.random.rand(pop_size, dim)
    else:
        rand = method(pop_size, dim)

    if isinstance(ub, (float, int)):
            pop = lb + rand * (ub - lb)
    else:
        for i in range(dim):
            pop[:, i] = rand[:, i] * (ub[i] - lb[i]) + lb[i]    
    
    return pop



if __name__ == "__main__":
    lb = [-1.5, -0.5]
    ub = [1.5, 2.5]
    x0 = initialize(5, 2, ub, lb, method=Cubic)
    print(initialize.__doc__)