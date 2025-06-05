from scipy.special import gamma
from ..typing import TensorLike
from fealpy.backend import backend_manager as bm
from .chaos import *

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

    if isinstance(ub, (list, tuple, TensorLike)) and isinstance(lb, (list, tuple, TensorLike)):
        if len(ub) != dim or len(lb) != dim:
            raise ValueError(f"Lengths of 'ub' and 'lb' must match 'dim'. "
                             f"Received: len(ub)={len(ub)}, len(lb)={len(lb)}, dim={dim}")
    elif not isinstance(ub, (float, int)) or not isinstance(lb, (float, int)):
        raise TypeError("Both 'ub' and 'lb' must be either scalars or lists/tuples/TensorLike of length 'dim'.")
    

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