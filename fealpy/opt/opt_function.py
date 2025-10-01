from scipy.special import gamma
from ..typing import TensorLike
from fealpy.backend import backend_manager as bm
from .chaos import *

def generate_reference_points_layer(m: int, H: int):
    """
    Generate reference points using Das and Dennis's simplex-lattice method.

    Parameters:
        m (int): Objective space dimension.
        H (int): Number of divisions along each axis.

    Returns:
        Tensor: Reference points array, shape (C(m+H-1, H), m).

    Notes:
        Produces points on the unit simplex in m-dimensional space.
    """
    def recursive_combination(m, H):
        if m == 1:
            yield [H]
        else:
            for i in range(H + 1):
                for tail in recursive_combination(m - 1, H - i):
                    yield [i] + tail
    points = bm.array(list(recursive_combination(m, H))) / H
    return points

def generate_reference_points_double_layer(M, H1, H2):
    """
    Generate two-layer reference points for better diversity.

    Returns:
        Tensor: Combined reference points, shape (n_points, M).
    """
    coarse_points = generate_reference_points_layer(M, H1)
    fine_points = generate_reference_points_layer(M, H2)

    all_points = []
    for cp in coarse_points:
        alpha = 1 / H1
        refined_points = (1 - alpha) * cp + alpha * fine_points
        all_points.append(refined_points)
    return bm.concatenate(all_points)

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

    # if type(ub) != type(lb):
    #     raise TypeError("'ub' and 'lb' must be of the same type.")

    # if isinstance(ub, (list, tuple, TensorLike)):
    #     if len(ub) != dim or len(lb) != dim:
    #         raise ValueError(f"Lengths of 'ub' and 'lb' must match 'dim'. "
    #                         f"Received: len(ub)={len(ub)}, len(lb)={len(lb)}, dim={dim}")
    # elif isinstance(ub, (int, float)):
    #     pass  
    # else:
    #     raise TypeError("Both 'ub' and 'lb' must be either scalars or lists/tuples/TensorLike of length 'dim'.")
    
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