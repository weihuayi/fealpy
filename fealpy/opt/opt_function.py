from fealpy.backend import backend_manager as bm
from scipy.special import gamma

def levy(n, m, beta):
    num = gamma(1 + beta) * bm.sin(bm.array(bm.pi * beta / 2))
    den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma_u = (num / den) ** (1 / beta)
    u = bm.random.randn(n, m) * sigma_u
    v = bm.random.randn(n, m)
    z = u / (bm.abs(v) ** (1 / beta))
    return z


def initialize(pop_size, dim, ub, lb, way: int = 0):
    if way == 0:
        pop = lb + bm.random.rand(pop_size, dim) * (ub - lb)
    
    return pop