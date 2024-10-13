from ..backend import backend_manager as bm
from scipy.stats import norm
from scipy.special import gamma

def levy(n, m, beta):
    num = gamma(1 + beta) * bm.sin(bm.pi * beta / 2)
    den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma_u = (num / den) ** (1 / beta)
    u = norm.rvs(0, sigma_u, size=(n, m))
    v = norm.rvs(0, 1, size=(n, m))
    z = u / (bm.abs(v) ** (1 / beta))
    return z