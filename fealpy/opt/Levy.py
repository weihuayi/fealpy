from ..backend import backend_manager as bm
from scipy.special import gamma

def levy(n, m, beta, Num, device):
    num = gamma(1 + beta) * bm.sin(bm.array(bm.pi * beta / 2))
    den = gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma_u = (num / den) ** (1 / beta)
    u = bm.random.randn(n, m, Num, device=device) * sigma_u
    v = bm.random.randn(n, m, Num, device=device)
    z = u / (abs(v) ** (1 / beta))
    return z