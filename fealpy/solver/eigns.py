import numpy as np
from numpy.linalg import norm
import pyamg


def picard(A, M, u0, tol=1e-12, atol = 1e-12, ml=None, sigma=None):
    if sigma is not None:
        A += sigma*M

    if ml is None:
        ml = pyamg.ruge_stuben_solver(A)
    else:
        if sigma is not None:
            print('Please make sure that you have shift matrix A!')

    d0 = (u0@A@u0)/(u0@M@u0)
    while True:
        u1 = ml.solve(d0*M@u0, x0=u0, tol=1e-12, accel='cg').reshape(-1)
        u1 /= np.max(np.abs(u1))
        L0 = u1@M@u1
        L1 = u1@A@u1
        d1 = L1/L0
        if np.max(np.abs(u1 - u0)) < tol or np.abs(d1 - d0) < atol:
            u0 = u1
            d0 = d1
            break
        else:
            u0 = u1
            d0 = d1

    if sigma is not None:
        d0 -= sigma
    return u0, d0



