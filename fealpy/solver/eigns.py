import numpy as np
from numpy.linalg import norm
import pyamg


def picard(A, M, u0, tol=1e-12, atol = 1e-12):
    ml = pyamg.ruge_stuben_solver(A)  
    d0 = (u0@A@u0)/(u0@M@u0)
    while True:
        u1 = ml.solve(d0*M@u0, x0=u0, tol=1e-12, accel='cg').reshape(-1)
        L0 = u1@M@u1
        L1 = u1@A@u1
        d1 = L1/L0
        if norm(u1 - u0, ord=1)/norm(u1, ord=1) < tol or np.abs(d1 - d0) < atol:
            u0 = u1/np.max(np.abs(u1))
            d0 = d1
            break
        else:
            u0 = u1/np.max(np.abs(u1))
            d0 = d1

    return u0, d0



