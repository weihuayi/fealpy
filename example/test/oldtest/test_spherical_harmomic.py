import numpy as np
from scipy.special import sph_harm 

class RodCoilModel():
    def __init__(self):
        pass

    def project(self, L, integrator):

        l = range(L+1)
        ll = np.repeat(l, range(1, 2*(L+1), 2))
        m = np.zeros(len(ll))
        start = 0
        for i in range(0, L+1):
            m[start:start+2*i+1]=range(-i, i+1) 
            start += 2*i+1
        
        # theta: [0, pi], phi: [0, 2*pi]
        a = sph_harm(m, ll, theta, phi) 



