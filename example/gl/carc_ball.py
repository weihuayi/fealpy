import numpy as np
from quaternion import Quaternion

class CArcBall():
    def __init__(self, w, h, x, y):
        self.r = np.min([w, h])/2
        self.center = np.array([w, h])/2
        self.position = self.project_to_ball(x, y)

    def update(self, x, y):
        p0 = self.position
        p1 = self.project_to_ball(x, y)
        q = np.zeros(4, dtype=np.float_)
        q[:-1] = np.cross(p0, p1)
        q[-1] = p0@p1
        q = Quaternion(q)
        self.position = p1
        return q

    def project_to_ball(self, x, y):
        p = np.array([x, y, 0], dtype=np.float_)/self.r
        r0 = p[0]**2 + p[1]**2
        if(r0 > 1):
            p = p/np.sqrt(r0)
        else:
            p[2] = 1 - r0
        return p
        
