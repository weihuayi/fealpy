from sympy import *
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian

class Parabolic2dData: 
    def __init__(self,u, x, y, t, D=[0, 1, 0, 1], T=[0, 1]):
        self._domain = D 
        self._duration = T 
        self.u = lambdify([x,y,t], sympify(u))
        self.f = lambdify([x,y,t],diff(u,t,1)-diff(u,x,2)-diff(u,y,2))
        self.t = t
        
    def domain(self):
        return self._domain

    def duration(self):
        return self._duration 

    def solution(self, p, t):
        x = bm.array(p[..., 0])
        y = bm.array(p[..., 1])
        t = bm.array(t)
        return self.u(x,y,t) 


    def init_solution(self, p):
        x = bm.array(p[..., 0])
        y = bm.array(p[..., 1])
        return self.u(x,y,self._duration[0])
    
    @cartesian
    def source(self, p, t):
        x = bm.array(p[..., 0])
        y = bm.array(p[..., 1])
        return self.f(x,y,t)
    
    def gradient(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pass
       # return self.dudx(p,t)
    
    def dirichlet(self, p, t):
        return self.solution(p, t)
        
    # def source(self, p, t):
    #     """
    #     @brief 方程右端项 

    #     @param[in] p numpy.ndarray, 空间点
    #     @param[in] t float, 时间点 

    #     @return 方程右端函数值
    #     """
    #     pi = np.pi
    #     x = p[..., 0]
    #     y = p[..., 1]
    #     return np.zeros(x.shape)