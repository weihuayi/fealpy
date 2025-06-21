from sympy import *
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
from fealpy.typing import TensorLike

class Parabolic2dData(): 
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

class Parabolic3dData():
    def __init__(self,u, x, y, z, t, D=[0, 1, 0, 1, 0, 1], T=[0, 1]):
        self._domain = D 
        self._duration = T 
        self.u = lambdify([x,y,z,t], sympify(u))
        self.f = lambdify([x,y,z,t],diff(u,t,1)-diff(u,x,2)-diff(u,y,2)-diff(u,z,2),)
        self.t = t
    
    def domain(self):
        return self._domain

    def duration(self):
        return self._duration
    
    @cartesian
    def source(self,p,t):
        x = bm.array(p[..., 0])
        y = bm.array(p[..., 1])
        z = bm.array(p[..., 2])
        return self.f(x,y,z,t)
    

    def solution(self, p, t):
        x = bm.array(p[..., 0])
        y = bm.array(p[..., 1])
        z = bm.array(p[..., 2])
        t = bm.array(t)
        return self.u(x,y,z,t) 
    
    def init_solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        return self.u(x,y,z,self._duration[0])

    def dirichlet(self, p, t):
        
        return self.solution(p, t)
    
    
# 平面应变问题
class BoxDomainPolyData2D():

    def domain(self):
        return [0, 1, 0, 1]

    @cartesian
    def source(self, points: TensorLike, index=None) -> TensorLike:
        x = points[..., 0]
        y = points[..., 1]
        
        val = bm.zeros(points.shape, dtype=points.dtype, device=bm.get_device(points))
        val[..., 0] = 35/13 * y - 35/13 * y**2 + 10/13 * x - 10/13 * x**2
        val[..., 1] = -25/26 * (-1 + 2 * y) * (-1 + 2 * x)
        
        return val
    
    @cartesian
    def solution(self, points: TensorLike) -> TensorLike:
        x = points[..., 0]
        y = points[..., 1]
        
        val = bm.zeros(points.shape, dtype=points.dtype, device=bm.get_device(points))
        val[..., 0] = x * (1 - x) * y * (1 - y)
        val[..., 1] = 0
        
        return val
    
    
    def dirichlet(self, points: TensorLike) -> TensorLike:

        return self.solution(points)

class BoxDomainTriData2D():

    def domain(self):
        return [0, 1, 0, 1]

    @cartesian
    def source(self, points: TensorLike, index=None) -> TensorLike:
        x = points[..., 0]
        y = points[..., 1]
        
        val = bm.zeros(points.shape, dtype=points.dtype, device=bm.get_device(points))
        val[..., 0] = (22.5 * bm.pi**2) / 13 * bm.sin(bm.pi * x) * bm.sin(bm.pi * y)
        val[..., 1] = - (12.5 * bm.pi**2) / 13 * bm.cos(bm.pi * x) * bm.cos(bm.pi * y)
        
        return val
    
    @cartesian
    def solution(self, points: TensorLike) -> TensorLike:
        x = points[..., 0]
        y = points[..., 1]
        
        val = bm.zeros(points.shape, dtype=points.dtype, device=bm.get_device(points))
        val[..., 0] = bm.sin(bm.pi * x) * bm.sin(bm.pi * y)
        val[..., 1] = 0
        
        return val
    
    def dirichlet(self, points: TensorLike) -> TensorLike:
        return self.solution(points)
    
class BoxDomainPolyUnloaded3d():
    def __init__(self):
        pass
        
    def domain(self):
        return [0, 1, 0, 1, 0, 1]
    
    @cartesian
    def solution(self, points: TensorLike):
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        val = bm.zeros(points.shape, dtype=points.dtype)
        val[..., 0] = 2*x**3 - 3*x*y**2 - 3*x*z**2
        val[..., 1] = 2*y**3 - 3*y*x**2 - 3*y*z**2
        val[..., 2] = 2*z**3 - 3*z*y**2 - 3*z*x**2
        
        return val

    @cartesian
    def source(self, points: TensorLike):
        val = bm.zeros(points.shape, dtype=points.dtype)
        
        return val
    
    def dirichlet(self, points: TensorLike) -> TensorLike:

        return self.solution(points)
    
class BoxDomainPolyLoaded3d():
    def domain(self):
        return [0, 1, 0, 1, 0, 1]
    
    @cartesian
    def source(self, points: TensorLike):
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        val = bm.zeros(points.shape, dtype=bm.float64)
        mu = 1
        factor1 = -400 * mu * (2 * y - 1) * (2 * z - 1)
        term1 = 3 * (x ** 2 - x) ** 2 * (y ** 2 - y + z ** 2 - z)
        term2 = (1 - 6 * x + 6 * x ** 2) * (y ** 2 - y) * (z ** 2 - z)
        val[..., 0] = factor1 * (term1 + term2)

        factor2 = 200 * mu * (2 * x - 1) * (2 * z - 1)
        term1 = 3 * (y ** 2 - y) ** 2 * (x ** 2 - x + z ** 2 - z)
        term2 = (1 - 6 * y + 6 * y ** 2) * (x ** 2 - x) * (z ** 2 - z)
        val[..., 1] = factor2 * (term1 + term2)

        factor3 = 200 * mu * (2 * x - 1) * (2 * y - 1)
        term1 = 3 * (z ** 2 - z) ** 2 * (x ** 2 - x + y ** 2 - y)
        term2 = (1 - 6 * z + 6 * z ** 2) * (x ** 2 - x) * (y ** 2 - y)
        val[..., 2] = factor3 * (term1 + term2)

        return val

    @cartesian
    def solution(self, points: TensorLike):
        x = points[..., 0]
        y = points[..., 1]
        z = points[..., 2]
        val = bm.zeros(points.shape, dtype=bm.float64)

        mu = 1
        val[..., 0] = 200*mu*(x-x**2)**2 * (2*y**3-3*y**2+y) * (2*z**3-3*z**2+z)
        val[..., 1] = -100*mu*(y-y**2)**2 * (2*x**3-3*x**2+x) * (2*z**3-3*z**2+z)
        val[..., 2] = -100*mu*(z-z**2)**2 * (2*y**3-3*y**2+y) * (2*x**3-3*x**2+x)

        return val
    
    def dirichlet(self, points: TensorLike) -> TensorLike:

        return bm.zeros(points.shape, dtype=points.dtype)

    

    
        
    
        

  