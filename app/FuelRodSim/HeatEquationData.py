import numpy as np

class Parabolic2dData:
    def domain(self):
        return [0, 1, 0, 1]

    def duration(self):
        return [0, 0.1]
    
    
    def solution(self,p,t):
        pi = np.pi
        x = p[..., 0]
        y = p[..., 1]
        return np.sin(pi*x)*np.sin(pi*y)*np.exp(-2*pi*t) 
    
    def init_solution(self, p):
        pi = np.pi
        x = p[..., 0]
        y = p[..., 1]
        return np.sin(pi*x)*np.sin(pi*y)
        
    
    def source(self, p, t):
        """
        @brief 方程右端项 

        @param[in] p numpy.ndarray, 空间点
        @param[in] t float, 时间点 

        @return 方程右端函数值
        """
        pi = np.pi
        x = p[..., 0]
        y = p[..., 1]
        return np.zeros(x.shape)

     
    def dirichlet(self, p,t):
        
        return self.solution(p,t)
    
class Parabolic3dData:
    def domain(self):
        return [0, 1, 0, 1, 0, 1]

    def duration(self):
        return [0, 1]

    def source(self,p,t):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        return np.zeros_like(x)

    def solution(self, p, t):
        pi = np.pi
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        return np.sin(pi * x) * np.sin(pi * y) * np.sin(pi * z) * np.exp(-3 * pi * t)
    
    def init_solution(self, p):
        pi = np.pi
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        return np.sin(pi*x)*np.sin(pi*y)*np.sin(pi * z) 

    def dirichlet(self, p, t):
        
        return self.solution(p, t)
    
class FuelRod3dData:
    def domain(self):
        return [0, 1, 0, 1, 0, 1]

    def duration(self):
        return [0, 1]

    def source(self,p,t):
        return 0

    def dirichlet(self, p, t):
        return np.array([500])
    
class FuelRod2dData:
    def domain(self):
        return [0, 1, 0, 1]

    def duration(self):
        return [0, 1]

    def source(self,p,t):
        return 0

    def dirichlet(self, p, t):
        return np.array([500])