import numpy as np
from ..mesh.TriangleMesh import TriangleMesh  

class CahnHilliardData1:
    def __init__(self, t0, t1, alpha=0.125):
        self.t0 = t0
        self.t1 = t1
        self.epsilon = alpha/(2*np.sqrt(2)*np.arctanh(0.9945))

    def space_mesh(self, n=4):
        point = np.array([
            (0, 0),
            (1, 0),
            (1, 1),
            (0, 1)], dtype=np.float)
        cell = np.array([
            (1, 2, 0),
            (3, 0, 2)], dtype=np.int)

        mesh = TriangleMesh(point, cell)
        mesh.uniform_refine(n)
        return mesh

    def time_mesh(self, tau):
        n = int(np.ceil(self.t1 - self.t0)/tau)
        tau = (self.t1 - self.t0)/n
        return np.linspace(self.t0, self.t1, num=n+1), tau

    def initdata(self, p):
        x = p[..., 0]
        y = p[..., 1]
        epsilon = self.epsilon
        u0 = np.tanh((0.6-np.sqrt((x-1.0)**2+(y-1.0)**2))/(np.sqrt(2)*epsilon)) \
                + np.tanh((0.5-np.sqrt((x-3.0)**2+(y-1.0)**2))/(np.sqrt(2)*epsilon))
        return u0
     
    def solution(self, p, t):
        pass
    
    def gradient(self, p, t):
        pass
    
    def laplace(self, p, t):
        pass
    
    def neumann(self, p):
        """
        Neumann boundary condition
        """
        return 0 
    
    def source(self, p):
        rhs = np.zeros(p.shape[0:-1]) 
        return rhs
    
class CahnHilliardData2:
    def __init__(self,h,m):
        self.h = 1/256
        self.m = 4
        
    def initdata(self,p):
        h = self.h
        m = self.m
        
        x = p[..., 0]
        y = p[..., 1]
        
        epsilon = (h*m)/(2*np.sqrt(2)*np.arctanh(0.9945))
        u0 = np.tanh((0.1-np.sqrt((x-0.5)**2+(y-0.5)**2))/(np.sqrt(2)*epsilon))
        return u0
    
    def solution(self,p):
        pass
    
    def gradient(self,p):
        pass
    
    def laplace(self,p):
        pass
    
    def neumann(self,p):
        """
        Neumann boundary condition
        """
        return self.solution(p)
        
    def source(self,p):
        
        rhs = np.zeros(p.shape[0:-1]) 
        return rhs

    
class CahnHilliardData3:
    def __init__(self,h,m):
        self.h = 1/128
        self.m = 4
        
    def initdata(self,p):
        
        h = self.h
        m = self.m
        
        x = p[..., 0]
        y = p[..., 1]

        epsilon = (h*m)/(2*np.sqrt(2)*np.arctanh(0.9945))
        if (-0.25<x<2.75) and (-0.4<y<0.6):
            u0 = 1
        else:
            u0 = -1
        return u0
    
    def solution(self,p):
        pass
    
    def gradient(self,p):
        pass
    
    def laplace(self,p):
        pass
    
    def neumann(self,p):
        """
        Neumann boundary condition
        """
        return self.solution(p)
    
    def source(self,p):
        rhs = np.zeros(p.shape[0:-1]) 
        return rhs


    
class CahnHilliardData4:
    def __init__(self,epsilon):
        self.epsilon = 0.08
        
    def initdata(self,p):
        
        epsilon = self.epsilon
        x = p[..., 0]
        y = p[..., 1]
        u0 = np.tanh(((x-0.3)**2+y**2-0.25**2)/epsilon)*np.tanh(((x+0.3)**2+y**2-0.3**2)/epsilon)
        return u0
    
    def solution(self,p):
        pass
    
    def gradient(self,p):
        pass
    
    def laplace(self,p):
        pass
    
    def neumann(self,p):
        """
        Neumann boundary condition
        """
        return self.solution(p)
    
    def source(self,p):
        rhs = np.zeros(p.shape[0:-1]) 
        return rhs

