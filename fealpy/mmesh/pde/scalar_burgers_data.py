from fealpy.backend import bm
from fealpy.decorator import cartesian
from fealpy.mesh import TriangleMesh, QuadrangleMesh
from sympy import *

class ScalarBurgersData:
    def __init__(self, u: str,var_list:list[str],
                 D = [0, 1, 0, 1],T = [0,1],Re = 1.0):
        '''
        标量Burgers方程
        '''
        u = sympify(u)
        self.TD = len(var_list)-1

        x = symbols(var_list[0])
        y = symbols(var_list[1])
        t = symbols(var_list[-1])
        
        self.u = lambdify(var_list, u, 'numpy')
        f_str = -1/Re*(diff(u, x, 2) + diff(u, y, 2)) + diff(u, t, 1)\
                + u*diff(u, x, 1) + u*diff(u, y, 1)
                
        self.grad_ux = lambdify(var_list, diff(u, x, 1))
        self.grad_uy = lambdify(var_list, diff(u, y, 1))
        
        if self.TD == 3:
            z = symbols(var_list[2])
            f_str -= diff(u, z, 2)
            self.grad_uz = lambdify(var_list, diff(u, z, 1))
            
        self.f = lambdify(var_list, f_str, 'numpy')
        
        self.D = D
        self.T = T
        self.u_str = u
        self.Re = Re
        
    def domain(self):
        return self.D
    
    def time_span(self):
        return self.T
    
    def set_mesh(self,nx = 40, ny = 40, meshtype='tri'):
        vertices = bm.array([[self.D[0], self.D[2]],
                             [self.D[1], self.D[2]],
                             [self.D[1], self.D[3]],
                             [self.D[0], self.D[3]]], dtype=bm.float64)
        if meshtype == 'tri':
            self.mesh = TriangleMesh.from_box(self.D, nx=nx, ny=ny)
        elif meshtype == 'quad':
            self.mesh = QuadrangleMesh.from_box(self.D, nx=nx, ny=ny)
        elif meshtype == 'cross_tri':
            self.mesh = TriangleMesh.from_box_cross_mesh(self.D, nx=nx, ny=ny)
        else:
            raise ValueError('meshtype error')
        self.mesh.meshdata['vertices'] = vertices
        return self.mesh
    
    @cartesian
    def solution(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        if self.TD == 3:
            z = p[..., 2]
            return self.u(x, y, z, t)
        return self.u(x, y, t)
    
    @cartesian
    def init_solution(self, p):
        return self.solution(p, self.T[0])
    
    @cartesian
    def moving_init_solution(self, p):
        return self.solution(p, self.T[0])
    
    @cartesian
    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        if self.TD == 3:
            z = p[..., 2]
            return self.f(x, y, z, t)
        return self.f(x, y, t)
    
    @cartesian
    def gradient(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        val = bm.zeros_like(p)
        if self.TD == 3:
            z = p[..., 2]
            val[..., 0] = self.grad_ux(x, y, z, t)
            val[..., 1] = self.grad_uy(x, y, z, t)
            val[..., 2] = self.grad_uz(x, y, z, t)
            return val
        val[..., 0] = self.grad_ux(x, y, t)
        val[..., 1] = self.grad_uy(x, y, t)
        return val

    @cartesian
    def dirichlet(self, p, t):
        return self.solution(p, t)