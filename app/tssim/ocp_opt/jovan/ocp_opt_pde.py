#!/usr/bin/python3
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian
import sympy as sp

class example_1:
    def __init__(self, c=1): 
        self.c = c
        self.manager, = bm._backends

        t, x1, x2 = sp.symbols('t, x1, x2', real=True)
        self.x1 = x1
        self.x2 = x2
        self.t = t

        self.y = (1-x1)*(1-x2)*x1*x2*sp.exp(t)
        self.z = (1-x1)*(1-x2)*x1*x2*(t-1)**3
        self.u = -(1-x1)*(1-x2)*x1*x2*(t-1)**3

        self.p0 = -(1+x1**2)*(1-x2)*x2*(1-2*x1)*sp.exp(t)
        self.p1= -(1+x2**2)*(1-x1)*x1*(1-2*x2)*sp.exp(t)
        self.p = sp.Matrix([self.p0, self.p1])
        
        self.q0 = -sp.pi*sp.cos(sp.pi*x1)*sp.sin(sp.pi*x2)*(t-1)**2
        self.q1 = -sp.pi*sp.sin(sp.pi*x1)*sp.cos(sp.pi*x2)*(t-1)**2
        self.q = sp.Matrix([self.q0, self.q1])
        
        self.A00 = 1+x1**2
        self.A11 = 1+x2**2
        self.A = sp.Matrix([[self.A00, 0], [0, self.A11]])
    
    def domain(self):
        return [0, 1, 0, 1]
    
    @cartesian
    def y_solution(self, space, time):
        x1 = self.x1
        x2 = self.x2
        t = self.t
        result = sp.lambdify([x1,x2,t], self.y, self.manager)
        return result(space[...,0], space[...,1], time)

    @cartesian
    def y_t_solution(self, space, time):
        x1 = self.x1
        x2 = self.x2
        t = self.t
        result = sp.lambdify([x1,x2,t], sp.diff(self.y, t), self.manager)
        return result(space[...,0], space[...,1], time)
         
    @cartesian
    def z_solution(self, space, time):
        x1 = self.x1
        x2 = self.x2
        t = self.t
        result = sp.lambdify([x1,x2,t], self.z, self.manager)
        return result(space[...,0], space[...,1], time)
    
    @cartesian
    def z_t_solution(self, space, time):
        x1 = self.x1
        x2 = self.x2
        t = self.t
        result = sp.lambdify([x1,x2,t], sp.diff(self.z, t), self.manager)
        return result(space[...,0], space[...,1], time)
    
    @cartesian
    def u_solution(self, space, time):
        x1 = self.x1
        x2 = self.x2
        t = self.t
        result = sp.lambdify([x1,x2,t], self.u, self.manager)
        return result(space[...,0], space[...,1], time)

    @cartesian
    def p_solution(self, space, time):
        '''
        val1 = -(1+x**2)*(1-y)*y*(1-2*x)*bm.exp(time)
        val2 = -(1+y**2)*(1-x)*x*(1-2*y)*bm.exp(time)   
        result = bm.stack([val1, val2], axis=-1)
        '''
        x1 = self.x1
        x2 = self.x2
        t = self.t
        
        x = space[..., 0]
        y = space[..., 1]
        result = bm.zeros_like(space)
        p0 = sp.lambdify([x1,x2,t], self.p0, self.manager)
        p1 = sp.lambdify([x1,x2,t], self.p1, self.manager)
        result[...,0] = p0(x, y, time) 
        result[...,1] = p1(x, y, time) 
        return result
    
    @cartesian
    def q_solution(self, space, time):
        x1 = self.x1
        x2 = self.x2
        t = self.t
        val = sp.Matrix([self.q0, self.q1])
        result = sp.lambdify([x1,x2,t], val, self.manager)
        output = result(space[..., 0], space[..., 1], time)  
        reshape_output = output.transpose(2, 0, 1).squeeze()
        return reshape_output

    
    @cartesian
    def A_matirx(self, space): 
        x = space[..., 0]
        y = space[..., 1]
        result = bm.zeros(space.shape[:-1]+(2,2)) 
        result[..., 0, 0] = 1+x**2 
        result[..., 1, 1] = 1+y**2
        return result 
    
    @cartesian
    def A_inverse(self, space):
        x = space[..., 0]
        y = space[..., 1]
        result = bm.zeros(space.shape[:-1]+(2,2)) 
        result[..., 0, 0] = 1/(1+x**2)
        result[..., 1, 1] = 1/(1+y**2)
        return result 

    @cartesian
    def f_fun(self, space, time, index=None):
        y = self.y
        p = self.p
        u = self.u
        t = self.t
        x1 = self.x1
        x2 = self.x2
        self.f = sp.diff(y, t, 2) + sp.diff(p[0], x1) + sp.diff(p[1], x2) + y - u
        result = sp.lambdify([x1,x2,t], self.f, self.manager) 
        return result(space[...,0], space[...,1], time)
    
    @cartesian
    def p_d_fun(self,  space, time):
        p = self.p
        q = self.q
        z = self.z
        A = self.A
        t = self.t
        x1 = self.x1
        x2 = self.x2
        grad_z = sp.Matrix([sp.diff(z, x1), sp.diff(z, x2)])
        A_inv = A.inv()
        p_d = p + A_inv*q + grad_z
        self.p_d = p_d
        result = sp.lambdify([x1,x2,t], self.p_d, self.manager)
        output = result(space[..., 0], space[..., 1], time)  # 原始形状为 (2, 1, 200, 10)
        output = output.transpose(2, 3, 0, 1).squeeze()  # 变为 (200, 10, 2)
        return output  
    
    @cartesian
    def y_d_fun(self,  space, time):
        q = self.q
        y = self.y
        z = self.z
        t = self.t
        x1 = self.x1
        x2 = self.x2
        #self.y_d = y - sp.diff(z, t, 2) + sp.diff(q[0], x1) + sp.diff(q[1], x2)
        self.y_d = y - sp.diff(z, t, 2) - sp.diff(q[0], x1) - sp.diff(q[1], x2) - z
        result = sp.lambdify([x1,x2,t], self.y_d, self.manager) 
        return result(space[...,0], space[...,1], time)
