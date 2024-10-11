#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: ocp_opt_pde.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Thu 05 Sep 2024 04:57:54 PM CST
	@bref 
	@ref 
'''  
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.decorator import cartesian
import sympy as sp

class example_1:
    def __init__(self): 
        t, x1, x2 = sp.symbols('t, x1, x2', real=True)
        self.x1 = x1
        self.x2 = x2
        self.t = t
        self.c = 1

        self.y = (1-x1)*(1-x2)*x1*x2*x1*x2*sp.exp(t)
        self.z = (1-x1)*(1-x2)*x1*x2*x1*x2*(t-1)**3
        self.u = -(1-x1)*(1-x2)*x1*x2*x1*x2*(t-1)**3
        p0 = -(1+x1**2)*(1-x2)*x2*(1-2*x1)*sp.exp(t)
        p1 = -(1+x2**2)*(1-x1)*x1*(1-2*x2)*sp.exp(t)
        self.p = sp.Matrix([p0, p1])
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
        manager, = bm._backends
        x1 = self.x1
        x2 = self.x2
        t = self.t
        result = sp.lambdify([x1,x2,t], self.y, manager)
        return result(space[...,0], space[...,1], time)

    @cartesian
    def y_t_solution(self, space, time):
        manager, = bm._backends
        x1 = self.x1
        x2 = self.x2
        t = self.t
        result = sp.lambdify([x1,x2,t], sp.diff(self.y, t), manager)
        return result(space[...,0], space[...,1], time)
         
    @cartesian
    def z_solution(self, space, time):
        manager, = bm._backends
        x1 = self.x1
        x2 = self.x2
        t = self.t
        result = sp.lambdify([x1,x2,t], self.z, manager)
        return result(space[...,0], space[...,1], time)
    
    @cartesian
    def z_t_solution(self, space, time):
        manager, = bm._backends
        x1 = self.x1
        x2 = self.x2
        t = self.t
        result = sp.lambdify([x1,x2,t], sp.diff(self.z, t), manager)
        return result(space[...,0], space[...,1], time)
    
    @cartesian
    def u_solution(self, space, time):
        manager, = bm._backends
        x1 = self.x1
        x2 = self.x2
        t = self.t
        result = sp.lambdify([x1,x2,t], self.u, manager)
        return result(space[...,0], space[...,1], time)

    @cartesian
    def px1_solution(self, space, time):
        x = space[..., 0]
        y = space[..., 1]
        result = -(1+x**2)*(1-y)*y*(1-2*x)*bm.exp(time)
        return result 
    
    @cartesian
    def px2_solution(self, space, time):
        x = space[..., 0]
        y = space[..., 1]
        result= -(1+y**2)*(1-x)*x*(1-2*y)*bm.exp(time)
        return result 
    
    @cartesian
    def qx1_solution(self, space, time):
        manager, = bm._backends
        x1 = self.x1
        x2 = self.x2
        t = self.t
        result = sp.lambdify([x1,x2,t], self.q0, manager)
        return result(space[...,0], space[...,1], time)
    
    @cartesian
    def qx2_solution(self, space, time):
        manager, = bm._backends
        x1 = self.x1
        x2 = self.x2
        t = self.t
        result = sp.lambdify([x1,x2,t], self.q1, manager)
        return result(space[...,0], space[...,1], time)
    
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
        manager, = bm._backends
        y = self.y
        p = self.p
        u = self.u
        t = self.t
        x1 = self.x1
        x2 = self.x2
        self.f = sp.diff(y, t, 2) + sp.diff(p[0], x1) + sp.diff(p[1], x2) + y - u
        result = sp.lambdify([x1,x2,t], self.f, manager) 
        return result(space[...,0], space[...,1], time)
    
    @cartesian
    def p_dx1_fun(self,  space, time):
        manager, = bm._backends
        p = self.p
        q = self.q
        z = self.z
        A = self.A
        t = self.t
        x1 = self.x1
        x2 = self.x2
        grad_z = sp.Matrix([sp.diff(z, x1), sp.diff(z, x2)])
        p_d = p + A**(-1)*q + grad_z 
        self.p_dx1 = p_d[0] 
        result = sp.lambdify([x1,x2,t], self.p_dx1, manager) 
        return result(space[...,0], space[...,1], time)
    
    @cartesian
    def p_dx2_fun(self,  space, time):
        manager, = bm._backends
        p = self.p
        q = self.q
        z = self.z
        A = self.A
        t = self.t
        x1 = self.x1
        x2 = self.x2
        grad_z = sp.Matrix([sp.diff(z, x1), sp.diff(z, x2)])
        p_d = p + A**(-1)*q + grad_z 
        self.p_dx2 = p_d[1] 
        result = sp.lambdify([x1,x2,t], self.p_dx2, manager) 
        return result(space[...,0], space[...,1], time)
    
    @cartesian
    def y_d_fun(self,  space, time):
        manager, = bm._backends
        q = self.q
        y = self.y
        z = self.z
        t = self.t
        x1 = self.x1
        x2 = self.x2
        self.y_d = y - sp.diff(z, t, 2) + sp.diff(q[0], x1) + sp.diff(q[1], x2)
        result = sp.lambdify([x1,x2,t], self.y_d, manager) 
        return result(space[...,0], space[...,1], time)
