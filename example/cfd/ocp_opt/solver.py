#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: solver.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Tue 10 Sep 2024 03:32:33 PM CST
	@bref 
	@ref 
'''  

from fealpy.experimental.fem import BilinearForm, ScalarMassIntegrator
from fealpy.experimental.fem import PressWorkIntegrator, PressWorkIntegrator1
from fealpy.experimental.fem import LinearForm, ScalarSourceIntegrator
from fealpy.experimental.decorator import barycentric, cartesian
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.sparse import COOTensor
from functools import partial

from ocp_opt_pde import example_1

class ocp_opt_solver():
    def __init__(self, mesh, yspace, pspace, pde, timeline,q=4):
        self.mesh = mesh
        self.yspace = yspace
        self.pspace = pspace
        self.pde = pde
        self.q = q
        self.dt = timeline.dt
        self.timeline = timeline
    
        @cartesian
        def coef_M00(p, index = None):
            return pde.A_inverse(p)[...,0,0]
        bform = BilinearForm(pspace)
        bform.add_integrator(ScalarMassIntegrator(coef=coef_M00, q=q))
        self.M00 = bform.assembly()
        
        @cartesian
        def coef_M01(p, index = None):
            return pde.A_inverse(p)[...,0,1]
        bform = BilinearForm(pspace)
        bform.add_integrator(ScalarMassIntegrator(coef=coef_M01, q=q))
        self.M01 = bform.assembly()
        
        @cartesian
        def coef_M10(p, index = None):
            return pde.A_inverse(p)[...,1,0]
        bform = BilinearForm(pspace)
        bform.add_integrator(ScalarMassIntegrator(coef=coef_M10, q=q))
        self.M10 = bform.assembly()
        
        @cartesian
        def coef_M11(p, index = None):
            return pde.A_inverse(p)[...,1,1]
        bform = BilinearForm(pspace)
        bform.add_integrator(ScalarMassIntegrator(coef=coef_M11, q=q))
        self.M11 = bform.assembly()
    
        bform = BilinearForm(pspace)
        bform.add_integrator(ScalarMassIntegrator(q=q))
        self.M = bform.assembly()

        bform = BilinearForm((pspace, pspace))
        bform.add_integrator(PressWorkIntegrator(q=q)) 
        self.S1 = bform.assembly()

        bform = BilinearForm((pspace, pspace))
        bform.add_integrator(PressWorkIntegrator1(q=q)) 
        self.S2 = bform.assembly()

    def A0n(self):
        dt = self.dt
        A0 = COOTensor.concat(((2/dt**2 + self.pde.c)*self.M, -self.S1, -self.S2), axis=0)
        A1 = COOTensor.concat((self.S1.T, self.M00, self.M01), axis=0)
        A2 = COOTensor.concat((self.S2.T, self.M10, self.M11), axis=0)
        A = COOTensor.concat((A0,A1,A2), axis=1)
        return A
    
    def forward_b0(self, u1):
        if u1 == None:
            u1 = self.yspace.function()

        @cartesian
        def fun(p, index=None):
            result = self.pde.y_solution(p, 0)
            result += self.dt * self.pde.y_t_solution(p, 0)
            result *= 2/self.dt**2
            return result
        @cartesian
        def coef(p, index=None):
            result = self.pde.f_fun(p, time=self.dt, index=index)
            return result
        lform = LinearForm(self.yspace) 
        lform.add_integrator(ScalarSourceIntegrator(coef,q=self.q))
        lform.add_integrator(ScalarSourceIntegrator(u1, q=self.q))
        lform.add_integrator(ScalarSourceIntegrator(fun, q=self.q))
        F0 = lform.assembly() 
        F1 = bm.zeros(self.pspace.number_of_global_dofs()) 
        b = bm.concatenate([F0,F1,F1])
        return b
    
    def backward_b0(self, yn1, pnx1, pnx2): 
        T =self.timeline.T1
        
        @barycentric
        def fun(bcs, index=None):
            result = yn1(bcs)
            return result  
        
        @cartesian
        def coef(p, index=None):
            result = self.pde.z_solution(p, T)
            result -= self.pde.z_t_solution(p, T)
            result *= 2/self.dt**2
            result = -self.pde.y_d_fun(p, time=T-self.dt)
            return result

        lform = LinearForm(self.yspace)
        lform.add_integrator(ScalarSourceIntegrator(coef,q=self.q))
        lform.add_integrator(ScalarSourceIntegrator(fun, q=self.q))
        F0 = lform.assembly() 
        
        @barycentric
        def funpx1(bcs, index=None):
            result = -pnx1(bcs)[..., 0]
            return result  
        
        @cartesian
        def coefpx1(p, index=None):
            result = self.pde.p_dx1_fun(p, time=T-self.dt)
            return result
        lform = LinearForm(self.pspace)
        lform.add_integrator(ScalarSourceIntegrator(coefpx1,q=self.q))
        lform.add_integrator(ScalarSourceIntegrator(funpx1, q=self.q))
        F1= lform.assembly() 
        
        @barycentric
        def funpx2(bcs, index=None):
            result = -pnx2(bcs)[..., 1]
            return result  
        
        @cartesian
        def coefpx2(p, index=None):
            result = self.pde.p_dx2_fun(p, time=T-self.dt)
            return result
        lform = LinearForm(self.pspace)
        lform.add_integrator(ScalarSourceIntegrator(coefpx2,q=self.q))
        lform.add_integrator(ScalarSourceIntegrator(funpx2, q=self.q))
        F2 = lform.assembly() 
        
        b = bm.concatenate([F0,F1,F2])
        return b
    
    def forward_boundary(self, A, b, isbdof, t):
        x0 = bm.zeros(self.yspace.number_of_global_dofs())
        yh, yisbdof = self.yspace.boundary_interpolate(
                partial(self.pde.y_solution, time=t), x0)
        x1 = bm.zeros(self.pspace.number_of_global_dofs())
        px1, pisbdof = self.pspace.boundary_interpolate(
                partial(self.pde.px1_solution, time=t), x1)
        x2 = bm.zeros(self.pspace.number_of_global_dofs())
        px2, pisbdof = self.pspace.boundary_interpolate(
                partial(self.pde.px2_solution, time=t), x2)
        xx = bm.concatenate([x0,x1,x2])
        
        b -= A@xx
        b[isbdof] = xx[isbdof]

        kwargs = A.values_context()
        indices = A.indices()
        new_values = bm.copy(A.values())
        IDX = isbdof[indices[0, :]] | isbdof[indices[1, :]]
        new_values[IDX] = 0
        A = COOTensor(indices, new_values, A.sparse_shape)
     
        index = bm.nonzero(isbdof)[0]
        shape = new_values.shape[:-1] + (len(index), )
        one_values = bm.ones(shape, **kwargs)
        one_indices = bm.stack([index, index], axis=0)
        A1 = COOTensor(one_indices, one_values, A.sparse_shape)
        A = A.add(A1).coalesce()

        return A,b 

    def backward_boundary(self, A, b, isbdof, t):
        x0 = bm.zeros(self.yspace.number_of_global_dofs())
        zh, zisbdof = self.yspace.boundary_interpolate(
                partial(self.pde.z_solution, time=t), x0)
        x1 = bm.zeros(self.pspace.number_of_global_dofs())
        qx1, qisbdof = self.pspace.boundary_interpolate(
                partial(self.pde.qx1_solution, time=t), x1)
        x2 = bm.zeros(self.pspace.number_of_global_dofs())
        qx2, qisbdof = self.pspace.boundary_interpolate(
                partial(self.pde.qx2_solution, time=t), x2)
        xx = bm.concatenate([x0,x1,x2])
        
        b -= A@xx
        b[isbdof] = xx[isbdof]

        kwargs = A.values_context()
        indices = A.indices()
        new_values = bm.copy(A.values())
        IDX = isbdof[indices[0, :]] | isbdof[indices[1, :]]
        new_values[IDX] = 0
        A = COOTensor(indices, new_values, A.sparse_shape)
     
        index = bm.nonzero(isbdof)[0]
        shape = new_values.shape[:-1] + (len(index), )
        one_values = bm.ones(shape, **kwargs)
        one_indices = bm.stack([index, index], axis=0)
        A1 = COOTensor(one_indices, one_values, A.sparse_shape)
        A = A.add(A1).coalesce()
        return A,b 

    def A(self):
        dt = self.dt
        A0 = COOTensor.concat(((1/dt**2 + self.pde.c)*self.M, -self.S1, -self.S2), axis=0)
        A1 = COOTensor.concat((self.S1.T, self.M00, self.M01), axis=0)
        A2 = COOTensor.concat((self.S2.T, self.M10, self.M11), axis=0)
        A = COOTensor.concat((A0,A1,A2), axis=1)
        return A
    
    def forward_b(self, y0, y1, u, t): 
        if u==None:
            u = self.yspace.function()

        @barycentric
        def fun(bcs, index=None):
            result = 2*y1(bcs) - y0(bcs)
            result *= 1/self.dt**2
            return result  
        @cartesian
        def coef(p, index=None):
            result = self.pde.f_fun(p, time=t, index=index)
            return result

        lform = LinearForm(self.yspace)
        lform.add_integrator(ScalarSourceIntegrator(coef,q=self.q))
        lform.add_integrator(ScalarSourceIntegrator(u, q=self.q))
        lform.add_integrator(ScalarSourceIntegrator(fun, q=self.q))
        F0 = lform.assembly() 
        F1 = bm.zeros(self.pspace.number_of_global_dofs()) 
        b = bm.concatenate([F0,F1,F1])
        return b
    
    def backward_b(self, zn0, zn1, yn1, pnx1, pnx2, t): 
        
        @barycentric
        def fun(bcs, index=None):
            result = 2*zn1(bcs) - zn0(bcs)
            result *= 1/self.dt**2
            result = yn1(bcs)
            return result  
        
        @cartesian
        def coef(p, index=None):
            result = -self.pde.y_d_fun(p, time=t)
            return result

        lform = LinearForm(self.yspace)
        lform.add_integrator(ScalarSourceIntegrator(coef,q=self.q))
        lform.add_integrator(ScalarSourceIntegrator(fun, q=self.q))
        F0 = lform.assembly() 
        
        @barycentric
        def funpx1(bcs, index=None):
            result = -pnx1(bcs)[..., 0]
            return result  
        
        @cartesian
        def coefpx1(p, index=None):
            result = self.pde.p_dx1_fun(p, time=t)
            return result
        lform = LinearForm(self.pspace)
        lform.add_integrator(ScalarSourceIntegrator(coefpx1,q=self.q))
        lform.add_integrator(ScalarSourceIntegrator(funpx1, q=self.q))
        F1= lform.assembly() 
        
        @barycentric
        def funpx2(bcs, index=None):
            result = -pnx2(bcs)[..., 1]
            return result  
        
        @cartesian
        def coefpx2(p, index=None):
            result = self.pde.p_dx2_fun(p, time=t)
            return result
        lform = LinearForm(self.pspace)
        lform.add_integrator(ScalarSourceIntegrator(coefpx2,q=self.q))
        lform.add_integrator(ScalarSourceIntegrator(funpx2, q=self.q))
        F2 = lform.assembly() 
        
        b = bm.concatenate([F0,F1,F2])
        return b
    
    def solve_u(self, z):
        result = bm.max(self.mesh.integral(z), 0) - z #积分子
        return result

    def mumps_solve(self, A, b):
        import scipy.sparse as sp
        values = A.values()
        indices = A.indices()
        A = sp.coo_matrix((values, (indices[0], indices[1])), shape=A.shape) 
        A = A.tocsr()
        x = sp.linalg.spsolve(A,b)
        return x

