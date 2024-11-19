from fealpy.fem import BilinearForm, ScalarMassIntegrator
from fealpy.fem import PressWorkIntegrator
from fealpy.fem import LinearForm, ScalarSourceIntegrator
from fealpy.decorator import barycentric, cartesian
from fealpy.backend import backend_manager as bm
from fealpy.sparse import COOTensor
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
        bform = BilinearForm(yspace)
        bform.add_integrator(ScalarMassIntegrator(coef=coef_M00, q=q))
        self.M00 = bform.assembly()
        
        @cartesian
        def coef_M01(p, index = None):
            return pde.A_inverse(p)[...,0,1]
        bform = BilinearForm(yspace)
        bform.add_integrator(ScalarMassIntegrator(coef=coef_M01, q=q))
        self.M01 = bform.assembly()
        
        @cartesian
        def coef_M10(p, index = None):
            return pde.A_inverse(p)[...,1,0]
        bform = BilinearForm(yspace)
        bform.add_integrator(ScalarMassIntegrator(coef=coef_M10, q=q))
        self.M10 = bform.assembly()
        
        @cartesian
        def coef_M11(p, index = None):
            return pde.A_inverse(p)[...,1,1]
        bform = BilinearForm(yspace)
        bform.add_integrator(ScalarMassIntegrator(coef=coef_M11, q=q))
        self.M11 = bform.assembly()
        
    
        bform = BilinearForm(yspace)
        bform.add_integrator(ScalarMassIntegrator(q=q))
        self.M = bform.assembly()

        bform = BilinearForm((yspace, pspace))
        bform.add_integrator(PressWorkIntegrator(-1,q=q)) 
        self.S1 = bform.assembly()
        
        bform = BilinearForm((yspace,pspace))
        bform.add_integrator(PressWorkIntegrator(-1,q=q)) 
        self.S2 = bform.assembly()
        
        bform = BilinearForm((yspace, pspace))
        bform.add_integrator(PressWorkIntegrator(-1,q=q)) 
        self.S = bform.assembly()
        
        
    def A0n(self):
        dt = self.dt
        M = self.M.tocoo()
        S = self.S.tocoo()
        M00 = self.M00.tocoo()
        M01 = self.M01.tocoo()
        M10 = self.M10.tocoo()
        M11 = self.M11.tocoo()
        M0 = COOTensor.concat((M00, M01), axis=1)
        M1 = COOTensor.concat((M10, M11), axis=1)
        M_A = COOTensor.concat((M0, M1), axis=0)
        A0 = COOTensor.concat(((2/dt**2 + self.pde.c)*M, -dt**2*S), axis=0)
        A1 = COOTensor.concat((S.T, M_A), axis=0)
        A = COOTensor.concat((A0,A1), axis=1)
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
        b = bm.concatenate([F0,F1]) 
        return b
    
    def backward_b0(self, yn1, pn): 
        T =self.timeline.T1

        @barycentric
        def fun(bcs, index=None):
            result = yn1(bcs)
            return result  
        
        @cartesian
        def coef(p, index=None):
            result = self.pde.z_solution(p, T)
            result -= self.pde.z_t_solution(p, T)*self.dt
            result *= 2/self.dt**2
            result += -self.pde.y_d_fun(p, time=T-self.dt)  
            return result

        lform = LinearForm(self.yspace)
        lform.add_integrator(ScalarSourceIntegrator(coef,q=self.q))
        lform.add_integrator(ScalarSourceIntegrator(fun, q=self.q))
        F0 = lform.assembly() 
        
        @barycentric
        def funp(bcs, index=None):
            result = -pn(bcs)
            return result  
        
        @cartesian
        def coefp(p, index=None):
            result = self.pde.p_d_fun(p, time=T-self.dt)
            return result
        lform = LinearForm(self.pspace) 
        lform.add_integrator(ScalarSourceIntegrator(coefp,q=self.q))
        lform.add_integrator(ScalarSourceIntegrator(funp, q=self.q))
        F1= lform.assembly()         
        
        b = bm.concatenate([F0,F1])
        return b
    

    
    def forward_boundary(self, A, b, isbdof, t):
        x0 = bm.zeros(self.yspace.number_of_global_dofs())
        yh, yisbdof = self.yspace.boundary_interpolate(
                partial(self.pde.y_solution, time=t), x0)
        x1 = bm.zeros(self.pspace.number_of_global_dofs())
        px1, pisbdof = self.pspace.boundary_interpolate(
                partial(self.pde.p_solution, time=t), x1)
        # x2 = bm.zeros(self.yspace.number_of_global_dofs())
        # px2, pisbdof = self.yspace.boundary_interpolate(
        #         partial(self.pde.px2_solution, time=t), x2)
        xx = bm.concatenate([x0,x1])
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
                partial(self.pde.qx_solution, time=t),x1)
        xx = bm.concatenate([x0,x1])
        
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
        S = self.S.tocoo()  
        M = self.M.tocoo()
        M01 = self.M01.tocoo()  
        M10 = self.M10.tocoo()
        M11 = self.M11.tocoo()
        M00 = self.M00.tocoo()
        M0 = COOTensor.concat((M00, M01), axis=1)  
        M1 = COOTensor.concat((M10, M11), axis=1)
        M_A = COOTensor.concat((M0, M1), axis=0)
        A0 = COOTensor.concat(((1/dt**2 + self.pde.c)*M, -dt**2*S), axis=0)
        A1 = COOTensor.concat((S.T,M_A ), axis=0)
        A = COOTensor.concat((A0,A1), axis=1)
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
        b = bm.concatenate([F0,F1])
        return b
    
    def backward_b(self, zn0, zn1, yn1, pn, t): 
        
        @barycentric
        def fun(bcs, index=None):
            result = 2*zn0(bcs) - zn1(bcs)
            result *= 1/self.dt**2
            result += yn1(bcs)
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
            result = -pn(bcs)
            return result  
        
        @cartesian
        def coefpx1(p, index=None):
            result = self.pde.p_d_fun(p, time=t)
            return result
        lform = LinearForm(self.pspace)
        lform.add_integrator(ScalarSourceIntegrator(coefpx1,q=self.q))
        lform.add_integrator(ScalarSourceIntegrator(funpx1, q=self.q))
        F1= lform.assembly() 
        
        b = bm.concatenate([F0,F1])
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

