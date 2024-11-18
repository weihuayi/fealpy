from fealpy.fem import BilinearForm, ScalarMassIntegrator
from fealpy.fem import PressWorkIntegrator
from fealpy.fem import LinearForm, ScalarSourceIntegrator
from fealpy.fem import BilinearForm, LinearForm, BlockForm,LinearBlockForm
from fealpy.decorator import barycentric, cartesian
from fealpy.backend import backend_manager as bm
from fealpy.sparse import COOTensor
from functools import partial
from tensor_mass_integrator import TensorMassIntegrator
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
        
    def FBForm_A0(self):
        pspace = self.pspace   
        yspace = self.yspace
        dt = self.dt
        pde = self.pde
        q = self.q
        c = pde.c
        """
        @cartesian
        def coef_M00(p, index = None):
            return pde.A_inverse(p)[...,0,0]
        M00 = BilinearForm(yspace)
        M00.add_integrator(ScalarMassIntegrator(coef=coef_M00, q=q))

        @cartesian
        def coef_M01(p, index = None):
            return pde.A_inverse(p)[...,0,1]
        M01 = BilinearForm(yspace)
        M01.add_integrator(ScalarMassIntegrator(coef=coef_M01, q=q))
        
        @cartesian
        def coef_M10(p, index = None):
            return pde.A_inverse(p)[...,1,0]
        M10 = BilinearForm(yspace)
        M10.add_integrator(ScalarMassIntegrator(coef=coef_M10, q=q))
        
        @cartesian
        def coef_M11(p, index = None):
            return pde.A_inverse(p)[...,1,1]
        M11 = BilinearForm(yspace)
        M11.add_integrator(ScalarMassIntegrator(coef=coef_M11, q=q))
        M_A_inv = BlockForm([[M00, M01], [M10, M11]]) 
        """
        M_inv = BilinearForm(yspace)
        M_inv.add_integrator(TensorMassIntegrator(coef=pde.A_inverse,q=q))
        
        
        
        M = BilinearForm(yspace)
        M.add_integrator(ScalarMassIntegrator(coef=2+c*dt**2,q=q))
    
        S = BilinearForm((yspace, pspace))
        S.add_integrator(PressWorkIntegrator(coef=-dt**2,q=q)) 
        
        S1 = BilinearForm((yspace, pspace))
        S1.add_integrator(PressWorkIntegrator(1,q=q))
        
        A0 = BlockForm([[M, S.T], [S1, M_inv]])
        
        return A0
    
    def FBform_A(self):    
        yspace = self.yspace
        pde = self.pde 
        q = self.q
        pspace = self.pspace
        dt = self.dt
        c = pde.c
        @cartesian
        def coef_M00(p, index = None):
            return pde.A_inverse(p)[...,0,0]
        M00 = BilinearForm(yspace)
        M00.add_integrator(ScalarMassIntegrator(coef=coef_M00, q=q))
        
        @cartesian
        def coef_M01(p, index = None):
            return pde.A_inverse(p)[...,0,1]
        M01 = BilinearForm(yspace)
        
        M01.add_integrator(ScalarMassIntegrator(coef=coef_M01, q=q))
        
        @cartesian
        def coef_M10(p, index = None):
            return pde.A_inverse(p)[...,1,0]
        M10 = BilinearForm(yspace)
        M10.add_integrator(ScalarMassIntegrator(coef=coef_M10, q=q))
        
        @cartesian
        def coef_M11(p, index = None):
            return pde.A_inverse(p)[...,1,1]
        M11 = BilinearForm(yspace)
        M11.add_integrator(ScalarMassIntegrator(coef=coef_M11, q=q))
        
        M_A_inv = BlockForm([[M00, M01], [M10, M11]])
        
        M = BilinearForm(yspace)
        M.add_integrator(ScalarMassIntegrator(coef=2+c*dt**2,q=q))
    
        S = BilinearForm((yspace, pspace))
        S.add_integrator(PressWorkIntegrator(coef=-dt**2,q=q)) 
        
        S1 = BilinearForm((yspace, pspace))
        S1.add_integrator(PressWorkIntegrator(1,q=q))
        
        A = BlockForm([[M, S.T], [S1, M_A_inv]])

        return A
    
    def FLform_b0(self,u1):
        yspace = self.yspace
        pspace = self.pspace    
        pde = self.pde 
        q = self.q
        
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
        
        L0 = LinearForm(yspace)
        f_coef =ScalarSourceIntegrator(coef,q=q)
        u_coef = ScalarSourceIntegrator(u1,q=q)
        fun_coef = ScalarSourceIntegrator(fun,q=q)
        L0.add_integrator(f_coef)
        L0.add_integrator(u_coef)
        L0.add_integrator(fun_coef)
        
        L1 = LinearForm(pspace)
        
        L = LinearBlockForm([L0, L1])
        
        return L
    
    def FLform_b(self):
        yspace = self.yspace
        pspace = self.pspace
        pde = self.pde
        q = self.q
        
        L0 = LinearForm(yspace)
        self.f_coef =ScalarSourceIntegrator(q=q)
        self.u_coef = ScalarSourceIntegrator(q=q)
        self.fun_coef = ScalarSourceIntegrator(q=q)
        L0.add_integrator(self.f_coef)
        L0.add_integrator(self.u_coef)
        L0.add_integrator(self.fun_coef)
        
        L1 = LinearForm(pspace)
        
        L = LinearBlockForm([L0, L1])
        return L
    
    def FLformb_update(self, y0, y1, u, t):
        if u==None:
            u = self.yspace.function()
        self.u_coef.source = u
        self.u_coef.clear()

        @barycentric
        def fun(bcs, index=None):
            result = 2*y1(bcs) - y0(bcs)
            result *= 1/self.dt**2
            return result  
        self.fun_coef.source = fun
        self.fun_coef.clear()
        
        @cartesian
        def coef(p, index=None):
            result = self.pde.f_fun(p, time=t, index=index)
            return result
        self.f_coef.source = coef
        self.f_coef.clear()
    
    def BLform_b0(self, yn1, pn):
        yspace = self.yspace
        pspace = self.pspace
        pde = self.pde
        q = self.q
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
        
        @barycentric
        def funp(bcs, index=None):
            result = -pn(bcs)
            return result  
        
        @cartesian
        def coefp(p, index=None):
            result = self.pde.p_d_fun(p, time=T-self.dt)
            return result
        
        L0 = LinearForm(yspace)
        f_coef =ScalarSourceIntegrator(q=q)
        f_coef.source = coef
        fun_coef = ScalarSourceIntegrator(q=q)
        fun_coef.source = fun
        L0.add_integrator(f_coef)
        L0.add_integrator(fun_coef)
        
        L1 = LinearForm(pspace)
        coefp_coef = ScalarSourceIntegrator(q=q)
        coefp_coef.source = coefp
        funp_coef = ScalarSourceIntegrator(q=q)
        funp_coef.source = funp
        L1.add_integrator(coefp_coef)
        L1.add_integrator(funp_coef)
        
        L = LinearBlockForm([L0, L1])
        return L
     
    def BLform_b(self):
        yspace = self.yspace
        pspace = self.pspace
        pde = self.pde
        q = self.q
        T =self.timeline.T1
        
        L0 = LinearForm(yspace)
        self.yd_coef =ScalarSourceIntegrator(q=q)
        self.fun_coef = ScalarSourceIntegrator(q=q)
        L0.add_integrator(self.yd_coef)
        L0.add_integrator(self.fun_coef)
        
        L1 = LinearForm(pspace)
        self.funpx1_coef = ScalarSourceIntegrator(q=q)
        self.coefpx1_coef = ScalarSourceIntegrator(q=q)
        L1.add_integrator(self.coefpx1_coef)
        L1.add_integrator(self.funpx1_coef)
        
        L = LinearBlockForm([L0, L1])
        return L
    
    def BLformb_update(self, zn0, zn1, yn1, pn, t):
        @barycentric
        def fun(bcs, index=None):
            result = 2*zn0(bcs) - zn1(bcs)
            result *= 1/self.dt**2
            result += yn1(bcs)
            return result  
        self.fun_coef.source = fun
        self.fun_coef.clear()
        @cartesian
        def coef(p, index=None):
            result = -self.pde.y_d_fun(p, time=t)
            return result
        self.yd_coef.source = coef
        self.yd_coef.clear()
        
        @barycentric
        def funpx1(bcs, index=None):
            result = -pn(bcs)
            return result
        self.funpx1_coef.source = funpx1
        self.funpx1_coef.clear()
          
        
        @cartesian
        def coefpx1(p, index=None):
            result = self.pde.p_d_fun(p, time=t)
            return result
        self.coefpx1_coef.source = coefpx1
        self.coefpx1_coef.clear()
        
    
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

