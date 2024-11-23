from fealpy.fem import BilinearForm, ScalarMassIntegrator
from fealpy.fem import PressWorkIntegrator
from fealpy.fem import LinearForm, ScalarSourceIntegrator
from fealpy.fem import BilinearForm, LinearForm, BlockForm,LinearBlockForm
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
        
    def Forward_BForm_A0(self):
        pspace = self.pspace   
        yspace = self.yspace
        dt = self.dt
        pde = self.pde
        q = self.q
        c = pde.c
        M_y = BilinearForm(yspace)
        M_y.add_integrator(ScalarMassIntegrator(coef=2+c*dt**2,q=q))

        M_p = BilinearForm(pspace)
        M_p.add_integrator(ScalarMassIntegrator(coef=pde.A_inverse,q=q))
                 
        S = BilinearForm((pspace, yspace))
        S.add_integrator(PressWorkIntegrator(coef=-dt**2,q=q)) 
        
        S1 = BilinearForm((pspace, yspace))
        S1.add_integrator(PressWorkIntegrator(1,q=q))
        
        A = BlockForm([[M_y, S], [S1.T, M_p]]) 
        return A
    
    Backward_BForm_An = Forward_BForm_A0

    def Forward_BForm_A(self):    
        pspace = self.pspace   
        yspace = self.yspace
        dt = self.dt
        pde = self.pde
        q = self.q
        c = pde.c
        
        M_y = BilinearForm(yspace)
        M_y.add_integrator(ScalarMassIntegrator(coef=1+c*dt**2,q=q))

        M_p = BilinearForm(pspace)
        M_p.add_integrator(ScalarMassIntegrator(coef=pde.A_inverse,q=q))
                 
        S = BilinearForm((pspace, yspace))
        S.add_integrator(PressWorkIntegrator(coef=-dt**2,q=q)) 
        
        S1 = BilinearForm((pspace, yspace))
        S1.add_integrator(PressWorkIntegrator(1,q=q))
        
        A = BlockForm([[M_y, S], [S1.T, M_p]]) 
        return A
    
    Backward_BForm_A = Forward_BForm_A
    
    def Forward_LForm_b0(self, u1=None):
        yspace = self.yspace
        pspace = self.pspace    
        pde = self.pde 
        q = self.q
        dt = self.dt
        
        
        @cartesian
        def coef(p, index=None):
            result = (dt**2)*self.pde.f_fun(p, time=dt)
            result += 2*(self.pde.y_solution(p, time=0) + dt * self.pde.y_t_solution(p, time=0))
            return result
        

        Ly = LinearForm(yspace)
        Ly.add_integrator(ScalarSourceIntegrator(coef,q=q))
        self.forward_0_b_coef = ScalarSourceIntegrator(q=q)
        Ly.add_integrator(self.forward_0_b_coef)
        
        Lp = LinearForm(pspace)
        L = LinearBlockForm([Ly, Lp])
        return L
    
    def Forward_0_update(self, u1): 
        dt = self.dt
        if u1 == None:
            u1 = self.yspace.function()
        
        @barycentric
        def coef_u1(bcs, index=None):
            result = (dt**2)*u1(bcs)
            return result
        self.forward_0_b_coef.source = coef_u1
        self.forward_0_b_coef.clear()

    def Forward_LForm_b(self):
        yspace = self.yspace
        pspace = self.pspace
        pde = self.pde
        q = self.q
        
        Ly = LinearForm(yspace)
        self.c_coef = ScalarSourceIntegrator(q=q)
        self.b_coef = ScalarSourceIntegrator(q=q)
        Ly.add_integrator(self.c_coef)
        Ly.add_integrator(self.b_coef)
        
        Lp = LinearForm(pspace)
        
        L = LinearBlockForm([Ly, Lp])
        return L
    
    def Forward_update(self, y0, y1, u2, t2):
        dt = self.dt

        if u2==None:
            u2 = self.yspace.function()

        @barycentric
        def coef_b(bcs, index=None):
            result = 2*y1(bcs, index) - y0(bcs, index)
            result += (dt**2)*u2(bcs)
            return result

        self.b_coef.source = coef_b
        self.b_coef.clear()

        @cartesian
        def coef_c(p, index=None):
            result = (dt**2) * self.pde.f_fun(p, time=t2, index=index)
            return result
        self.c_coef.source = coef_c
        self.c_coef.clear()
    

    def Backward_LForm_bn(self):
        yspace = self.yspace
        pspace = self.pspace
        pde = self.pde
        q = self.q
        T =self.timeline.T1
        dt = self.dt

        @cartesian
        def z_coef_c(p, index=None):
            result = 2*self.pde.z_solution(p, T) - 2*dt*self.pde.z_t_solution(p, T)
            result -= dt**2 * self.pde.y_d_fun(p, time=T-dt)  
            return result
         
        Lz = LinearForm(yspace)
        Lz.add_integrator(ScalarSourceIntegrator(z_coef_c,q=q))
        self.backward_n_z_coef = ScalarSourceIntegrator(q=q) 
        Lz.add_integrator(self.backward_n_z_coef)
        
        @cartesian
        def q_coef_c(p, index=None):
            result = self.pde.p_d_fun(p, T-dt)
            return result
        Lq = LinearForm(pspace)
        Lq.add_integrator(ScalarSourceIntegrator(q_coef_c,q=q))
        self.backward_n_q_coef = ScalarSourceIntegrator(q=q) 
        Lq.add_integrator(self.backward_n_q_coef) 
        L = LinearBlockForm([Lz, Lq])
        return L
     
    def Backward_n_update(self, y2, p2):
        dt = self.dt
        
        @barycentric
        def coef_z(bcs, index=None):
            result = dt**2 * y2(bcs, index)
            return result
        self.backward_n_z_coef.source = coef_z
        self.backward_n_z_coef.clear()
        
        @barycentric
        def coef_q(bcs, index=None):
            result = - p2(bcs, index)
            return result
        self.backward_n_q_coef.source = coef_q
        self.backward_n_q_coef.clear()

        
    def Backward_LForm_b(self):
        yspace = self.yspace
        pspace = self.pspace
        pde = self.pde
        q = self.q
        dt = self.dt 
        
        Lz = LinearForm(yspace)
        self.backward_z_c_coef = ScalarSourceIntegrator(q=q)
        self.backward_z_b_coef = ScalarSourceIntegrator(q=q)
        Lz.add_integrator(self.backward_z_b_coef)
        Lz.add_integrator(self.backward_z_c_coef)

        Lq = LinearForm(pspace)
        self.backward_q_c_coef = ScalarSourceIntegrator(q=q)
        self.backward_q_b_coef = ScalarSourceIntegrator(q=q)
        Lq.add_integrator(self.backward_q_b_coef)
        Lq.add_integrator(self.backward_q_c_coef)
        
        L = LinearBlockForm([Lz, Lq])   
    
        return L
    
    def Backward_update(self, zn0, zn1, yn2, p2, t2):
        dt = self.dt
        
        @cartesian
        def coef_c_z(p, index=None):
            result = -dt**2 * self.pde.y_d_fun(p, time=t2)
            return result 

        @barycentric
        def coef_b_z(bcs, index=None):
            result = 2*zn1(bcs) - zn0(bcs)
            result += dt**2 * yn2(bcs)
            return result  
        self.backward_z_c_coef.source = coef_c_z
        self.backward_z_c_coef.clear()
        self.backward_z_b_coef.source = coef_b_z
        self.backward_z_b_coef.clear()
 
        @cartesian
        def coef_c_q(p, index=None):
            result = self.pde.p_d_fun(p, time=t2)
            return result 

        @barycentric
        def coef_b_q(bcs, index=None):
            result = -p2(bcs) 
            return result  
        self.backward_q_c_coef.source = coef_c_q
        self.backward_q_c_coef.clear()
        self.backward_q_b_coef.source = coef_b_q
        self.backward_q_b_coef.clear()
          
    def solve_z_bar(self, allz):
        dt = self.dt
        integral_z = bm.array([self.mesh.integral(i, q=self.q) for i in allz],dtype=bm.float64)
        z_bar = bm.sum((dt/2)*(integral_z[:-1] + integral_z[1:]))
        z_bar /= (self.timeline.T1 - self.timeline.T0)*self.mesh.integral(1, q=self.q)
        return z_bar

