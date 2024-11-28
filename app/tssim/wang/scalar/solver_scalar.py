#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: solver.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Sat 26 Oct 2024 04:18:00 PM CST
	@bref 
	@ref 
'''  
from fealpy.backend import backend_manager as bm
from fealpy.decorator import cartesian,barycentric
from fealpy.fem import BilinearForm, LinearForm, BlockForm, LinearBlockForm
from fealpy.fem import (ScalarConvectionIntegrator, 
                        ScalarDiffusionIntegrator, 
                        ScalarMassIntegrator,
                        SourceIntegrator,
                        PressWorkIntegrator,
                        PressWorkIntegratorX,
                        PressWorkIntegratorY)
from fealpy.fem import (BoundaryFaceMassIntegrator,
                        BoundaryFaceSourceIntegrator)
#from tangent_face_mass_integrator import TangentFaceMassIntegrator
                        
class Solver():
    def __init__(self, pde, mesh, pspace, phispace, uspace, dt, q=5):
        self.mesh = mesh
        self.phispace = phispace
        self.pspace = pspace
        self.uspace = uspace
        self.pde = pde
        self.dt = dt
        self.q = q
    
    def CH_BForm(self):
        phispace = self.phispace
        dt = self.dt
        L_d = self.pde.L_d
        epsilon = self.pde.epsilon
        s = self.pde.s
        V_s = self.pde.V_s
        q = self.q

        A00 = BilinearForm(phispace)
        M = ScalarMassIntegrator(coef=3, q=q)
        self.phi_C = ScalarConvectionIntegrator(q=q)
        A00.add_integrator(M)
        A00.add_integrator(self.phi_C)

        A01 = BilinearForm(phispace)
        A01.add_integrator(ScalarDiffusionIntegrator(coef=2*dt*L_d, q=q))
        
        A10 = BilinearForm(phispace)
        A10.add_integrator(ScalarDiffusionIntegrator(coef=-epsilon, q=q))
        A10.add_integrator(ScalarMassIntegrator(coef=-s/epsilon, q=q))
        A10.add_integrator(BoundaryFaceMassIntegrator(coef=-3/(2*dt*V_s), q=q, threshold=self.pde.is_wall_boundary))     

        A11 = BilinearForm(phispace)
        A11.add_integrator(ScalarMassIntegrator(coef=1, q=q))

        A = BlockForm([[A00, A01], [A10, A11]]) 
        return A

    def CH_LForm(self):
        phispace = self.phispace
        dt = self.dt
        L_d = self.pde.L_d
        epsilon = self.pde.epsilon
        s = self.pde.s
        q = self.q

        L0 = LinearForm(phispace)
        self.phi_SI = SourceIntegrator(q=q)
        L0.add_integrator(self.phi_SI)

        L1 = LinearForm(phispace)
        self.mu_SI = SourceIntegrator(q=q)
        self.mu_BF_SI = BoundaryFaceSourceIntegrator(q=q, threshold=self.pde.is_wall_boundary)
        L1.add_integrator(self.mu_SI)
        L1.add_integrator(self.mu_BF_SI)

        L = LinearBlockForm([L0, L1])
        return L

    def CH_update(self, u_0x, u_0y, u_1x, u_1y, phi_0, phi_1):
        dt = self.dt
        s = self.pde.s
        epsilon =self.pde.epsilon
        tangent = self.mesh.edge_unit_tangent()
        V_s = self.pde.V_s
        theta_s = self.pde.theta_s

        # BilinearForm
        @barycentric
        def phi_C_coef(bcs, index):
            return 4*dt*bm.stack((u_1x(bcs, index), u_1y(bcs, index)), axis=-1)
        self.phi_C.coef = phi_C_coef
        self.phi_C.clear()
        
        # LinearForm 
        @barycentric
        def phi_coef(bcs, index):
            result = 4*phi_1(bcs, index) - phi_0(bcs, index) 
            result += 2*dt*bm.einsum('ji, ji->ji', u_0x(bcs, index), phi_1.grad_value(bcs, index)[...,0])
            result += 2*dt*bm.einsum('ji, ji->ji', u_0y(bcs, index), phi_1.grad_value(bcs, index)[...,1])
            return result
        self.phi_SI.source = phi_coef
        self.phi_SI.clear()
       
        @barycentric
        def mu_coef(bcs, index):
            result = -2*(1+s)*phi_1(bcs, index) + (1+s)*phi_0(bcs, index)
            result += 2*phi_1(bcs, index)**3 - phi_0(bcs, index)**3
            result /= epsilon
            return result
        self.mu_SI.source = mu_coef
        self.mu_SI.clear()
        
        
        @barycentric
        def mu_BF_coef(bcs, index):
            result0 = (-4*phi_1(bcs, index) + phi_0(bcs, index))/(2*dt)

            result10 = (2*bm.einsum('el, e->el', u_1x(bcs, index), tangent[index,0]) + 
                        2*bm.einsum('el, e->el', u_1y(bcs, index), tangent[index,1]))
            result10 *= (bm.einsum('el, e->el', phi_1.grad_value(bcs, index)[...,0], tangent[index,0])+ \
                        bm.einsum('el, e->el', phi_1.grad_value(bcs, index)[...,1], tangent[index,1]))
            
            result11 = (bm.einsum('el, e->el', u_0x(bcs, index), tangent[index,0]) + 
                        bm.einsum('el, e->el', u_0y(bcs, index), tangent[index,1]))
            result11 *= (bm.einsum('el, e->el', phi_0.grad_value(bcs, index)[...,0], tangent[index,0])+
                        bm.einsum('el, e->el', phi_0.grad_value(bcs, index)[...,1], tangent[index,1]))

            result1 = result10 - result11

            result2 = -2*(bm.sqrt(2)/6) * bm.pi * bm.cos(theta_s) * bm.cos((bm.pi/2) * phi_1(bcs, index))
            result2 += (bm.sqrt(2)/6) * bm.pi * bm.cos(theta_s) * bm.cos((bm.pi/2)*phi_0(bcs, index))
            
            result = (1/V_s)*(result0 + result1) + result2
            return result
        self.mu_BF_SI.source = mu_BF_coef
        self.mu_BF_SI.clear()

    def NS_BForm(self):
        pspace = self.pspace
        uspace = self.uspace
        dt = self.dt
        R = self.pde.R
        q = self.q
        L_s = self.pde.L_s

        A00 = BilinearForm(uspace)
        M = ScalarMassIntegrator(coef=3*R, q=q)
        self.u_C = ScalarConvectionIntegrator(q=q)
        D = ScalarDiffusionIntegrator(coef=2*dt, q=q)
        ## TODO:和老师确认一下这个边界积分子 
        #FM = TangentFaceMassIntegrator(coef=2*dt/L_s, q=q, threshold=self.pde.is_wall_boundary)
        A00.add_integrator(M)
        A00.add_integrator(self.u_C)
        A00.add_integrator(D)
        #A00.add_integrator(FM)

        A02 = BilinearForm((pspace, uspace))
        A02.add_integrator(PressWorkIntegratorX(coef=-2*dt, q=q))
 
        A12 = BilinearForm((pspace, uspace))
        A12.add_integrator(PressWorkIntegratorY(coef=-2*dt, q=q))
        
        A = BlockForm([[A00, None, A02], 
                       [None, A00, A12],
                       [A02.T, A12.T, None]]) 
        return A

    def NS_LForm(self, q=None):
        pspace = self.pspace
        uspace = self.uspace
        dt = self.dt
        R = self.pde.R
        q = self.q
        L_s = self.pde.L_s

        L0 = LinearForm(uspace) 
        self.u_SI_x = SourceIntegrator(q=q)
        self.u_BF_SI_x = BoundaryFaceSourceIntegrator(q=q, threshold=self.pde.is_wall_boundary)
        
        @cartesian
        def ux_BF_SI_coef(p):
            return 2*dt*self.pde.u_w(p)[..., 0]/L_s
        uw_BF_SI = BoundaryFaceSourceIntegrator(source=ux_BF_SI_coef, q=q, threshold=self.pde.is_wall_boundary)
        
        L0.add_integrator(self.u_SI_x)
        L0.add_integrator(self.u_BF_SI_x)
        L0.add_integrator(uw_BF_SI)
        
        L1 = LinearForm(uspace) 
        self.u_SI_y = SourceIntegrator(q=q)
        self.u_BF_SI_y = BoundaryFaceSourceIntegrator(q=q, threshold=self.pde.is_wall_boundary)
         
        L1.add_integrator(self.u_SI_y)
        L1.add_integrator(self.u_BF_SI_y)

        L2 = LinearForm(pspace)
        L = LinearBlockForm([L0, L1, L2])
        return L

    def NS_update(self, u_0x, u_0y, u_1x, u_1y, mu_2, phi_2, phi_1):
        dt = self.dt
        R = self.pde.R
        lam = self.pde.lam
        epsilon = self.pde.epsilon
        n = self.mesh.edge_unit_normal()
        t = self.mesh.edge_unit_tangent()
        theta_s = self.pde.theta_s
        
        ## BilinearForm
        def u_C_coef(bcs, index):
            return 4*R*dt*bm.stack((u_1x(bcs, index), u_1y(bcs, index)), axis=-1)
        self.u_C.coef = u_C_coef
        self.u_C.clear()
        
        ## LinearForm
        def u_SI_coef_x(bcs, index):
            result = R*(4*u_1x(bcs, index) - u_0x(bcs, index))
            result += 2*R*dt*(bm.einsum('cq, cq->cq', u_0x(bcs, index), u_1x.grad_value(bcs, index)[...,0])
                              +bm.einsum('cq, cq->cq', u_0y(bcs, index), u_1x.grad_value(bcs, index)[...,1]))
            result += 2*lam*dt*mu_2(bcs, index)*phi_2.grad_value(bcs, index)[...,0]
            return result

        self.u_SI_x.source = u_SI_coef_x
        self.u_SI_x.clear()
        
        def u_SI_coef_y(bcs, index):
            result = R*(4*u_1y(bcs, index) - u_0y(bcs, index))
            result += 2*R*dt*(bm.einsum('cq, cq->cq', u_0x(bcs, index), u_1y.grad_value(bcs, index)[...,0])
                              +bm.einsum('cq, cq->cq', u_0y(bcs, index), u_1y.grad_value(bcs, index)[...,1]))
            result += 2*lam*dt*mu_2(bcs, index)*phi_2.grad_value(bcs, index)[...,1]
            return result

        self.u_SI_y.source = u_SI_coef_y
        self.u_SI_y.clear()
        
        def u_BF_SI_coef(bcs, index):
            result = 2*dt*lam*bm.einsum('eld, ed -> el', phi_2.grad_value(bcs, index), t[index,:])
            L_phi = epsilon*bm.einsum('eld, ed -> el', phi_2.grad_value(bcs, index), n[index,:])
            L_phi -= 2*(bm.sqrt(2)/6)*bm.pi*bm.cos(theta_s)*bm.cos((bm.pi/2)*phi_2(bcs, index))
            L_phi += (bm.sqrt(2)/6)*bm.pi*bm.cos(theta_s)*bm.cos((bm.pi/2)*phi_1(bcs, index))
            result *= L_phi
            return result
        self.u_BF_SI_x.source = u_BF_SI_coef
        self.u_BF_SI_x.clear()
        self.u_BF_SI_y.source = u_BF_SI_coef
        self.u_BF_SI_y.clear()
