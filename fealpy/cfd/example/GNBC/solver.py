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
                        ViscousWorkIntegrator)
from fealpy.fem import (BoundaryFaceMassIntegrator,
                        BoundaryFaceSourceIntegrator,
                        TangentFaceMassIntegrator)
                        
                        
                        


class Solver():
    def __init__(self, pde, mesh, pspace, phispace, uspace, dt, q=5):
        self.mesh = mesh
        self.phispace = phispace
        self.pspace = pspace
        self.uspace = uspace
        self.pde = pde
        self.dt = dt
        self.q = q
    
    # def CH_BForm_0(self):
    #     phispace = self.phispace
    #     dt = self.dt
    #     L_d = self.pde.L_d
    #     epsilon = self.pde.epsilon
    #     s = self.pde.s
    #     V_s = self.pde.V_s
    #     q = self.q
    #     return A

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

    def CH_update(self, u_0, u_1, phi_0, phi_1):
        dt = self.dt
        s = self.pde.s
        epsilon =self.pde.epsilon
        tangent = self.mesh.edge_unit_tangent()
        tangent[..., 0] = 1
        V_s = self.pde.V_s
        theta_s = self.pde.theta_s

        # BilinearForm
        @barycentric
        def phi_C_coef(bcs, index):
            return 4*dt*u_1(bcs, index)
        self.phi_C.coef = phi_C_coef
        self.phi_C.clear()
        
        # LinearForm 
        @barycentric
        def phi_coef(bcs, index):
            result = 4*phi_1(bcs, index) - phi_0(bcs, index) 
            result += 2*dt*bm.einsum('jid, jid->ji', u_0(bcs, index), phi_1.grad_value(bcs, index))
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

            result10 = 2*bm.einsum('eld, ed->el', u_1(bcs, index), tangent[index,:])
            result10 *= bm.einsum('eld, ed->el', phi_1.grad_value(bcs, index), tangent[index,:])
            result11 = bm.einsum('eld, ed->el', u_0(bcs, index), tangent[index,:])
            result11 *= bm.einsum('eld, ed->el', phi_0.grad_value(bcs, index), tangent[index,:])
            result1 = result10 - result11

            result2 = -2*(bm.sqrt(bm.array(2))/6) * bm.pi * bm.cos(theta_s) * bm.cos((bm.pi/2) * phi_1(bcs, index))
            result2 +=   (bm.sqrt(bm.array(2))/6) * bm.pi * bm.cos(theta_s) * bm.cos((bm.pi/2) * phi_0(bcs, index))
            
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
        # D = ScalarDiffusionIntegrator(coef=2*dt, q=q)
        D = ViscousWorkIntegrator(coef=2*dt, q=q)
        ## TODO:和老师确认一下这个边界积分子 
        FM = TangentFaceMassIntegrator(coef=2*dt/L_s, q=q, threshold=self.pde.is_wall_boundary)
        A00.add_integrator(M)
        A00.add_integrator(self.u_C)
        A00.add_integrator(D)
        A00.add_integrator(FM)

        A01 = BilinearForm((pspace, uspace))
        A01.add_integrator(PressWorkIntegrator(coef=-2*dt, q=q))
 
        A10 = BilinearForm((pspace, uspace))
        A10.add_integrator(PressWorkIntegrator(coef=1, q=q))
        
        A = BlockForm([[A00, A01], [A10.T, None]]) 
        return A

    def NS_LForm(self, q=None):
        pspace = self.pspace
        uspace = self.uspace
        dt = self.dt
        R = self.pde.R
        q = self.q
        L_s = self.pde.L_s

        L0 = LinearForm(uspace) 
        self.u_SI = SourceIntegrator(q=q)
        self.u_BF_SI = BoundaryFaceSourceIntegrator(q=q, threshold=self.pde.is_wall_boundary)
        
        @cartesian
        def uw_BF_SI_coef(p):
            return (2*dt/L_s)*self.pde.u_w(p)
        uw_BF_SI = BoundaryFaceSourceIntegrator(source=uw_BF_SI_coef, q=q, threshold=self.pde.is_wall_boundary)
        
        L0.add_integrator(uw_BF_SI)
        L0.add_integrator(self.u_SI)
        L0.add_integrator(self.u_BF_SI)

        L1 = LinearForm(pspace)
        L = LinearBlockForm([L0, L1])
        return L

    def NS_update(self, u_0, u_1, mu_2, phi_2, phi_1):
        dt = self.dt
        R = self.pde.R
        lam = self.pde.lam
        epsilon = self.pde.epsilon
        normal = self.mesh.edge_unit_normal()
        tangent = self.mesh.edge_unit_tangent()
        tangent[..., 0] = 1
        
        L_s = self.pde.L_s
        theta_s = self.pde.theta_s
        
        ## BilinearForm
        def u_C_coef(bcs, index):
            return 4*R*dt*u_1(bcs, index)
        self.u_C.coef = u_C_coef
        self.u_C.clear()
        
        def u_SI_coef(bcs, index):
            result = R*(4*u_1(bcs, index) - u_0(bcs, index))
            result += 2*R*dt*bm.einsum('cqij, cqj->cqi', u_1.grad_value(bcs, index), u_0(bcs, index))
            result += 2*lam*dt*mu_2(bcs, index)[...,bm.newaxis]*phi_2.grad_value(bcs, index)
            return result

        self.u_SI.source = u_SI_coef
        self.u_SI.clear()
        
        def u_BF_SI_coef(bcs, index):
            L_phi = epsilon*bm.einsum('eld, ed -> el', phi_2.grad_value(bcs, index), normal[index,:])
            L_phi -= 2*(bm.sqrt(bm.array(2))/6)*bm.pi*bm.cos(theta_s)*bm.cos((bm.pi/2)*phi_2(bcs, index))
            L_phi +=   (bm.sqrt(bm.array(2))/6)*bm.pi*bm.cos(theta_s)*bm.cos((bm.pi/2)*phi_1(bcs, index))
            
            result = 2*dt*lam*L_phi*bm.einsum('eld, ed -> el', phi_2.grad_value(bcs, index), tangent[index,:])
            result = bm.repeat(result[..., bm.newaxis], 2, axis=-1) 
            return result
        self.u_BF_SI.source = u_BF_SI_coef
        self.u_BF_SI.clear()

    def reinit_phi(self, phi):
       tag0 = phi[:] > 2.1*self.pde.epsilon
       tag1 = phi[:] < -2.1*self.pde.epsilon
       phi[tag0] = 1
       phi[tag1] = -1
       return phi[:]
