#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: ns_fem_solver_new.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Thu 05 Dec 2024 10:22:30 AM CST
	@bref 
	@ref 
'''  
from ...backend import backend_manager as bm
from ...decorator import cartesian,barycentric
from ...fem import BilinearForm, LinearForm, BlockForm, LinearBlockForm
from ...fem import (ScalarConvectionIntegrator, 
                        ScalarDiffusionIntegrator, 
                        ScalarMassIntegrator,
                        SourceIntegrator,
                        PressWorkIntegrator,
                        FluidBoundaryFrictionIntegrator,
                        ViscousWorkIntegrator,
                        GradSourceIntegrator)
from ...fem import (BoundaryFaceMassIntegrator,
                        BoundaryFaceSourceIntegrator)

class NSFEMSolver():
    def __init__(self, pde, mesh, pspace, uspace, dt, q=5, keep_data=True):        
        self.pde = pde
        self.mesh = mesh
        self.pspace = pspace
        self.uspace = uspace
        self.dt = dt
        self.q = q
        self.keep_data = keep_data

    def Ossen_BForm(self):
        pspace = self.pspace
        uspace = self.uspace
        dt = self.dt
        R = self.pde.R
        q = self.q

        A00 = BilinearForm(uspace)
        M = ScalarMassIntegrator(coef=R , q=q)
        self.u_C = ScalarConvectionIntegrator(q=q)
        self.u_C.keep_data(self.keep_data)
        D = ScalarDiffusionIntegrator(coef=dt, q=q)
        A00.add_integrator(M)
        A00.add_integrator(self.u_C)
        A00.add_integrator(D)

        A01 = BilinearForm((pspace, uspace))
        A01.add_integrator(PressWorkIntegrator(coef=-dt, q=q))
 
        A10 = BilinearForm((pspace, uspace))
        A10.add_integrator(PressWorkIntegrator(coef=1, q=q))
        
        A = BlockForm([[A00, A01], [A10.T, None]]) 
        return A

    def Ossen_LForm(self, q=None):
        pspace = self.pspace
        uspace = self.uspace
        dt = self.dt
        R = self.pde.R
        q = self.q

        L0 = LinearForm(uspace) 
        self.u_SI = SourceIntegrator(q=q)
        self.u_SI.keep_data(self.keep_data)
        L0.add_integrator(self.u_SI)
        L1 = LinearForm(pspace)

        L = LinearBlockForm([L0, L1])
        return L

    def Ossen_update(self, u_0):
        dt = self.dt
        R = self.pde.R
         
        ## BilinearForm
        def u_C_coef(bcs, index):
            return R*dt*u_0(bcs, index)
        self.u_C.coef = u_C_coef
        self.u_C.clear()
        
        def u_SI_coef(bcs, index):
            result = R*u_0(bcs, index)
            return result
        self.u_SI.source = u_SI_coef
        self.u_SI.clear()
     
    def Newton_BForm(self, threshold=None):
        pspace = self.pspace
        uspace = self.uspace
        dt = self.dt
        R = self.pde.R
        q = self.q

        A00 = BilinearForm(uspace)
        M = ScalarMassIntegrator(coef=R , q=q)
        self.newton_C = ScalarConvectionIntegrator(q=q)
        self.newton_C.keep_data(self.keep_data)
        self.newton_M = ScalarMassIntegrator(q=q)##
        self.newton_M.keep_data(self.keep_data)
        D = ScalarDiffusionIntegrator(coef=dt, q=q)
        A00.add_integrator(M)
        A00.add_integrator(self.newton_C)
        A00.add_integrator(self.newton_M)
        A00.add_integrator(D)

        A01 = BilinearForm((pspace, uspace))
        A01.add_integrator(PressWorkIntegrator(coef=-dt, q=q))

        A10 = BilinearForm((pspace, uspace))
        A10.add_integrator(PressWorkIntegrator(coef=1, q=q))

        A = BlockForm([[A00, A01], [A10.T, None]]) 
        return A


    def Newton_LForm(self, pthreshold=None):

        pspace = self.pspace
        uspace = self.uspace
        dt = self.dt
        R = self.pde.R
        q = self.q

        L0 = LinearForm(uspace)
        self.newton_lform_SI = SourceIntegrator(q=q)
        self.newton_lform_SI.keep_data(self.keep_data)
        L0.add_integrator(self.newton_lform_SI) 
        L1 = LinearForm(pspace)
        L = LinearBlockForm([L0, L1])
        return L

    def Newton_update(self, u0, p0):
        dt = self.dt
        R = self.pde.R
        gd = self.mesh.geo_dimension()
        ## BilinearForm
        def u_C_coef(bcs, index):
            return R*dt*u0(bcs, index)
        self.newton_C.coef = u_C_coef

        self.newton_C.clear()
        def u_M_coef(bcs,index):
            return R*dt*u0.grad_value(bcs, index)
        self.newton_M.coef=u_M_coef


        ##linearform
        def coef(bcs, index):
            result = R*u0(bcs, index)
            result += R*dt*bm.einsum('...j, ...ij -> ...i', u0(bcs, index), u0.grad_value(bcs, index))
            return result
        self.newton_lform_SI.source = coef
       

    def IPCS_BForm_0(self, threshold=None):
        uspace = self.uspace
        dt = self.dt
        R = self.pde.R
        q = self.q
        
        Bform = BilinearForm(uspace)
        M = ScalarMassIntegrator(coef=R/dt, q=q)
        F = FluidBoundaryFrictionIntegrator(coef=-1, q=q, threshold=threshold)
        VW = ViscousWorkIntegrator(coef=2, q=q)
        Bform.add_integrator(VW)
        Bform.add_integrator(F)
        Bform.add_integrator(M)
        return Bform
    
    def IPCS_LForm_0(self, pthreshold=None):
        uspace = self.uspace
        dt = self.dt
        R = self.pde.R
        q = self.q
        
        Lform = LinearForm(uspace)
        
        self.ipcs0_lform_SI = SourceIntegrator(q=q)
        self.ipcs0_lform_SI.keep_data(self.keep_data)
        Lform.add_integrator(self.ipcs0_lform_SI)
        
        self.ipcs0_lform_GSI = GradSourceIntegrator(q=q)
        self.ipcs0_lform_GSI.keep_data(self.keep_data)
        Lform.add_integrator(self.ipcs0_lform_GSI)

        self.ipcs0_lform_BSI = BoundaryFaceSourceIntegrator(q=q, threshold=pthreshold)
        self.ipcs0_lform_BSI.keep_data(self.keep_data)
        Lform.add_integrator(self.ipcs0_lform_BSI)
        return Lform
   
    def update_ipcs_0(self, u0, p0):
        dt = self.dt
        R = self.pde.R
        gd = self.mesh.geo_dimension()
        
        def coef(bcs, index):
            result = R/dt*u0(bcs, index)
            result -= R * bm.einsum('...j, ...ij -> ...i', u0(bcs, index), u0.grad_value(bcs, index))
            return result
        
        def G_coef(bcs, index):
            I = bm.eye(gd)
            result = bm.repeat(p0(bcs,index)[...,bm.newaxis], gd, axis=-1)
            result = bm.expand_dims(result, axis=-1) * I
            return result

        def B_coef(bcs, index):
            result = -bm.einsum('...i, ...j->...ij', p0(bcs, index), self.mesh.face_unit_normal(index=index))
            result += bm.einsum('eqij, ej->...i', u0.grad_value(bcs, index), self.mesh.face_unit_normal(index=index))
            return result

        self.ipcs0_lform_SI.source = coef
        self.ipcs0_lform_BSI.source = B_coef
        self.ipcs0_lform_GSI.source = G_coef


    def IPCS_BForm_1(self):
        pspace = self.pspace
        uspace = self.uspace
        dt = self.dt
        R = self.pde.R
        q = self.q

        Bform = BilinearForm(pspace)
        D = ScalarDiffusionIntegrator(coef=1, q=q)
        Bform.add_integrator(D)
        return Bform 
    
    def IPCS_LForm_1(self):
        pspace = self.pspace
        dt = self.dt
        q = self.q

        Lform = LinearForm(pspace)
        self.ipcs1_lform_SI = SourceIntegrator(q=q)
        self.ipcs1_lform_SI.keep_data(self.keep_data)
        self.ipcs1_lform_GSI = GradSourceIntegrator(q=q)
        self.ipcs1_lform_GSI.keep_data(self.keep_data)
        Lform.add_integrator(self.ipcs1_lform_SI) 
        Lform.add_integrator(self.ipcs1_lform_GSI)
        return Lform

    def update_ipcs_1(self, us, p0):
        dt = self.dt
        R = self.pde.R

        def grad_coef(bcs, index=None):
            result = p0.grad_value(bcs, index)
            return result
        
        def coef(bcs, index=None):
            result = -1/dt*bm.trace(us.grad_value(bcs, index), axis1=-2, axis2=-1)
            return result

        self.ipcs1_lform_GSI.source = grad_coef
        self.ipcs1_lform_SI.source = coef

    def IPCS_BForm_2(self):
        uspace = self.uspace
        R = self.pde.R
        q = self.q

        Bform = BilinearForm(uspace)
        M = ScalarMassIntegrator(coef=R, q=q)
        Bform.add_integrator(M)
        return Bform

    def IPCS_LForm_2(self):
        uspace = self.uspace
        dt = self.dt
        R = self.pde.R
        q = self.q

        Lform = LinearForm(uspace)
        self.ipcs2_lform_SI = SourceIntegrator(q=q)
        self.ipcs2_lform_SI.keep_data(self.keep_data)
        Lform.add_integrator(self.ipcs2_lform_SI)
        return Lform

    def update_ipcs_2(self, us, p0, p1):
        dt = self.dt
        R = self.pde.R

        def coef(bcs, index):
            result = R*us(bcs, index)
            result -= dt*(p1.grad_value(bcs, index) - p0.grad_value(bcs, index))
            return result

        self.ipcs2_lform_SI.source = coef
