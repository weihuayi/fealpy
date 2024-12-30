#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: ns_fem_solver_new.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Thu 05 Dec 2024 10:22:30 AM CST
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
                        FluidBoundaryFrictionIntegrator,
                        ViscousWorkIntegrator)
from fealpy.fem import (BoundaryFaceMassIntegrator,
                        BoundaryFaceSourceIntegrator)

class NSFEMSolver():
    def __init__(self, pde, mesh, pspace, uspace, dt, q=5):        
        self.pde = pde
        self.mesh = mesh
        self.pspace = pspace
        self.uspace = uspace
        self.dt = dt
        self.q = q

    def Ossen_BForm(self):
        pspace = self.pspace
        uspace = self.uspace
        dt = self.dt
        R = self.pde.R
        q = self.q

        A00 = BilinearForm(uspace)
        M = ScalarMassIntegrator(coef=R , q=q)
        self.u_C = ScalarConvectionIntegrator(q=q)
        self.u_C.keep_data()
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
        self.u_SI.keep_data()
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
    
    def IPCS_BForm_0(self, threshold):
        pspace = self.pspace
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
        pspace = self.pspace
        uspace = self.uspace
        dt = self.dt
        R = self.pde.R
        q = self.q
        
        Lform = LinearForm(uspace)
        self.ipcs_lform_SI = SourceIntegrator(q=q)
        self.ipcs_lform_SI.keep_data()
        self.ipcs_lform_BSI = BoundaryFaceSourceIntegrator(q=q, threshold=pthreshold)
        self.ipcs_lform_BSI.keep_data()
        return Lform

    def update_ipcs_0(self, u0, p0):
        dt = self.dt
        R = self.pde.R

        def coef(bcs, index):
            result = 1/dt*u0(bcs, index)
            result += np.einsum('...j, ....,ij -> ...i', u0(bcs, index), u0.grad_value(bcs, index))
            result += np.repeat(p0(bcs,index)[...,np.newaxis], 2, axis=-1)
            return 

        def B_coef(bcs, index):
            result = np.einsum('..ij, ....j->...ij', p(bcs, index), self.mesh.edge_unit_normal(bcs, index))
            return

        self.ipcs_lform_SI.source = coef
        self.ipcs_lform_BSI.source = B_coef
    

