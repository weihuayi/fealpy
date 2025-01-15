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
                        PressWorkIntegrator)
from fealpy.fem import (BoundaryFaceMassIntegrator,
                        BoundaryFaceSourceIntegrator)
from tangent_face_mass_integrator import TangentFaceMassIntegrator
                        
                        
                        


class Solver():
    def __init__(self, pde, mesh, pspace, phispace, uspace, dt, q=5):
        self.mesh = mesh
        self.phispace = phispace
        self.pspace = pspace
        self.uspace = uspace
        self.pde = pde
        self.dt = dt
        self.q = q
        self.is_bd = pde.is_wall_boundary
    
    def CH_BForm_0(self):
        phispace = self.phispace
        dt = self.dt
        L_d = self.pde.L_d
        epsilon = self.pde.epsilon
        s = self.pde.s
        V_s = self.pde.V_s
        q = self.q
        return A

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
        self.phi_C.keep_data()
        A00.add_integrator(M)
        A00.add_integrator(self.phi_C)

        A01 = BilinearForm(phispace)
        A01.add_integrator(ScalarDiffusionIntegrator(coef=2*dt*L_d, q=q))
        
        A10 = BilinearForm(phispace)
        A10.add_integrator(ScalarDiffusionIntegrator(coef=-epsilon, q=q))
        A10.add_integrator(ScalarMassIntegrator(coef=-s/epsilon, q=q))
        A10.add_integrator(BoundaryFaceMassIntegrator(coef=-3/(2*dt*V_s), q=q, threshold=self.is_bd))     

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
        self.mu_BF_SI = BoundaryFaceSourceIntegrator(q=q, threshold=self.is_bd)
        self.mu_SI.keep_data()
        self.mu_BF_SI.keep_data()
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
        self.u_C.keep_data()
        D = ScalarDiffusionIntegrator(coef=2*dt, q=q)
        ## TODO:和老师确认一下这个边界积分子 
        FM = TangentFaceMassIntegrator(coef=2*dt/L_s, q=q, threshold=self.is_bd)
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
        self.u_BF_SI = BoundaryFaceSourceIntegrator(q=q, threshold=self.is_bd)
        self.u_SI.keep_data()
        self.u_BF_SI.keep_data()

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
    ''' 
    def stress(self, u):
        bcs = bm.array([[1/3, 1/3, 1/3]], dtype=bm.float64)
        stress = u.grad_value(bcs)
        return stress
    '''

    def plot_change_on_y(self, fun, y, space=None):
        import matplotlib.pyplot as plt
        if space is None:
            space = fun.space
        mesh = self.mesh
        ip = space.interpolation_points()
        tag = ip[..., 1] == y
        x = ip[tag, 0]
        plt.plot(bm.sort(x), fun[tag][bm.argsort(x)])
        plt.show()
    
    def interface_on_boundary(self, phi):
        inteface_phi = bm.abs(phi[:])<0.9
        space = phi.space
        ip = space.interpolation_points()

        up = space.is_boundary_dof(self.pde.is_up_boundary, method='interp')
        down = space.is_boundary_dof(self.pde.is_down_boundary, method='interp')
        up_tag = up & inteface_phi
        down_tag = down & inteface_phi
        up_node = bm.mean(ip[up_tag], axis=0)
        down_node = bm.mean(ip[down_tag], axis=0)
        return up_node, down_node
    
    def slip_dof(self, up_node, down_node, h):
        @cartesian
        def up_dof(p):
            x = p[..., 0]
            y = p[..., 1]
            tag0 = bm.abs((p[..., 1] - up_node[1])) < 1e-10
            tag1 = (p[..., 0] > up_node[0]) & (p[..., 0] < up_node[0] + h)  
            return tag0 & tag1
        
        @cartesian
        def down_dof(p):
            x = p[..., 0]
            y = p[..., 1]
            tag0 = bm.abs((p[..., 1] - down_node[1])) < 1e-10
            tag1 = (p[..., 0] > down_node[0]) & (p[..., 0] < down_node[0] + h)  
            return tag0 & tag1
        updof = self.uspace.scalar_space.is_boundary_dof(up_dof, method='interp')
        downdof = self.uspace.scalar_space.is_boundary_dof(down_dof, method='interp')
        
        return updof, downdof 

    def stress(self, uh):
        GD = self.mesh.GD
        sspace = self.uspace.scalar_space
        cell2dof = sspace.cell_to_dof()
        gdof = sspace.number_of_global_dofs()
        ldof = sspace.number_of_local_dofs()
        p = sspace.p
        bc = bm.multi_index_matrix(p,GD,dtype=sspace.ftype)/p
        guh = uh.grad_value(bc)
        NC = self.mesh.number_of_cells()
        
        rguh = bm.zeros((gdof, GD, GD), dtype=sspace.ftype)
        
        deg = bm.bincount(cell2dof.flatten(), minlength = gdof)
        bm.index_add(rguh, cell2dof, guh)
        
        rguh /= deg[:, None, None]
        result = 0.5*(rguh + rguh.swapaxes(-1,-2))
        return result
