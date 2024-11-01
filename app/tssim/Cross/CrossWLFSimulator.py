#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: cross_solver.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: Fri 17 May 2024 04:05:10 PM CST
	@bref 
	@ref 
'''  
import numpy as np
from fealpy.cfd import NSFEMSolver
from fealpy.levelset.ls_fem_solver import LSFEMSolver

class CrossSolver():
    def __init__(self, pde, mesh, uspace, pspace, phispace):
        self.pde = pde
        self.mesh = mesh
        self.uspace = uspace
        self.pspace = pspace
        self.phispace = phispace


    def Heavside(self, phi, epsilon=None):
        if epsilon is None:
            #epsilon = np.max(self.mesh.entity_measure('edge'))
            epsilon = np.min(self.mesh.entity_measure('edge'))/2
        result = self.phispace.function()
        result[phi<-epsilon] = 0
        result[phi>epsilon] = 1
        tag = (phi>=-epsilon) & (phi<=epsilon)
        result[tag] = 0.5*(1 + phi[tag]/epsilon + np.sin(np.pi*phi[tag]/epsilon))/np.pi
        return result
        
    
    '''
    计算参数函数rho,c,lambda,eta
    '''
    def parfunction(self, fun, phi):
        result = self.phispace.function() 
        result[:] = fun + (1-fun)*self.Heavside(phi)
        return result
    
    '''
    sdof函数梯度
    '''
    def grad_fun(self, fun):
        GD = self.mesh.geo_dimension()
        space = fun.space
        p = space.p
        ndof = space.number_of_global_dofs()
        dim = len(fun.shape)
        multindex = self.mesh.multi_index_matrix(p=p, etype=2)/GD
        grad = fun.grad_value(multindex)
        cell2dof = space.cell_to_dof()
        if dim==1:
            '''
            标量函数
            '''
            result = space.function(dim=2)
            result[0,:][np.transpose(cell2dof)] = grad[:,:,0]
            result[1,:][np.transpose(cell2dof)] = grad[:,:,1]
        if dim==2:
            '''
            向量函数
            '''
            result = space.function(array = np.zeros((2,2,ndof)))
            result[0,0,:][np.transpose(cell2dof)] = grad[:,0,0,:]
            result[0,1,:][np.transpose(cell2dof)] = grad[:,0,1,:]
            result[1,0,:][np.transpose(cell2dof)] = grad[:,1,0,:]
            result[1,1,:][np.transpose(cell2dof)] = grad[:,1,1,:]
        return result

    '''
    计算剪切速率
    '''
    def delta_epsion(self, phi, epsilon=None): 
        if epsilon is None:
            epsilon = np.max(self.mesh.entity_measure('edge'))
            #epsilon = np.min(self.mesh.entity_measure('edge'))/2
        result = self.phispace.function()
        tag = (phi>-epsilon) & (phi<epsilon)
        result[tag] = 0.5*(1 + np.cos(np.pi*phi[tag]/epsilon))/epsilon
        return result 

    '''
    计算eta_l
    '''
    def eta_l(self, T, p, u, bcs=None):
        #参数选择为PC的参数
        D1 = 1.9e11
        D2 = 417.15
        D3 = 0
        A1 = 27.396
        A2_wave = 51.6
        tau = 182680
        n = 0.574
        lam = 0.173 
        
        tag = bcs
        if bcs is None:
            bcs = self.mesh.multi_index_matrix(p=2, etype=2)/2
            cell2dof = self.uspace.cell_to_dof()
            result = self.uspace.function()
        deformnation = u.grad_value(bcs)
        deformnation = 0.5*(deformnation + deformnation.transpose(0,2,1,3))
        gamma = np.sqrt(2*np.einsum('iklj,iklj->ij',deformnation,deformnation))

        T_s = D2 + D3*p(bcs)
        A2 = A2_wave + D3*p(bcs)
        eta0 = D1 * np.exp(-A1*(T(bcs)-T_s)/(A2 + (T(bcs)-T_s))) 
        eta = eta0/(1 + (eta0 * gamma/tau)**(1-n))
        if tag is None:
            result[np.transpose(cell2dof)] = eta
        else:
            result = eta
        return result
    

    def Re(self, eta_l):
        pde = self.pde
        rho_l = self.rho_l
        U = self.U
        L = self.L
        result = self.phispace.function()
        result[:] = rho_l*L*U/eta_l
        return result
    
    def Re(self, eta_l):
        pde = self.pde
        rho_l = pde.rho_l
        U = pde.U
        L = pde.L
        result = self.phispace.function()
        result[:] = rho_l*L*U/eta_l
        return result
    
    def Br(self, eta_l):
        pde = self.pde
        U = pde.U
        T0 = pde.T0
        lambda_l = pde.lambda_l
        result = self.phispace.function()
        result[:] = eta_l *U*U/(lambda_l*T0) 
        return result

    '''
    界面法向散度
    '''
    def kappa(self, phi):
        gphi = self.grad_fun(phi)
        ggphi = self.grad_fun(gphi)
        normgphi = np.sqrt(gphi[0,:]**2 + gphi[1,:]**2)
        result = self.phispace.function()
        result[:] = (ggphi[0,0,:]+ggphi[1,1,:])/normgphi
        return result
        

    
    '''
    计算温度方程
    '''
    def temperature_A(self, u, C, rho, lam, pe):
        bform = BilinearForm(self.Tspace)
        dt = self.dt

        @barycentric
        def coef(bcs, index): 
            return pe*rho(bcs,index) * C(bcs,index)

        @barycentric
        def coef1(bcs, index):
            result = np.einsum('ikj,ij->ikj',u(bcs,index),coef(bcs,index))
            return dt*result

        @barycentric
        def coef2(bcs, index):
            return dt*lam(bcs,index)
        bform.add_domain_integrator(ScalarMassIntegrator(c=coef, q=self.q))
        bform.add_domain_integrator(ScalarConvectionIntegrator(c=coef1, q=self.q))
        bform.add_domain_integrator(ScalarDiffusionIntegrator(c=coef2, q=self.q))
        A = bform.assembly()         

    def temperature_b(self, Tn, u, p, eta, br, pe, rho, c):
        dt = self.dt

        @barycentric
        def coef(bcs, index): 
            gradu = u.grad_value(bcs,index)
            D = 0.5*(gradu + gradu.transpose(0,2,1,3))
            D = np.einsum('ij, imnj->imnj',eta(bcs,index), D)
            D[:,0,0,:] -= p(bcs,index)
            D[:,1,1,:] -= p(bcs,index)
            result = br*np.einsum('imnj,imnj->ij',D, gradu)
            result = pe*c(bcs,index)*rho(bcs,index)*Tn(bcs,index) + dt*result
            return result
        L = LinearForm(self.Tspace)
        L.add_domain_integrator(ScalarSourceIntegrator(coef, self.q))
        b = L.assembly()
        return b
    
     

    


