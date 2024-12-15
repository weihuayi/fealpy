import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import spdiags, bmat
import os

from fealpy.decorator import barycentric
from ..functionspace import LagrangeFESpace
from ..fem import ScalarDiffusionIntegrator, VectorMassIntegrator
from ..fem import ScalarMassIntegrator, ScalarConvectionIntegrator
from ..fem import PressWorkIntegrator
from ..fem import BilinearForm, MixedBilinearForm
from ..fem import LinearForm
from ..fem import VectorSourceIntegrator, ScalarSourceIntegrator
from ..fem import VectorConvectionIntegrator, VectorEpsilonSourceIntegrator
from ..fem import VectorBoundarySourceIntegrator, FluidBoundaryFrictionIntegrator

class NSFEMSolver:
    def __init__(self, mesh, dt, uspace, pspace, Tspace=None, rho=1.0, mu=1.0, q=5):
        #self.model = model
        self.mesh = mesh
        self.uspace = uspace 
        self.pspace = pspace
        self.Tspace = Tspace
        self.rho = rho
        self.mu = mu
        self.q = q
        self.dt = dt

        ##\rho u
        bform = BilinearForm((self.uspace,)*2)
        bform.add_domain_integrator(VectorMassIntegrator(c=rho, q=q))
        self.M = bform.assembly() 
        
        ##mu * \laplace u
        bform = BilinearForm((self.uspace,)*2)
        bform.add_domain_integrator(ScalarDiffusionIntegrator(c=self.mu, q=q))
        self.S = bform.assembly() 
        
        ##\laplace p
        bform = BilinearForm(self.pspace)
        bform.add_domain_integrator(ScalarDiffusionIntegrator(q=q))
        self.SP = bform.assembly() 
        
        ##\nabla p
        bform = MixedBilinearForm((self.pspace,), 2*(self.uspace,)) 
        bform.add_domain_integrator(PressWorkIntegrator(q=q)) 
        self.AP = bform.assembly()
    
        ##mu * (epslion(u), epslion(v))
        bform = BilinearForm((self.uspace,)*2)
        bform.add_domain_integrator(DiffusionIntegrator(mu=mu, q=self.q))
        self.epS = bform.assembly() 

        self._solver = {
            "chorin": self.chorin, 
            "netwon": self.netwon,
            "ipcs": self.ipcs}

        #self.solver = self._solver['']

    def options(self):
        """
        @brief 解法器控制参数
        """
        pass

    def run(self):
        pass

    def chorin(self):
        pass

    def netwon(self):
        pass

    def ipcs(self):
        pass
    
    # 求解中间速度u_star
    def chorin_A_0(self, mu=None, rho=None):
        dt = self.dt
        if rho is None:
            M = self.M
        else:
            bform = BilinearForm((self.uspace,)*2)
            bform.add_domain_integrator(VectorMassIntegrator(c=rho, q=self.q))
            M = bform.assembly() 
        
        if mu is None:
            S = self.S
        else:
            bform = BilinearForm((self.uspace,)*2)
            bform.add_domain_integrator(ScalarDiffusionIntegrator(c=mu, q=self.q))
            S = bform.assembly()
         
        A = M + dt*S
        return A

    def chorin_b_0(self, un, source,rho=None):
        dt = self.dt
        
        if rho is None:
            rho = self.rho

        @barycentric
        def coef(bcs, index): 
            if callable(rho):
                result = np.einsum('imnc, inc->imc',un.grad_value(bcs, index), un(bcs, index))
                result = un(bcs, index) - dt*result
                result = np.einsum('ic,ijc->ijc',rho(bcs,index),result)
                result += dt*source(bcs, index)
                return result
            else:
                result = np.einsum('imnc, inc->imc',un.grad_value(bcs, index), un(bcs, index))
                result = un(bcs, index) - dt*result + dt*source(bcs, index)
                return rho * result
        
        L = LinearForm((self.uspace,)*2)
        L.add_domain_integrator(VectorSourceIntegrator(coef, q=self.q))
        b = L.assembly()
        return b

    # 求压力
    def chorin_A_1(self):
        return self.dt*self.SP
    
    def chorin_b_1(self, us, rho=None):
        
        if rho is None:
            rho = self.rho

        @barycentric
        def coef(bcs, index): 
            if callable(rho):
                result = us.grad_value(bcs, index)[:,0,0,:] + us.grad_value(bcs,index)[:,1,1,:]
                result = np.einsum('ij,ij->ij',rho(bcs,index), result)
                return -result
            else:
                result = us.grad_value(bcs, index)[:,0,0,:] + us.grad_value(bcs,index)[:,1,1,:]
                return -rho * result
        
        L = LinearForm(self.pspace)
        L.add_domain_integrator(ScalarSourceIntegrator(coef, self.q))
        b = L.assembly()
        return b

    #求下一步速度
    def chorin_A_2(self, rho=None):
        if rho is None:
            A = self.M
        else:
            bform = BilinearForm((self.uspace,)*2)
            bform.add_domain_integrator(VectorMassIntegrator(c=rho, q=self.q))
            A = bform.assembly()         
        return A
    
    def chorin_b_2(self, us, p, rho=None):
        dt = self.dt
        
        if rho is None:
            rho = self.rho

        @barycentric
        def coef(bcs, index): 
            if callable(rho):
                result =  np.einsum('ij,ikj->ikj',rho(bcs,index), us(bcs,index))
                result -= p.grad_value(bcs,index).transpose(0,2,1)
                return result
            else:
                result = rho*us(bcs,index) - dt*p.grad_value(bcs,index).transpose(0,2,1)
                return result
        
        L = LinearForm((self.uspace,)*2)
        L.add_domain_integrator(VectorSourceIntegrator(coef, self.q))
        b = L.assembly()
        return b
    
    # 求解中间速度u_star
    def ipcs_A_0(self, mu=None, rho=None, threshold=None, Re=1):
        dt = self.dt
        if rho is None:
            rho = self.rho
            M = rho*self.M
        else:
            bform = BilinearForm((self.uspace,)*2)
            bform.add_domain_integrator(VectorMassIntegrator(c=rho, q=self.q))
            M = bform.assembly() 
        
        if mu is None:
            S = self.epS
        else:
            bform = BilinearForm((self.uspace,)*2)
            bform.add_domain_integrator(DiffusionIntegrator(mu=mu, q=self.q))
            S = bform.assembly()
            self.epS = S
        A = M + (1/Re)*dt*S
        
        if threshold is not None:
            bform = BilinearForm((self.uspace,)*2)
            bform.add_boundary_integrator(FluidBoundaryFrictionIntegrator(mu=mu, q=self.q, threshold=threshold))
            B = bform.assembly()
            self.bfS = B
            A -= (1/Re)*dt*0.5*B
        return A

    def ipcs_b_0(self, un, p0, source, Re=1, rho=None, threshold=None):
        dt = self.dt
        
        if rho is None:
            rho = self.rho
            
        '''
        (\rho(u^n - dt * u^n \cdot \nabla u^n) , v)
        '''
        @barycentric
        def coef(bcs, index): 
            if callable(rho):
                result = np.einsum('imnc, inc->imc',un.grad_value(bcs, index), un(bcs, index))
                result = un(bcs, index) - dt*result
                result = np.einsum('ic,ijc->ijc',rho(bcs,index),result)
                result += dt*source(bcs, index)
                return result
            else:
                result = un(bcs, index)
                result -= dt * np.einsum('imnc, inc->imc',un.grad_value(bcs, index), un(bcs, index))
                result +=  dt*source(bcs, index)
                return rho * result
        '''
        (p^n I, \nabla b)
        '''
        @barycentric
        def coefp(bcs, index):
            result = np.repeat(p0(bcs,index)[...,np.newaxis], 2, axis=-1)
            return dt*result
        
        L = LinearForm((self.uspace,)*2)
        L.add_domain_integrator(VectorSourceIntegrator(coef, q=self.q))
        L.add_domain_integrator(VectorEpsilonSourceIntegrator(coefp, q=self.q))
        
        if threshold is not None:
            '''
            <p^n \cdot n, v>
            '''
            @barycentric
            def coefpn(ebcs, index):
                val = p0(ebcs,index=index)
                n = self.mesh.face_unit_normal(index=index)
                result = np.einsum('ij,jk->ijk',val,n)
                return -dt*result
            L.add_boundary_integrator(VectorBoundarySourceIntegrator(coefpn, q=self.q, threshold=threshold))
        
        b = L.assembly()
        b -= (1/Re)*dt*self.epS@un.flatten() 
        b += (1/Re)*0.5*dt*self.bfS@un.flatten() 
        return b

    # 求压力
    def ipcs_A_1(self):
        return self.dt*self.SP
    
    def ipcs_b_1(self, us, p0, rho=None):
        
        if rho is None:
            rho = self.rho

        @barycentric
        def coef(bcs, index): 
            if callable(rho):
                result = us.grad_value(bcs, index)[:,0,0,:] + us.grad_value(bcs,index)[:,1,1,:]
                result = np.einsum('ij,ij->ij',rho(bcs,index), result)
                return -result
            else:
                result = us.grad_value(bcs, index)[:,0,0,:] + us.grad_value(bcs,index)[:,1,1,:]
                return -rho * result
        
        L = LinearForm(self.pspace)
        L.add_domain_integrator(ScalarSourceIntegrator(coef, self.q))
        b = L.assembly()
        b += self.dt*self.SP@p0 
        return b
    
    #求下一步速度
    def ipcs_A_2(self, rho=None):
        if rho is None:
            A = self.M
        else:
            bform = BilinearForm((self.uspace,)*2)
            bform.add_domain_integrator(VectorMassIntegrator(c=rho, q=self.q))
            #bform.add_domain_integrator(VectorMassIntegrator(q=self.q))
            A = bform.assembly()         
        return A
    
    def ipcs_b_2(self, us, p1, p0, rho=None):
        dt = self.dt
        
        if rho is None:
            rho = self.rho

        @barycentric
        def coef(bcs, index): 
            if callable(rho):
                result0 = p1.grad_value(bcs, index) - p0.grad_value(bcs, index)
                result =  np.einsum('ij,ikj->ikj',rho(bcs,index), us(bcs,index))
                #result =  us(bcs,index)
                result -= dt * result0.transpose(0,2,1)
                return result
            else:
                result = p1.grad_value(bcs, index) - p0.grad_value(bcs, index)
                result = rho*us(bcs,index) - dt*result.transpose(0,2,1)
                return result
        
        L = LinearForm((self.uspace,)*2)
        L.add_domain_integrator(VectorSourceIntegrator(coef, self.q))
        b = L.assembly()
        return b
    
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


    #u \cdot \nabla u   \approx   u^n \cdot \nabla u^{n+1}
    def ossen_A(self,un, mu=None ,rho=None):
        AP = self.AP
        if rho is None:
            M = self.M
        else:
            bform = BilinearForm((self.uspace,)*2)
            bform.add_domain_integrator(VectorMassIntegrator(c=rho, q=self.q))
            M = bform.assembly() 

        if mu is None:
            S = self.S
        else:
            bform = BilinearForm((self.uspace,)*2)
            bform.add_domain_integrator(VectorDiffusionIntegrator(c=mu, q=self.q))
            S = bform.assembly()

        SP = self.SP
        dt = self.dt
        
        @barycentric
        def coef(bcs, index):
            if callable(rho):
                return rho(bcs,index)[:,None,:]*un(bcs,index)
            else:
                return rho*un(bcs,index)
            
        bform = BilinearForm((self.uspace,)*2)
        bform.add_domain_integrator(VectorConvectionIntegrator(c=coef, q=self.q))
        C = bform.assembly() 

        A0 = 1/dt*M+S+C
        A = bmat([[1/dt*M+S+C,  -AP],\
                [AP.T, None]], format='csr')
        return A

    def ossen_b(self, un, rho=None): 
        dt = self.dt
        pgdof = self.pspace.number_of_global_dofs()
        if rho is None:
            M = self.M
        else:
            bform = LinearForm((self.uspace,)*2)
            bform.add_domain_integrator(VectorMassIntegrator(c=rho, q=self.q))
            M = bform.assembly() 
        
        b = 1/dt * M@un.flatten()
        b = np.hstack((b,[0]*pgdof))
        return b
    
    def slip_stick_boundary(self, A, b, stick_dof=None):
        pass
    
    def netwon_sigma(self, u, mu):
        doforder = self.uspace.doforder
        mid_bcs = np.array([1/3,1/3,1/3],dtype=np.float64)
        if doforder == 'sdofs':
            result = mu(mid_bcs)*u.grad_value(mid_bcs) + mu(mid_bcs)*u.grad_value(mid_bcs).transpose(1,0,2)
        else:
            print("还没开发")
        return result

    def netwon_A(self, un):
        M = self.M
        AP = self.AP
        rho = self.rho
        S = self.S
        SP = self.SP
        dt = self.dt
 
        @barycentric
        def coef(bcs, index):
            if callable(rho):
                return rho(bcs,index)[:,None,:]*un(bcs,index)
            else:
                return rho*un(bcs,index)
            
        bform = BilinearForm((self.uspace,)*2)
        bform.add_domain_integrator(VectorConvectionIntegrator(c=coef, q=self.q))
        C = bform.assembly() 

        A0 = 1/dt*M+S+C
        A = bmat([[1/dt*M+S+C,  -AP],\
                [AP.T, None]], format='csr')
        return A
    
    def cross_wlf(self, p, u, bcs, T=200):
        #参数选择为PC的参数
        D1 = 1.9e11
        D2 = 417.15
        D3 = 0
        A1 = 27.396
        A2_wave = 51.6
        tau = 182680
        n = 0.574 
        
        deformnation = u.grad_value(bcs)
        deformnation = 0.5*(deformnation + deformnation.transpose(0,2,1,3))
        gamma = np.sqrt(2*np.einsum('iklj,iklj->ij',deformnation,deformnation))

        T_s = D2 + D3*p(bcs)
        A2 = A2_wave + D3*p(bcs)
        eta0 = D1 * np.exp(-A1*(T-T_s)/(A2 + (T-T_s))) 
        eta = eta0/(1 + (eta0 * gamma/tau)**(1-n))
        return eta
    
    def output(self, name, variable, timestep, output_dir='./', filename_prefix='test'):
        mesh = self.mesh
        gdof = self.uspace.number_of_global_dofs()
        NC = mesh.number_of_cells()
        assert len(variable) == len(name)
        for i in range(len(name)):
            if variable[i].shape[-1] == gdof:
                mesh.nodedata[name[i]] = np.swapaxes(variable[i],0,-1)
            elif variable[i].shape[-1] == NC:
                mesh.celldata[name[i]] = variable[i] 
        fname = os.path.join(output_dir, f'{filename_prefix}_{timestep:010}.vtu')
        mesh.to_vtk(fname=fname)
