from fealpy.backend import bm
from fealpy.fem import (ScalarDiffusionIntegrator,ScalarConvectionIntegrator,
                        ScalarSourceIntegrator,ScalarMassIntegrator)
from fealpy.fem import LinearForm, BilinearForm , DirichletBC
from fealpy.functionspace import LagrangeFESpace
from fealpy.mmesh import MMesher, Config
from fealpy.decorator import barycentric
from fealpy.solver import spsolve
from fealpy.mmesh.pde.scalar_burgers_data import ScalarBurgersData


class Burgers_MMsolver:
    def __init__(self, pde: ScalarBurgersData ,p = 1 , nt = 500 , method = 'default'):
        self.pde = pde
        self.nt = nt
        self.method = method
        
        self.dt = pde.T[1]/nt
        self.q = p + 2
        self.mesh = pde.mesh
        self.Re = pde.Re
        
        gamma = 1- bm.sqrt(2)/2 
        self.tau1 = gamma
        self.tau2 = 1
        self.a11 = gamma
        self.a21 = 1- gamma
        self.a22 = gamma
        self.b1 = 1 - gamma
        self.b2 = gamma
        self.sub_steps = 10
    
    def linear_system(self):
        self.space = LagrangeFESpace(self.mesh, p=self.q)
        self.bform0 = BilinearForm(self.space)
        self.bform1 = BilinearForm(self.space)
        self.lform = LinearForm(self.space)
        
        # 积分子定义
        self.SSI = ScalarSourceIntegrator(q=self.q)
        self.SMI = ScalarMassIntegrator(q=self.q)
        self.SDI = ScalarDiffusionIntegrator(q=self.q)
        # self.SCI = ScalarConvectionIntegrator(q=self.q)
        
        self.bform0.add_integrator(self.SDI)
        self.bform1.add_integrator(self.SMI)
        self.lform.add_integrator(self.SSI)
        
        self.u = self.space.function()
        self.u1 = self.space.function()
        self.u2 = self.space.function()
        
        self.bc = DirichletBC(self.space)
    
    def update(self,uh , t ,mv):
        delta = self.dt / self.sub_steps
        mesh = self.mesh
        a = 1 / self.Re
        SDI = self.SDI
        # SCI = self.SCI
        SSI = self.SSI
        SMI = self.SMI
        bc = self.bc
        space = self.space
        node0 = mesh.node - mv * self.dt
        v0 = space.function(mv[:,0])
        v1 = space.function(mv[:,1])
        
        for j in range(self.sub_steps):
            t_hat = t + j * delta
            mesh.node = node0 + self.tau1 * mv * delta
            M = self.bform1.assembly()
            
            SDI.coef = a * delta * self.a11
            SMI.coef = 1.0
            
            @barycentric
            def coef(bcs , index):
                guh = uh.grad_value(bcs , index)
                v0_uh = v0(bcs, index) - uh(bcs , index)
                v1_uh = v1(bcs, index) - uh(bcs , index)
                result = guh[...,0] * v0_uh + guh[...,1] * v1_uh
                result *= delta * self.a11
                result += uh(bcs , index)
                return result
            SSI.source = coef
            
            A = self.bform0.assembly()
            A += M
            
            b = self.lform.assembly()
            bc.gd = lambda p: self.pde.dirichlet(p , t_hat + self.a11 * delta)
            A,b = bc.apply(A , b)
            self.u1[:] = spsolve(A , b , 'scipy')
            
            mesh.node = node0 + delta * mv
            k1 = (self.u1 - uh) / (self.tau1 * delta)
            k1 = space.function(k1)

            SDI.coef = a * delta * self.a22
            SMI.coef = 1.0
            
            @barycentric
            def coef2(bcs , index):
                guh = uh.grad_value(bcs , index)
                v0_uh = v0(bcs, index) - uh(bcs , index)
                v1_uh = v1(bcs, index) - uh(bcs , index)
                result = guh[...,0] * v0_uh + guh[...,1] * v1_uh
                result *= delta * self.a22
                result += uh(bcs , index) + delta * self.a21 * k1(bcs , index)
                return result
            SSI.source = coef2
            A = self.bform0.assembly()
            M = self.bform1.assembly()
            A += M
            
            b = self.lform.assembly()
            bc.gd = lambda p: self.pde.dirichlet(p , t_hat + self.tau2 * delta)
            A,b = bc.apply(A , b)
            self.u2[:] = spsolve(A , b , 'scipy')
            
            k2 = (self.u2 - uh - delta * self.a21 * k1) / (self.a22 * delta)
            self.u[:] = uh + delta * (self.b1 * k1 + self.b2 * k2)
            
            node0 = mesh.node.copy()
            uh[:] = self.u[:]
            
    
            