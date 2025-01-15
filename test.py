from fealpy.backend import backend_manager as bm
import numpy as np
from fealpy.mesh import TriangleMesh
from fealpy.fem import ScalarMassIntegrator,ScalarSourceIntegrator,BilinearForm,LinearForm
from fealpy.mesh.lagrange_triangle_mesh import LagrangeTriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.functionspace import ParametricLagrangeFESpace

from sympy import *
class PDEData():
    def __init__(self , u :str ,x : str ,y : str, D = [0,1,0,1]):
        u = sympify(u)
        self.u = lambdify([x,y], u ,'numpy')
        self.f = lambdify([x,y], diff(u,x,2) + diff(u,y,2))
        self.grad_ux = lambdify([x,y], diff(u,x,1))
        self.grad_uy = lambdify([x,y], diff(u,y,1))
        self.grad_uxx = lambdify([x,y] , diff(u,x,2))
        self.grad_uyy = lambdify([x,y] , diff(u,y,2))
        self.grad_uxy = lambdify([x,y] , diff(diff(u,x,1) ,y ,1))
        self.domain = D

    def domain(self):
        return self.domain
    
    def solution(self, p):
        x = p[...,0]
        y = p[...,1]

        return self.u(x,y)
    
    def source(self,p):
        x = p[...,0]
        y = p[...,1]
        return self.f(x,y)
    
    def gradient(self,p):
        x = p[...,0]
        y = p[...,1]
        val = bm.zeros_like(p)
        val[...,0] = self.grad_ux(x,y)
        val[...,1] = self.grad_uy(x,y)
        return val
    
    def gradient_twice(self,p):
        x = p[...,0]
        y = p[...,1]
        val = bm.zeros((len(p),3))
        val[...,0] = self.grad_uxx(x,y)
        val[...,1] = self.grad_uxy(x,y)
        val[...,2] = self.grad_uyy(y,y)
        return val
    
    def dirichlet(self,p):
        return self.solution(p)


def example(Lmesh:LagrangeTriangleMesh,uh,space:LagrangeFESpace):
    p = space.p
    qf = mesh.quadrature_formula(p+1)
    bcs,ws = qf.get_quadrature_points_and_weights()
    SMI = ScalarMassIntegrator(q=p+1)
    SSI = ScalarSourceIntegrator(q=p+1)
    bform = BilinearForm(space)
    lform = LinearForm(space)
    bform.add_integrator(SMI)
    lform.add_integrator(SSI)

    SSI.source = uh
    
    phi = space.basis(bcs)
    M = bform.assembly()
    b = lform.assembly()


u = 'sin(x)*sin(y)'
PDE = PDEData(u, 'x', 'y')
mesh = TriangleMesh.from_box(nx=5, ny=5)
Lmesh = LagrangeTriangleMesh.from_triangle_mesh(mesh, p=2)

space = ParametricLagrangeFESpace(Lmesh, p=2)
uh = space.interpolate(PDE.solution)
print(space.number_of_global_dofs())
print(uh.shape)

example(Lmesh,uh , space)

