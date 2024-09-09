from fealpy.experimental.backend import backend_manager as bm
import matplotlib.pyplot as plt
from app.mmpde.harmap_mmpde import Harmap_MMPDE
from fealpy.experimental.mesh import TriangleMesh
from fealpy.mesh import TriangleMesh as TM
from app.mmpde.harmap_mmpde_data import *
from sympy import *
from fealpy.experimental.functionspace import LagrangeFESpace
from fealpy.experimental.fem import (BilinearForm 
                                     ,ScalarDiffusionIntegrator
                                     ,LinearForm
                                     ,ScalarSourceIntegrator
                                     ,DirichletBC)
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix


class PDEData():
    def __init__(self , u :str ,x : str ,y : str, D = [0,1,0,1]):
        u = sympify(u)
        self.u = lambdify([x,y], u ,'numpy')
        f_str = -diff(u,x,2) - diff(u,y,2)
        self.f = lambdify([x,y], f_str)
        self.grad_ux = lambdify([x,y], diff(u,x,1))
        self.grad_uy = lambdify([x,y], diff(u,y,1))
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
    
    def dirichlet(self,p):
        return self.solution(p)
    

def test_harmap_mmpde(beta , mol_times , redistribute):
    mesh = mesh_data['from_box']
    pde = PDEData(function_data['u0'] , x='x',y='y' , D = [0,1,0,1])
    print('Number of points:', mesh.number_of_nodes())
    print('Number of cells:', mesh.number_of_cells())

    p = 1
    space = LagrangeFESpace(mesh, p=p)
    bform = BilinearForm(space)
    bform.add_integrator(ScalarDiffusionIntegrator(q=p+2))
    A = bform.assembly()
    lform = LinearForm(space)
    lform.add_integrator(ScalarSourceIntegrator(source = pde.source, q=p+2))
    F = lform.assembly()
    bc = DirichletBC(space = space, gd = pde.dirichlet) # 创建 Dirichlet 边界条件处理对象
    uh0 = bm.zeros_like(mesh.node[:,0],dtype=bm.float64) # 创建有限元函数对象
    A, F = bc.apply(A, F, uh0)
    uh0[:] = spsolve(csr_matrix(A.toarray()), F)

    HMP = Harmap_MMPDE(mesh,uh0 ,beta = beta , mol_times= mol_times , redistribute=redistribute)
    mesh0 = HMP()
    node= mesh0.node
    cell = mesh0.cell

    mesh1 = TM(node,cell)
    mesh1.show_function(plt,pde.solution(node))
    fig = plt.figure()
    axes = fig.gca()
    mesh1.add_plot(axes)
    plt.show()

if __name__ == '__main__':
    beta_ = 1
    mol_times = 3
    redistribute = True
    test_harmap_mmpde(beta_ , mol_times , redistribute)
    

