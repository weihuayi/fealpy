from fealpy.backend import backend_manager as bm
import matplotlib.pyplot as plt
from app.mmpde.harmap_mmpde import Harmap_MMPDE
from fealpy.mesh import TriangleMesh
from fealpy.mesh import TriangleMesh as TM
from app.mmpde.harmap_mmpde_data import *
from sympy import *
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import (BilinearForm 
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
    mesh = mesh_data['from_domain_distmesh']
    pde = PDEData(function_data['u2'] , x='x',y='y' , D = [0,1,0,1])
    print('Number of points:', mesh.number_of_nodes())
    print('Number of cells:', mesh.number_of_cells())

    p = 1
    space = LagrangeFESpace(mesh, p=p)
    uh0 = space.interpolate(pde.solution)

    HMP = Harmap_MMPDE(mesh,uh0, pde=pde ,beta = beta , mol_times= mol_times , redistribute=redistribute)
    mesh0 = HMP.solve_elliptic_Equ()
    node= mesh0.node
    cell = mesh0.cell
    space = LagrangeFESpace(mesh0, p=p)
    uh = space.interpolate(pde.solution)
    error0 = mesh.error(space.function(array = uh0) ,pde.solution)
    error1 = mesh0.error(space.function(array = uh) ,pde.solution)
    
    print('旧网格插值误差:',error0)
    print('新网格插值误差:',error1)
    
    mesh1 = TM(node,cell)# 新网格
    mesh2 = TM(mesh.node,mesh.cell)# 旧网格
    error0_color = mesh.error(space.function(array = uh0) ,pde.solution,celltype=True)
    error1_color = mesh0.error(space.function(array = uh) ,pde.solution,celltype=True)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_trisurf(node[:, 0], node[:, 1], pde.solution(node),
                     triangles = cell, cmap='viridis', 
                     edgecolor='blue',linewidth=0.2)
    
    fig , axes0 = plt.subplots(1,2)
    mesh2.add_plot(axes0[0])
    mesh2.add_plot(axes0[1] ,cellcolor=error0_color)
    fig , axes1 = plt.subplots(1,2)
    mesh1.add_plot(axes1[0])
    mesh1.add_plot(axes1[1] ,cellcolor=error1_color)
    plt.show()
    
if __name__ == '__main__':
    beta_ = 0.42
    mol_times = 1
    redistribute = False
    test_harmap_mmpde(beta_ , mol_times , redistribute)
    

