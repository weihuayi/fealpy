#!/usr/bin/python3
import numpy as np
from scipy.sparse.linalg import spsolve
from fealpy.mesh.uniform_mesh_2d import UniformMesh2d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fealpy.cfd import NSMacSolver
from tarlor_green_pde import taylor_greenData 

Re = 1

#PDE 模型 
pde = taylor_greenData(Re)
domain = pde.domain()

#空间离散
nx = 4
ny = 4
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny
mesh_u = UniformMesh2d([0, nx, 0, ny-1], h=(hx, hy), origin=(domain[0], domain[2]+hy/2))
mesh_v = UniformMesh2d([0, nx-1, 0, ny], h=(hx, hy), origin=(domain[0]+hx/2, domain[2]))
mesh_p = UniformMesh2d([0, nx-1, 0, ny-1], h=(hx, hy), origin=(domain[0]+hx/2, domain[2]+hy/2))

#时间离散
duration = pde.duration()
nt = 128
tau = (duration[1] - duration[0])/nt

#初值
nodes_u = mesh_u.entity('node') #[20.2]
nodes_v = mesh_v.entity('node') #[20,2]
nodes_p = mesh_p.entity('node') #[16,2]

#计算网格u,v,p节点的总数
num_nodes_u = nodes_u.shape[0] #20
num_nodes_v = nodes_v.shape[0] #20
num_nodes_p = nodes_p.shape[0] #16

def solution_u_0(p):
    return pde.solution_u(p,t=0)
def solution_v_0(p):
    return pde.solution_v(p,t=0)
def solution_p_0(p):
    return pde.solution_p(p,t=0)

solution_u = mesh_u.interpolate(solution_u_0) #[4,5]
solution_u_values = solution_u.reshape(-1)
solution_v = mesh_v.interpolate(solution_v_0)
solution_v_values = solution_v.reshape(-1)
solution_p = mesh_p.interpolate(solution_p_0)
solution_p_values = solution_p.reshape(-1)

solver = NSMacSolver(mesh_u, mesh_v, mesh_p)
gradux = solver.grad_ux() @ solution_u_values
graduy = solver.grad_uy() @ solution_u_values
gradvx = solver.Tuv() @ solution_v_values
laplaceu = solver.laplace_u()

print(solver.source_Fx(pde,0.1).shape)
print(gradvx.shape)
AD_xu_1 = solution_u_values * gradux + gradvx * graduy
#print(solver.grad_ux().toarray())
