#!/usr/bin/python3
import numpy as np
from scipy.sparse.linalg import spsolve
from fealpy.mesh.uniform_mesh_2d import UniformMesh2d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fealpy.cfd import NSMacSolver
from tarlor_green_pde import taylor_greenData 
from scipy.sparse import spdiags

Re = 1
nu = 1/Re
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
solution_u_values0 = solution_u.reshape(-1)
solution_v = mesh_v.interpolate(solution_v_0)
solution_v_values0 = solution_v.reshape(-1)
solution_p = mesh_p.interpolate(solution_p_0)
solution_p_values0 = solution_p.reshape(-1)

solver = NSMacSolver(mesh_u, mesh_v, mesh_p)
gradux0 = solver.grad_ux() @ solution_u_values0
graduy0 = solver.grad_uy() @ solution_u_values0
gradvx0 = solver.Tuv() @ solution_v_values0
laplaceu = solver.laplace_u()

AD_xu_0 = solution_u_values0 * gradux0 + gradvx0 * graduy0

def solution_u_1(p):
    return pde.solution_u(p,t=tau)
def solution_v_1(p):
    return pde.solution_v(p,t=tau)
def solution_p_1(p):
    return pde.solution_p(p,t=tau)

solution_u = mesh_u.interpolate(solution_u_1) #[4,5]
solution_u_values1 = solution_u.reshape(-1)
solution_v = mesh_v.interpolate(solution_v_1)
solution_v_values1 = solution_v.reshape(-1)
solution_p = mesh_p.interpolate(solution_p_1)
solution_p_values1 = solution_p.reshape(-1)

gradux1 = solver.grad_ux() @ solution_u_values1
graduy1 = solver.grad_uy() @ solution_u_values1
gradvx1 = solver.Tuv() @ solution_v_values1

AD_xu_1 = solution_u_values1 * gradux1 + gradvx1 * graduy1

I = np.zeros_like(laplaceu.toarray())
row, col = np.diag_indices_from(I)
I[row,col] = 1
A = I - (nu*tau*laplaceu)/2
F = solver.source_Fx(pde,t=tau)
Fx = F[:,0]
b = tau*(-3/2*AD_xu_1-1/2*AD_xu_0+nu/2*(laplaceu@solution_u_values1)+Fx-solver.grand_uxp()@solution_p_values1)

nxu = mesh_u.node.shape[1]
is_boundaryu = np.zeros(num_nodes_u,dtype='bool')
is_boundaryu[:nxu] = True
is_boundaryu[-nxu:] = True
dirchiletu = pde.dirichlet_u(nodes_u[is_boundaryu], 0)
b[is_boundaryu] = dirchiletu

bdIdxu = np.zeros(A.shape[0], dtype=np.int_)
bdIdxu[is_boundaryu] = 1
Tbdu = spdiags(bdIdxu, 0, A.shape[0], A.shape[0])
T = spdiags(1-bdIdxu, 0, A.shape[0], A.shape[0])
A = A@T + Tbdu
u_1 = spsolve(A, b)


