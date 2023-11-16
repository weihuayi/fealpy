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

#0时间层的值
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
#u网格上的算子矩阵
gradux0 = solver.grad_ux() @ solution_u_values0
graduy0 = solver.grad_uy() @ solution_u_values0
Tuv0 = solver.Tuv() @ solution_v_values0
laplaceu = solver.laplace_u()
#v网格上的算子矩阵
gradvx0 = solver.grad_vx() @ solution_v_values0
gradvy0 = solver.grad_vy() @ solution_v_values0
Tvu0 = solver.Tvu() @ solution_u_values0
laplacev = solver.laplace_v()
#0时间层的 Adams-Bashforth 公式逼近的对流导数
AD_xu_0 = solution_u_values0 * gradux0 + Tuv0 * graduy0
BD_yv_0 = solution_v_values0 * gradvy0 + Tvu0 * gradvx0

#tau时间层的值
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
#u网格上的算子矩阵
gradux1 = solver.grad_ux() @ solution_u_values1
graduy1 = solver.grad_uy() @ solution_u_values1
Tuv1 = solver.Tuv() @ solution_v_values1
#v网格上的算子矩阵
gradvx1 = solver.grad_vx() @ solution_v_values1
gradvy1 = solver.grad_vy() @ solution_v_values1
Tvu1 = solver.Tvu() @ solution_u_values1
#tau时间层的 Adams-Bashforth 公式逼近的对流导数
AD_xu_1 = solution_u_values1 * gradux1 + Tuv1 * graduy1
BD_yv_1 = solution_v_values1 * gradvy1 + Tvu1 * gradvx1

#组装A、b矩阵
I = np.zeros_like(laplaceu.toarray())
row1, col1 = np.diag_indices_from(I)
I[row1,col1] = 1
A = I - (nu*tau*laplaceu)/2
F = solver.source_Fx(pde,t=tau)
Fx = F[:,0]
b = tau*(-3/2*AD_xu_1-1/2*AD_xu_0+nu/2*(laplaceu@solution_u_values1)+Fx-solver.grand_uxp()@solution_p_values1)

#组装B、c矩阵
E = np.zeros_like(laplacev.toarray())
row2, col2 = np.diag_indices_from(E)
E[row2,col2] = 1
B = E - (nu*tau*laplacev)/2
Fy = F[:,1]
c = tau*(-3/2*BD_yv_1-1/2*BD_yv_0+nu/2*(laplacev@solution_v_values1)+Fy-solver.grand_vyp()@solution_p_values1)

#A,b矩阵边界处理并解方程
nxu = mesh_u.node.shape[1]
is_boundaryu = np.zeros(num_nodes_u,dtype='bool')
is_boundaryu[:nxu] = True
is_boundaryu[-nxu:] = True
dirchiletu = pde.dirichlet_u(nodes_u[is_boundaryu], 0)
b[is_boundaryu] = dirchiletu

bdIdxu = np.zeros(A.shape[0], dtype=np.int_)
bdIdxu[is_boundaryu] = 1
Tbdu = spdiags(bdIdxu, 0, A.shape[0], A.shape[0])
T1 = spdiags(1-bdIdxu, 0, A.shape[0], A.shape[0])
A = A@T1 + Tbdu

u_1 = spsolve(A, b)

#B,c矩阵边界处理并解方程
nyv = mesh_v.node.shape[1]
is_boundaryv = np.zeros(num_nodes_v,dtype='bool')
is_boundaryv[(np.arange(num_nodes_v) % nyv == 0)] = True
indices = np.where(is_boundaryv)[0] - 1
is_boundaryv[indices] = True
dirchiletv = pde.dirichlet_v(nodes_v[is_boundaryv], 0)
c[is_boundaryv] = dirchiletv

bdIdyv = np.zeros(B.shape[0],dtype=np.int_)
bdIdyv[is_boundaryv] = 1
Tbdv = spdiags(bdIdyv,0,B.shape[0],B.shape[0])
T2 = spdiags(1-bdIdyv,0,B.shape[0],B.shape[0])
B = B@T2 + Tbdv

v_1 = spsolve(B,c)

#求解修正项
