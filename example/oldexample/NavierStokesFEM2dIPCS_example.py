import argparse
import sys
import numpy as np
import matplotlib

from scipy.sparse import spdiags, bmat
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix,hstack,vstack,spdiags,bmat
from mumps import DMumpsContext

import matplotlib.pyplot as plt
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.timeintegratoralg import UniformTimeLine

from fealpy.pde.navier_stokes_equation_2d import FlowPastCylinder as PDE

# 参数设置
parser = argparse.ArgumentParser(description=
        """
        有限元方法求解NS方程
        """)

parser.add_argument('--udegree',
        default=2, type=int,
        help='运动有限元空间的次数, 默认为 2 次.')

parser.add_argument('--pdegree',
        default=1, type=int,
        help='压力有限元空间的次数, 默认为 1 次.')

parser.add_argument('--h',
        default=0.01, type=float,
        help='单元尺寸')

parser.add_argument('--nt',
        default=5000, type=int,
        help='时间剖分段数，默认剖分 5000 段.')

parser.add_argument('--T',
        default=5, type=float,
        help='演化终止时间, 默认为 5')

parser.add_argument('--output',
        default='./', type=str,
        help='结果输出目录, 默认为 ./')

parser.add_argument('--step',
        default=20, type=int,
        help='隔多少步输出一次')

args = parser.parse_args()
udegree = args.udegree
pdegree = args.pdegree
nt = args.nt
T = args.T
h = args.h
output = args.output
step = args.step


# 网格,空间,函数
rho = 1
mu=0.001
udim = 2
pde = PDE()
#mesh = pde.mesh()
mesh = pde.mesh1(h)
tmesh = UniformTimeLine(0,T,nt)
dt = tmesh.dt

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()

uspace = LagrangeFiniteElementSpace(mesh,p=udegree)
pspace = LagrangeFiniteElementSpace(mesh,p=pdegree)

u0 = uspace.function(dim=udim)
us = uspace.function(dim=udim)
u1 = uspace.function(dim=udim)

p0 = pspace.function()
p1 = pspace.function()

fname = output + 'test_'+ str(0).zfill(10) + '.vtu'
mesh.nodedata['velocity'] = u0 
mesh.nodedata['pressure'] = p1
mesh.to_vtk(fname=fname)

ugdof = uspace.number_of_global_dofs()
pgdof = pspace.number_of_global_dofs()
gdof = pgdof+2*ugdof
mesh.grad_lambda()

##矩阵组装准备
qf = mesh.integrator(9,'cell')
bcs,ws = qf.get_quadrature_points_and_weights()
cellmeasure = mesh.entity_measure('cell')

## 速度空间
uphi = uspace.basis(bcs)
ugphi = uspace.grad_basis(bcs)
ucell2dof = uspace.cell_to_dof()
NC = mesh.number_of_cells()

epqf = uspace.integralalg.edgeintegrator
epbcs,epws = epqf.get_quadrature_points_and_weights()

## 压力空间
pphi = pspace.basis(bcs)
pgphi = pspace.grad_basis(bcs)
pcell2dof = pspace.cell_to_dof()

index = mesh.ds.boundary_face_index()
ebc = mesh.entity_barycenter('face',index=index)
flag = pde.is_outflow_boundary(ebc)
index = index[flag]# p边界条件的index

emeasure = mesh.entity_measure('face',index=index)
face2dof = uspace.face_to_dof()[index]
n = mesh.face_unit_normal(index=index)

def edge_matrix(pfun, gfun, nfun): 
    n = nfun(index=index)

    edge2cell = mesh.ds.edge2cell[index]
    egphi = gfun(epbcs,edge2cell[:,0],edge2cell[:,2])
    ephi = pfun(epbcs)
    
    pgx0 = np.einsum('i,ijk,jim,j,j->jkm',epws,ephi,egphi[...,0],n[:,0],emeasure)
    pgy1 = np.einsum('i,ijk,jim,j,j->jkm',epws,ephi,egphi[...,1],n[:,1],emeasure)
    pgx1 = np.einsum('i,ijk,jim,j,j->jkm',epws,ephi,egphi[...,0],n[:,1],emeasure)
    pgy0 = np.einsum('i,ijk,jim,j,j->jkm',epws,ephi,egphi[...,1],n[:,0],emeasure)

    J1 = np.broadcast_to(face2dof[:,:,None],shape =pgx0.shape) 
    tag = edge2cell[:,0]
    I1 = np.broadcast_to(ucell2dof[tag][:,None,:],shape = pgx0.shape)
    
    D00 = csr_matrix((pgx0.flat,(J1.flat,I1.flat)),shape=(ugdof,ugdof))
    D11 = csr_matrix((pgy1.flat,(J1.flat,I1.flat)),shape=(ugdof,ugdof))
    D01 = csr_matrix((pgy0.flat,(J1.flat,I1.flat)),shape=(ugdof,ugdof))
    D10 = csr_matrix((pgx1.flat,(J1.flat,I1.flat)),shape=(ugdof,ugdof))

    matrix = vstack([hstack([D00,D10]),hstack([D01,D11])]) 
    return matrix

#组装第一个方程的左端矩阵
H = mesh.cell_phi_phi_matrix(udegree, udegree)
H = mesh.construct_matrix(udegree, udegree, H)
H = bmat([[H, None], [None, H]], format='csr')

E00 = mesh.cell_gphix_gphix_matrix(udegree, udegree)
E11 = mesh.cell_gphiy_gphiy_matrix(udegree, udegree)
E01 = mesh.cell_gphix_gphiy_matrix(udegree, udegree)
E10 = mesh.cell_gphiy_gphix_matrix(udegree, udegree)

E00 = mesh.construct_matrix(udegree, udegree, E00)
E11 = mesh.construct_matrix(udegree, udegree, E11)
E01 = mesh.construct_matrix(udegree, udegree, E01)
E10 = mesh.construct_matrix(udegree, udegree, E10)
E = bmat([[E00+1/2*E11, 1/2*E10], [1/2*E01, E11+1/2*E00]], format='csr')

A = rho/dt*H + mu*E 

##边界处理
xx = np.zeros(2*ugdof, np.float64)

is_u_bdof = uspace.is_boundary_dof()
is_uin_bdof = uspace.is_boundary_dof(threshold = pde.is_inflow_boundary)
is_uout_bdof = uspace.is_boundary_dof(threshold = pde.is_outflow_boundary)

is_u_bdof[is_uout_bdof] = False 

ipoint = uspace.interpolation_points()[is_uin_bdof]
uinfow = pde.u_inflow_dirichlet(ipoint)
xx[0:ugdof][is_uin_bdof] = uinfow[:,0]
xx[ugdof:2*ugdof][is_uin_bdof] = uinfow[:,1]

isBdDof = np.hstack([is_u_bdof,is_u_bdof])
bdIdx = np.zeros(2*ugdof, dtype=np.int_)
bdIdx[isBdDof] = 1
Tbd = spdiags(bdIdx, 0, 2*ugdof, 2*ugdof)
T = spdiags(1-bdIdx, 0, 2*ugdof, 2*ugdof)
A = T@A + Tbd

#组装第二个方程的左端矩阵
B1 = mesh.cell_stiff_matrix(pdegree, pdegree)
B1 = mesh.construct_matrix(pdegree, pdegree, B1)

is_p_bdof = pspace.boundary_dof(threshold=pde.is_outflow_boundary)
bdIdx = np.zeros((B1.shape[0],), np.int_)
bdIdx[is_p_bdof] = 1
Tbd = spdiags(bdIdx, 0, B1.shape[0], B1.shape[0])
T = spdiags(1-bdIdx, 0, B1.shape[0], B1.shape[0])
B =  T@B1 + Tbd

#组装第三个方程的左端矩阵
C = H

ctx = DMumpsContext()
ctx.set_silent()
errorMatrix = np.zeros((4,nt),dtype=np.float64)

for i in range(0, nt):
    # 下一个的时间层 t1
    t1 = tmesh.next_time_level()
    print("t1=", t1)

    #组装第一个方程的右端向量
    cu0x = np.array(u0[...,0])
    cu0y = np.array(u0[...,1])
    cp0 = np.array(p0[:]) 
    
    fb1x = mesh.cell_phi_phi_matrix(udegree, udegree, cu0x)
    fb1y = mesh.cell_phi_phi_matrix(udegree, udegree, cu0y)
    fb1x = mesh.construct_vector(udegree, fb1x)
    fb1y = mesh.construct_vector(udegree, fb1y)
    fb1 = np.array([fb1x,fb1y]).T

    fb2xxx = mesh.cell_phi_gphix_phi_matrix(udegree, udegree, udegree, cu0x, cu0x)
    fb2yxy = mesh.cell_phi_gphiy_phi_matrix(udegree, udegree, udegree, cu0x, cu0y)
    fb2xyx = mesh.cell_phi_gphix_phi_matrix(udegree, udegree, udegree, cu0y, cu0x)
    fb2yyy = mesh.cell_phi_gphiy_phi_matrix(udegree, udegree, udegree, cu0y, cu0y)
    fb2xxx = mesh.construct_vector(udegree, fb2xxx)
    fb2yxy = mesh.construct_vector(udegree, fb2yxy)
    fb2xyx = mesh.construct_vector(udegree, fb2xyx)
    fb2yyy = mesh.construct_vector(udegree, fb2yyy)
    fb2 = np.array([fb2xxx+fb2yxy,fb2xyx+fb2yyy]).T

    fb3 = E@u0.flatten(order='F')
    
    fb4x = mesh.cell_gphix_phi_matrix(udegree, pdegree, c2=cp0)    
    fb4y = mesh.cell_gphiy_phi_matrix(udegree, pdegree, c2=cp0)
    fb4x = mesh.construct_vector(udegree, fb4x)
    fb4y = mesh.construct_vector(udegree, fb4y)
    fb4 = np.hstack((fb4x, fb4y))

    b1 = (rho/dt*fb1 - rho*fb2).flatten(order='F')
    b1 = b1 + fb4 - mu*fb3 
     
    b1[isBdDof] = xx[isBdDof]
        
    ctx.set_centralized_sparse(A)
    x = b1.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)
    
    us[:,0] = x[0:ugdof]
    us[:,1] = x[ugdof:]
    #组装第二个方程的右端向量
    cusx = np.array(us[..., 0]) 
    cusy = np.array(us[..., 1]) 
    b21 = B1@p0
    b22x = mesh.cell_gphix_phi_matrix(udegree, pdegree, c1=cusx)
    b22y = mesh.cell_gphiy_phi_matrix(udegree, pdegree, c1=cusy)
    b22x = mesh.construct_vector(pdegree, b22x) 
    b22y = mesh.construct_vector(pdegree, b22y) 
    b22 = b22x + b22y
    b2 = b21 -1/dt*b22
    
    b2[is_p_bdof] = 0

    ctx.set_centralized_sparse(B)
    x = b2.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)
    p1[:] = x[:]

    #组装第三个方程的右端向量
    tb1 = C@us.flatten(order='F')
    
    cp1p0 = np.array(p1[:]-p0[:])
    tb2x = mesh.cell_gphix_phi_matrix(pdegree, udegree, c1=cp1p0)
    tb2y = mesh.cell_gphiy_phi_matrix(pdegree, udegree, c1=cp1p0)
    tb2x = mesh.construct_vector(udegree, tb2x)
    tb2y = mesh.construct_vector(udegree, tb2y)
    tb2 = np.array([tb2x, tb2y]).T
    
    b3 = tb1 - dt*(tb2.flatten(order='F')) 
    
    ctx.set_centralized_sparse(C)
    x = b3.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)
    u1[:,0] = x[0:ugdof]
    u1[:,1] = x[ugdof:]
    if i%step == 0:
        fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
        mesh.nodedata['velocity'] = u1
        mesh.nodedata['pressure'] = p1
        mesh.to_vtk(fname=fname) 
    
    u0[:] = u1 
    p0[:] = p1
    # 时间步进一层 
    tmesh.advance()

ctx.destroy()
