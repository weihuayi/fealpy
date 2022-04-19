import argparse
import sys
import numpy as np
import matplotlib

from scipy.sparse import spdiags, bmat
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix,hstack,vstack,spdiags,bmat
from mumps import DMumpsContext

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
        default=10, type=int,
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
mesh = pde.mesh(h)
tmesh = UniformTimeLine(0,T,nt)
dt = tmesh.dt

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

##矩阵组装准备
#组装第一个方程的左端矩阵
H = mesh.cell_phi_phi_matrix(udegree, udegree)
H = mesh.construct_matrix(udegree, udegree, H)
S = mesh.cell_stiff_matrix(udegree, udegree)
S = mesh.construct_matrix(udegree, udegree, S)

C1 = mesh.cell_gphix_phi_matrix(udegree, pdegree)
C2 = mesh.cell_gphiy_phi_matrix(udegree, pdegree)
C1 = mesh.construct_matrix(udegree, pdegree, C1)
C2 = mesh.construct_matrix(udegree, pdegree, C2)


##边界处理
xx = np.zeros(2*ugdof, np.float64)

u_isbddof_u0 = uspace.is_boundary_dof()
u_isbddof_in = uspace.is_boundary_dof(threshold = pde.is_inflow_boundary)
u_isbddof_out = uspace.is_boundary_dof(threshold = pde.is_outflow_boundary)

u_isbddof_u0[u_isbddof_in] = False 
xx[0:ugdof][u_isbddof_u0] = 0
xx[ugdof:2*ugdof][u_isbddof_u0] = 0

u_isbddof = u_isbddof_u0
u_isbddof[u_isbddof_in] = True
ipoint = uspace.interpolation_points()[u_isbddof_in]

ipoint = uspace.interpolation_points()[u_isbddof_in]
uinfow = pde.u_inflow_dirichlet(ipoint)
xx[0:ugdof][u_isbddof_in] = uinfow[:,0]
xx[ugdof:2*ugdof][u_isbddof_in] = uinfow[:,1]

isBdDof = np.hstack([u_isbddof, u_isbddof])
bdIdx = np.zeros(2*ugdof, dtype=np.int_)
bdIdx[isBdDof] = 1
Tbd = spdiags(bdIdx, 0, 2*ugdof, 2*ugdof)
T = spdiags(1-bdIdx, 0, 2*ugdof, 2*ugdof)
A = T@A + Tbd

#组装第二个方程的左端矩阵
B1 = mesh.cell_stiff_matrix(pdegree, pdegree)
B1 = mesh.construct_matrix(pdegree, pdegree, B1)

ispBDof = pspace.boundary_dof(threshold=pde.is_outflow_boundary)
bdIdx = np.zeros((B1.shape[0],), np.int_)
bdIdx[ispBDof] = 1
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

    ##p边界
    ep = p0(epbcs)[...,index]
    value = np.einsum('ij,jk->ijk',ep,n)
    ephi = uspace.face_basis(epbcs)
    evalue = np.einsum('i,ijk,ijm,j->jkm',epws,ephi,value,emeasure)
    fb5 = np.zeros((ugdof,2))
    np.add.at(fb5,(face2dof,np.s_[:]),evalue)
    
    fb6 = D@u0.flatten(order='F') 

    b1 = (rho/dt*fb1 - rho*fb2-dt*fb5).flatten(order='F')
    b1 = b1 + fb4 - mu*fb3 + mu/2*fb6
     
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
    
    ispBDof = pspace.is_boundary_dof(threshold=pde.is_outflow_boundary)
    b2[ispBDof] = 0

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
