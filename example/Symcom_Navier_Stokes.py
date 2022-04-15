#!/usr/bin/python3
'''!    	
	@Author: wpx
	@File Name: ti-navier.py
	@Mail: wpx15673207315@gmail.com 
	@Created Time: 2022年04月13日 星期三 12时06分13秒
	@bref 
	@ref 
'''  
import argparse
import sys
import numpy as np
from mumps import DMumpsContext
import matplotlib.pyplot as plt
import taichi as ti
from scipy.sparse import bmat,csr_matrix,spdiags

from fealpy.ti import TriangleMesh 
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.mesh import MeshFactory as MF
from fealpy.geometry import DistDomain2d
from fealpy.geometry import huniform
from fealpy.functionspace import LagrangeFiniteElementSpace

from fealpy.decorator import cartesian,barycentric

from fealpy.pde.navier_stokes_equation_2d import FlowPastCylinder as PDE

ti.init()
# 参数设置
parser = argparse.ArgumentParser(description=
        """
        有限元方法求解绕柱流问题
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

args = parser.parse_args()
udegree = args.udegree
pdegree = args.pdegree
nt = args.nt
T = args.T
h = args.h
output = args.output
rho = 1
mu=0.001


# 网格
points = np.array([[0.0, 0.0], [2.2, 0.0], [2.2, 0.41], [0.0, 0.41]],
        dtype=np.float64)
facets = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int_)


p, f = MF.circle_interval_mesh([0.2, 0.2], 0.05, 0.01) 

points = np.append(points, p, axis=0)
facets = np.append(facets, f+4, axis=0)

fm = np.array([0, 1, 2, 3])

smesh = MF.meshpy2d(points, facets, h, hole_points=[[0.2, 0.2]], facet_markers=fm, meshtype='tri')

uspace = LagrangeFiniteElementSpace(smesh,p=udegree)
pspace = LagrangeFiniteElementSpace(smesh,p=pdegree)

u0 = uspace.function(dim=2)
us = uspace.function(dim=2)
u1 = uspace.function(dim=2)

p0 = pspace.function()
p1 = pspace.function()

node = smesh.entity('node')
cell = smesh.entity('cell')
mesh = TriangleMesh(node,cell)
tmesh = UniformTimeLine(0,T,nt)
'''
fig = plt.figure()
axes = fig.gca()
smesh.add_plot(axes)
plt.show()
'''

dt = tmesh.dt
pde = PDE()

ugdof = mesh.number_of_global_interpolation_points(udegree)
pgdof = mesh.number_of_global_interpolation_points(pdegree)
gdof = pgdof+2*ugdof

##矩阵组装准备
qf = smesh.integrator(4,'cell')
bcs,ws = qf.get_quadrature_points_and_weights()
cellmeasure = smesh.entity_measure('cell')

## 速度空间
uphi = uspace.basis(bcs)
ugphi = uspace.grad_basis(bcs)
ucell2dof = uspace.cell_to_dof()
NC = smesh.number_of_cells()

epqf = uspace.integralalg.edgeintegrator
epbcs,epws = epqf.get_quadrature_points_and_weights()

## 压力空间
pphi = pspace.basis(bcs)
pgphi = pspace.grad_basis(bcs)
pcell2dof = pspace.cell_to_dof()

index = smesh.ds.boundary_face_index()
ebc = smesh.entity_barycenter('face',index=index)
flag = pde.is_outflow_boundary(ebc)
index = index[flag]# p边界条件的index

epqf = uspace.integralalg.edgeintegrator
epbcs,epws = epqf.get_quadrature_points_and_weights()
emeasure = smesh.entity_measure('face',index=index)
face2dof = uspace.face_to_dof()[index]
n = smesh.face_unit_normal(index=index)
ucell2dof = uspace.cell_to_dof()
def edge_matrix(pfun,gfun,nfun): 
    n = nfun(index=index)

    edge2cell = smesh.ds.edge2cell[index]
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

    matrix = bmat([[D00,D10],[D01,D11]],format='csr') 
    return matrix

# 组装第一个方程左端矩阵
H =  mesh.construct_matrix(udegree,udegree,'mass')
H = bmat([[H,None],[None,H]],format='csr')
E00 = mesh.construct_matrix(udegree,udegree,'gpx_gpx')
E01 = mesh.construct_matrix(udegree,udegree,'gpx_gpy')
E10 = mesh.construct_matrix(udegree,udegree,'gpy_gpx')
E11 = mesh.construct_matrix(udegree,udegree,'gpy_gpy')
E = bmat([[E00+1/2*E11,1/2*E10],[1/2*E01,E11+1/2*E00]],format='csr')
D = edge_matrix(uspace.face_basis,uspace.edge_grad_basis,smesh.face_unit_normal)
A = rho/dt*H + mu*E -1/2*mu*D
##边界处理
xx = np.zeros(2*ugdof, np.float64)

u_isbddof_u0 = uspace.is_boundary_dof()
u_isbddof_in = uspace.is_boundary_dof(threshold = pde.is_inflow_boundary)
u_isbddof_out = uspace.is_boundary_dof(threshold = pde.is_outflow_boundary)

u_isbddof_u0[u_isbddof_in] = False 
u_isbddof_u0[u_isbddof_out] = False

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
# 组装第二个方程左端矩阵
B1 = mesh.construct_matrix(pdegree,pdegree,'stiff')
ispBDof = pspace.boundary_dof(threshold=pde.is_outflow_boundary)
bdIdx = np.zeros((B1.shape[0],),np.int_)
bdIdx[ispBDof] = 1
Tbd = spdiags(bdIdx,0,B1.shape[0],B1.shape[0])
T = spdiags(1-bdIdx,0,B1.shape[0],B1.shape[0])
B =  T@B1 + Tbd
# 组转第三个方程左端矩阵
C = H

ctx = DMumpsContext()
ctx.set_silent()
errorMatrix = np.zeros((4,nt),dtype=np.float64)

for i in range(0,nt):
    # 下一个的时间层 t1
    t1 = tmesh.next_time_level()
    print("t1=", t1)

    #组装第一个方程的右端向量
    cu0 = np.array(u0[:,0])
    cu1 = np.array(u0[:,1])
    fb10 = mesh.source_mass_vector(udegree,udegree,cu0)
    fb11 = mesh.source_mass_vector(udegree,udegree,cu1)
    fb1 = np.array([fb10,fb11]).T

    fuu = u0(bcs)
    fgu = u0.grad_value(bcs)
    fbb2 = np.einsum('i,ijn,ijk,ijmk,j -> jnm',ws,uphi,fuu,fgu,cellmeasure)
    fb2 = np.zeros((ugdof,2))
    np.add.at(fb2,(ucell2dof,np.s_[:]),fbb2)
    
    fb3 = E@u0.flatten(order='F')
     
    cp0 = np.array(p0)
    fb4 =mesh.source_gphix_phi_vector(2,1,cp0)
    fb5 =mesh.source_gphiy_phi_vector(2,1,cp0)
    fb4 = np.hstack((fb4,fb5)) 
    
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
    
    b21 = B1@p0
    cusx = np.array(us[:,0])
    cusy = np.array(us[:,1])
    b220 = mesh.source_gphixx_phi_vector(2,1,cusx)
    b221 = mesh.source_gphiyy_phi_vector(2,1,cusy)
    b22 = b220+b221
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
    gp = p1.grad_value(bcs)-p0.grad_value(bcs)
    tbb2 = np.einsum('i,ijk,ijm,j -> jkm',ws,uphi,gp,cellmeasure)
    tb2 = np.zeros((ugdof,2))
    np.add.at(tb2,(ucell2dof,np.s_[:]),tbb2)
    b3 = tb1 - dt*(tb2.flatten(order='F')) 
    
    ctx.set_centralized_sparse(C)
    x = b3.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)
    u1[:,0] = x[0:ugdof]
    u1[:,1] = x[ugdof:]
    
    fname = output + 'test_'+ str(i+1).zfill(10) + '.vtu'
    smesh.nodedata['velocity'] = u1
    smesh.nodedata['pressure'] = p1
    smesh.to_vtk(fname=fname) 
    
    u0[:] = u1 
    p0[:] = p1
    # 时间步进一层 
    tmesh.advance()

ctx.destroy()

