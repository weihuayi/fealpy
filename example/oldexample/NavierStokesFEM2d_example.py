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

parser.add_argument('--method',
        default='Netwon', type=str,
        help='非线性化方法')

args = parser.parse_args()
udegree = args.udegree
pdegree = args.pdegree
nt = args.nt
T = args.T
h = args.h
output = args.output
step = args.step
method = args.method

# 网格,空间,函数
rho = 1
mu=0.001
udim = 2
pde = PDE()
mesh = pde.mesh1(h)
tmesh = UniformTimeLine(0,T,nt)
dt = tmesh.dt

mesh.grad_lambda()
uspace = LagrangeFiniteElementSpace(mesh,p=udegree)
pspace = LagrangeFiniteElementSpace(mesh,p=pdegree)

u0 = uspace.function(dim=udim)
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
M = mesh.cell_phi_phi_matrix(udegree, udegree)
M = mesh.construct_matrix(udegree, udegree, M)
S = mesh.cell_stiff_matrix(udegree, udegree)
S = mesh.construct_matrix(udegree, udegree, S)

C1 = mesh.cell_gphix_phi_matrix(udegree, pdegree)
C2 = mesh.cell_gphiy_phi_matrix(udegree, pdegree)
C1 = mesh.construct_matrix(udegree, pdegree, C1)
C2 = mesh.construct_matrix(udegree, pdegree, C2)

#边界处理
xx = np.zeros(gdof, np.float64)

is_u_bdof = uspace.is_boundary_dof()
is_uin_bdof = uspace.is_boundary_dof(threshold = pde.is_inflow_boundary)
is_uout_bdof = uspace.is_boundary_dof(threshold = pde.is_outflow_boundary)
is_p_bdof = pspace.boundary_dof(threshold=pde.is_outflow_boundary)

is_u_bdof[is_uout_bdof] = False 

ipoint = uspace.interpolation_points()[is_uin_bdof]
uinfow = pde.u_inflow_dirichlet(ipoint)
xx[0:ugdof][is_uin_bdof] = uinfow[:,0]
xx[ugdof:2*ugdof][is_uin_bdof] = uinfow[:,1]

isBdDof = np.hstack([is_u_bdof,is_u_bdof,is_p_bdof])
bdIdx = np.zeros(gdof, dtype=np.int_)
bdIdx[isBdDof] = 1
Tbd = spdiags(bdIdx, 0, gdof, gdof)
T = spdiags(1-bdIdx, 0, gdof, gdof)


ctx = DMumpsContext()
ctx.set_silent()
errorMatrix = np.zeros((4,nt),dtype=np.float64)

for i in range(0, nt):
    # 下一个的时间层 t1
    t1 = tmesh.next_time_level()
    print("t1=", t1)
    
    #组装左端矩阵
    cu0x = u0[..., 0]
    cu0y = u0[..., 1]

    D1 = mesh.cell_phi_gphix_phi_matrix(udegree, udegree, udegree, c3=cu0x)
    D2 = mesh.cell_phi_gphiy_phi_matrix(udegree, udegree, udegree, c3=cu0y)
    D1 = mesh.construct_matrix(udegree, udegree, D1)
    D2 = mesh.construct_matrix(udegree, udegree, D2)
    
    E1 = mesh.cell_phi_gphix_phi_matrix(udegree, udegree, udegree, c2=cu0x)
    E2 = mesh.cell_phi_gphiy_phi_matrix(udegree, udegree, udegree, c2=cu0x)
    E3 = mesh.cell_phi_gphix_phi_matrix(udegree, udegree, udegree, c2=cu0y)
    E4 = mesh.cell_phi_gphiy_phi_matrix(udegree, udegree, udegree, c2=cu0y)   
    E1 = mesh.construct_matrix(udegree, udegree, E1)
    E2 = mesh.construct_matrix(udegree, udegree, E2)
    E3 = mesh.construct_matrix(udegree, udegree, E3)
    E4 = mesh.construct_matrix(udegree, udegree, E4)
    
    if method == 'Netwon' :
        A = bmat([[1/dt*M + mu*S+D1+D2+E1, E2, -C1],\
                [E3, 1/dt*M + mu*S +D1+D2+E4, -C2],\
                [C1.T, C2.T, None]], format='csr')
    elif method == 'Ossen':
        A = bmat([[1/dt*M + mu*S+D1+D2, None, -C1],\
                [None, 1/dt*M + mu*S +D1+D2, -C2],\
                [C1.T, C2.T, None]], format='csr')
    elif method == 'Eular':
        A = bmat([[1/dt*M + mu*S, None, -C1],\
                [None, 1/dt*M + mu*S, -C2],\
                [C1.T, C2.T, None]], format='csr')
    #组装右端向量
    
    fb2xxx = mesh.cell_phi_gphix_phi_matrix(udegree, udegree, udegree, cu0x, cu0x)
    fb2yxy = mesh.cell_phi_gphiy_phi_matrix(udegree, udegree, udegree, cu0x, cu0y)
    fb2xyx = mesh.cell_phi_gphix_phi_matrix(udegree, udegree, udegree, cu0y, cu0x)
    fb2yyy = mesh.cell_phi_gphiy_phi_matrix(udegree, udegree, udegree, cu0y, cu0y)
    fb2xxx = mesh.construct_vector(udegree, fb2xxx)
    fb2yxy = mesh.construct_vector(udegree, fb2yxy)
    fb2xyx = mesh.construct_vector(udegree, fb2xyx)
    fb2yyy = mesh.construct_vector(udegree, fb2yyy)
    fb2 = np.hstack((fb2xxx+fb2yxy,fb2xyx+fb2yyy)).T

    fb1 = np.hstack((M@u0[:,0],M@u0[:,1])).T
     
    if method == 'Netwon' :
        b = 1/dt*fb1 + fb2
        b = np.hstack((b,[0]*pgdof))
    elif method == 'Ossen':
        b = 1/dt*fb1
        b = np.hstack((b,[0]*pgdof))
    elif method == 'Eular':
        b =  1/dt*fb1 - fb2
        b = np.hstack((b,[0]*pgdof))
    
    ## 边界条件处理
    A = T@A + Tbd
    b[isBdDof] = xx[isBdDof]
    ctx.set_centralized_sparse(A)
    x = b.copy()
    ctx.set_rhs(x)
    ctx.run(job=6)
    
    u1[:,0] = x[0:ugdof]
    u1[:,1] = x[ugdof:2*ugdof]
    p1[:] = x[2*ugdof:]
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
