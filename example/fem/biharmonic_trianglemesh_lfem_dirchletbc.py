#!/usr/bin/env python3
# 

import argparse
import ipdb
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from scipy.sparse.linalg import spsolve, cg, lgmres
from scipy.sparse import csr_matrix, spdiags, eye, bmat

from fealpy import logger
from fealpy.mesh import TriangleMesh 
from fealpy.mesh.halfedge_mesh import HalfEdgeMesh2d
from fealpy.functionspace import InteriorPenaltyBernsteinFESpace2d

from fealpy.fem import ScalarBiharmonicIntegrator
from fealpy.fem import ScalarInteriorPenaltyIntegrator
from fealpy.fem import BilinearForm

from fealpy.fem import ScalarSourceIntegrator
from fealpy.fem import LinearForm

from pde import DoubleLaplacePDE 

class SinSinData:

    def domain(self):
        return [0, 1, 0, 1]

    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi  = np.pi
        sin = np.sin
        val = sin(2*pi*x)**2*sin(2*pi*y)**2
        return val 

    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi  = np.pi
        sin = np.sin
        cos = np.cos
        val0 = 4*pi*sin(2*pi*x)*sin(2*pi*y)**2*cos(2*pi*x)
        val1 = 4*pi*sin(2*pi*x)**2*sin(2*pi*y)*cos(2*pi*y)
        return np.concatenate([val0[..., None], val1[..., None]], axis=-1) 

    def hessian(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi  = np.pi
        sin = np.sin
        cos = np.cos
        val0 = -8*pi**2*sin(2*pi*x)**2*sin(2*pi*y)**2 + 8*pi**2*sin(2*pi*y)**2*cos(2*pi*x)**2
        val1 = 16*pi**2*sin(2*pi*x)*sin(2*pi*y)*cos(2*pi*x)*cos(2*pi*y)
        val2 = -8*pi**2*sin(2*pi*x)**2*sin(2*pi*y)**2 + 8*pi**2*sin(2*pi*x)**2*cos(2*pi*y)**2
        return np.concatenate([val0[..., None], val1[..., None], val1[...,
            None], val2[..., None]], axis=-1).reshape(p.shape+(2, 2))

    def dirichlet(self, p):
        """ Dilichlet boundary condition
        """
        return self.solution(p)
    
    def neuman(self, p, n):
        """ Neuman boundary condition
        """
        val = self.gradient(p)
        return np.sum(val*n, axis=-1)

    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi  = np.pi
        sin = np.sin
        cos = np.cos
        val = 64*pi**4*(3*sin(2*pi*x)**2*sin(2*pi*y)**2 - 3*sin(2*pi*x)**2*cos(2*pi*y)**2 - sin(2*pi*y)**2*cos(2*pi*x)**2 + cos(2*pi*x)**2*cos(2*pi*y)**2) + 64*pi**4*(3*sin(2*pi*x)**2*sin(2*pi*y)**2 - sin(2*pi*x)**2*cos(2*pi*y)**2 - 3*sin(2*pi*y)**2*cos(2*pi*x)**2 + cos(2*pi*x)**2*cos(2*pi*y)**2)
        return val

    def is_boundary_dof(self, p):
        eps = 1e-14 
        return (p[...,0] < eps) | (p[...,1] < eps) | (p[..., 0] > 1.0 - eps) | (p[..., 1] > 1.0 - eps)

def is_boundary_dof(p):
    eps = 1e-14 
    return (p[...,0] < eps) | (p[...,1] < eps) | (p[..., 0] > 1.0 - eps) | (p[..., 1] > 1.0 - eps)


def apply_dbc(A, f, uh, isDDof):
    f = f - A@uh.reshape(-1)
    bdIdx = np.zeros(A.shape[0], dtype=np.int_)
    bdIdx[isDDof.reshape(-1)] = 1
    D0 = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
    D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
    A = D0@A@D0 + D1
    
    f[isDDof.reshape(-1)] = uh[isDDof].reshape(-1)
    return A, f

def node_g_h_value(uh, nidx):
    cell = mesh.entity('cell')
    NC = cell.shape[0]
    cidx = np.where(np.any(cell == nidx, axis=1))[0]
    k = np.argmax(cell[cidx] == nidx, axis=1)

    bc = np.identity(3)
    guh = uh.grad_value(bc)
    huh = uh.hessian_value(bc)

    gx0 = np.mean(guh[k, cidx, :], axis=0)
    hx0 = np.mean(huh[k, cidx, :], axis=0)
    return gx0, hx0

def find_nodes(nodes):
    x_min, x_max = 0.25, 0.75
    y_min, y_max = 0.25, 0.75
    condition = (nodes[:, 0] >= x_min) & (nodes[:, 0] <= x_max) & (nodes[:, 1] >= y_min) & (nodes[:, 1] <= y_max)
    node0 = nodes[condition]
    return node0


## 参数解析
parser = argparse.ArgumentParser(description=
        """
        TriangleMesh 上任意次内罚有限元方法求解双调和方程
        """)

parser.add_argument('--degree',
        default=2, type=int,
        help='Bernstein 有限元空间的次数, 默认为 2 次.')

parser.add_argument('--nx',
        default=4, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--ny',
        default=4, type=int,
        help='初始网格剖分段数.')

parser.add_argument('--maxit',
        default=4, type=int,
        help='默认网格加密求解的次数, 默认加密求解 4 次')

parser.add_argument('--gamma',
        default=5, type=int,
        help='默认内罚参数，默认为3')

args = parser.parse_args()

p = args.degree
nx = args.nx
ny = args.ny
maxit = args.maxit
gamma = args.gamma

print("P : ", p)

x = sp.symbols("x")
y = sp.symbols("y")
u = (sp.sin(sp.pi*y)*sp.sin(sp.pi*x))**2
pde = DoubleLaplacePDE(u)

vertice = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float_)
mesh  = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=nx, ny=ny)
#mesh  = TriangleMesh.from_polygon_gmsh(vertice, 0.5)
space = InteriorPenaltyBernsteinFESpace2d(mesh, p = p)

errorType = ['$|| Ax-b ||_{\\Omega,0}$']
errorMatrix = np.zeros((3, maxit), dtype=np.float64)
error = np.zeros((2, maxit), dtype=np.float64)
Rerror = np.zeros((1, maxit), dtype=np.float64)

NDof = np.zeros(maxit, dtype=np.int_)

node = mesh.entity('node')
node0 = find_nodes(node)

# 插值点
points = mesh.interpolation_points(p=p)
points0 = find_nodes(points)

GD = mesh.geo_dimension()
Rhu = np.zeros((maxit, len(points), GD, GD), dtype=np.float_)

gx0 = np.zeros((len(points0), GD), dtype=np.float_)
hx0 = np.zeros((len(points0), GD, GD), dtype=np.float_)

h = np.zeros(maxit, dtype=np.float64)
for i in range(maxit):
    h[i] = np.sqrt(np.max(mesh.cell_area()))

    bform = BilinearForm(space)
    L = ScalarBiharmonicIntegrator()

    bform.add_domain_integrator(L)
    A0 = bform.assembly()
    
    P0 = ScalarInteriorPenaltyIntegrator(gamma=gamma)
    P  = P0.assembly_face_matrix(space)  
    A  = A0 + P
    
    lform = LinearForm(space)
    F = ScalarSourceIntegrator(pde.source, q=p+4)
    lform.add_domain_integrator(F)
    b = lform.assembly()
    
    x = pde.solution(mesh.interpolation_points(p=p))
    
    Bd = is_boundary_dof(mesh.interpolation_points(p=p))
    gd = np.zeros_like(x)
    gd[Bd] = x[Bd]
    
    A, f = apply_dbc(A, b, gd, is_boundary_dof(mesh.interpolation_points(p=p)))

    uh = space.function()
    uh[:] = spsolve(A, f)
    #uh[:], tol = cg(A, f, atol=1e-15)
    #print("AAAA : ", np.max(A0.data))
    #print("AAA : ", np.max(A.data))
    #print("AAA : ", np.max(P.data))

#    errorMatrix[0, i] = np.max(np.abs(uh-x))
    errorMatrix[0, i] = mesh.error(uh, pde.solution)
    errorMatrix[1, i] = mesh.error(uh.grad_value, pde.gradient) 
    errorMatrix[2, i] = mesh.error(uh.hessian_value, pde.hessian) 
    print(errorMatrix)
    
    # 计算某点处的误差情况
    node = mesh.entity('node')
    points = mesh.interpolation_points(p=p)
    mesh0 = HalfEdgeMesh2d.from_mesh(mesh, NV=3)
#    nidx = np.where(np.isin(node[:, 0], node0[:, 0]) & np.isin(node[:, 1], node0[:, 1]))[0]
    
    print(points0.shape)
    scell, pbc = mesh0.find_point_in_triangle_mesh(points0)
    print('dddd:', scell.shape, pbc)

    x = pde.solution(points0)
    gx = pde.gradient(points0)
    hx = pde.hessian(points0)
     
    for j in range(len(points0)):
        if np.all(pbc[j, :] >1e-5):
            guh[j] = uh.grad_value(pbc[j])
            huh[j] = uh.hessian_value(pbc[j])
            print('ddddddd:', j)
        elif np.sum(pbc[j, :] >1e-10) == 1:
            nidx = np.where(np.isin(points[:, 0], points0[j, 0]) &
                    np.isin(points[:, 1], points0[j, 1]))[0]
            gx0[j], hx0[j] = node_g_h_value(uh, nidx)
        elif np.sum(pbc[j, :] > 1e-10) == 2:
            e2c = mesh0.ds.edge_to_cell()
            if pbc[j, 0] < 1e-10:
                if (scell[j] == e2c[:, 0]):
                    pass
            print('scell:, ', scell[j])
            print('cbc:', j, pbc[j])
            print(e2c)
            ee2c = mesh.ds.edge_to_cell()
            print(ee2c)

    
    error[0, i] = np.max(np.abs(gx0-gx))
    error[1, i] = np.max(np.abs(hx0-hx))
    
    if p > 2:
        k = 2**p/2.0
        Rhu[i] -= hx0/(k-1)
        if i > 0:
            Rhu[i-1] += k*hx0/(k-1)
            Rerror[0, i-1] = np.max(np.abs(Rhu[i-1]-hx))
    else:
        Rerror[0, i] = error[1, i]
    
#    node = mesh.entity('node')
#    cell = mesh.entity('cell')
#    cell = np.array(cell)

#    NN = len(node)
#    node0 = np.array(node)
#    node0 = np.c_[node0, x[:NN, None]]
#    node1 = np.array(node)
#    node1 = np.c_[node1, uh[:NN, None]]
#    meshv0 = TriangleMesh(node0, cell)
#    meshv1 = TriangleMesh(node1, cell)
#    meshv0.to_vtk(fname='aaa.vtu')
#    meshv1.to_vtk(fname='bbb.vtu')

    if i < maxit-1:
        nx = nx*2
        ny = ny*2
        mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=nx, ny=ny)
        #mesh  = TriangleMesh.from_polygon_gmsh(vertice, 0.5/2**i)
        space = InteriorPenaltyBernsteinFESpace2d(mesh, p = p)
        

print('error:', errorMatrix)
print('d_error:', errorMatrix[:, 0:-1]/errorMatrix[:, 1:])

def compute_order(errors, h):
    orders = np.zeros_like(errors)
    for i in range(orders.shape[-1]-1):
        if np.any(errors[:, i] == 0) or np.any(errors[:, i+1] == 0):
            orders[:, i+1] = 0
        else:
            orders[:, i+1] = np.log(errors[:, i] / errors[:, i+1]) / np.log(h[i]/h[i+1])
    return orders

#order = np.zeros((2, maxit-1), dtype=np.float64) 
order = compute_order(errorMatrix, h)
print('order:', order)

print('x0_error:', error)
x0order = compute_order(error, h)
print('x0order:', x0order)
    
print('Rx0_error:', Rerror)
Rx0order = compute_order(Rerror, h)
print('Rx0order:', Rx0order)
    

