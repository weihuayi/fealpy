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
from fealpy.functionspace import LagrangeFESpace

from fealpy.fem import ScalarMassIntegrator
from fealpy.fem import ScalarSourceIntegrator         # (f, v)
from fealpy.fem import ScalarBiharmonicIntegrator
from fealpy.fem import ScalarInteriorPenaltyIntegrator
from fealpy.fem import BilinearForm

from fealpy.fem import ScalarSourceIntegrator
from fealpy.fem import LinearForm

from pde import DoubleLaplacePDE 
from scipy.optimize import minimize

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
    return np.where(condition)[0] 

def grad_max_error(uh, pde, nidx):
    """
    @brief: 计算 nidx 的节点上梯度值的最大值误差
    """
    node = mesh.entity('node')
#    print(nidx)
    
    x = pde.solution(node[nidx])
    gx = pde.gradient(node[nidx])
    hx = pde.hessian(node[nidx])
    gx0 = np.zeros((len(nidx), GD), dtype=np.float_)
    hx0 = np.zeros((len(nidx), GD, GD), dtype=np.float_)
    
    for j in range(len(nidx)):
        gx0[j], hx0[j] = node_g_h_value(uh, nidx[j])

    error0 = np.max(np.abs(gx0-gx))
    error1 = np.max(np.abs(hx0-hx))
    return error0, error1 

# 计算两点之间的欧几里得距离
def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# 找到半径为 d 的区域内的所有点
def points_in_radius(node, center, d):
    Bindices = []
    Bpoints = []
    for m, n in enumerate(node):
        if distance(n, center) <= d:
            Bindices.append(m)
            Bpoints.append(n)
    return Bindices, Bpoints


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
        default=6, type=int,
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

q = 11

x = sp.symbols("x")
y = sp.symbols("y")
#u = (sp.sin(sp.pi*y)*sp.sin(sp.pi*x))/(4*sp.pi**4)
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
ratio = np.zeros(maxit, dtype=np.float64)

NDof = np.zeros(maxit, dtype=np.int_)

node = mesh.entity('node')
#nidx = find_nodes(node)

GD = mesh.geo_dimension()
#Rhu = np.zeros((maxit, len(nidx), GD, GD), dtype=np.float_)
#gx0 = np.zeros((len(nidx), GD), dtype=np.float_)
#hx0 = np.zeros((len(nidx), GD, GD), dtype=np.float_)

h = np.zeros(maxit, dtype=np.float64)
for i in range(maxit):
    h[i] = np.sqrt(np.max(mesh.cell_area()))

    bform = BilinearForm(space)
    L = ScalarBiharmonicIntegrator(q=q)

    bform.add_domain_integrator(L)
    A0 = bform.assembly()
    
    P0 = ScalarInteriorPenaltyIntegrator(gamma=gamma, q=q)
    P  = P0.assembly_face_matrix(space)  
    A  = A0 + P
    
    lform = LinearForm(space)
    F = ScalarSourceIntegrator(pde.source, q=q)
    lform.add_domain_integrator(F)
    b = lform.assembly()
    
    x = pde.solution(mesh.interpolation_points(p=p))
    
    Bd = is_boundary_dof(mesh.interpolation_points(p=p))
    gd = np.zeros_like(x)
    gd[Bd] = x[Bd]
    
    A, f = apply_dbc(A, b, gd, is_boundary_dof(mesh.interpolation_points(p=p)))

    uh = space.function()
    uh[:] = spsolve(A, f)
#    uh[:] = lgmres(A, f, atol=1e-18)[0]

    errorMatrix[0, i] = mesh.error(uh, pde.solution)
    errorMatrix[1, i] = mesh.error(uh.grad_value, pde.gradient) 
    errorMatrix[2, i] = mesh.error(uh.hessian_value, pde.hessian) 
    print(errorMatrix)
    
    # 计算某点处的误差情况
    error[:, i] = grad_max_error(uh, pde, np.arange(node.shape[0]))
    print(error)
    
    lspace = LagrangeFESpace(mesh, p=p)

    lbform = BilinearForm(lspace)
    lbform.add_domain_integrator(ScalarMassIntegrator(q=p+2)) 
    M = bform.assembly()

    llform = LinearForm(lspace)
    # (f, v)
    si = ScalarSourceIntegrator(uh, q=p+2)
    llform.add_domain_integrator(si)
    F = lform.assembly()
    Lx = lspace.function()

    Lx[:] = spsolve(M, F)

    # 找到每个点在半径 d 内的节点
    for a in range(1, 5):
        d = 3 * a * h[i]
        Lerror1 = np.linalg.norm(Lx)/d**2
        if d > 0.8:
            break
        node0 = np.array((0.5, 0.5))
        Bidx, Bnode = points_in_radius(node, node0, d)

        def f(V):
            uI = space.function()
            uI[:] = V
            gerror = grad_max_error(uI, pde, Bidx)[0]
#            print('error:', gerror)
            return gerror

        gdof = space.dof.number_of_global_dofs()
        uI = space.function()
        uI[:] = minimize(f, uh[:]).x
        Berror = np.max(uI)
        print('Berror', Berror)
        if a == 1:
            ratio[i] = error[0,
                    i]/(Berror+Lerror1)/np.sqrt(np.abs(np.log(h[i])))**3
        else:
            ratio[i] = max(ratio[i], error[0, i]/(Berror+Lerror1)/np.sqrt(np.abs(np.log(h[i])))**3)

#    def ff(x):
#        return x**2

    print(ratio)

    if i < maxit-1:
        nx = nx*2
        ny = ny*2
        mesh.uniform_refine()
        space = InteriorPenaltyBernsteinFESpace2d(mesh, p = p)
        
print(ratio)
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
    

