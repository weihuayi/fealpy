# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 16:11:29 2021

@author: 86188
"""


# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 02:15:13 2021

@author: 86188
"""



import numpy as np
import matplotlib.pyplot as plt

from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.quadrature import FEMeshIntegralAlg
from scipy.sparse import csr_matrix
from fealpy.mesh.interface_mesh_generator import msign
from scipy.sparse.linalg import spsolve
from fealpy.tools.show import showmultirate
from fealpy.mesh import MeshFactory as MF
from fealpy.mesh.TriangleMesh import TriangleMesh,TriangleMeshDataStructure
from fealpy.decorator import cartesian, barycentric

class LineInterfaceData():
    def __init__(self, a0=10, a1=1, b0=1, b1=0):
        self.a0 = a0
        self.a1 = a1
        self.b0 = b0
        self.b1 = b1
        self.interface = Line()
        
    def mesh(self):
        node = np.array([
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (0, 2),
            (1, 2)], dtype=np.float64)
        cell = np.array([
                 (1, 3, 0),
                 (2, 0, 3),
                 (3, 5, 2), 
                 (4, 2, 5)], dtype=np.int_)
        return Interface2dMesh(self.interface, node, cell)
        
    #根据不同的区域确定方程的系数项  
    @cartesian
    def subdomain(self, p):
        sdflag = [self.interface(p) < 0, self.interface(p) > 0]
        return sdflag
    
    @cartesian
    def A_coefficient(self, p):
        #p(NC, GD)
        flag = self.subdomain(p)  #flag(2,)其中每个元素都是(NC,)的数组
        A_coe = np.zeros(p.shape[:-1], dtype = np.float64)
        A_coe[flag[0]] = self.a0
        A_coe[flag[1]] = self.a1
        #A_coe(NC,)
        return A_coe
    
    @cartesian
    def B_coefficient(self, p):
        #p(NC, GD)
        flag = self.subdomain(p)
        B_coe = np.zeros(p.shape[:-1], dtype = np.float64)
        B_coe[flag[0]] = self.b0
        B_coe[flag[1]] = self.b1
        #B_coe(NC,)
        return B_coe
    
    #真解
    @cartesian
    def solution(self, p):
        flag = self.subdomain(p)
        pi = np.pi
        #u0 = sin(pi*x)*sin(pi*y)
        #u1 = -sin(pi*x)*sin(pi*y)
        sol = np.zeros(p.shape[:-1], dtype = np.float64)
        x = p[..., flag[0], 0]
        y = p[..., flag[0], 1]
        sol[flag[0]] = np.sin(pi*x)*np.sin(pi*y)
        x = p[..., flag[1], 0]
        y = p[..., flag[1], 1]
        sol[flag[1]] = -np.sin(pi*x)*np.sin(pi*y)
        #sol(NC,)
        return sol
    
    #真解的梯度
    @cartesian
    def gradient(self, p):
        flag = self.subdomain(p)
        #u0x = pi*cos(pi*x)*sin(pi*y)
        #u1y = pi*sin(pi*x)*cos(pi*y)
        #u0x = -pi*cos(pi*x)*sin(pi*y)
        #u1y = -pi*sin(pi*x)*cos(pi*y)
        pi = np.pi
        grad = np.zeros(p.shape, dtype = np.float64)
        x = p[...,flag[0], 0]
        y = p[...,flag[0], 1]
        grad[flag[0], 0] = pi*np.cos(pi*x)*np.sin(pi*y)
        grad[flag[0], 1] = pi*np.sin(pi*x)*np.cos(pi*y)
        x = p[...,flag[1], 0]
        y = p[...,flag[1], 1]
        grad[flag[1], 0] = -pi*np.cos(pi*x)*np.sin(pi*y)
        grad[flag[1], 1] = -pi*np.sin(pi*x)*np.cos(pi*y)
        #grad(NC,GD)
        return grad
    
    #源项
    @cartesian
    def source(self, p):
        flag = self.subdomain(p)
        pi = np.pi
        a0 = self.a0
        a1 = self.a1
        b0 = self.b0
        b1 = self.b1
        
        sol = self.solution(p)
        #f0 = 2*a0*pi^2*u0 + b0*u0^3
        #f1 = 2*a1*pi^2*u1 + b1*u1^3
        b = np.zeros(p.shape[:-1], dtype = np.float64)
        b[flag[0]] = 2*a0*pi**2*sol[flag[0]]+b0*sol[flag[0]]**3
        b[flag[1]] = 2*a1*pi**2*sol[flag[1]]+b1*sol[flag[1]]**3
        #b(NC,)
        return b
    
    #边界条件
    @cartesian
    def neumann(self, p,n):
        #p(NE,GD)
        flag = self.subdomain(p)
        a0 = self.a0
        a1 = self.a1
        grad = self.gradient(p)
        #n = self.normal(p)#每一个点对应一个边，边对应一个法向量n(NE,)
        
        #neu0 = a0*grad0*n0
        #neu1 = a1*grad1*n1
        #grad0 表示在(NE[flag[0]], GD)的数组, n0 表示在(NE[flag[0]], GD)的数组
        #grad1 表示在(NE[flag[1]], GD)的数组, n1 表示在(NE[flag[1]], GD)的数组
        neu = np.zeros(p.shape[:-1], dtype = np.float64)
        n = np.broadcast_to(n, grad.shape)
        neu[flag[0]] = a0*np.sum(grad[flag[0], :] * n[flag[0], :],axis = -1)
        neu[flag[1]] = a1*np.sum(grad[flag[1], :] * n[flag[1], :],axis = -1)
        #neu(NE,)
        return neu
    
    @cartesian
    def dirichlet(self, p):
        return self.solution(p)
    
    
    #界面条件
    @cartesian
    def interfaceFun(self, p):
        #p(NIE,GD)
        a0 = self.a0
        a1 = self.a1
        pi = np.pi
        x = p[..., 0]
        y = p[..., 1]
        #gI = a0*grad[flag[0],:]*n[flag[0], :]+a1*grad[flag[1].:]*n[flag[1],:]
        #为方便，将方程中界面函数 gI 前面的负号移动到界面函数中
        grad0 = np.zeros(p.shape, dtype = np.float64)
        grad1 = np.zeros(p.shape, dtype = np.float64)
        grad0[..., 0] = pi*np.cos(pi*x)*np.sin(pi*y)
        grad0[..., 1] = pi*np.sin(pi*x)*np.cos(pi*y)
        grad1[..., 0] = -pi*np.cos(pi*x)*np.sin(pi*y)
        grad1[..., 1] = -pi*np.sin(pi*x)*np.cos(pi*y)
        n0 = np.zeros(p.shape, dtype = np.float64)
        n1 = np.zeros(p.shape, dtype = np.float64)
        n0[..., 0] = 0
        n0[..., 1] = 1
        n1[..., 0] = 0
        n1[..., 1] = -1
        
        gI = np.zeros(p.shape[:-1], dtype = np.float64)
        gI = a0*np.sum(grad0*n0,axis=-1)+a1*np.sum(grad1*n1,axis=-1)
        return gI

class Line():
    def __init__(self):
        self.box = [0, 1, 0, 2]

    @cartesian
    def __call__(self, pp):
        return pp[..., 1] - 1
    
    @cartesian
    def value(self, pp):
        return self(pp)
    

class Interface2dMesh(TriangleMesh):   
    
    def __init__(self, interface, node, cell): 
        super().__init__(node, cell)
        self.node = node
        self.cell = self.ds.cell
        self.domain = [0, 1, 0, 2]
        self.edge = self.ds.edge
        self.interface = interface
        self.phi = self.interface(self.node)
        self.phiSign = msign(self.phi)
    
    def interface_edge_flag(self):
        node = self.node
        edge = self.ds.edge
        interface = self.interface
        EdgeMidnode = 1/2*(node[edge[:,0],:]+node[edge[:,1],:])
        isInterfaceEdge = (interface(EdgeMidnode) == 0)
        return isInterfaceEdge
    
    def interface_edge_index(self):
        isInterfaceEdge = self.interface_edge_flag()
        InterfaceEdgeIdx = np.nonzero(isInterfaceEdge)
        InterfaceEdgeIdx = InterfaceEdgeIdx[0]
        return InterfaceEdgeIdx
    
    def interface_edge(self):
        isInterfaceEdge = self.interface_edge_flag()
        edge = self.ds.edge
        InterfaceEdge = edge[isInterfaceEdge]
        return InterfaceEdge
    
    def interface_node_flag(self):
        phiSign = self.phiSign
        isInterfaceNode = np.zeros(self.N, dtype=np.bool_)
        isInterfaceNode = (phiSign == 0)
        #另外一种想法
        #N = self.N
        #isInterfaceNode = np.zeros(N, dtype=np.float64)
        #isInterfaceNode[edge[isInterfaceEdge,:]] = True
        return isInterfaceNode
    
    def interface_node_index(self):
        isInterfaceNode = self.interface_node_flag()
        InterfaceNodeIdx = np.nonzero(isInterfaceNode)
        return InterfaceNodeIdx
    
    def interface_node(self):
        isInterfaceNode = self.interface_node_flag()
        node = self.node
        InterfaceNode = node[isInterfaceNode]
        return InterfaceNode
    

def nonlinear_matrix(uh, b):
    space = uh.space
    mesh = space.mesh
    
    qf = mesh.integrator(q=2, etype='cell')
    bcs,ws = qf.get_quadrature_points_and_weights()
    cellmeasure = mesh.entity_measure('cell')
    pp = mesh.bc_to_point(bcs)
    cval = 3*b(pp)*uh(bcs)**2#(NQ, NC)
    phii = space.basis(bcs)
    phij = space.basis(bcs)
    
    B = np.einsum('q, qci, qc, qcj, c->cij',ws,phii,cval,phij,cellmeasure)
    
    gdof = space.number_of_global_dofs()
    cell2dof = space.cell_to_dof()
    I = np.broadcast_to(cell2dof[:,:,None],shape=B.shape)
    J = np.broadcast_to(cell2dof[:,None,:],shape=B.shape)
    B = csr_matrix((B.flat,(I.flat,J.flat)),shape=(gdof,gdof))
    return B

def nonlinear_matrix_right(uh, b):
    space = uh.space
    mesh = space.mesh
    
    qf = mesh.integrator(q=2, etype='cell')
    bcs, ws = qf.get_quadrature_points_and_weights()
    cellmeasure = mesh.entity_measure('cell')
    cell2dof = space.cell_to_dof()
    pp = mesh.bc_to_point(bcs)
    cval = b(pp)*uh(bcs)**3
    phi = space.basis(bcs)
    bb = np.einsum('q, qc, qcj, c->cj',ws,cval,phi,cellmeasure)
    #bb(NC,ldof)
    
    gdof = space.number_of_global_dofs()
    cell2dof = space.cell_to_dof()
    B0 = np.zeros(gdof, dtype=np.float64)
    np.add.at(B0, cell2dof ,bb)
    return B0
    


def Set_interface(gI, F, p):
    index = mesh.interface_edge_index()
    face2dof = space.face_to_dof()[index]  
    measure = mesh.entity_measure('edge', index=index)
    
    qf = mesh.integrator(q=2, etype='edge')
    bcs, ws = qf.get_quadrature_points_and_weights()    
    phi = space.face_basis(bcs)

    pp = mesh.bc_to_point(bcs, index=index)
    val = gI(pp) # (NQ, NIE)，NIE表示界面对应的边
    
    bb = np.einsum('q, qe, qei, e->ei', ws, val, phi, measure)
    np.add.at(F, face2dof, bb)

    return F

##### 主函数 #####
p = 1
maxit = 4 
tol = 1e-8

pde = LineInterfaceData()
mesh = pde.mesh()

index = mesh.interface_edge_index()
print(index)
fig3 = plt.figure()
axes = fig3.gca()
mesh.add_plot(axes)
mesh.find_node(axes, showindex=True)
mesh.find_edge(axes, showindex=True)
mesh.find_cell(axes, showindex=True)
mesh.find_edge(axes, color='r',multiindex=index)
plt.show()

NDof = np.zeros(maxit, dtype=np.int_)
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
errorType = ['$|| u - u_h||_{\Omega, 0}$', '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$']

##### 网格迭代 #####
for i in range(maxit):
    print(i, ":")
    space = LagrangeFiniteElementSpace(mesh, p=p)
    NDof[i] = space.number_of_global_dofs()
    
    u0 = space.function()   
    du = space.function() 
    
    isDDof = space.set_dirichlet_bc(pde.dirichlet, u0)
    isIDof = ~isDDof

    b = space.source_vector(pde.source)
    #b = space.set_neumann_bc(pde.neumann, b)  
    b = Set_interface(pde.interfaceFun, b, p=p)   
    
    ##### 非线性迭代 #####
    while True:
        A = space.stiff_matrix(c=pde.A_coefficient)
        B = nonlinear_matrix(u0,b=pde.B_coefficient)
        B0 = nonlinear_matrix_right(u0, b=pde.B_coefficient)
        U = A + B
        F = b - A@u0 -B0
        du[isIDof] = spsolve(U[isIDof, :][:, isIDof], F[isIDof]).reshape(-1)
        #du[:] = spsolve(U, F).reshape(-1)
        u0 += du
        err = np.max(np.abs(du))
        print(err)
        if err < tol:
            break
     
    ##### 误差估计 #####
    errorMatrix[0, i] = space.integralalg.error(pde.solution, u0.value)
    errorMatrix[1, i] = space.integralalg.error(pde.gradient, u0.grad_value)
    if i < maxit-1:
        mesh.uniform_refine()
    
print(errorMatrix)
showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)   
plt.show()

#数值解
# fig2 = plt.figure()
# axes = fig2.gca(projection='3d')
# u0.add_plot(axes, cmap='rainbow')
# plt.show()

#网格
# fig3 = plt.figure()
# axes = fig3.gca()
# mesh.add_plot(axes)
# mesh.find_node(axes, showindex=True)
# mesh.find_edge(axes, showindex=True)
# mesh.find_cell(axes, showindex=True)
# plt.show()

#真解
# Node = mesh.entity("node")
# uI = pde.solution(Node)
# uI = space.function(array=uI)
# fig1 = plt.figure()
# axes = fig1.gca(projection='3d')
# uI.add_plot(axes, cmap='rainbow')
# plt.show()


