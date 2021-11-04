---
title: 半线性界面问题
permalink: /docs/zh/example/num-nonlinear-interface
key: docs-num-nonlinear-interface-zh
author: zjk
---

本文基于一个半线性非齐次界面问题所做的一个编程说明文档。原方程如下：
$$
-\nabla\left(a\nabla u\right)+bu^3 = f
$$
系数满足如下条件：
$$
a=\left\{\begin{aligned}a_{1},&\quad \text{in} \quad\Omega_{1}\\
a_{2},&\quad \text{in} \quad\Omega_{2}
\end{aligned}\right.
$$

$$
b=\left\{\begin{aligned}1,&\quad \text{in} \quad\Omega_{1}\\
0,&\quad \text{in} \quad\Omega_{2}
\end{aligned}\right.
$$

满足如下边界条件：
$$
u = g_D, \quad\text{on }\quad\Gamma_D\leftarrow \text{\bf Dirichlet }
$$

$$
a\frac{\partial u}{\partial\bf n}  = g_N, \quad\text{on }\quad\Gamma_{N} \leftarrow \text{\bf Neumann}
$$

$$
a\frac{\partial u}{\partial\bf n} + \kappa u = g_R, \quad\text{on }\quad\Gamma_R \leftarrow \text{\bf Robin}
$$



满足如下齐次界面条件：
$$
[u]=u_{1}-u_{2}=0,\quad \text{on}\quad \Gamma\\
[a\frac{\partial u}{\partial\bf n}] = a_{1}\frac{\partial u_{1}}{\partial \bf n_{1}}+a_{2}\frac{\partial u_{2}}{\partial \bf n_{2}}=g_{I,1}+g_{I,2}=g_{I},\quad \text{on}\quad \Gamma
$$

## Newton-Galerkin方法

牛顿法是用来解决非线性问题的常用方法。以上面的方程为例，在方程两边同时乘上试探函数$v \in H_{D,0}^1(\Omega)$,利用分部积分法，可以得到**连续弱形式**
$$
(a_{1}\nabla u_{1},\nabla v)+(b_{1} u_{1}^3,v)+<\kappa u_{1},v>_{\Gamma_{R,1}} \\= (f_{1},v)+<g_{I,1},v>_{\Gamma}+<g_{N,1},v>_{\Gamma_{N,1}}+<g_{R,1},v>_{\Gamma_{R,1}},\quad\text{in}\quad\Omega_{1}
$$
$$
(a_{2}\nabla u_{2},\nabla v)+(b_{2} u_{2}^3,v)+<\kappa u_{2},v>_{\Gamma_{R,2}} \\= (f_{2},v)+<g_{I,2},v>_{\Gamma}+<g_{N,2},v>_{\Gamma_{N,2}}+<g_{R,2},v>_{\Gamma_{R,2}},\quad \text{in}\quad\Omega_{2}
$$

设 $u^0$ 是 $u$ 的一个逼近，$u = u^{0}+\delta u$ ,代入弱形式中
$$
(a_{1}\nabla (u_{1}^{0}+\delta u_{1},\nabla v)+(b_{1} (u_{1}^{0}+\delta u_{1})^3,v)+<\kappa (u_{1}^{0}+\delta u_{1}),v>_{\Gamma_{R,1}} \\= (f_{1},v)+<g_{I,1},v>_{\Gamma}+<g_{N,1},v>_{\Gamma_{N,1}}+<g_{R,1},v>_{\Gamma_{R,1}}
$$
$$
(a_{2}\nabla (u_{2}^{0}+\delta u_{2}),\nabla v)+(b_{2} (u_{2}^{0}+\delta u_{2})^3,v)+<\kappa (u_{2}^{0}+\delta u_{2}),v>_{\Gamma_{R,2}} \\= (f_{2},v)+<g_{I,2},v>_{\Gamma}+<g_{N,2},v>_{\Gamma_{N,2}}+<g_{R,2},v>_{\Gamma_{R,2}}
$$

其中 $b (u^{0}+\delta u)^3$ 在 $u^0$ 处泰勒展开，可得
$$
bu^{3}=b (u^{0}+\delta u)^3 = b(u^{0})^3+3b(u^{0})^{2}\delta u+O(\delta u)^{2}
$$
忽略高阶项 $O(\delta u)^{2}$，可得
$$
(a_{1}\nabla \delta u_{1},\nabla v_{1})+(3b_{1} (u_{1}^{0})^{2}\delta u_{1},v_{1})+<\kappa\delta u_{1},v_{1}>_{\Gamma_{R,1}}\\ = (f_{1},v_{1})+<g_{I,1},v>_{\Gamma}+<g_{N,1},v_{1}>_{\Gamma_{N,1}}+<g_{R,1},v_{1}>_{\Gamma_{R,1}}-(a_{1}\nabla u_{1}^{0},\nabla v_{1})-(b_{1} ( u_{1}^{0})^3,v_{1})-<\kappa u_{1}^{0},v_{1}>_{\Gamma_{R,1}}
$$
$$
(a_{2}\nabla \delta u_{2},\nabla v_{2})+(3b_{2} (u_{2}^{0})^{2}\delta u_{2},v_{2})+<\kappa\delta u_{2},v_{2}>_{\Gamma_{R,2}} \\= (f_{2},v_{2})+<g_{I,2},v>_{\Gamma}+<g_{N,2},v_{2}>_{\Gamma_{N,2}}+<g_{R,2},v_{2}>_{\Gamma_{R,2}}-(a_{2}\nabla u_{2}^{0},\nabla v_{2})-(b_{2} ( u_{2}^{0})^3,v_{2})-<\kappa u_{2}^{0},v_{2}>_{\Gamma_{R,2}}
$$

由此可得
$$
(a_{1}\nabla \delta u_{1},\nabla v_{1})+(a_{2}\nabla \delta u_{2},\nabla v_{2})+(3b_{1} (u_{1}^{0})^{2}\delta u_{1},v_{1})+(3b_{2} (u_{2}^{0})^{2}\delta u_{2},v_{2})\\+<\kappa\delta u_{1},v_{1}>_{\Gamma_{R,1}}+<\kappa\delta u_{2},v_{2}>_{\Gamma_{R,2}}
$$

$$
=(f_{1},v_{1})+(f_{2},v_{2})+<g_{N,1},v_{N,1}>_{\Gamma_{N,1}}+<g_{N,2},v_{N,2}>_{\Gamma_{N,2}}+<g_{I,1},v>_{\Gamma}+<g_{I,2},v>_{\Gamma}+\\<g_{R,1},v_{R,1}>_{\Gamma_{R,1}}+<g_{R,2},v_{R,2}>_{\Gamma_{R,2}}-(a_{1}\nabla u_{1}^{0},\nabla v_{1})-(a_{2}\nabla u_{2}^{0},\nabla v_{2})-(b_{1} ( u_{1}^{0})^3,v_{1})-(b_{2} ( u_{2}^{0})^3,v_{2})\\-<\kappa u_{1}^{0},v>_{\Gamma_{R,1}}-<\kappa u_{2}^{0},v>_{\Gamma_{R,2}}
$$

并且利用界面条件与$v_{I,1}=v_{I,2}$，由此可得
$$
(a\nabla \delta u,\nabla v)+(3b(u^{0})^{2}\delta u,v)+<\kappa \delta u,v>_{\Gamma_{R}}\\=(f,v)+<g_{I},v_{I}>_{\Gamma}+<g_{N},v_{N}>_{\Gamma_{N}}+<g_{R},v_{R}>_{\Gamma_{R}}-(a\nabla u^{0},\nabla v)-(b(u^{0})^{3},v)-<\kappa u^{0},v>_{\Gamma_{R}}
$$
其中
$$
a_0 = \begin{bmatrix}
a_{0} \\
\vdots \\
a_{0} \\
0\\
\vdots\\
0\\
\end{bmatrix},\quad
a_1 = \begin{bmatrix}
0\\
\vdots\\
0\\
a_{1}\\
\vdots\\
a_{1}\\
\end{bmatrix},\quad
$$


$$
b_0 =\begin{bmatrix}
b_{0} \\
\vdots \\
b_{0} \\
0\\
\vdots\\
0\\
\end{bmatrix},
\quad 
b_{0} = \begin{bmatrix}
0\\
\vdots\\
0\\
b_{1}\\
\vdots\\
b_{1}\\
\end{bmatrix},\quad 
$$

$$
f_0 = \begin{bmatrix}
f_{0} \\
\vdots \\
f_{0} \\
0\\
\vdots\\
0\\
\end{bmatrix},
\quad 
f_1 = \begin{bmatrix}
0\\
\vdots\\
0\\
f_{1}\\
\vdots\\
f_{1}\\
\end{bmatrix},\quad
$$



给定求解区域 $\Omega$ 上的一个剖分 $\mathcal{T}$ ，并构造相应的有限元空间 $V_{h}$ 

其**N**个向量的**全局基函数**组成的**行向量函数**记为
$$
\phi(x) = \left[\phi_0(x), \phi_1(x), \cdots, \phi_{N-1}(x)\right], x \in \Omega
$$
在实际的计算中，往往要在每个单元$\tau$使用**局部基函数**
$$
\varphi( x) = \left[\varphi_0( x), \varphi_1( x), \cdots, \varphi_{l-1}( x)\right],  x \in \tau
$$
所有基函数梯度对应的向量为
$$
\nabla \varphi( x) = \left[\nabla \varphi_0( x), \nabla \varphi_1( x), \cdots, \nabla \varphi_{l-1}(x)\right], x \in \tau
$$
其中
$$
\nabla \varphi_i = \begin{bmatrix}
\frac{\partial \varphi_i}{\partial x_0} \\
\frac{\partial \varphi_i}{\partial x_1} \\
\vdots \\
\frac{\partial \varphi_i}{\partial x_{d-1}} \\
\end{bmatrix},
\quad i= 0, 1, \cdots, l-1.
$$
则 $(a_{k}\nabla \delta u_{k},\nabla v)$ 对应的局部单元矩阵为
$$
A_\tau = \int_\tau a_{k}(\nabla \varphi)^T\nabla\varphi\mathrm d  x
$$
$(3b_{k} (u_{k}^{0})^{2}\delta u_{k},v)$ 对应的局部单元矩阵为
$$
B_\tau = \int_\tau 3b_{k}(u_{k}^{0})^{2} \varphi^T\varphi\mathrm d  x
$$
$(f_{k},v)$ 对应的局部单元矩阵为
$$
F = \int_\tau f_{k}\varphi^T\mathrm d  x
$$
$(a_{k}\nabla u_{k}^{0},\nabla v)$对应的局部单元矩阵为
$$
(A^{0})_\tau = \int_e a_{k}\nabla u_{k}^0\nabla\varphi^T\mathrm d  x, \forall e \subset \Gamma_{R}
$$
$(b_{k} ( u_{k}^{0})^3,v)$对应的局部单元矩阵为
$$
(B^{0})_\tau = \int_e b_{k} ( u_{k}^{0})^3\varphi^T\mathrm d  x, \forall e \subset \Gamma_{R}
$$

下面讨论边界条件相关的矩阵和向量组装。设网格边界边上的**局部基函数**个数为 $m$
个，其组成的**行向量函数**记为
$$
\omega ( x) = \left[\omega_0( x), \omega_1( x), \cdots, \omega_{m-1}( x)\right]
$$
设 $e$ 是一个局部边界边，则$<\kappa\delta u_{k},v>_{\Gamma_R}$ 对应的局部单元矩阵为
$$
R_\tau = \int_e \kappa\omega^T\omega\mathrm d  x, \forall e \subset \Gamma_{R}
$$
$<g_{I,k},v>_{\Gamma}$ 对应的局部单元矩阵为
$$
b_I = \int_e g_{I,k}\omega^T\mathrm d  x, \forall e \subset \Gamma
$$
$<g_{N,k},v>_{\Gamma_{N}}$对应的局部单元矩阵为
$$
b_N = \int_e g_{N,k}\omega^T\mathrm d  x, \forall e \subset \Gamma_{N}
$$
$<g_{R,k},v>_{\Gamma_R}$ 对应的局部单元矩阵为
$$
b_R = \int_e g_{R,k}\omega^T\mathrm d  x, \forall e \subset \Gamma_{R}
$$
$<\kappa u_{k}^{0},v>_{\Gamma_R}$对应的局部单元矩阵为
$$
(R^0)_{\tau} = \int_e \kappa u_{k}^{0}\omega^T\mathrm d  x, \forall e \subset \Gamma_{R}
$$

则原方程对应的矩阵形式为
$$
B =\begin{bmatrix}
B_{1}\quad 0\\
0\quad B_{2}
\end{bmatrix}
$$

$$
A=\begin{bmatrix}
A_{1} \quad 0\\
0 \quad A_{2}\\
\end{bmatrix}
$$

$$
R=\begin{bmatrix}
R_{1} \quad 0\\
0 \quad R_{2}\\
\end{bmatrix}
$$

$$
F=\begin{bmatrix}
F_{1}\\
F_{2}\\
\end{bmatrix}
$$

$$
A^{0}=\begin{bmatrix}
A_{1}^0\\
A_{2}^0\\
\end{bmatrix}
$$

$$
B^0=\begin{bmatrix}
B_{1}^0\\
B_{2}^0\\
\end{bmatrix}
$$

$$
R^0=\begin{bmatrix}
R_{1}^0\\
R_{2}^0\\
\end{bmatrix}
$$



## 基于fealpy的程序实现

设求解区域为 $\Omega=[0,1]\times[0,2]$ ,其中 $\Omega = \Omega_{0}+\Omega_{1}+\Gamma$,

$\Omega_{0}=[0,1]\times[0,1]$,

$\Omega_{1}=[0,1]\times[1,2]$,

界面为 $\Gamma = \{(x,y)|y=1\}$  ，

网格图

<img src="C:\Users\86188\Desktop\test3_mesh.png"  />

方程

$$
-\nabla\cdot\left(a\nabla u\right)+bu^3 = f
$$
系数
$$
a=\left\{\begin{aligned}10,&\quad \text{in} \quad\Omega_{0}\\
1,&\quad \text{in} \quad\Omega_{1}
\end{aligned}\right.
$$

$$
b=\left\{\begin{aligned}1,&\quad \text{in} \quad\Omega_{0}\\
0,&\quad \text{in} \quad\Omega_{1}
\end{aligned}\right.
$$



真解
$$
u_{0}=sin(\pi x)sin(\pi y)\\
u_{1}=-sin(\pi x)sin(\pi y)
$$
真解的梯度
$$
\nabla u_{0}=[\pi cos(\pi x)sin(\pi y), \pi sin(\pi x)cos(\pi y)]
$$

$$
\nabla u_{1}=[-\pi cos(\pi x)sin(\pi y), -\pi sin(\pi x)cos(\pi y)]
$$

界面处的法向量
$$
n_{0}=[0,1]\\
n_{1}=[0,-1]
$$

源项
$$
f_{0} = 20\pi^2 sin(\pi x)sin(\pi y)+(sin(\pi x)sin(\pi y))^3，\quad x\in \Omega_{0}\\
f_{1} = -2\pi^2 sin(\pi x)sin(\pi y), \quad x\in \Omega_{1}
$$
其中界面条件
$$
[u]=0
$$

$$
[a\frac{\partial u}{\partial\bf n}]=a_{0}\frac{\partial u_{0}}{\partial n_{0}}+a_{1} \frac{\partial u_{1}}{\partial n_{1}}=g_{I}
$$

$$
\begin{aligned}=&10[\pi cos(\pi x)sin(\pi y), \pi sin(\pi x)cos(\pi y)]|_{y=1}\cdot[0,1]\\&+1[-\pi cos(\pi x)sin(\pi y), -\pi sin(\pi x)cos(\pi y)]|_{y=1}\cdot[0,-1]\\
=&-11\pi sin(\pi x)
\end{aligned}
$$

边界条件为 Dirichlet 条件，将原方程化简为
$$
(a\nabla \delta u,\nabla v)+(3b(u^{0})^2\delta u,v)\\=(f,v)-<g_{I},v>_{\Gamma}-(a\nabla u^{0},\nabla v)-(b(u^{0})^3,v)
$$

对应的矩阵形式为
$$
(A+B)\delta u=F-A^{0}-B^{0}
$$



## 代码

首先给出模型数据代码

```python
import numpy as np
from fealpy.decorator import cartesian, barycentric
class LineInterfaceData():
    def __init__(self, a0=10, a1=1, b0=1, b1=0):
        self.a0 = a0
        self.a1 = a1
        self.b0 = b0
        self.b1 = b1
        self.interface = Line()
        
    #根据不同的区域确定方程的系数项 
    def mesh(self):
        node = np.array([
            (0,0),
            (1,0),
            (0,1),
            (1,1),
            (0,2),
            (1,2)],dtype=np.float64)
        cell = np.array([
            (1,3,0),
            (2,0,3),
            (3,5,2),
            (4,2,5)],dtype=np.int_)
        return Interface2dMesh(self.interface, node, cell)
    
    #每个子区域对应的单元全局编号的布尔值
    @cartesian
    def subdomain(self, p):
        sdflag = [self.interface(p) < 0, self.interface(p) > 0]
        #sdflag是一个二元组，其中的每一项都是一个(NC,)的数组,
        #sdflag[0]表示的是\Omega_0处的单元，sdflag[1]表示的是\Omega_!处的单元
        return sdflag
    
    #刚度矩阵的系数
    @cartesian
    def A_coefficient(self, p):
        #p(NC, GD)
        flag = self.subdomain(p) 
        A_coe = np.zeros(p.shape[:,-1], dtype = np.float64)
        A_coe[flag[0]] = self.a0
        A_coe[flag[1]] = self.a1
        #A_coe(NC,)
        return A_coe
    
    #质量矩阵的系数
    @cartesian
    def B_coefficient(self, p):
        #p(NC, GD)
        flag = self.subdomain(p)
        B_coe = np.zeros(p.shape[:,-1], dtype = np.float64)
        B_coe[flag[0]] = self.b0
        B_coe[flag[1]] = self.b1
        #B_coe(NC,)
        return B_coe
    
    #真解
    @cartesian
    def solution(self, p):
        flag = self.subdomain(p)
        #u0 = sin(pi*x)*sin(pi*y)
        #u1 = -sin(pi*x)*sin(pi*y)
        sol = np.zeros(p.shape[:,-1], dtype = np.float64)
        x = p[..., 0]
        y = p[..., 1]
        sol[flag[0]] = np.sin(np.pi*x)*np.sin(pi*y)
        sol[flag[1]] = -np.sin(np.pi*x)*np.sin(pi*y)
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
        x = p[..., 0]
        y = p[..., 1]
        grad[flag[0], 0] = pi*np.cos(pi*x)*np.sin(pi*y)
        grad[flag[0], 1] = pi*np.sin(pi*x)*np.cos(pi*y)
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
        b = np.zeros(p.shape[:,-1], dtype = np.float64)
        b[flag[0]] = 2*a0*pi**2*sol[flag[0]]+b0*sol[flag[0]]**3
        b[flag[1]] = 2*a1*pi**2*sol[flag[1]]+b1*sol[flag[1]]**3
        #b(NC,)
        return b
    
    #边界条件
    @cartesian
    def neumann(self, p):
        #p(NE,GD)
        flag = self.subdomain(p)
        a0 = self.a0
        a1 = self.a1
        grad = self.gradient(p)
        n = self.normal(p)#n(NE,)
        #neu0 = a0*grad0*n0
        #neu1 = a1*grad1*n1
        #grad0 表示在(NE[flag[0]], GD)的数组, n0 表示在(NE[flag[0]], GD)的数组
        #grad1 表示在(NE[flag[1]], GD)的数组, n1 表示在(NE[flag[1]], GD)的数组
        neu = np.zeros(p.shape[:,-1], dtype = np.float64)
        neu[flag[0]] = a0*sum(grad[flag[0], :] * n[flag[0], :],axis = -1)
        neu[flag[1]] = a1*(sum(grad[flag[1], :] * n[flag[1], :],axis = -1)
        #neu(NE,)
        return neu
    
    @cartesian                       
    def dirichlet(self,p):
        return self.solution(p)
    
    #界面条件
    @cartesian
    def interface(self, p):
        #p(NIE,GD)
        a0 = self.a0
        a1 = self.a1
        pi = np.pi
        x = p[..., 0]
        y = p[..., 1]
        #gI = a0*grad[flag[0],:]*n[flag[0], :]+a1*grad[flag[1].:]*n[flag[1],:]
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
        
        gI = np.zeros(p.shape[:,-1], dtype = np.float64)
        gI = a0*sum(grad0*n0,axis=-1)+a1*sum(grad1*n1,axis=-1)
        #gI(NIE,)
        return gI
```

构造$\Omega = [0,1]\times[0,2]$上的三角形网格，以及在处理界面界面时可能会用到的方法。

```python
from fealpy.mesh import MeshFactory as MF
from fealpy.mesh.TriangleMesh import TriangleMesh
class Interface2dMesh(TriangleMesh):
    def __init__(self, interface):
        self.domain = [0, 1, 0, 2]
        super()._init_(node,cell)
        self.node = node
        self.cell = self.ds.cell
        self.edge = self.ds.edge
        self.interface =interface
        self.phi = self.interface(self.node)
        self.phiSign = msign(self.phi)
        
    
    def interface_edge_flag(self):
        node = self.node
        edge = self.edge
        interface = self.interface
        EdgeMidnode = 1/2*(node[edge[:,0],:]+node[edge[:,1],:])
        isInterfaceEdge = (interface(EdgeMidnode) == 0)
        return isInterfaceEdge
    
    def interface_edge_index(self):
        isInterfaceEdge = self.interface_edge_flag
        InterfaceEdgeIdx = np.nonzero(isInterfaceEdge)
        InterfaceEdgeIdx = InterfaceEdgeIdx[0]
        return InterfaceEdgeIdx
    
    def interface_edge(self):
        InterfaceEdgeIdx = self.interface_edge_index
        edge = self.edge
        InterfaceEdge = edge[InterfaceEdgeIdx]
        return InterfaceEdge
        
    def interface_node_flag(self):
        phiSign = self.phiSign
        isInterfaceNode = np.zeros(self.N, dtype=np.bool)
        isInterfaceNode = (phiSign == 0)
        return isInterfaceNode
    
    def interface_node_index(self):
        isInterfaceNode = self.interface_node_flag
        InterfaceNodeIdx = np.nonzero(isInterfaceNode)
        return InterfaceNodeIdx
    
    def interface_node(self):
        InterfaceNodeIdx = self.interface_node_index
        node = self.node
        InterfaceNode = node[isInterfaceNode]
        return InterfaceNode
    
```

构造`mesh`上的$p$次Lagrange有限元空间space，并定义一个该空间的函数`u0`,

```python
from fealpy.functionspace import LagrangeFiniteElementSpace

space = LagrangeFiniteElementSpace(mesh, p=1)
u0 = space.function()
du = space.function()
```

其中 $(3b(u^{0})^2\delta u,v)$ 对应的组装代码为

```python
def nonlinear_matrix(uh, b):
    space = uh.space
    mesh = space.mesh
    
    qf = mesh.integrator(q=2, etype='cell')
    bcs,ws = qf.get_quadrature_points_and_weights()
    cellmeasure = mesh.entity_measure('cell')
    pp = mesh.bc_to_point(bcs)
    cval = 3*b(pp)*uh(bcs)**2 #(NQ,NC)
    phii = space.basis(bcs)
    phij = space.basis(bcs)
    
    B = np.einsum('q, qci, qc, qcj, c->cij',ws,phii,cval,phij,cellmeasure)
    
    gdof = space.number_of_global_dofs()
    cell2dof = space.cell_to_dof()
    I = np.broadcast_to(cell2dof[:,:,None],shape=B.shape)
    J = np.broadcast_to(cell2dof[:,None,:],shape=B.shape)
    B = csr_matrix((B.flat,(I.flat,J.flat)),shape=(gdof,gdof))
    return B
```

$(b(u^{0})^3,v)$对应的组装代码

```python
def nonlinear_matrix_right(uh, b):
    space = uh.space
    mesh = space.mesh
    
    qf = mesh.integrator(q=2, etype='cell')
    bcs, ws = qf.get_quadrature_points_and_weights()
    cellmeasure = mesh.entity_measure('cell')
    pp = mesh.bc_to_point(bcs)
    cval = b(pp)*uh(bcs)**3
    phii = space.basis(bcs)
    phij = space.basis(bcs)
    
    bb = np.einsum('q, qci, qc, qcj, c->cij',ws,phii,cval,phij,cellmeasure)
    #bb(NC,ldof)
    
    gdof = space.number_of_global_dofs()
    cell2dof = space.cell_to_dof()
    B0 = np.zeros(gdof, dtype=np.float64)
    np.add.at(B0, cell2dof, bb)
    return B0
```

下面来处理边界条件和界面条件

```python
from fealpy.quadrature import FEMeshIntegralAlg
from fealpy.functionspace import LagrangeFiniteElementSpace

def Set_interface(gI, F):
    #gI是界面函数，F是右端项
    index = mesh.interface_edge_index() # 选取界面对应的边的编号#(NIE,)
    face2dof = mesh.face_to_dof()[index] # 选取界面边转化为点，face2dof(NIE,GD)
    measure = mesh.entity_measure('edge', index=index)

    qf = mesh.integrator(q=2, etype='edge') ##代表选择第q个积分公式，积分类型为线积分，qf是类  
    bcs, ws = qf.get_quadrature_points_and_weights() #(NQ) bcs是局部单元上的重心坐标，ws是数值积分对应的权    
    phi = space.face_basis(bcs) # (NQ,NIE,ldof)

    pp = mesh.interface_to_point(bcs, index=index)## 将重心坐标转化为笛卡尔坐标
    val = gI(pp) # (NQ, NIE, ...) NIE表示界面对应的边

    bb = np.einsum('q, qe, qei, e->ei', ws, val, phi, measure) # 求界面边对应的数值积分
    np.add.at(F, face2dof, bb) # 将线积分添加到右端项中

    return F
```

接下来是求解方程

```python
#主函数
p = 1
maxit = 4 
tol = 1e-8

pde = LineInterfaceData()
mesh = pde,mesh()

NDof = np.zeros(maxit, dtype=np.int_)
errorMatrix = np.zeros((2, maxit), dtype=np.float64)
errorType = ['$|| u - u_h||_{\Omega, 0}$', '$||\\nabla u - \\nabla u_h||_{\Omega, 0}$']

#网格迭代
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
    b = Set_interface(pde.gI, b, p=p)
    
    
    # 非线性迭代
    while True:
        A =space.stiff_matrix(c=pde.A_coefficient)
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

    errorMatrix[0, i] = space.integralalg.error(pde.solution, u0.value)
    errorMatrix[1, i] = space.integralalg.error(pde.gradient, u0.grad_value)
    if i < maxit-1:
        mesh.uniform_refine()
        
#收敛阶    
print(errorMatrix)
showmultirate(plt, 0, NDof, errorMatrix,  errorType, propsize=20)   
plt.show()

#真解
Node = mesh.entity("node")
uI = pde.solution(Node)
uI = space.function(array=uI)
fig1 = plt.figure()
axes = fig1.gca(projection='3d')
uI.add_plot(axes, cmap='rainbow')
plt.show()

#数值解
fig2 = plt.figure()
axes = fig2.gca(projection='3d')
u0.add_plot(axes, cmap='rainbow')
plt.show()

```

真解

![](C:\Users\86188\Desktop\test3_truesol.png)

![](C:\Users\86188\Desktop\test3_truesol1.png)

数值解

![](C:\Users\86188\Desktop\test3_numsol.png)

![](C:\Users\86188\Desktop\test3_numsol1.png)

收敛阶

![](C:\Users\86188\Desktop\test3_1.png)
