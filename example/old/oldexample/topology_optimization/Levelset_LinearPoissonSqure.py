import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve

from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.mesh import MeshFactory as MF
from fealpy.boundarycondition import DirichletBC 
from fealpy.decorator import cartesian, barycentric

'''
水平集方程求解方形区域下由 PDEs 控制的形状优化问题
状态方程：
-\Delta u = f on D
        u = 0 on \partial D
能量泛函：
J(u) = \int_D (|\nabla u|^2 - fu) dx
体积约束：
|\Omega| = \gamma
'''

lag = 1.0 # Lagrange 乘子
TT = 0.0005 # 重新初始化的时间步长
eps1 = 0.001 # 向量计算时避免除 0 的小参数
eps2 = 0.01 # 速度的正则化参数
output = './result/' # 将所有的输出结果保存在名为 result 的文件夹中

volTarg = 0.5 # 目标体积
lagrangesetp = 0.8 # Lagrange 乘子下降步长

plotNum = int(10)
N = int(100) # 梯度算法的迭代次数
NiterLSE = int(10) # 状态方程的求解次数
ReInitFreq = int(5) # 控制初始化频率的参数
Ninit = int(10) # 重新初始化的迭代次数

### 绘制网格(米字型矩形网格) ###
mesh = MF.special_boxmesh2d([0, 1, 0, 1], n=80, meshtype='rice')
fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
plt.show()

### 初始的水平集函数 ###
@cartesian
def Phi0(p):
    x = p[..., 0]
    y = p[..., 1]
    val = 0.2 - (np.abs(x-0.5)-0.5)*(np.abs(y-0.5)-0.5)
    return val

### 定义 u=0 的 Dirichlet 边界条件 ###
@cartesian
def Dirichlet(p):
    return 0

### 定义有限元空间 ###
space = LagrangeFiniteElementSpace(mesh, p=1) # 定义线性有限元空间

phi = space.interpolation(Phi0) # 当前水平集函数

h = max(mesh.entity_measure('edge'))
print("最大网格尺寸:", h)

SS = space.stiff_matrix()
MM = space.mass_matrix()

gdof = space.number_of_global_dofs()
bc = DirichletBC(space, Dirichlet) 

qf = mesh.integrator(7)
bcs, ws = qf.get_quadrature_points_and_weights()
cellmeasure = mesh.entity_measure('cell')

### 特征线方法求解对流方程 ###
def convect(v, dt, fun):
    gradfun = space.grad_recovery(fun)
    result = fun[:] + dt*(v[0]*gradfun[:, 0] + v[1]*gradfun[:, 1])
    return result

nabla = space.function()
S = space.function() 
M1 = space.function()
M2 = space.function()
phiinit = space.function()

for i in range(N): 
    print("迭代第", i, "步")
    ### 水平集方程重新初始化 ###
    if i%ReInitFreq == 0: # 第 0 步也会初始化
        for j in range(Ninit):
            gradphi = space.grad_recovery(phi)

            nabla[:] = gradphi[:, 0]**2 + gradphi[:, 1]**2
            S[:] = phi[:]/np.sqrt(phi[:]**2 + h*h*nabla[:])
            M1[:] = gradphi[:, 0]/np.sqrt(nabla[:] + eps1**2)
            M2[:] = gradphi[:, 1]/np.sqrt(nabla[:] + eps1**2)
            phiinit[:] = convect([-S[:]*M1[:], -S[:]*M2[:]], TT, phi) 
            phiinit[:] += TT*S[:]
            phi[:] = phiinit[:]

    # 计算法向
    gradphi = space.grad_recovery(phi)
    nabla[:] = gradphi[:,0]**2 + gradphi[:,1]**2

    N1 = space.function()
    N2 = space.function()

    N1[:] = gradphi[:,0]/np.sqrt(nabla[:] + eps1**2)
    N2[:] = gradphi[:,1]/np.sqrt(nabla[:] + eps1**2)

    X = space.function()
    X[phi<0] = 1

    Dirac = space.function()
    Dirac[np.abs(phi) < 0.01] = 1

    # 解状态方程
    u = space.function()
    A0 = SS
    b0 = space.source_vector(X)    

    A0, b0 = bc.apply(A0, b0, u)
    x0 = spsolve(A0, b0)
    u[:] = x0
    print("u_max", np.max(u[:]), "---u_min", np.min(u[:]))

    # 绘制 shape
    if i%plotNum == 0:
        fname = output + 'levelset_'+ str(i+1).zfill(10) + '.vtu'
        mesh.nodedata['X'] = X
        mesh.nodedata['U'] = u
        mesh.nodedata['Phi'] = phi
        mesh.to_vtk(fname=fname)

    # 计算目标函数
    compliance = -0.5 * np.einsum('i, ij..., ij..., j-> ', ws, X(bcs), u(bcs), cellmeasure)
    objective = compliance
    print("objective", objective)

    # 计算体积误差
    volume = space.integralalg.integral(X)
    volErr = np.abs(volume - volTarg)
    print("体积误差:", volErr)

    # 解速度场光滑化方程
    @barycentric
    def V(bcs):
        val = X(bcs) - u(bcs) - lag*X(bcs)
        return val
    
    Vreg = space.function()
    A1 = eps2*SS + MM
    b1 = space.source_vector(V)

    Vreg[:] = spsolve(A1, b1)
    print("V_max", np.max(Vreg[:]), "---V_min", np.min(Vreg[:]))
    Vmax = np.max(np.abs(Vreg[:]))

    V = space.function()
    V[:] = Vreg[:]

    T = 0.05*h / Vmax
    if(T < 0.0001):
        T = 0.0001
    print("下降步长 T=", T)
    
    for j in range(NiterLSE):
        phi[:] = convect([-V[:]*N1[:], -V[:]*N2[:]], T, phi) 

    # Lagrange 乘子更新
    perimeter = np.einsum('i, ij..., ij..., j-> ', ws, Dirac(bcs), nabla(bcs)**0.5, cellmeasure)
    temp = np.einsum('i, ij..., j-> ', ws, -Dirac(bcs)*nabla(bcs)**0.5*u(bcs), cellmeasure)
    lag = 0.5*lag - 0.5*temp/perimeter + lagrangesetp*(volume - volTarg)/volTarg
