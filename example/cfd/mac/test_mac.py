import numpy as np
from scipy.sparse.linalg import spsolve
from fealpy.mesh.uniform_mesh_2d import UniformMesh2d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fealpy.cfd import NSMacSolver
from taylor_green_pde import taylor_greenData 
from scipy.sparse import spdiags
from scipy.sparse import diags
from fealpy.timeintegratoralg import UniformTimeLine

Re = 1
nu = 1/Re
T = 5
#PDE 模型 
pde = taylor_greenData(Re,T=[0,T])
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
nt = 128
tmesh = UniformTimeLine(0, T, nt)
tau = tmesh.dt

#网格点的位置
nodes_u = mesh_u.entity('node') #[20.2]
nodes_v = mesh_v.entity('node') #[20,2]
nodes_p = mesh_p.entity('node') #[16,2]

#计算网格u,v,p节点的总数
num_nodes_u = nodes_u.shape[0] #20
num_nodes_v = nodes_v.shape[0] #20
num_nodes_p = nodes_p.shape[0] #16

def nodes_v_ub(mesh):
        Nrow = mesh.node.shape[1]
        nodes_v_ub = np.zeros_like(nodes_v)
        indexl = np.zeros(nodes_v.shape[0],dtype='bool')
        indexr = np.zeros(nodes_v.shape[0],dtype='bool')
        indexl[1:Nrow-1] = True
        indexr[-Nrow+1:-1] = True
        nodes_v_ub[indexr] = nodes_v[indexr]
        nodes_v_ub[indexl] = nodes_v[indexl]
        nodes_v_ub[indexr,0] -= (hx/2)
        nodes_v_ub[indexl,0] += (hx/2)
        return nodes_v_ub

def nodes_u_ub(mesh):
    Nrow = mesh.node.shape[1]
    Ncol = mesh.node.shape[0]
    N = Nrow * Ncol
    nodes_u_ub = np.zeros_like(nodes_u)
    indexb = np.zeros(nodes_u.shape[0],dtype='bool')
    indexu = np.zeros(nodes_u.shape[0],dtype='bool')
    indexb[Nrow:N-Nrow][np.arange(Nrow,N-Nrow) % Nrow == 0] = True
    indexu[Nrow:N-Nrow][np.arange(Nrow,N-Nrow) % Nrow == Nrow-1] = True
    nodes_u_ub[indexb] = nodes_u[indexb]
    nodes_u_ub[indexu] = nodes_u[indexu]
    nodes_u_ub[indexb,1] -= (hy/2)
    nodes_u_ub[indexu,1] += (hy/2)
    return nodes_u_ub

def negation(mesh,k):
    Nrow = mesh.node.shape[1]
    Ncol = mesh.node.shape[0]
    N = Nrow * Ncol
    index = np.zeros(nodes_u.shape[0],dtype='bool')
    index[Nrow:N-Nrow][np.arange(Nrow,N-Nrow) % Nrow == 0] = True
    k[index] *= -1
    return  k

def laplaplace_phi(mesh):
    Nrow = mesh.node.shape[1]
    Ncol = mesh.node.shape[0]
    N = Nrow*Ncol
    result = diags([-4,1,1,1,1],[0,1,-1,Nrow,-Nrow],(N,N), format='lil')
    index = np.arange(0,N)
    index0 = np.where(index%Nrow ==0)
    index1 = np.ones_like(index0)
    result[index0,index0-index1] = 0
    result[index0-index1,index0] = 0
    result[index[:Nrow],index[:Nrow]] = -3
    result[index[-Nrow:],index[-Nrow:]] = -3
    result[index0,index0] = -3
    result[index0-index1,index0-index1] = -3
    result[0,0] = -2
    result[Nrow-1,Nrow-1] = -2
    result[N-1,N-1] = -2
    result[N-Nrow,N-Nrow] = -2
    return result

def result_phi(p,mesh):
    Nrow = mesh.node.shape[1]
    K = p
    phi = K.reshape((Nrow,Nrow)).T[::-1] #改变形状为(4,4)并且每个值对应网格节点上的phi值
    return phi

solver = NSMacSolver(mesh_u, mesh_v, mesh_p)
for i in range(5):
    # 下一个的时间层 ti
    tl = tmesh.next_time_level()
    print("tl=", tl)
    print("i=", i)

    #0时间层的值
    def solution_u_0(p):
        return pde.solution_u(p,t=i*tau)
    def solution_v_0(p):
        return pde.solution_v(p,t=i*tau)
    def solution_p_0(p):
        return pde.solution_p(p,t=i*tau)
    
    u_ub0 = pde.solution_u(nodes_u_ub(mesh_u),t=i*tau)
    u_ub0= negation(mesh_u,u_ub0)
    v_ub0 = pde.solution_v(nodes_v_ub(mesh_v),t=i*tau)
    M = v_ub0.shape[0] // 2
    v_ub0[:M] *= -1
    
    solution_u = mesh_u.interpolate(solution_u_0) #[4,5]
    solution_u_values0 = solution_u.reshape(-1)
    solution_v = mesh_v.interpolate(solution_v_0)
    solution_v_values0 = solution_v.reshape(-1)
    solution_p = mesh_p.interpolate(solution_p_0)
    solution_p_values0 = solution_p.reshape(-1)

    #u网格上的算子矩阵
    gradux0 = solver.grad_ux() @ solution_u_values0
    graduy0 = solver.grad_uy() @ solution_u_values0 + (8*u_ub0/3)/(2*hy)
    Tuv0 = solver.Tuv() @ solution_v_values0
    laplaceu = solver.laplace_u()
    #v网格上的算子矩阵
    gradvx0 = solver.grad_vx() @ solution_v_values0 + (8*v_ub0/3)/(2*hx)
    gradvy0 = solver.grad_vy() @ solution_v_values0
    Tvu0 = solver.Tvu() @ solution_u_values0
    laplacev = solver.laplace_v()
    #0时间层的 Adams-Bashforth 公式逼近的对流导数
    AD_xu_0 = solution_u_values0 * gradux0 + Tuv0 * graduy0
    BD_yv_0 = solution_v_values0 * gradvy0 + Tvu0 * gradvx0
    
    #tau时间层的值
    def solution_u_1(p):
        return pde.solution_u(p,t=(i+1)*tau)
    def solution_v_1(p):
        return pde.solution_v(p,t=(i+1)*tau)
    def solution_p_1(p):
        return pde.solution_p(p,t=(i+1)*tau)
    
    u_ub11 = pde.solution_u(nodes_u_ub(mesh_u),t=(i+1)*tau)
    u_ub1 = np.copy(u_ub11)
    u_ub1= negation(mesh_u,u_ub1)
    v_ub11 = pde.solution_v(nodes_v_ub(mesh_v),t=(i+1)*tau)
    v_ub1 = np.copy(v_ub11)
    H = v_ub1.shape[0] // 2
    v_ub1[:M] *= -1
    
    solution_u = mesh_u.interpolate(solution_u_1) #[4,5]
    solution_u_values1 = solution_u.reshape(-1)
    solution_v = mesh_v.interpolate(solution_v_1)
    solution_v_values1 = solution_v.reshape(-1)
    solution_p = mesh_p.interpolate(solution_p_1)
    solution_p_values1 = solution_p.reshape(-1)

    #u网格上的算子矩阵
    gradux1 = solver.grad_ux() @ solution_u_values1
    graduy1 = solver.grad_uy() @ solution_u_values1 + (8*u_ub1/3)/(2*hy)
    Tuv1 = solver.Tuv() @ solution_v_values1
    #v网格上的算子矩阵
    gradvx1 = solver.grad_vx() @ solution_v_values1 + (8*v_ub1/3)/(2*hx)
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
    F = solver.source_Fx(pde,t=(i+1)*tau)
    Fx = F[:,0]
    b = tau*(-3/2*AD_xu_1-1/2*AD_xu_0+nu/2*(laplaceu@solution_u_values1 + (8*u_ub11/3)/(hx*hy))+Fx-solver.grand_uxp()@solution_p_values1)

    #组装B、c矩阵
    E = np.zeros_like(laplacev.toarray())
    row2, col2 = np.diag_indices_from(E)
    E[row2,col2] = 1
    B = E - (nu*tau*laplacev)/2
    Fy = F[:,1]
    c = tau*(-3/2*BD_yv_1-1/2*BD_yv_0+nu/2*(laplacev@solution_v_values1 + (8*v_ub11/3)/(hx*hy))+Fy-solver.grand_vyp()@solution_p_values1)
    
    #A,b矩阵边界处理并解方程（时间层用0还是tau？）
    nxu = mesh_u.node.shape[1]
    is_boundaryu = np.zeros(num_nodes_u,dtype='bool')
    is_boundaryu[:nxu] = True
    is_boundaryu[-nxu:] = True
    dirchiletu = pde.dirichlet_u(nodes_u[is_boundaryu], i*tau)
    b[is_boundaryu] = dirchiletu

    bdIdxu = np.zeros(A.shape[0], dtype=np.int_)
    bdIdxu[is_boundaryu] = 1
    Tbdu = spdiags(bdIdxu, 0, A.shape[0], A.shape[0])
    T1 = spdiags(1-bdIdxu, 0, A.shape[0], A.shape[0])
    A = A@T1 + Tbdu
    u_1 = spsolve(A, b) #(20,)

    #B,c矩阵边界处理并解方程（时间层用0还是tau？）
    nyv = mesh_v.node.shape[1]
    is_boundaryv = np.zeros(num_nodes_v,dtype='bool')
    is_boundaryv[(np.arange(num_nodes_v) % nyv == 0)] = True
    indices = np.where(is_boundaryv)[0] - 1
    is_boundaryv[indices] = True
    dirchiletv = pde.dirichlet_v(nodes_v[is_boundaryv], i*tau)
    c[is_boundaryv] = dirchiletv

    bdIdyv = np.zeros(B.shape[0],dtype=np.int_)
    bdIdyv[is_boundaryv] = 1
    Tbdv = spdiags(bdIdyv,0,B.shape[0],B.shape[0])
    T2 = spdiags(1-bdIdyv,0,B.shape[0],B.shape[0])
    B = B@T2 + Tbdv
    v_1 = spsolve(B,c) #(20,)
    #中间速度
    u_1_reshape = u_1.reshape(-1,1) #(20,1)
    v_1_reshape = v_1.reshape(-1,1) #(20,1)
    w_1 = np.concatenate((u_1_reshape,v_1_reshape),axis=1) #(20,2)

    #求解修正项
    H = laplaplace_phi(mesh_p).toarray()
    M = np.dot(solver.grad_pux().toarray(),u_1)+np.dot(solver.grad_pvy().toarray(),v_1)
    L = M.reshape(-1,1)/tau
    K = spsolve(H,L) #(16,1)
    phi = result_phi(K,mesh_p)
    
    #计算修正项的梯度
    grad_phix = np.gradient(phi,axis=1) #x方向(4,4)
    grad_phiy = np.gradient(phi,axis=0)  #y方向(4,4)
    reshape_grad_phix = grad_phix[::-1].flatten('F') #(16,1)phix0-phix15
    reshape_grad_phiy = grad_phiy[::-1].flatten('F') #(16,1)phiy0-phiy15
    
    #更新速度
    def phi_x(p,mesh):
        Nrow = mesh.node.shape[1]
        Ncol = mesh.node.shape[0]
        N = Nrow*Ncol
        K = p.reshape(-1,1)
        L = np.zeros((Nrow,1))
        phi_x = np.vstack((K,L))
        index = np.arange(0,N)
        phi_x[index[:Nrow]] = 0
        return phi_x
    phi_x = phi_x(reshape_grad_phix,mesh_u) #(20,1)
    solution_u_values2 = w_1[:,0].reshape(-1,1)-tau*phi_x #(20,1)

    def phi_y(p,mesh):
        Nrow = mesh.node.shape[1]
        Ncol = mesh.node.shape[0]
        N = Nrow*Ncol
        K = p.reshape(-1,1)
        phi_y = np.zeros((N,1))
        index0 = np.arange(N)
        index0 = np.where(index0 % Nrow != Nrow-1)
        phi_y[index0] = K
        index1 = np.arange(N)
        index1 = np.where(index1 % Nrow == 0)
        phi_y[index1] = 0
        return phi_y
    phi_y = phi_y(reshape_grad_phiy,mesh_v) #(20,1)
    solution_v_values2 = w_1[:,1].reshape(-1,1)-tau*phi_y #(20,1)

    w_2 = np.concatenate((solution_u_values2,solution_v_values2),axis=1)

    #更新压力及其梯度
    solution_p_values2 = solution_p_values1+K-(nu*tau*M)/2
    def reshape_p(p,mesh):
        Nrow = mesh.node.shape[1]
        Ncol = mesh.node.shape[0]
        N = Nrow*Ncol
        reshape_p = p.reshape((Nrow,Nrow)).T[::-1] ##改变形状为(4,4)并且每个值对应网格节点上的p值
        return reshape_p
    reshape_p = reshape_p(solution_p_values2,mesh_p)
    gradpx = np.gradient(reshape_p,axis=1) #压力在x方向上的梯度(4,4)
    gradpy = np.gradient(reshape_p,axis=0) #压力在y方向上的梯度(4,4)
    grad_px = gradpx[::-1].flatten('F') #(16,1)gradpx0-gradpx15
    grad_py = gradpy[::-1].flatten('F') #(16,1)gradpy0-gradpy15
    
    uu = pde.solution_u(nodes_u,tl)
    vv = pde.solution_v(nodes_v,tl)
    pp = pde.solution_p(nodes_p,tl)
    erru = np.sqrt((uu-w_2[:.0])**2+(vv-w_2[:,1])**2)
    print(erru)
    

     # 时间步进一层 
    tmesh.advance()
