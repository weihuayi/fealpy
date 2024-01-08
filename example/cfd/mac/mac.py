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
from scipy.sparse import csr_matrix

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
mesh = UniformMesh2d([0,nx,0,ny],h=(hx,hy),origin=(0,0))

#时间离散
nt = 1000
tmesh = UniformTimeLine(0, T, nt)
tau = tmesh.dt

solver = NSMacSolver(Re, mesh)
np.set_printoptions(linewidth=1000)

#网格点的位置
nodes_u = solver.umesh.entity('node') 
nodes_v = solver.vmesh.entity('node') 
nodes_p = solver.pmesh.entity('node') 

#计算网格u,v,p节点的总数
num_nodes_u = nodes_u.shape[0] 
num_nodes_v = nodes_v.shape[0] 
num_nodes_p = nodes_p.shape[0] 

#把phi插值到u，v网格
def result_phi(p,mesh):
    Nrow = mesh.node.shape[1]
    phi = p.reshape((Nrow,Nrow)).T[::-1] 
    return phi

for i in range(2):
    # 下一个的时间层 ti
    tl = tmesh.next_time_level()
    print("tl=", tl)
    print("i=", i)

    #n-1时间层的值
    def solution_u_0(p):
        return pde.solution_u(p,t=i*tau)
    def solution_v_0(p):
        return pde.solution_v(p,t=i*tau)
    def solution_p_0(p):
        return pde.solution_p(p,t=i*tau)
    
    ##u,v网格u_b的值 
    u_ub0 = pde.solution_u(solver.unodes_ub(),t=i*tau)
    v_ub0 = pde.solution_v(solver.vnodes_ub(),t=i*tau)
    ##n-1时间层各网格节点对应的取值
    solution_u = solver.umesh.interpolate(solution_u_0) 
    uvalues0 = solution_u.reshape(-1)
    solution_v = solver.vmesh.interpolate(solution_v_0)
    vvalues0 = solution_v.reshape(-1)
    solution_p = solver.pmesh.interpolate(solution_p_0)
    pvalues0 = solution_p.reshape(-1)

    ##u网格上的AD_x算子矩阵
    dux,duy = solver.du()
    dux0 = dux @ uvalues0
    duy0 = duy @ uvalues0 + (8*u_ub0/3)/(2*hy)
    Tuv0 = solver.Tuv() @ vvalues0
    ADxu0 = uvalues0 * dux0 + Tuv0 * duy0
    ##v网格上的AD_y算子矩阵
    dvx,dvy = solver.dv()
    dvx0 = dvx @ vvalues0 + (8*v_ub0/3)/(2*hx)
    dvy0 = dvy @ vvalues0
    Tvu0 = solver.Tvu() @ uvalues0
    ADyv0 = Tvu0 * dvx0 + vvalues0 * dvy0

    #n时间层的值
    def solution_u_1(p):
        return pde.solution_u(p,t=(i+1)*tau)
    def solution_v_1(p):
        return pde.solution_v(p,t=(i+1)*tau)
    def solution_p_1(p):
        return pde.solution_p(p,t=(i+1)*tau)
    
    ##u,v网格u_b的值 
    u_ub1 = pde.solution_u(solver.unodes_ub(),t=(i+1)*tau)
    v_ub1 = pde.solution_v(solver.vnodes_ub(),t=(i+1)*tau)
    ##n时间层各网格节点对应的取值
    solution_u = solver.umesh.interpolate(solution_u_1) 
    uvalues1 = solution_u.reshape(-1)
    solution_v = solver.vmesh.interpolate(solution_v_1)
    vvalues1 = solution_v.reshape(-1)
    solution_p = solver.pmesh.interpolate(solution_p_1)
    pvalues1 = solution_p.reshape(-1)
    
    ##u网格上的AD_x算子矩阵
    dux1 = dux @ uvalues1
    duy1 = duy @ uvalues1 + (8*u_ub1/3)/(2*hy)
    Tuv1 = solver.Tuv() @ vvalues1
    ADxu1 = uvalues1 * dux1 + Tuv1 * duy1
    
    ##v网格上的AD_y算子矩阵
    dvx1 = dvx @ vvalues1 + (8*v_ub1/3)/(2*hx)
    dvy1 = dvy @ vvalues1
    Tvu1 = solver.Tvu() @ uvalues1
    ADyv1 = Tvu1 * dvx1 + vvalues1 * dvy1
    
    #组装A_u、b_u矩阵
    laplaceu = solver.laplace_u()
    A_u = solver.laplace_u(1-tau/(2*Re))
    F = solver.source_Fx(pde,t=(i+1)*tau)
    Fx = F[:,0]
    b_u = uvalues1+tau*(-3*ADxu1/2+ADxu0/2-nu*solver.dp_u()@pvalues1\
        +nu/2*(laplaceu@uvalues1+(8*u_ub1/3)/(hx*hy))+nu*Fx)
    #组装A_v、b_v矩阵
    laplacev = solver.laplace_v()
    A_v = solver.laplace_v(1-tau/(2*Re))
    Fy = F[:,1]
    b_v = vvalues1+tau*(-3*ADyv1/2+ADyv0/2-nu*solver.dp_v()@pvalues1\
        +nu/2*(laplacev@vvalues1+(8*v_ub1/3)/(hx*hy))+nu*Fy)
    
    ##A_u,b_u矩阵边界处理并解方程
    nxu = solver.umesh.node.shape[1]
    is_boundaryu = np.zeros(num_nodes_u,dtype='bool')
    is_boundaryu[:nxu] = True
    is_boundaryu[-nxu:] = True
    dirchiletu = pde.dirichlet_u(nodes_u[is_boundaryu], (i+2)*tau)
    b_u[is_boundaryu] = dirchiletu

    bdIdxu = np.zeros(A_u.shape[0], dtype=np.int_)
    bdIdxu[is_boundaryu] = 1
    Tbdu = spdiags(bdIdxu, 0, A_u.shape[0], A_u.shape[0])
    T1 = spdiags(1-bdIdxu, 0, A_u.shape[0], A_u.shape[0])
    A_u = A_u@T1 + Tbdu

    u_1 = spsolve(A_u, b_u) 
    
    ##A_v,b_v矩阵边界处理并解方程
    nyv = solver.vmesh.node.shape[1]
    is_boundaryv = np.zeros(num_nodes_v,dtype='bool')
    is_boundaryv[(np.arange(num_nodes_v) % nyv == 0)] = True
    indices = np.where(is_boundaryv)[0] - 1
    is_boundaryv[indices] = True
    dirchiletv = pde.dirichlet_v(nodes_v[is_boundaryv], (i+2)*tau)
    b_v[is_boundaryv] = dirchiletv

    bdIdyv = np.zeros(A_v.shape[0],dtype=np.int_)
    bdIdyv[is_boundaryv] = 1
    Tbdv = spdiags(bdIdyv,0,A_v.shape[0],A_v.shape[0])
    T2 = spdiags(1-bdIdyv,0,A_v.shape[0],A_v.shape[0])
    A_v = A_v@T2 + Tbdv

    v_1 = spsolve(A_v,b_v) 
    
    #中间速度
    w_1 = np.stack((u_1,v_1),axis=1) 
    
    #求解修正项
    H = solver.laplace_phi() 
    dpm_u,dpm_v = solver.dpm()
    M = (dpm_u*u_1+dpm_v*v_1)/tau 
    K = spsolve(H,M) 
    mean = np.sum(K*hx*hy)/(nx*hx*ny*hy)
    K -= mean
    phi = result_phi(K,solver.pmesh) 
    
    ##计算修正项的梯度
    dphi_x = np.gradient(phi,axis=1) 
    dphi_y = np.gradient(phi,axis=0)  
    rdphi_x = dphi_x[::-1].flatten('F') 
    rdphi_y = dphi_y[::-1].flatten('F') 
    
    #更新速度
    def phi_x(p,mesh):
        Nrow = mesh.node.shape[1] 
        Ncol = mesh.node.shape[0] 
        N = Nrow*Ncol 
        L = np.zeros((Nrow,))
        phi_x = np.concatenate((p,L))
        index = np.arange(0,N)
        phi_x[index[:Nrow]] = 0
        return phi_x
    def phi_y(p,mesh):
        Nrow = mesh.node.shape[1] 
        Ncol = mesh.node.shape[0] 
        N = Nrow*Ncol 
        phi_y = np.zeros((N,))
        index0 = np.arange(N)
        index0 = np.where(index0 % Nrow != Nrow-1)
        phi_y[index0] = p
        index1 = np.arange(N)
        index1 = np.where(index1 % Nrow == 0)
        phi_y[index1] = 0
        return phi_y

    phi_x = phi_x(rdphi_x,solver.umesh) 
    uvalues2 = u_1-tau*phi_x 
    phi_y = phi_y(rdphi_y,solver.vmesh) 
    vvalues2 = v_1-tau*phi_y 
    ##u、v方向速度
    uvvalues2 = np.stack((uvalues2,vvalues2),axis=1)
    
    #更新压力
    pvalues2 = pvalues1+K-(nu*tau*(dpm_u*u_1+dpm_v*v_1))/2
    
    uu = pde.solution_u(nodes_u,tl)
    vv = pde.solution_v(nodes_v,tl)
    pp = pde.solution_p(nodes_p,tl)
    erru = np.sqrt(np.sum((uu-uvalues2)**2+(vv-vvalues2)**2))/num_nodes_u
    errp = np.sqrt(np.sum(pp-pvalues2)**2)/num_nodes_p
    

    # 时间步进一层 
    tmesh.advance()
