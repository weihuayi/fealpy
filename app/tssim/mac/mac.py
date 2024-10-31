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
from scipy.sparse import csr_matrix, eye

Re =1
nu = 1/Re
T = 5
#PDE 模型 
pde = taylor_greenData(Re,T=[0,T])
domain = pde.domain()

#建立网格
nx = 16
ny = 16
hx = (domain[1] - domain[0])/nx
hy = (domain[3] - domain[2])/ny
mesh = UniformMesh2d([0,nx,0,ny],h=(hx,hy),origin=(0,0))

#时间离散
nt = 10000
tmesh = UniformTimeLine(0, T, nt)
tau = tmesh.dt

#建立Solver
solver = NSMacSolver(Re, mesh)

#网格点的位置
nodes_u = solver.umesh.entity('node') 
nodes_v = solver.vmesh.entity('node') 
nodes_p = solver.pmesh.entity('node') 

#计算网格u,v,p节点的总数
num_nodes_u = nodes_u.shape[0] 
num_nodes_v = nodes_v.shape[0] 
num_nodes_p = nodes_p.shape[0] 

#0 时间层各网格节点对应的取值
values_0_u = solver.umesh.interpolate(lambda p:pde.solution_u(p,0)).reshape(-1)
values_0_v = solver.vmesh.interpolate(lambda p:pde.solution_v(p,0)).reshape(-1)
values_0_p = solver.pmesh.interpolate(lambda p:pde.solution_p(p,0)).reshape(-1)

#1 时间层各网格节点对应的取值
values_1_u = solver.umesh.interpolate(lambda p:pde.solution_u(p,tau)).reshape(-1)
values_1_v = solver.vmesh.interpolate(lambda p:pde.solution_v(p,tau)).reshape(-1)
values_1_p = solver.pmesh.interpolate(lambda p:pde.solution_p(p,tau)).reshape(-1)
values_half_p = (values_0_p + values_1_p)/2

# 初始化误差数组
erru_values = np.zeros(nt)
errp_values = np.zeros(nt)

for i in range(nt):
    # 更新时间层 ti
    tl = tmesh.next_time_level()
    print("当前时间", tl)
    print("当前时间层", i)

    #0 时间层u,v网格边界上速度的值 
    b_0_u = pde.solution_u(solver.unodes_ub(),t=(i)*tau)
    b_0_v = pde.solution_v(solver.vnodes_ub(),t=(i)*tau)
    
    #1 时间层u,v网格边界上速度的值 
    b_1_u = pde.solution_u(solver.unodes_ub(),t=(i+1)*tau)
    b_1_v = pde.solution_v(solver.vnodes_ub(),t=(i+1)*tau)
    
    # 第一步左端矩阵组装
    laplaceu = solver.laplace_u()
    laplace_u_s = solver.laplace_u((nu*tau)/2)
    I_u = eye(laplace_u_s.shape[0], format='coo')
    A_u = I_u - laplace_u_s

    # 第一步右端向量组装
    dux,duy = solver.du()
    dux0 = dux @ values_0_u
    duy0 = duy @ values_0_u + (8*b_0_u/3)/(2*hy)
    Tuv0 = solver.Tuv() @ values_0_v
    ADxu0 = values_0_u * dux0 + Tuv0 * duy0

    dux1 = dux @ values_1_u
    duy1 = duy @ values_1_u + (8*b_1_u/3)/(2*hy)
    Tuv1 = solver.Tuv() @ values_1_v
    ADxu1 = values_1_u * dux1 + Tuv1 * duy1

    F = solver.source_Fx(pde,t=(i+1)*tau)
    Fx = F[:,0]
    b_u = values_1_u+tau*(-3*ADxu1/2+ADxu0/2-solver.dp_u()@values_half_p\
        +nu/2*(laplaceu@values_1_u+(8*b_1_u/3)/(hx*hy))+Fx)
    
    # 第一步求解
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
    A_u = T1@A_u + Tbdu

    u_s = spsolve(A_u, b_u)  
    
    # 第二步左端矩阵组装
    laplacev = solver.laplace_v()
    laplace_v_s = solver.laplace_v((nu*tau)/2)
    I_v = eye(laplace_v_s.shape[0], format='coo')
    A_v = I_v - laplace_v_s

    # 第二步右端向量组装
    dvx,dvy = solver.dv()
    dvx0 = dvx @ values_0_v + (8*b_0_v/3)/(2*hx)
    dvy0 = dvy @ values_0_v
    Tvu0 = solver.Tvu() @ values_0_u
    ADyv0 = Tvu0 * dvx0 + values_0_v * dvy0
 
    dvx1 = dvx @ values_1_v + (8*b_1_v/3)/(2*hx)
    dvy1 = dvy @ values_1_v
    Tvu1 = solver.Tvu() @ values_1_u
    ADyv1 = Tvu1 * dvx1 + values_1_v * dvy1
    
    Fy = F[:,1]
    b_v = values_1_v+tau*(-3*ADyv1/2+ADyv0/2-solver.dp_v()@values_half_p\
        +nu/2*(laplacev@values_1_v+(8*b_1_v/3)/(hx*hy))+Fy)
    
    # 第二步求解
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
    A_v = T2@A_v + Tbdv
    
    v_s = spsolve(A_v,b_v)
    
    # 第三步左端矩阵组装
    H = solver.laplace_phi()
    
    # 第三步右端向量组装
    dpm_u,dpm_v = solver.dpm()
    M = (dpm_u*u_s+dpm_v*v_s)/tau 

    # 第三步求解
    phi = spsolve(H,M) 
    phi = phi - np.mean(phi)
    
    # 更新变量
    values_2_u = u_s-tau*solver.dp_u()@phi

    values_2_v = v_s-tau*solver.dp_v()@phi

    values_2_p = values_half_p + phi - (nu*(dpm_u*u_s+dpm_v*v_s))/2
    values_2_p = values_2_p - np.mean(values_2_p)
    
    # 计算误差
    uu = pde.solution_u(nodes_u,(i+2)*tau)
    vv = pde.solution_v(nodes_v,(i+2)*tau)
    pp = pde.solution_p(nodes_p,(i+3/2)*tau)
    erru = np.sqrt(np.sum((uu-values_2_u)**2+(vv-values_2_v)**2))/(num_nodes_u+num_nodes_v)
    errp = np.sqrt(np.sum(pp-values_2_p)**2)/num_nodes_p
    #errp = np.abs(np.mean(pp-values_2_p))/num_nodes_p

    print("u",erru)
    print("p",errp)
    
    values_0_u[:] = values_1_u
    values_1_u[:] = values_2_u
    values_0_v[:] = values_1_v
    values_1_v[:] = values_2_v
    values_half_p[:] = values_2_p

    # 记录误差
    erru_values[i] = erru
    errp_values[i] = errp

    # 时间层进一步
    tmesh.advance()

'''
# 绘制误差随时间的变化图像
time_levels = np.linspace(0, T, nt)
plt.figure(figsize=(10, 6))
plt.plot(time_levels, erru_values, label='Error in u')
plt.plot(time_levels, errp_values, label='Error in p')
plt.xlabel('Time')
plt.ylabel('Error')
plt.title('Error Evolution over Time')
plt.legend()
plt.grid(True)
plt.show()
'''
