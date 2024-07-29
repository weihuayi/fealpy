import jax
import jax.numpy as jnp
import numpy as np
from fealpy.jax.mesh.node_mesh import NodeMesh
from fealpy.jax.sph import SPHSolver
from fealpy.jax.sph.solver import Tag
from fealpy.jax.sph import partition 
from fealpy.jax.sph.jax_md.partition import Sparse
from fealpy.jax.sph.kernel_function import QuinticKernel
from jax_md import space
from jax import vmap
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True) # 启用 float64 支持
jax.config.update('jax_platform_name', 'cpu')

EPS = jnp.finfo(float).eps
dx = 1.25e-4 
h = 0.5 * dx
rho0 = 737.54 
uin = jnp.array([5.0, 0.0])
n = 0.3083
tau_s = 16834.4
mu0 = 938.4118
c1 = 0.0894
B = 5.914e7
eta = 0.5 #动量方程中的参数
dt = 1e-7
domain=[0,0.05,0,0.005] 
init_domain=[0.0,0.005,0,0.005] 

#初始化
mesh = NodeMesh.from_long_rectangular_cavity_domain(init_domain=init_domain, domain=domain, uin=uin, dx=dx)
solver = SPHSolver(mesh)
kernel = QuinticKernel(h=h, dim=2)
displacement, shift = space.free()

#计算区域
x0 = jnp.min(mesh.nodedata['position'][mesh.nodedata['tag']==3][:,0])
x1 = jnp.max(mesh.nodedata['position'][mesh.nodedata['tag']==1][:,0])
y0 = jnp.min(mesh.nodedata['position'][mesh.nodedata['tag']==2][:,1])
y1 = jnp.max(mesh.nodedata['position'][mesh.nodedata['tag']==2][:,1])
box_size = jnp.array([x1, y1])

#生成固壁粒子与虚粒子之间的对应关系
w_idx = jnp.where(mesh.nodedata['tag'] == 1)[0]
d_w_idx = jnp.repeat(w_idx, 3)

#邻近搜索
neighbor_fn = partition.neighbor_list(
        displacement,
        box_size,
        r_cutoff=kernel.cutoff_radius,
        backend="jaxmd_vmap",
        capacity_multiplier=1.25,
        mask_self=False,
        format=Sparse,
        num_particles_max=mesh.nodedata["position"].shape[0],
        num_partitions=1,
        pbc=np.array([False, False]),
    )
#所有粒子
num_particles = mesh.nodedata['position'].shape[0]
neighbors = neighbor_fn.allocate(mesh.nodedata["position"], num_particles=num_particles)

#流体粒子和固壁粒子
fw_particles = ((mesh.nodedata["tag"] == 0) | (mesh.nodedata["tag"] == 1)).sum()
fw_neighbors = neighbor_fn.allocate(mesh.nodedata["position"][(mesh.nodedata["tag"] == 0)\
             | (mesh.nodedata["tag"] == 1)], num_particles=fw_particles)

#流体粒子
f_particles = (mesh.nodedata["tag"] == 0).sum()
f_neighbors = neighbor_fn.allocate(mesh.nodedata["position"][mesh.nodedata["tag"]==0], num_particles=f_particles)

state_history = []
for i in range(10000): 
    print("i",i)
    state_history.append(mesh.nodedata.copy())

    #所有粒子（删除自身索引）
    i, j = neighbors.idx 
    i, j = i[i != jnp.max(i)], j[j != jnp.max(j)] 
    vstack_s = jnp.vstack((i, j))
    mask = vstack_s[0] != vstack_s[1]
    mask_idx = vstack_s[:, mask]
    i_s, j_s = mask_idx[0], mask_idx[1]
    r_i_s, r_j_s = mesh.nodedata["position"][jnp.array(i_s)], mesh.nodedata["position"][jnp.array(j_s)]
    dr_i_j = vmap(displacement)(r_i_s, r_j_s)
    dist = space.distance(dr_i_j)
    w_dist = vmap(kernel)(dist)

    e_s = dr_i_j / (dist[:, None] + EPS)
    grad_w_dist_norm = vmap(kernel.grad_value)(dist)
    grad_w_dist = grad_w_dist_norm[:, None] * e_s

    #流体粒子和固壁粒子
    fwi_s, fwj_s = fw_neighbors.idx
    fwi_s, fwj_s = fwi_s[fwi_s != jnp.max(fwi_s)], fwj_s[fwj_s != jnp.max(fwj_s)]
    fw_position = mesh.nodedata["position"][(mesh.nodedata["tag"] == 0) | (mesh.nodedata["tag"] == 1)]
    fwr_i_s, fwr_j_s = fw_position[jnp.array(fwi_s)], fw_position[jnp.array(fwj_s)]
    fwdr_i_j = vmap(displacement)(fwr_i_s, fwr_j_s)
    fwdist = space.distance(fwdr_i_j)
    fww_dist = vmap(kernel)(fwdist)
    
    #流体粒子（删除自身索引）
    fi, fj = f_neighbors.idx
    fi, fj = fi[fi != jnp.max(fi)], fj[fj != jnp.max(fj)]
    fvstack_s = jnp.vstack((fi, fj))
    fmask = fvstack_s[0] != fvstack_s[1]
    fmask_idx = fvstack_s[:, fmask]
    fi_s, fj_s = fmask_idx[0], fmask_idx[1]
    fr_i_s, fr_j_s = mesh.nodedata["position"][fi_s], mesh.nodedata["position"][fj_s]
    fdr_i_j = vmap(displacement)(fr_i_s, fr_j_s)
    fdist = space.distance(fdr_i_j)
    fw_dist = vmap(kernel)(fdist)

    fe_s = fdr_i_j / (fdist[:, None] + EPS)
    fgrad_w_dist_norm = vmap(kernel.grad_value)(fdist)
    fgrad_w_dist = fgrad_w_dist_norm[:, None] * fe_s

    #更新固壁粒子外推速度赋值到虚粒子上
    v_ext = solver.wall_extrapolation(mesh.nodedata, fww_dist, fwi_s, fwj_s)
    v_repeat = jnp.repeat(v_ext, repeats=3, axis=0)
    mesh.nodedata["v"] = mesh.nodedata["v"].at[mesh.nodedata["tag"] == 2].set(v_repeat)
    
    #更新半步压力和声速
    mesh.nodedata = solver.change_p(mesh.nodedata, fww_dist, fwi_s, fwj_s, d_w_idx, B=B, rho0=rho0, c1=c1)

    #更新半步密度
    drho = solver.continue_equation(mesh.nodedata, i_s, j_s, grad_w_dist)
    rho_1 = mesh.nodedata["rho"][mesh.nodedata["tag"] == 0] + 0.5*dt*drho
    f_tag = jnp.where(mesh.nodedata["tag"] == 0)[0]
    mesh.nodedata["rho"] = mesh.nodedata["rho"].at[f_tag].set(rho_1)
    
    #更新 mu
    mu = solver.mu_wlf(mesh.nodedata, i_s, j_s, grad_w_dist, mu0=mu0, tau=tau_s, n=n)

    #更新半步速度
    dv = solver.momentum_equation(mesh.nodedata, i_s, j_s, grad_w_dist, mu=mu, eta=eta, h=h)
    v_1 = mesh.nodedata["v"][mesh.nodedata["tag"] == 0] + 0.5*dt*dv
    mesh.nodedata["v"] = mesh.nodedata["v"].at[f_tag].set(v_1)
    
    #更新固壁粒子外推速度赋值到虚粒子上
    v_ext = solver.wall_extrapolation(mesh.nodedata, fww_dist, fwi_s, fwj_s)
    v_repeat = jnp.repeat(v_ext, repeats=3, axis=0)
    mesh.nodedata["v"] = mesh.nodedata["v"].at[mesh.nodedata["tag"] == 2].set(v_repeat)

    #更新半步位置
    dr = solver.change_position(mesh.nodedata, i_s, j_s, w_dist)
    r_1 = mesh.nodedata["position"][mesh.nodedata["tag"] == 0] + 0.5*dt*dr
    mesh.nodedata["position"] = mesh.nodedata["position"].at[f_tag].set(r_1)

    #更新半步后的质量
    mesh.nodedata["mass"] = dx**2 * mesh.nodedata["rho"]
    
    #更新压力和声速
    mesh.nodedata = solver.change_p(mesh.nodedata, fww_dist, fwi_s, fwj_s, d_w_idx, B=B, rho0=rho0, c1=c1)

    #更新密度
    drho = solver.continue_equation(mesh.nodedata, i_s, j_s, grad_w_dist)
    rho_1 = mesh.nodedata["rho"][mesh.nodedata["tag"] == 0] + 0.5*dt*drho
    f_tag = jnp.where(mesh.nodedata["tag"] == 0)[0]
    mesh.nodedata["rho"] = mesh.nodedata["rho"].at[f_tag].set(rho_1)

    #更新 mu
    mu = solver.mu_wlf(mesh.nodedata, i_s, j_s, grad_w_dist, mu0=mu0, tau=tau_s, n=n)

    #更新速度
    dv = solver.momentum_equation(mesh.nodedata, i_s, j_s, grad_w_dist, mu=mu, eta=eta, h=h)
    v_1 = mesh.nodedata["v"][mesh.nodedata["tag"] == 0] + 0.5*dt*dv
    mesh.nodedata["v"] = mesh.nodedata["v"].at[f_tag].set(v_1)

    #更新位置
    dr = solver.change_position(mesh.nodedata, i_s, j_s, w_dist)
    r_1 = mesh.nodedata["position"][mesh.nodedata["tag"] == 0] + 0.5*dt*dr
    mesh.nodedata["position"] = mesh.nodedata["position"].at[f_tag].set(r_1)

    #更新质量
    mesh.nodedata["mass"] = dx**2 * mesh.nodedata["rho"]

    #增加门粒子并更新位置
    mesh.nodedata = solver.gate_change(mesh.nodedata, domain, dt=dt, dx=dx, uin=uin)

    #索引更新
    num_particles = mesh.nodedata['position'].shape[0]
    neighbors = neighbor_fn.allocate(mesh.nodedata["position"], num_particles=num_particles)

    fw_particles = ((mesh.nodedata["tag"] == 0) | (mesh.nodedata["tag"] == 1)).sum()
    fw_neighbors = neighbor_fn.allocate(mesh.nodedata["position"][(mesh.nodedata["tag"] == 0)\
             | (mesh.nodedata["tag"] == 1)], num_particles=fw_particles)

    f_particles = (mesh.nodedata["tag"] == 0).sum()
    f_neighbors = neighbor_fn.allocate(mesh.nodedata["position"][mesh.nodedata["tag"]==0], num_particles=f_particles)

solver.create_animation(state_history, 'animation.gif')

'''
# 获取不同类型粒子的位置
fluid = mesh.nodedata["position"][mesh.nodedata["tag"] == 0]  # 流体粒子
solid = mesh.nodedata["position"][mesh.nodedata["tag"] == 1]  # 固壁粒子
dummy = mesh.nodedata["position"][mesh.nodedata["tag"] == 2]  # 虚粒子
gate = mesh.nodedata["position"][mesh.nodedata["tag"] == 3] #门粒子

# 创建图形
plt.figure(figsize=(10, 5))
plt.scatter(fluid[:, 0], fluid[:, 1], color='blue', label='Fluid Particles (tag=0)', s=10)
plt.scatter(solid[:, 0], solid[:, 1], color='red', label='Solid Particles (tag=1)', s=10)
plt.scatter(dummy[:, 0], dummy[:, 1], color='green', label='Dummy Particles (tag=2)', s=10)
plt.scatter(gate[:, 0], gate[:, 1], color='black', label='Gate Particles (tag=3)', s=10)

# 设置图形属性
plt.title('Particle Types Visualization')
plt.xlabel('X ')
plt.ylabel('Y ')
plt.xlim(x0, x1)
plt.ylim(y0, y1)
plt.legend()
plt.grid()
plt.show()
'''
