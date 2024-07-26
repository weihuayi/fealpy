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
dx = 1.25e-4 * 10
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
domain=[0,0.05,0,0.005] * 10
init_domain=[0.0,0.005,0,0.005] * 10

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

for i in range(1):

    #所有粒子
    i_s, j_s = neighbors.idx
    i_s, j_s = i_s[i_s != jnp.max(i_s)], j_s[j_s != jnp.max(j_s)]
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
    dist = space.distance(fwdr_i_j)
    fww_dist = vmap(kernel)(dist)
    
    #更新固壁粒子外推速度(？待确认：是只更新虚粒子的外推速度，固壁粒子仍然为0还是固壁粒子和虚粒子一起更新)
    v_ext = solver.wall_extrapolation(mesh.nodedata, fww_dist, fwi_s, fwj_s)
    mesh.nodedata["v"] = mesh.nodedata["v"].at[mesh.nodedata["tag"] == 1].set(v_ext)
    mesh.nodedata["v"] = mesh.nodedata["v"].at[mesh.nodedata["tag"] == 2].set(mesh.nodedata["v"][d_w_idx])

    #更新半步压力和声速
    mesh.nodedata = solver.change_p(mesh.nodedata, fww_dist, fwi_s, fwj_s, d_w_idx, B=B, rho0=rho0, c1=c1)
    


    #增加门粒子并更新位置
    jnp.set_printoptions(threshold=jnp.inf)
    mesh.nodedata = solver.gate_change(mesh.nodedata, domain, dt=dt, dx=dx, uin=uin)
    '''
    #流体粒子
    fi_s, fj_s = f_neughbors.idx
    fi_s, fj_s = fi_s[fi_s != jnp.max(fi_s)], fj_s[fj_s != jnp.max(fj_s)]    
    fr_i_s, fr_j_s = mesh.nodedata["position"][fi_s], mesh.nodedata["position"][fj_s]
    fdr_i_j = vmap(displacement)(fr_i_s, fr_j_s)
    fdist = space.distance(fdr_i_j)
    fw_dist = vmap(kernel)(fdist)

    fe_s = fdr_i_j / (fdist[:, None] + EPS)
    fgrad_w_dist_norm = vmap(kernel.grad_value)(fdist)
    fgrad_w_dist = fgrad_w_dist_norm[:, None] * fe_s
    

    jnp.set_printoptions(threshold=jnp.inf)
    #solver.free_surface(mesh.nodedata, fi_s, fj_s, fw_dist, fgrad_w_dist, h=h)
    '''

'''
# 获取不同类型粒子的位置
fluid_particles = mesh.nodedata["position"][mesh.nodedata["tag"] == 0]  # 流体粒子
solid_particles = mesh.nodedata["position"][mesh.nodedata["tag"] == 1]  # 固壁粒子
dummy_particles = mesh.nodedata["position"][mesh.nodedata["tag"] == 2]  # 虚粒子
gate_particles = mesh.nodedata["position"][mesh.nodedata["tag"] == 3] #门粒子

# 创建图形
plt.figure(figsize=(10, 5))
plt.scatter(fluid_particles[:, 0], fluid_particles[:, 1], color='blue', label='Fluid Particles (tag=0)', s=10)
plt.scatter(solid_particles[:, 0], solid_particles[:, 1], color='red', label='Solid Particles (tag=1)', s=10)
plt.scatter(dummy_particles[:, 0], dummy_particles[:, 1], color='green', label='Dummy Particles (tag=2)', s=10)
plt.scatter(gate_particles[:, 0], gate_particles[:, 1], color='black', label='Gate Particles (tag=3)', s=10)

# 设置图形属性
plt.title('Particle Types Visualization')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.xlim(x0, x1)
plt.ylim(y0, y1)
plt.legend()
plt.grid()
plt.show()
'''
