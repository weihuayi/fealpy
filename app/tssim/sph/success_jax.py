import jax.numpy as jnp
from jax import jit
from fealpy.cfd.sph import partition
from fealpy.cfd.sph.jax_md.partition import Sparse
from fealpy.cfd.sph.particle_kernel_function import QuinticKernel
from jax_md import space
from jax import ops, vmap, lax
import matplotlib.pyplot as plt

dx = 0.025
dy = 0.025
rho0 = 1000
H = 0.92 * jnp.sqrt(dx**2 + dy**2)
dt = 0.001
c0 = 10
g = jnp.array([0.0, -9.8])
gamma = 7
alpha = 0.3
maxstep = 1
box_size = jnp.array([5.0,5.0])

def initial_position(dx, dy):
    pp = jnp.mgrid[dx:1+dx:dx, dy:2+dy:dy].reshape(2, -1).T
    
    # 墙体粒子
    wbp = jnp.mgrid[0:4+dx:dx, 0:dy:dy].reshape(2, -1).T
    wlp = jnp.mgrid[0:dx:dx, dy:4+dy:dy].reshape(2, -1).T
    wrp = jnp.mgrid[4:4+dx/2:dx, dy:4+dy:dy].reshape(2, -1).T
    wp = jnp.vstack((wbp, wlp, wrp))
     
    # 虚粒子
    dbp = jnp.mgrid[-3*dx:4+4*dx:dx, -3*dy:-dy:dy].reshape(2, -1).T
    wlp = jnp.mgrid[-3*dx:-dx:dx, 0:4+dy:dy].reshape(2, -1).T
    wrp = jnp.mgrid[4+dx:4+dx*4:dx, 0:4+dy:dy].reshape(2, -1).T
    dp = jnp.vstack((dbp, wlp, wrp))

    return pp, wp, dp

# 初始化
pp, wp, dp = initial_position(dx, dy)
num_particles = pp.shape[0] + wp.shape[0] + dp.shape[0]

#创建标签
tag = jnp.zeros(num_particles, dtype=int)
tag = tag.at[pp.shape[0]:pp.shape[0] + wp.shape[0]].set(1) 
tag = tag.at[pp.shape[0] + wp.shape[0]:].set(2)

nodedata = {
    "position": jnp.vstack((pp, wp, dp)),
    "velocity": jnp.zeros((num_particles, 2)),
    "rho": jnp.full(num_particles, rho0),
    "mass": jnp.full(num_particles, 2 * rho0 / pp.shape[0]),
    "pressure": jnp.zeros(num_particles),
    "sound": jnp.zeros(num_particles),
    "tag": tag
}

kernel = QuinticKernel(h=H, dim=2)
displacement, shift = space.periodic(side=box_size)
'''
# 可视化
colors = ['red', 'green', 'blue']  # 分别对应不同的标签
plt.figure(figsize=(8, 6))
plt.scatter(nodedata['position'][:, 0], nodedata['position'][:, 1], 
            c=nodedata['tag'], cmap=plt.cm.get_cmap('RdYlGn', 3), s=5)
plt.colorbar(ticks=[0, 1, 2], label='Particle Type')
plt.clim(-0.5, 2.5)  
plt.grid(True)
ax = plt.gca()
ax.set_aspect('equal')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Particle Distribution by Type')
plt.show()
'''
neighbor_fn = partition.neighbor_list(
    displacement,
    box_size,
    r_cutoff=QuinticKernel(h=H, dim=2).cutoff_radius,
    backend="jaxmd_vmap",
    capacity_multiplier=1.25,
    mask_self=False,
    format=Sparse,
    num_particles_max=nodedata["position"].shape[0],
    num_partitions=nodedata["position"].shape[0],
    pbc=[True, True, True],
)

neighbors = neighbor_fn.allocate(nodedata["position"], num_particles=nodedata["position"].shape[0])


for i in range(maxstep):
    print("i:", i)
    neighbors = neighbors.update(mesh.nodedata["position"], num_particles=mesh.nodedata["position"].shape[0])
    
    '''
    idx = find_neighbors_within_distance(nodedata["position"], 2 * H)
    idx = [jnp.array(neighbors) for neighbors in idx]
    
    # 更新半步密度和半步质量
    rho_0 = nodedata['rho'].copy()
    F_rho_0 = continue_equation(nodedata, idx)
    
    rho_half = rho_0 + 0.5 * dt * F_rho_0

    # 更新半步速度
    velocity_0 = nodedata['velocity'].copy()
    F_velocity_0 = momentum_equation(nodedata, idx)
    velocity_half = velocity_0 + 0.5 * dt * F_velocity_0
    velocity_half[nodedata['isDd']] = wall_extrapolation(nodedata, idx, velocity_half)[dummy_idx]

    # 更新半步位置
    position_0 = nodedata['position'].copy()
    F_position_0 = change_position(nodedata, idx)
    position_half = position_0 + 0.5 * dt * F_position_0

    nodedata['rho'] = rho_half
    nodedata['velocity'] = velocity_half
    nodedata['position'] = position_half
    state_equation(nodedata)

    # 更新密度
    F_rho_1 = continue_equation(nodedata, idx)
    rho_1 = rho_0 + 0.5 * dt * F_rho_1

    # 更新速度
    F_velocity_1 = momentum_equation(nodedata, idx)
    velocity_1 = velocity_0 + 0.5 * dt * F_velocity_1

    # 更新半步位置
    F_position_1 = change_position(nodedata, idx)
    position_1 = position_0 + 0.5 * dt * F_position_1

    nodedata['rho'] = 2 * rho_1 - rho_0 
    nodedata['velocity'] = 2 * velocity_1 - velocity_0
    nodedata['velocity'][nodedata['isDd']] = wall_extrapolation(nodedata, idx, nodedata['velocity'])[dummy_idx]
    nodedata['position'] = 2 * position_1 - position_0
    state_equation(nodedata)

    if i % 30 == 0 and i != 0:
        nodedata['rho'] = rein_rho(nodedata, idx)
    draw(nodedata, i)
    '''
