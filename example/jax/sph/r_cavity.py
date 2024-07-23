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
dx = 1
dy = dx
h = 1.5 * dx
rho0 = 737.54 
n = 0.3083
tau_s = 16834.4
mu0 = 938.4118
c1 = 0.0894
B = 5.914e7
eta = 0.5 #动量方程中的参数
dt = 1e-7

#计算区域
box_size = jnp.array([50, 5 + 4 * dx])

#初始化
mesh = NodeMesh.from_long_rectangular_cavity_domain(dx=dx)
solver = SPHSolver(mesh)
kernel = QuinticKernel(h=h, dim=2)
displacement, shift = space.free()

#生成固壁粒子与虚粒子之间的对应关系
wp = mesh.nodedata["position"][mesh.nodedata["tag"] == 1]  # 固壁粒子
dp = mesh.nodedata["position"][mesh.nodedata["tag"] == 2]  # 虚粒子
dis = solver.compute_distances(wp, dp)
dummy_idx = jnp.argmin(dis, axis=1)

#新增粒子物理量
new_node = jnp.mgrid[0:dx:dx, dy:5:dy].reshape(2, -1).T
new_tag = jnp.full((new_node.shape[0],), 0, dtype=int)
new_mv = jnp.tile(jnp.array([5, 0]), (new_node.shape[0], 1))
new_rho = jnp.ones(new_node.shape[0]) * rho0
new_mass = jnp.ones(new_node.shape[0]) * dx * dy * rho0
mesh.add_node_data(['position', 'tag', 'mv', 'tv', 'dmvdt', 'dtvdt', 'drhodt', 'rho', 'p', 'sound', 'mass'],
                    [new_node, new_tag, new_mv,new_mv,jnp.zeros_like(new_mv),jnp.zeros_like(new_mv),
                    jnp.zeros_like(new_rho), new_rho, jnp.zeros_like(new_rho), jnp.zeros_like(new_rho), new_mass])

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
num_particles = (mesh.nodedata["tag"] != -1).sum()
neighbors = neighbor_fn.allocate(mesh.nodedata["position"], num_particles=num_particles)

for i in range(1):

    i_s, j_s = neighbors.idx
    i_s0 = [(len(mesh.nodedata["position"])-1) if x == len(mesh.nodedata["position"]) else x for x in i_s]
    j_s0 = [(len(mesh.nodedata["position"])-1) if x == len(mesh.nodedata["position"]) else x for x in j_s]
    r_i_s, r_j_s = mesh.nodedata["position"][jnp.array(i_s0)], mesh.nodedata["position"][jnp.array(j_s0)]
    dr_i_j = vmap(displacement)(r_i_s, r_j_s)
    dist = space.distance(dr_i_j)
    w_dist = vmap(kernel)(dist)

    e_s = dr_i_j / (dist[:, None] + EPS)
    grad_w_dist_norm = vmap(kernel.grad_value)(dist)
    grad_w_dist = grad_w_dist_norm[:, None] * e_s

    jnp.set_printoptions(threshold=jnp.inf)
    

'''
# 获取不同类型粒子的位置
fluid_particles = mesh.nodedata["position"][mesh.nodedata["tag"] == 0]  # 流体粒子
solid_particles = mesh.nodedata["position"][mesh.nodedata["tag"] == 1]  # 固壁粒子
virtual_particles = mesh.nodedata["position"][mesh.nodedata["tag"] == 2]  # 虚粒子
# 创建图形
plt.figure(figsize=(10, 5))
plt.scatter(fluid_particles[:, 0], fluid_particles[:, 1], color='blue', label='Fluid Particles (tag=0)', s=10)
plt.scatter(solid_particles[:, 0], solid_particles[:, 1], color='red', label='Solid Particles (tag=1)', s=10)
plt.scatter(virtual_particles[:, 0], virtual_particles[:, 1], color='green', label='Virtual Particles (tag=2)', s=10)

# 设置图形属性
plt.title('Particle Types Visualization')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.xlim(0, box_size[0])
plt.ylim(-5, box_size[1])
plt.legend()
plt.grid()
plt.show()
'''
