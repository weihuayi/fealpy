import jax
import jax.numpy as jnp
from fealpy.jax.sph.node_mesh import NodeMesh
import matplotlib.pyplot as plt
# 启用 float64 支持
jax.config.update("jax_enable_x64", True)

EPS = jnp.finfo(float).eps
dx = 0.02
dy = 0.2
m = dx * dy #质量
h = dx #平滑长度
Vmax = 1 #预期最大速度
c0 =10 * Vmax #声速
rho0 = 1 #参考密度
gamma = 1
p0 = c0**2*rho0/gamma #参考压力
nu = 1 #运动粘度
g = jnp.array([0.0, 0.0])
T = 5
dt = 0.0004

fluid_node_set, dummy_node_set = NodeMesh.from_tgv_domain(dx, dy)

# 初始化数组
position = jnp.vstack((fluid_node_set.node, dummy_node_set.node))
NN = position.shape[0]
mv = jnp.zeros((NN, 2), dtype=jnp.float64)
tv = jnp.zeros((NN, 2), dtype=jnp.float64)
mass = jnp.ones(NN, dtype=jnp.float64) * m
rho = jnp.ones(NN, dtype=jnp.float64) * rho0
volume = jnp.zeros(NN, dtype=jnp.float64)
pressure = jnp.zeros(NN, dtype=jnp.float64)
external_force = jnp.zeros((NN, 2), dtype=jnp.float64) * g
isFd = jnp.zeros(NN, dtype=bool)
isDd = jnp.zeros(NN, dtype=bool)

#计算初始速度
x = position[:,0]
y = position[:,1]
u0 = -jnp.cos(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y)
v0 = jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi *y)
mv = mv.at[:,0].set(u0)
mv = mv.at[:,1].set(v0)
tv = mv

#设置粒子标签
isFd = isFd.at[:fluid_node_set.node.shape[0]].set(True)
isDd = isDd.at[fluid_node_set.node.shape[0]:].set(True)
'''
plt.scatter(position[isFd, 0], position[isFd, 1], c='blue', label='Fluid Nodes')
plt.scatter(position[isDd, 0], position[isDd, 1], c='red', label='Dummy Nodes')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Particle Tag')
plt.show()
'''
#放入元组中存储
mesh.nodedate = {
    "position": position,
    "tag": isFd,
    "mv": mv,
    "tv": tv,
    "drhodt": drhodt,
    "dudt": jnp.zeros_like(mv),
    "dvdt": jnp.zeros_like(mv),
    "rho": rho,
    "p": p,
    "mass": mass,
    "eta": eta,
}




'''
for i in state:
    print(i)

fig, ax = plt.subplots()
fluid_node_set.add_plot(ax, color='red', markersize=25)
dummy_node_set.add_plot(ax,color='blue', markersize=25)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('NodeSet from TGV Domain')
plt.show()
'''
