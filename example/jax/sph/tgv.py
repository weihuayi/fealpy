import jax
import enum
import jax.numpy as jnp
from fealpy.jax.sph.node_mesh import NodeMesh
from fealpy.jax.sph.kernel_function import QuinticKernel
from jax_md import space
from jax import ops, vmap
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True) # 启用 float64 支持

EPS = jnp.finfo(float).eps
dx = 0.02
dy = 0.02
h = dx #平滑长度
Vmax = 1 #预期最大速度
c0 =10 * Vmax #声速
rho0 = 1 #参考密度
eta0 = 0.01
gamma = 1 #用于压力计算
p0 = c0**2*rho0/gamma #参考压力
pb = 0 #背景压力
nu = 1 #运动粘度
T = 5 #终止时间
dt = 0.0004 #时间间隔
t_num = int(T / dt)
box_size = jnp.array([1.0,1.0]) #模拟区域

mesh = NodeMesh.from_tgv_domain(box_size, dx)

position = mesh.node
NN = position.shape[0]
mv = jnp.zeros((NN, 2), dtype=jnp.float64)
tv = jnp.zeros((NN, 2), dtype=jnp.float64)
volume = jnp.ones(NN, dtype=jnp.float64) * dx*dy
rho = jnp.ones(NN, dtype=jnp.float64) * rho0
mass = jnp.ones(NN, dtype=jnp.float64) * dx*dy * rho0
eta = jnp.ones(NN, dtype=jnp.float64) * eta0
external_force = jnp.zeros_like(position)
displacement, shift = space.periodic(side=box_size)

#状态方程更新压力
def tait_eos(rho, p0=p0, rho0=rho0, gamma=gamma, pb=pb):
    return p0 * ((rho/rho0)**gamma - 1) + pb
p = tait_eos(rho)

#设置标签
class Tag(enum.IntEnum):
    fill_value = -1 #当粒子数量变化时，用 -1 填充
    fluid = 0
    solid_wall = 1 #固壁墙粒子
    moving_wall = 2 #移动墙粒子
    dirichlet_wall = 3 #温度边界条件的狄利克雷墙壁粒子
def tag_set(pos):
    tag = jnp.full(len(pos), Tag.fluid, dtype=int)
    return tag
tag = tag_set(position)

#计算初始速度
x = position[:,0]
y = position[:,1]
u0 = -jnp.cos(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y)
v0 = jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi *y)
mv = mv.at[:,0].set(u0)
mv = mv.at[:,1].set(v0)
tv = mv

#初始化 nodedate 字典
mesh.nodedate = {
    "position": position,
    "tag": tag,
    "mv": mv,
    "tv": tv,
    "dmvdt": jnp.zeros_like(mv),
    "dtvdt": jnp.zeros_like(mv),
    "drhodt": jnp.zeros_like(rho),
    "rho": rho,
    "p": p,
    "mass": mass,
    "eta": eta,
}

kernel = QuinticKernel(h=h,dim=2)

for i in range(t_num):
    '''
    solver
    '''

    mesh.nodedata['mv'] += dt*mesh.nodedata["dmvdt"]
    mesh.nodedata['tv'] = mesh.nodedata['mv'] + 0.5*dt*mesh.nodata['mv']


    mesh.nodedata["position"] = shift_fn(mesh.nodedata["position"], 1.0 * dt * mesh.nodedata["tv"])
    
    '''
    更新neighbor
    '''
    '''
    forward
    '''
    '''
    不需要边界处理
    '''






