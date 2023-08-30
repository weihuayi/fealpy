import numpy as np

from fealpy.mesh import UniformMesh2d
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import animation

def p2m(particles):
    xp = particles["position"]/hx
    base  = (xp - 0.5).astype(np.int_)
    fx = xp-base
    w = np.array([0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2])
    stress = -dt * 4 * E * particles['vol'] * (particles['J'] - 1)/hx**2
    affine = particles['mass'][..., None, None] * particles['C']
    affine[..., 0, 0] += stress
    affine[..., 1, 1] += stress
    
    for i in range(3):
        for j in range(3):
            offset = np.array([i, j])
            dp = (offset - fx) * hx
            weight = w[i][:, 0]*w[j][:, 1]
            idx = base + offset
            val = np.einsum('ijk,ik->ij', affine, dp)
            val += particles['mass'][:, None]* particles['velocity'] 
            val *= weight[:, None]
            mv[idx[:, 0], idx[:, 1], :] += val 
            mm[idx[:, 0], idx[:, 1]] += weight * particles['mass']

def set_bound():
    mv[mm >0] /= mm[mm>0, None]
    mv[..., 1] -= dt*gravity
    
    mv00 = mv[0:bound]
    mv00[mv00[..., 0] < 0, 0] = 0.0

    mv01 = mv[nx-bound:]
    mv01[mv01[..., 0] > 0, 0] = 0.0


    mv10 = mv[:, 0:bound]
    mv10[mv10[..., 1] < 0, 1] = 0.0
    
    mv11 = mv[:, nx-bound:]
    mv11[mv11[..., 1] > 0, 1] = 0.0

def m2p(particles):
    xp = particles["position"]/hx
    base  = (xp - 0.5).astype(np.int_) 
    fx = xp-base
    w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
    particles['velocity'] = 0.0
    particles['C'] = 0.0
    for i in range(3):
        for j in range(3):
            offset = np.array([i, j])
            dp = (offset - fx) * hx
            weight = w[i][:, 0]*w[j][:, 1]
            idx = base + offset
            pv = mv[idx[:, 0], idx[:, 1]]
            particles['velocity'] += weight[:, None] * pv
            particles['C'] += np.einsum('p, pi, pj->pij', weight, pv, dp)
    particles['C'] *= 4/hx**2
    particles['position'] += dt * particles['velocity']
    particles['J'] *= 1 + dt * particles['C'].trace(axis1=1,axis2=2)

dt = 1e-4 # 时间步长
maxit = 9000  # 模拟步数

# 背景网格
nx = 100
ny = 100
extent = [0, nx, 0, ny]
hx = 1/nx
hy = 1/ny
mesh = UniformMesh2d(extent, h=(hx, hy))
mv = mesh.function(dim=2) # 网格速度，定义在节点上
mm = mesh.function() # 网格质量，定义在节点上

# 离子参数
num_particles = 1000 # 粒子个数
rho = 1
vol = hx*hy/4.0 
mass = rho*vol
gravity = 9.8
bound = 5
E = 400


# 离子数据结构
dtype = [("position", "float64", (2, )), 
         ("velocity", "float64", (2, )),
         ("rho", "float64"),
         ("mass", "float64"),
         ("vol", "float64"),
         ("C", "float64", (2, 2)),
         ("J", "float64")]
particles = np.zeros(num_particles, dtype=dtype)
print(particles[0]["position"])
# 初始化
particles["position"] = 0.4*np.random.rand(num_particles, 2) + 0.2
particles["velocity"] = np.array([0, -1])
particles["rho"] = rho
particles["mass"] = mass
particles["vol"] = vol
particles["J"] = 1

xx = []
yy = []

for i in range(maxit):
    mm[:] = 0.0
    mv[:] = 0.0
    p2m(particles)
    set_bound()
    m2p(particles)      
    xx.append(list(particles['position'][:,0]))
    yy.append(list(particles['position'][:,1]))


print("finish")
fig, ax = plt.subplots()
sc = ax.scatter([],[])
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('X')
plt.ylabel('Y')

def update(i):
    x = xx[i]
    y = yy[i]
    sc.set_offsets(np.c_[x, y])  # 更新散点图的位置
    return sc,

anim = FuncAnimation(fig, update, frames=maxit, interval=1, blit=True)
#FFwriter = animation.FFMpegWriter(fps=300, extra_args=['-vcodec', 'libx264'])
anim.save('basic_animation.mp4',writer="ffmpeg")
plt.show()
