import numpy as np

from fealpy.mesh import UniformMesh2d

nx = 100
num_particles = 1000 # 粒子个数
dt = 2e-4 # 时间步长
maxit = 1000  

p_rho = 1
p_vol = (dx*0.5)**2
p_mass = p_rho*p_vol
gravity = 9.8
bound = 5
E = 400
max_step = 9000 # 模拟步数



nx = 100
ny = 100
extent = [0, nx, 0, ny]
hx = 1/nx
hy = 1/ny
mesh = UniformMesh2d(extent, h=(hx, hy))
mv = mesh.function()


dtype = [("position", "float64", (2, )), 
         ("velocity", "float64", (2, )),
         ("rho", "float64"),
         ("mass", "float64"),
         ("vol", "float64"),
         ("C", "float64", (2, 2)),
         ("J", "float64")]


particles = np.zeros(num_particles, dtype=dtype)

particles["position"] = 0.4*np.random.rand(num_particles, 2) + 0.2
particles["velocity"] = np.array([0, -1])
particles["rho"] = 1
particles["rho"] = p_rho
particles["mass"] = p_mass
particles["vol"] = p_vol
particles["J"] = 1



