import numpy as np

from fealpy.mesh import UniformMesh2d


num_particles = 1000 # 粒子个数
dt = 2e-4 # 时间步长
maxit = 1000  # 模拟步数


mesh = UniformMesh2d()


dtype = [("position", "float64", (2, )), 
         ("velocity", "float64", (2, )),
         ("rho", "float64"),
         ("vol", "float64"),
         ("C", "float64", (2, 2)),
         ("J", "float64")]


particles = np.zeros(num_particles, dtype=dtype)

particles["position"] = 0.4*np.random.rand(num_particles, 2) + 0.2
particles["velocity"] = np.array([0, -1])
particles["rho"] = 1
particles["vol"] = 



