import numpy as np
from fealpy.mesh import Sphere
from fealpy.mesh import SurfaceTriangleMesh


surface = Sphere()
mesh = surface.init_mesh()
N = 8
e = np.zeros(N-1, dtype=np.float)
for i in range(1, N):
    smesh = SurfaceTriangleMesh(mesh, surface, p=i)
    area = smesh.area()
    e[i-1] = np.abs(4*np.pi - sum(area))

print(e[0:-1]/e[1:])

