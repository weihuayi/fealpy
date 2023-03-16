
from matplotlib import pyplot as plt

from fealpy.mesh import MeshFactory as Mf
from fealpy.pinn.sampler import TetrahedronMeshSampler


mesh = Mf.boxmesh3d([0, 1, 0, 1, 0, 1], nx=1, ny=1, nz=1, meshtype='tet')
sampler = TetrahedronMeshSampler(100, mesh)
result = sampler.run()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

mesh.find_edge(ax)
ax.scatter(result[:, 0], result[:, 1], result[:, 2])

plt.show()
