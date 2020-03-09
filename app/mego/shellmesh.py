import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from fealpy.mesh import TriangleMesh
from fealpy.mesh.implicit_surface import Sphere
from fealpy.mesh.implicit_curve import Circle

c = np.zeros(3)
pi = np.pi
rc = 15
alpha = pi*40/180

r = rc/np.sin(alpha)
h = r*np.cos(alpha)

node = np.zeros((7, 3), dtype=np.float)
theta = np.linspace(0, 2*pi, num=6, endpoint=False)

node[0, :] = [0, 0, r]
node[1:, 0] = 15*np.cos(theta)
node[1:, 1] = 15*np.sin(theta)
node[1:, 2] = h

cell = np.array([
    (0, 1, 2), (0, 2, 3), (0, 3, 4), 
    (0, 4, 5), (0, 5, 6), (0, 6, 1)
    ], dtype=np.int32)

surface = Sphere(c, r)
circle = Circle(radius=rc)

mesh = TriangleMesh(node, cell)
for i in range(8):
    mesh.uniform_refine() 
    isBdNode = mesh.ds.boundary_node_flag()
    r0 = np.sqrt(np.sum(mesh.node[isBdNode, 0:2]**2, axis=1))
    mesh.node[isBdNode, 0:2] *=rc/r0.reshape(-1, 1)

mesh.node, _ = surface.project(mesh.node)

mesh0 = TriangleMesh(mesh.node.copy(), mesh.ds.cell.copy())
mesh0.node *=(r+0.03)/r

mesh1 = TriangleMesh(mesh.node.copy(), mesh.ds.cell.copy())
mesh1.node *=(r-0.03)/r

fig = plt.figure()
axes = fig.add_subplot(111, projection='3d')
#mesh.add_plot(axes)
mesh0.add_plot(axes)
mesh1.add_plot(axes)
plt.show()

