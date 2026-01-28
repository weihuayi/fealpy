from typing import Type
import matplotlib.pyplot as plt

from fealpy.cgraph.mesh import BoxMinusCylinder

#mesh=TriangleMesh.from_square_hole(box=[0,1,0,2], scenter=[0.2,0.2], r=0.1 , h = 0.02)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# mesh.add_plot(ax)
# plt.show()




# domin = [0, 1,0, 1]
# X = 0.5
# Y = 0.5
# r = 0.2
# h = 0.05
# mesh = SquareHole.run("triangle", domin, X, Y, r, h)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# mesh.add_plot(ax)
# plt.show()

# domain=[0, 1, 0, 1, 0, 1]
# X = 0.0
# Y = 0.0
# Z = 0.0
# r = 0.5
# h = 0.1
# mesh = CubeSphericalHole.run("tetrahedron", domain, X, Y, Z, r, h)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# mesh.add_plot(ax)
# plt.show()

domain=[0, 1, 0, 1, 0, 1]
X = 1
Y = 1
Z = 0.5
ax = 1.0
ay = 0.0
az = 0.0
cyl_radius = 0.5
cyl_height = None
h = 0.1
mesh = BoxMinusCylinder.run("tetrahedron", domain, X, Y, Z, ax, ay, az, cyl_radius, cyl_height, h)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
mesh.add_plot(ax)
plt.show()