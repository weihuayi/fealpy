import matplotlib.pyplot  as plt
from mpl_toolkits.mplot3d import Axes3D
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh, LagrangeTriangleMesh


p = 2
box = [0, 1, 0, 1]
mesh = LagrangeTriangleMesh.from_box(box, p, nx=25, ny=25)

node = mesh.entity('node')
cell = mesh.entity('cell')

print("node.shape = ", node.shape)
print("cell.shape = ", cell.shape)

mesh.vtkview(etype='edge')

#fig = plt.figure()
#axes = fig.add_subplot(111, projection='3d')
#mesh.add_plot(axes)
#plt.show()
