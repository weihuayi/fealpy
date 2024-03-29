import numpy as np

from fealpy.mesh import TriangleMesh
from fealpy.plotter.gl import OpenGLPlotter


# Given parameters
ss = np.array([5, -576.3797, 0, 0.0007185556, -3.39907e-07, 5.242219e-10])
xc, yc = 559.875074, 992.836922
c, d, e = 1.000938, 0.000132, -0.000096
width, height = 1920, 1080

# Example usage with dummy 3D points (M)
M_example = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Convert
m_converted = world2cam(M_example, ss, xc, yc, width, height, c, d, e)
m_converted


#mesh, U, V = TriangleMesh.from_ellipsoid_surface(80, 800, 
#        radius=(4, 2, 1), 
#        theta=(np.pi/2, np.pi/2+np.pi/3),
#        returnuv=True)

mesh = TriangleMesh.from_unit_sphere_surface()
node = mesh.entity('node')
cell = mesh.entity('cell')

bc = mesh.entity_barycenter('cell')
cell = cell[bc[:, 2] > 0]

NN = len(node)
isValidNode = np.zeros(NN, dtype=np.bool_)
isValidNode[cell] = True
node = node[isValidNode]
idxMap = np.zeros(NN, dtype=cell.dtype)
idxMap[isValidNode] = range(isValidNode.sum())
cell = idxMap[cell]

#U = (U - np.min(U))/(np.max(U)-np.min(U))
#V = (V - np.min(V))/(np.max(V)-np.min(V))
#nodes = np.hstack((node, V.reshape(-1, 1), U.reshape(-1, 1)), dtype=np.float32)

nodes = np.array(node, dtype=np.float32)
cells = np.array(cell, dtype=np.uint32)

plotter = OpenGLPlotter()
plotter.load_mesh(nodes, cells)
#plotter.load_texture('/home/why/we.jpg')
plotter.run()
