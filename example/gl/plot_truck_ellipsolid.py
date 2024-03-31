
import numpy as np

from fealpy.mesh import TriangleMesh
from fealpy.plotter.gl import OpenGLPlotter

cam = np.array([
    [ 8.35/2.0, -3.47/2.0, 1.515-3.0], # 右前
    [-8.25/2.0, -3.47/2.0, 1.505-3.0], # 右后
    [-17.5/2.0,       0.0, 1.295-3.0], # 后
    [-8.35/2.0,  3.47/2.0, 1.495-3.0], # 左后
    [ 8.35/2.0,  3.47/2.0, 1.495-3.0], # 左前
    [ 17.5/2.0,       0.0, 1.345-3.0]  # 前
    ], dtype=np.float64)

mesh, U, V = TriangleMesh.from_ellipsoid_surface(80, 800, 
        radius=(17.5, 3.47, 3), 
        theta=(np.pi/2, np.pi/2+np.pi/3),
        returnuv=True)

node = mesh.entity('node')
cell = mesh.entity('cell')

U = (U - np.min(U))/(np.max(U)-np.min(U))
V = (V - np.min(V))/(np.max(V)-np.min(V))
nodes = np.hstack((node, V.reshape(-1, 1), U.reshape(-1, 1)), dtype=np.float32)

#nodes = np.array(node, dtype=np.float32)
cells = np.array(cell, dtype=np.uint32)

plotter = OpenGLPlotter()
plotter.load_mesh(nodes, cells)
plotter.load_texture('/home/why/frame1_0.jpg')
plotter.run()
