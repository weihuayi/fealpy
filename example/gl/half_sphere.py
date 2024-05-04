import ipdb
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from fealpy.mesh import TriangleMesh
from fealpy.plotter.gl import OpenGLPlotter, OCAMModel


cmodel = OCAMModel()


mesh = TriangleMesh.from_unit_sphere_surface(refine=3)
#mesh = TriangleMesh.from_cylinder_surface()
#mesh = TriangleMesh.from_torus_surface(2, 1, 10, 10)

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

uv = cmodel.sphere_to_cam(node)

#plt.scatter(uv[:, 0], uv[:, 1])
#plt.scatter(cmodel.xc, cmodel.yc)
#plt.show()

uv[:, 0] = (uv[:, 0] - np.min(uv[:, 0]))/(np.max(uv[:, 0])-np.min(uv[:, 0]))
uv[:, 1] = (uv[:, 1] - np.min(uv[:, 1]))/(np.max(uv[:, 1])-np.min(uv[:, 1]))

node = np.hstack((node, uv), dtype=np.float32)
cell = np.array(cell, dtype=np.uint32)
vertices = node[cell].reshape(-1, node.shape[1])

"""
# 定义顶点数据和UV坐标
node = np.array([
    [-0.5, -0.5, 0.0,  0.0, 0.0],  # 左下角
    [ 0.5, -0.5, 0.0,  1.0, 0.0],  # 右下角
    [ 0.5,  0.5, 0.0,  1.0, 1.0],  # 右上角
    [-0.5,  0.5, 0.0,  0.0, 1.0]   # 左上角
], dtype=np.float32)

cell = np.array([
    0, 1, 2,
    2, 3, 0
], dtype=np.uint32)

"""

plotter = OpenGLPlotter()
plotter.add_mesh(vertices, cell=None, texture_path='/home/why/frame1_0.jpg')
#plotter.add_mesh(node, cell=cell, texture_path='/home/why/frame1_0.jpg')
plotter.run()
