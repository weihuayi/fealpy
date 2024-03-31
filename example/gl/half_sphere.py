import ipdb
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from fealpy.mesh import TriangleMesh
from fealpy.plotter.gl import OpenGLPlotter, OCAMModel 


cmodel = OCAMModel()


image = Image.open('/home/why/frame1_0.jpg')
idata = np.asarray(image)
uimage = cmodel.undistort(np.asarray(image))

plt.imshow(uimage)
plt.title("Undistorted Image")
plt.show()


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
uv = cmodel.world2cam_fast(node)

plt.scatter(uv[:, 0], uv[:, 1])
plt.scatter(cmodel.xc, cmodel.yc)
plt.show()

uv[:, 0] = (uv[:, 0] - np.min(uv[:, 0]))/(np.max(uv[:, 0])-np.min(uv[:, 0]))
uv[:, 1] = (uv[:, 1] - np.min(uv[:, 1]))/(np.max(uv[:, 1])-np.min(uv[:, 1]))

nodes = np.hstack((node, uv), dtype=np.float32)
#nodes = np.array(node, dtype=np.float32)
cells = np.array(cell, dtype=np.uint32)

plotter = OpenGLPlotter()
plotter.load_mesh(nodes, cells)
plotter.load_texture('/home/why/frame1_0.jpg')
plotter.run()
