

import numpy as np
from fealpy.mesh import MeshFactory as MF

box = [0, 1, 0, 1]
mesh = MF.boxmesh2d(box, nx=1, ny=1, meshtype='tri')

NC = mesh.number_of_cells() # NC = 2

node = mesh.entity('node')
cell = mesh.entity('cell') # shape=(NC, 3)

v0 = node[cell[:, 2]] - node[cell[:, 1]] # (NC, 2) 
v1 = node[cell[:, 0]] - node[cell[:, 2]] # (NC, 2)
v2 = node[cell[:, 1]] - node[cell[:, 0]] # (NC, 2)

nv = np.cross(v1, v2) # shape=(NC, ) 

W = np.array([[0, 1], [-1, 0]]) # (2, 2)

Dlambda = np.zeros((NC, 3, 2), dtype=np.float64)

Dlambda[:, 0, :] = v0@W/nv[:, None] 
Dlambda[:, 1, :] = v1@W/nv[:, None] 
Dlambda[:, 2, :] = v2@W/nv[:, None] 

Dl = mesh.grad_lambda()

print("Dl:", Dl)
print("Dlambda:", Dlambda)
