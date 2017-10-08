
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri

import matplotlib.colors as colors
import matplotlib.cm as cm 

import numpy as np
import scipy.io as sio

from fealpy.mesh.TriangleMesh import TriangleMesh 
from fealpy.mesh.vtkMeshIO import write_vtk_mesh 

def load_scft_data(f):
    data = sio.loadmat(f)
    point = np.array(data['cmp_mesh']['node'][0, 0], dtype=np.float)
    cell = np.array(data['cmp_mesh']['elem'][0, 0] - 1, dtype=np.int)
    rho = np.array(data['scft']['rho'][0, 0], dtype=np.float)
    tri = TriangleMesh(point, cell)
    tri.pointData['rhoA'] = rho[1, :]
    return tri



f = sys.argv[1]
smesh = load_scft_data(f)


write_vtk_mesh(smesh, 'test.vtk')
point = smesh.point
cell = smesh.ds.cell
rho = smesh.pointData['rhoA']
cellcolor = rho[cell].sum(axis=1)/3

cmap = plt.get_cmap('rainbow')
fig = plt.figure()
axes = fig.add_subplot(1, 1, 1, projection='3d')
m = cm.ScalarMappable(cmap=cm.jet)
m.set_array(cellcolor)
p = axes.plot_trisurf(point[:, 0], point[:, 1], point[:, 2], triangles=cell, linewidth=0.0)
p.set_array(cellcolor)
plt.colorbar(m)
p.autoscale()
axes.set_aspect('equal')
axes.set_axis_off()
plt.show()
