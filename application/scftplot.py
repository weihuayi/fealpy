import sys

#import matplotlib.colors as colors
#import matplotlib.cm as cm 
#import pylab as pl
#import mpl_toolkits.mplot3d as a3

from mayavi import mlab 
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

surf1 = mlab.triangular_mesh(point[:, 0], point[:, 1], point[:, 2], cell,
        scalars=rho, mask= rho>0.5)
lut = np.array([(0, 0, 255, 0), (255, 0, 0, 255)])
surf1.module_manager.scalar_lut_manager.lut.table = lut

surf2 = mlab.triangular_mesh(0.8*point[:, 0], 0.8*point[:, 1], 0.8*point[:, 2],
        cell, color=(1, 1, 1))

#lut = np.array([(0, 0, 255, 120), (0, 0, 255, 120)])
#surf2.module_manager.scalar_lut_manager.lut.table = lut

mlab.draw()
mlab.view(40, 85)
mlab.show()

#colorsList = [(0, 0, 0), (1, 0, 0)]
#cmap = colors.ListedColormap(colorsList)
#axes = a3.Axes3D(pl.figure())
#axes.plot_trisurf(point[:, 0], point[:, 1], point[:, 2], triangles=cell)
#axes.set_aspect('equal')
#axes.set_axis_off()
#pl.show()
