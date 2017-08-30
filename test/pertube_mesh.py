import numpy as np
import sys
from fealpy.mesh.TriangleMesh import TriangleMesh 
import matplotlib.pyplot as plt

from fealpy.mesh.meshio import load_mat_mesh, write_mat_mesh 

alpha = 0.3 

for i in range(4):
    mesh = load_mat_mesh('../data/square'+str(i+2)+'.mat')
    area = mesh.area()
    h = np.sqrt(area.sum()/area.shape[0])
    point = mesh.point
    isInPoint = ~mesh.ds.boundary_point_flag()
    NN = isInPoint.sum()
    p = alpha*(-1 + 2*np.random.rand(NN, 2))*h
    point[isInPoint, :] += p
    area = mesh.area()
    print(np.all(area > 0))
    write_mat_mesh(mesh, '../data/sqaureperturb'+str(i+2)+'.'+str(alpha) + '.mat')


fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
axes.set_axis_on()
plt.show()
