import numpy as np
from fealpy import function_space
from fealpy import LaplaceSymetricForm, MassForm
from fealpy import TriangleMesh

import matplotlib.pyplot as plt
import sys


degree = int(sys.argv[1])
mesh = TriangleMesh(None, None)
mesh.point = np.array([
    [0,0],
    [1,0],
    [0,1]], dtype=np.float)
mesh.cell = np.array([[0, 1, 2]], dtype=np.int)
V = function_space(mesh, 'Lagrange', degree)
a  = LaplaceSymetricForm(V, 9)
m = MassForm(V, 6)
A = a.get_matrix()
M = m.get_matrix()
points = V.interpolation_points()
cell2dof = V.cell_to_dof()
points = points[cell2dof[0],:]

np.savetxt('stff'+str(degree), A.toarray(), fmt='%.5e', delimiter='\t')
np.savetxt('mass'+str(degree), M.toarray(), fmt='%.5e', delimiter='\t')

fig = plt.figure()
axes = fig.gca()
mesh.add_plot(axes)
axes.plot(points[:,0], points[:,1], 'ro')
for i in range(points.shape[0]):
    axes.annotate(str(i), points[i,:], fontsize=24) 
plt.show()

