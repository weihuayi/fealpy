import numpy as np
import triangle
import triangle.plot as plot
import matplotlib.pyplot as plt

import scipy.io as sio

from fealpy.mesh.TriangleMesh import TriangleMesh

def corner(x, y, marker=0):
    yv, xv = np.meshgrid(y, x)
    verts = np.concatenate([xv.reshape(-1, 1), yv.reshape(-1, 1)], axis=1)
    segs = np.zeros((4, 2), dtype=np.int32)
    segs[0] = [0, 2]
    segs[1] = [2, 3]
    segs[2] = [3, 1]
    segs[3] = [1, 0]
    m = marker*np.ones((4, 1))
    return verts, segs, m

h=2.0e-4
dd=12*h

x1 = np.array([0, 0.07])
y1 = np.array([0, 0.064])

x0 = x1 + np.array([-dd, dd])
y0 = y1 + np.array([-dd, dd])

x2 = np.array([0.024, 0.054])
y2 = np.array([0.002, 0.062])

verts0, segs0, m0= corner(x0, y0, 1)
verts1, segs1, m1 = corner(x1, y1, 2)
verts2, segs2, m2 = corner(x2, y2, 3)

verts3 = np.array([
        [0.004, 0.025],
        [0.004, 0.035]
    ])
segs3 = np.array([[0, 1]], dtype=np.int32)
m3= np.array([[4]])

verts = np.concatenate([verts0, verts1, verts2, verts3], axis=0)
segs = np.concatenate([segs0, segs1+4, segs2+8, segs3+12], axis=0)
marker = np.concatenate([m0, m1, m2, m3], axis=0)

plsg = {'segments':segs, 'vertices':verts, 'segment_markers':marker}
t = triangle.triangulate(plsg, 'pq30a0.000004')
tmesh = TriangleMesh(t['vertices'], t['triangles'])

tmesh.uniform_refine()

data = {
        'node':tmesh.point, 
        'elem':tmesh.ds.cell + 1, 
        'node_marker': t['vertex_markers']
        }

sio.matlab.savemat('trimesh', data)

#plot.plot(plt.axes(), **t)
#plt.show()
fig = plt.figure()
axes = fig.gca()
tmesh.add_plot(axes)
plt.show()
