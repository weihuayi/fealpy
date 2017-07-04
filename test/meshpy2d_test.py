import sys

import numpy as np
from meshpy.triangle import MeshInfo, build
from fealpy.mesh.TriangleMesh import TriangleMesh
from fealpy.graph import metis
import scipy.io as sio

def mesh2dpy(mesh_info, h, n):
    mesh = build(mesh_info, max_volume=(h)**2)
    point = np.array(mesh.points, dtype=np.float)
    cell = np.array(mesh.elements, dtype=np.int)
    tmesh = TriangleMesh(point, cell)

    edgecuts, parts = metis.part_mesh(tmesh, nparts=n, entity='cell')
    point = tmesh.point                                                             
    edge = tmesh.ds.edge                                                            
    cell = tmesh.ds.cell                                                            
    cell2edge = tmesh.ds.cell_to_edge()                                             
    edge2cell = tmesh.ds.edge_to_cell()                                             
    isBdPoint = tmesh.ds.boundary_point_flag()
    data = {'Point':point, 'Edge':edge+1, 'Elem':cell+1,'Edge2Elem':edge2cell+1,
            'isBdPoint':isBdPoint,'Partitions':parts+1}
    result = sio.matlab.savemat('test'+str(n)+'parts'+str(h)+'.mat', data)
    return result

mesh_info = MeshInfo()
mesh_info.set_points([(0,0), (1,0), (1,1), (0,1)])
mesh_info.set_facets([[0,1], [1,2], [2,3], [3,0]])             

h = np.array([0.1, 0.05, 0.025,0.0125,0.00625])
n = np.array([9,16,25,36,49,64])

N = len(h)
M = len(n)

for i in range(len(h)):
    for j in range(len(n)):
        result = mesh2dpy(mesh_info,h[i],n[j])
        



