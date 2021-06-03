import sys
import numpy as np
from meshpy.tet import MeshInfo, build
from fealpy.mesh.TetrahedronMesh import TetrahedronMesh 
from fealpy.graph import metis

import scipy.io as sio

h = float(sys.argv[1])
n = int(sys.argv[2])

mesh_info = MeshInfo()

# Set the vertices of the domain [0, 1]^3
mesh_info.set_points([
    (0,0,0), (1,0,0), (1,1,0), (0,1,0),
    (0,0,1), (1,0,1), (1,1,1), (0,1,1),
    ])

# Set the facets of the domain [0, 1]^3
mesh_info.set_facets([
    [0,1,2,3],
    [4,5,6,7],
    [0,4,5,1],
    [1,5,6,2],
    [2,6,7,3],
    [3,7,4,0],
    ])

# Generate the tet mesh
mesh = build(mesh_info, max_volume=(h)**3)

point = np.array(mesh.points, dtype=np.float)
cell = np.array(mesh.elements, dtype=np.int)

tmesh = TetrahedronMesh(point, cell)

# Partition the mesh cells into n parts 
edgecuts, parts = metis.part_mesh(tmesh, nparts=n, entity='cell')

point = tmesh.point
edge = tmesh.ds.edge
face = tmesh.ds.face
cell = tmesh.ds.cell
face2cell = tmesh.ds.face2cell
cell2edge = tmesh.ds.cell_to_edge()
face2edge = tmesh.ds.face_to_edge()
isBdPoint = tmesh.ds.boundary_point_flag()

data = {'Point':point, 'Face':face+1, 'Elem':cell+1,
        'Face2Elem':face2cell+1, 'isBdPoint':isBdPoint, 'Partitions':parts+1}

sio.matlab.savemat('test'+str(n)+'parts'+str(h)+'.mat', data)




