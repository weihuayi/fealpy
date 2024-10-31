

import numpy as np
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors
import pylab as pl

from fealpy.mesh.TetrahedronMesh import TetrahedronMesh
from meshpy.tet import MeshInfo, build

def unstructure_mesh(h):
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

    return TetrahedronMesh(point, cell)

def structure_mesh():
    point = np.array([
        [-1,-1,-1],
        [ 1,-1,-1], 
        [ 1, 1,-1],
        [-1, 1,-1],
        [-1,-1, 1],
        [ 1,-1, 1], 
        [ 1, 1, 1],
        [-1, 1, 1]], dtype=np.float) 

    cell = np.array([
        [0,1,2,6],
        [0,5,1,6],
        [0,4,5,6],
        [0,7,4,6],
        [0,3,7,6],
        [0,2,3,6]], dtype=np.int)

    return TetrahedronMesh(point, cell)

mesh = unstructure_mesh(0.05)
for i in range(3):
    angle = mesh.dihedral_angle()
    print("max:", np.max(angle))
    print("min:", np.min(angle))
    mesh.uniform_refine()

#ax0 = a3.Axes3D(pl.figure())
#mesh.add_plot(ax0)
#pl.show()
