
import numpy as np

import sys

from meshpy.tet import MeshInfo, build
from fealpy.mesh.TetrahedronMesh  import TetrahedronMesh 
from fealpy.mesh.vtkMeshIO import write_vtk_mesh

example = 2
if example == 1:
    mesh_info = MeshInfo()
    mesh_info.set_points([
        (0,0,0), (1,0,0), (1,1,0), (0,1,0),
        (0,0,1), (1,0,1), (1,1,1), (0,1,1),
        ])
    mesh_info.set_facets([
        [0,1,2,3],
        [4,5,6,7],
        [0,4,5,1],
        [1,5,6,2],
        [2,6,7,3],
        [3,7,4,0],
        ])
    mesh = build(mesh_info, max_volume=0.1**3)

    point = np.array(mesh.points, dtype=np.float)
    cell = np.array(mesh.elements, dtype=np.int)
elif example == 2:
    point = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [-0.5, -0.5, 0.5],
        [-0.5, -0.5,-0.5]], dtype=np.float)

    cell = np.array([
        [0, 1, 2, 3],
        [0, 2, 1, 4]], dtype=np.int)
elif example == 3:
    point = np.array([
        ( 0.0, -np.sqrt(3)/3,          0.0), 
        ( 0.5,  np.sqrt(3)/6,          0.0),
        (-0.5,  np.sqrt(3)/6,          0.0),
        ( 0.0,           0.0, np.sqrt(6)/3)], dtype=np.float)
    cell = np.array([[0, 1, 2, 3]], dtype=np.int)


tmesh = TetrahedronMesh(point, cell)

#isFreePoint = ~tmesh.ds.boundary_point_flag()
#q0 = tmesh.quality()
#for i in range(100):
#    q = tmesh.quality()
#    maxq0 = np.max(q)
#    print(maxq0)
#    print('mean:', np.mean(q))
#    point0 = tmesh.point.copy()
#    grad = tmesh.grad_quality()
#    alpha = 1
#    tmesh.point[isFreePoint] -= alpha*grad[isFreePoint]
#    maxq1 = np.max(tmesh.quality())
#    while (~tmesh.is_valid()) or (maxq1 > maxq0): 
#        alpha /=2 
#        print(alpha)
#        tmesh.point = point0.copy()
#        tmesh.point[isFreePoint] -= alpha*grad[isFreePoint]
#        maxq1 = np.max(tmesh.quality())

q = tmesh.quality()
print(np.max(q))
grad = tmesh.grad_quality()
print(grad)
tmesh.cellData["quality1"] = 1/q
tmesh.pointData["gradient_quality"] = -grad
tmesh.pointData["flag"] = np.array(isFreePoint, dtype=np.int)

write_vtk_mesh(tmesh, 'test.vtk')
