import numpy as np

def meshpy3d(points, facets, h, 
        hole_points=None, 
        facet_markers=None, 
        point_markers=None):

    from meshpy.tet import MeshInfo, build
    from .TetrahedronMesh import TetrahedronMesh

    mesh_info = MeshInfo()
    mesh_info.set_points([
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
        ])

    mesh_info.set_facets([
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 4, 5, 1],
        [1, 5, 6, 2],
        [2, 6, 7, 3],
        [3, 7, 4, 0],
    ])
    mesh = build(mesh_info, max_volume=h**3/6.0)

    node = np.array(mesh.points, dtype=np.float64)
    cell = np.array(mesh.elements, dtype=np.int_)

    return TetrahedronMesh(node, cell)
