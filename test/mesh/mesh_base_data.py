import numpy as np

sub_mesh_class1 = [
    {"class_name": "TriangleMesh",
     "sub_class_name": "IntervalMesh"},
    {"class_name": "TetrahedronMesh",
     "sub_class_name": "TriangleMesh"},
    {"class_name": "QuadrangleMesh",
     "sub_class_name": "IntervalMesh"},
    {"class_name": "HexahedronMesh",
     "sub_class_name": "QuadrangleMesh"},
]

sub_mesh_class2 = [
    # {"class_name": "UniformMesh2d",
    #  "sub_class_name": "IntervalMesh"},
    {"class_name": "UniformMesh3d",
     "sub_class_name": "QuadrangleMesh"},
]