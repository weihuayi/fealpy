import numpy as np
from fealpy.mesh import TriangleMesh
from .harmonic_map import HarmonicMapData, sphere_harmonic_map


def test_sphere_harmonic_map():
    mesh = TriangleMesh.from_box([0, 1, 0, 1], nx=10, ny=10) 
    didx = mesh.ds.boundary_node_index()
    data = HarmonicMapData()
    sphere_harmonic_map(data)

test_sphere_harmonic_map()
