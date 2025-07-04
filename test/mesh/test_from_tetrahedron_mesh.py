
from fealpy.backend import backend_manager as bm
from fealpy.mesh import HexahedronMesh, TetrahedronMesh

import pytest
import numpy as np

from from_tetrahedron_mesh_data import *

@pytest.mark.parametrize("backend", ['numpy','jax','pytorch'])
def test_from_tetrahedron_mesh(backend): 
    bm.set_backend(backend)
    node = bm.array([[-1.0, 0.0, 0.0], 
                    [2.2, 0.0, 0.0],
                    [0.11, 3, 0.0],
                    [1.0, 1.0, 4.0]], dtype=bm.float64)
    cell = bm.array([[0, 1, 2, 3]], dtype=bm.int32)
    mesh = TetrahedronMesh(node, cell)
    mesh0 = HexahedronMesh.from_one_tetrahedron()
    mesh1 = HexahedronMesh.from_tetrahedron_mesh(mesh)
    new_node0 = bm.to_numpy(mesh0.entity('node'))
    new_cell0 = bm.to_numpy(mesh0.entity('cell'))
    new_node1 = bm.to_numpy(mesh1.entity('node'))
    new_cell1 = bm.to_numpy(mesh1.entity('cell'))
    np.testing.assert_allclose(new_node0, node0)
    np.testing.assert_allclose(new_cell0, cell0)
    np.testing.assert_allclose(new_node1, node1)
    np.testing.assert_allclose(new_cell1, cell1)


