import numpy as np
import matplotlib.pyplot as plt
import pytest

from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.edge_mesh import EdgeMesh
from fealpy.experimental.tests.mesh.edge_mesh_data import *



class EdgeMeshInterfaces:

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_mesh_data)
    def test_init(self, meshdata, backend):
        bm.set_backend(backend)
        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])

        mesh = EdgeMesh(node, cell)

        assert mesh.number_of_nodes() == meshdata["NN"] 
        assert mesh.number_of_edges() == meshdata["NE"] 
        assert mesh.number_of_cells() == meshdata["NC"] 
        
        face2cell = mesh.face_to_cell()
        face2cell = face2cell.toarray()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), meshdata["face2cell"])

#if __name__ == "__main__":
#    pytest.main(["test_edge_mesh.py"])



    






