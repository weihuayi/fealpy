import ipdb
import numpy as np
import matplotlib.pyplot as plt
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh.triangle_mesh import TriangleMesh
from fealpy.experimental.tests.mesh.triangle_mesh_data import *

# 测试不同的后端
backends = ['numpy', 'pytorch', 'jax', 'mindspore']

class TestTriangleMeshInterfaces:
    @pytest.fixture(scope="class", params=backends)
    def backend(self, request):
        bm.set_backend(request.param)
        return request.param

    @pytest.mark.parametrize("meshdata", init_mesh_data)
    def test_init(self, meshdata, backend):

        node = bm.from_numpy(meshdata['node'])
        cell = bm.from_numpy(meshdata['cell'])

        mesh = TriangleMesh(node, cell)

        assert mesh.number_of_nodes() == meshdata["NN"] 
        assert mesh.number_of_edges() == meshdata["NE"] 
        assert mesh.number_of_faces() == meshdata["NF"] 
        assert mesh.number_of_cells() == meshdata["NC"] 
        
        face2cell = mesh.face_to_cell()
        np.testing.assert_array_equal(bm.to_numpy(face2cell), meshdata["face2cell"])
'''
    @pytest.mark.parametrize("meshdata", from_box_data)
    def test_from_box():
        mesh = TriangleMesh.from_box(nx=2, ny=2)
        assert mesh.node.shape == (9, 2)
        assert mesh.cell.shape == (8, 3)

        node = mesh.entity('node')
        cell = mesh.entity('cell')
        
        if False:
            tmesh = TMesh(node, cell)
            fig = plt.figure()
            axes = fig.gca()
            tmesh.add_plot(axes)
            tmesh.find_node(axes, showindex=True)
            tmesh.find_cell(axes, showindex=True)
            plt.show()
'''
if __name__ == "__main__":
    pytest.main()
