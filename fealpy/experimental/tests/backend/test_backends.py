import ipdb
import numpy as np 
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.tests.backend.backend_data import *

# 测试不同的后端
backends = ['numpy', 'pytorch', 'jax']

class TestBackendInterfaces:
    @pytest.fixture(scope="class", params=backends)
    def backend(self, request):
        bm.set_backend(request.param)
        return request.param

    @pytest.mark.parametrize("data", unique_data)
    def test_unique(self, backend, data):
        name = ('result', 'indices', 'inverse', 'counts')
        indata = data["input"]
        result = data["result"]
        test_result = bm.unique(bm.from_numpy(indata), return_index=True, 
                             return_inverse=True, 
                             return_counts=True, axis=0) 
        
        for r, e, s in zip(result, test_result, name):
            np.testing.assert_array_equal(r, bm.to_numpy(e), 
                                          err_msg=f"The {s} of `bm.unique` function is not equal to real result in backend {backend}")

    @pytest.mark.parametrize("data", multi_index_data)
    def test_multi_index_matrix(self, backend, data):
        p = data["p"]
        dim = data["dim"]
        result = data["result"]
        test_result = bm.multi_index_matrix(p, dim)
        np.testing.assert_array_equal(result, bm.to_numpy(test_result), 
                                      err_msg=f" `bm.multi_index_matrix` function is not equal to real result in backend {backend}")

    @pytest.mark.parametrize("data", triangle_mesh2d_data)
    def test_edge_length(self, backend, data):
        edge = bm.from_numpy(data["edge"])
        node = bm.from_numpy(data["node"])
        result = data["edge_length"]
        test_result = bm.edge_length(edge, node)
        np.testing.assert_array_equal(result, bm.to_numpy(test_result), 
                                      err_msg=f" `bm.edge_length` function is not equal to real result in backend {backend}")
    
    @pytest.mark.parametrize("data", triangle_mesh2d_data)
    def test_edge_normal(self, backend, data):
        edge = bm.from_numpy(data["edge"])
        node = bm.from_numpy(data["node"])
        result = data["edge_normal"]
        test_result = bm.edge_normal(edge, node, unit=False)
        np.testing.assert_array_equal(result, bm.to_numpy(test_result), 
                                      err_msg=f" `bm.edge_normal` function is not equal to real result in backend {backend}")
    
    @pytest.mark.parametrize("data", triangle_mesh2d_data)
    def test_edge_tangent(self, backend, data):
        edge = bm.from_numpy(data["edge"])
        node = bm.from_numpy(data["node"])
        result = data["edge_tangent"]
        test_result = bm.edge_tangent(edge, node, unit=False)
        np.testing.assert_array_equal(result, bm.to_numpy(test_result), 
                                     err_msg=f" `bm.edge_tangent` function is not equal to real result in backend {backend}")

    @pytest.mark.parametrize("data", triangle_mesh2d_data)
    def test_bc_to_points(self, backend, data):
        bcs = bm.from_numpy(data["bcs"])
        cell = bm.from_numpy(data["cell"])
        node = bm.from_numpy(data["node"])
        result = data["bc_to_points"]
        test_result = bm.bc_to_points(bcs, node, cell)
        np.testing.assert_array_equal(result, bm.to_numpy(test_result), 
                                     err_msg=f" `bm.bc_to_points` function is not equal to real result in backend {backend}")
    
    @pytest.mark.parametrize("data", triangle_mesh2d_data)
    def test_barycenter(self, backend, data):
        edge = bm.from_numpy(data["edge"])
        cell = bm.from_numpy(data["cell"])
        node = bm.from_numpy(data["node"])
        result_cell = data["cell_barycenter"]
        result_edge = data["edge_barycenter"]
        test_result_cell = bm.barycenter(cell, node)
        test_result_edge = bm.barycenter(edge, node)
        np.testing.assert_array_equal(result_cell, bm.to_numpy(test_result_cell), 
                                     err_msg=f" cell of `bm.barycenter` function is not equal to real result in backend {backend}")
        np.testing.assert_array_equal(result_edge, bm.to_numpy(test_result_edge), 
                                     err_msg=f" edge of `bm.barycenter` function is not equal to real result in backend {backend}")
    
    @pytest.mark.parametrize("data", triangle_mesh2d_data)
    def test_simple_measure(self, backend, data):
        cell = bm.from_numpy(data["cell"])
        node = bm.from_numpy(data["node"])
        result = data["simple_measure"]
        test_result = bm.simplex_measure(cell, node)
        np.testing.assert_almost_equal(result, bm.to_numpy(test_result),decimal=15, 
                                     err_msg=f" `bm.simple_measure` function is not equal to real result in backend {backend}")
    
    @pytest.mark.parametrize("data", triangle_mesh2d_data)
    def test_simple_shape_function(self, backend, data):
        bcs = bm.from_numpy(data["bcs"])
        p = data["p"]
        result = data["simple_shape_function"]
        test_result = bm.simplex_shape_function(bcs, p)
        np.testing.assert_almost_equal(result, bm.to_numpy(test_result),decimal=7, 
                                     err_msg=f" `bm.simple_shape_function` function is not equal to real result in backend {backend}")
    
    @pytest.mark.parametrize("data", triangle_mesh2d_data)
    def test_simple_grad_shape_function(self, backend, data):
        bcs = bm.from_numpy(data["bcs"])
        p = data["p"]
        result = data["simple_grad_shape_function"]
        test_result = bm.simplex_grad_shape_function(bcs, p)
        np.testing.assert_almost_equal(result, bm.to_numpy(test_result),decimal=7, 
                                     err_msg=f" `bm.simple_grad_shape_function` function is not equal to real result in backend {backend}")
    
    ##TODO:HESS_SHAPE_FUNCTION TEST    
    
    @pytest.mark.parametrize("data", interval_mesh_data)
    def test_interval_grad_lambda(self, backend, data): 
        line = bm.from_numpy(data["line"])
        node = bm.from_numpy(data["node"])
        result = data["interval_grad_lambda"]
        test_result = bm.interval_grad_lambda(line, node)
        np.testing.assert_almost_equal(result, bm.to_numpy(test_result),decimal=7, 
                                     err_msg=f" `bm.interval_grad_lambda` function is not equal to real result in backend {backend}")
    
    
    @pytest.mark.parametrize("data", triangle_mesh3d_data)
    def test_triangle_area_3d(self, backend, data): 
        cell = bm.from_numpy(data["cell"])
        node = bm.from_numpy(data["node"])
        result = data["triangle_area_3d"]
        test_result = bm.triangle_area_3d(cell, node)
        np.testing.assert_almost_equal(result, bm.to_numpy(test_result),decimal=7, 
                                     err_msg=f" `bm.triangle_area_3d` function is not equal to real result in backend {backend}")

    @pytest.mark.parametrize("data", triangle_mesh2d_data)
    def test_triangle_grad_lambda_2d(self, backend, data): 
        cell = bm.from_numpy(data["cell"])
        node = bm.from_numpy(data["node"])
        result = data["triangle_grad_lambda_2d"]
        test_result = bm.triangle_grad_lambda_2d(cell, node)
        np.testing.assert_almost_equal(result, bm.to_numpy(test_result),decimal=7, 
                                     err_msg=f" `bm.triangle_grad_lambda_2d` function is not equal to real result in backend {backend}")
    
    @pytest.mark.parametrize("data", triangle_mesh3d_data)
    def test_triangle_grad_lambda_3d(self, backend, data): 
        cell = bm.from_numpy(data["cell"])
        node = bm.from_numpy(data["node"])
        result = data["triangle_grad_lambda_3d"]
        test_result = bm.triangle_grad_lambda_3d(cell, node)
        np.testing.assert_almost_equal(result, bm.to_numpy(test_result),decimal=7, 
                                     err_msg=f" `bm.triangle_grad_lambda_3d` function is not equal to real result in backend {backend}")
    

    @pytest.mark.parametrize("data", tetrahedron_mesh_data)
    def test_tetrahedron_grad_lambda_3d(self, backend, data): 
        cell = bm.from_numpy(data["cell"])
        node = bm.from_numpy(data["node"])
        localface = bm.from_numpy(data["localface"])
        result = data["tetrahedron_grad_lambda_3d"]
        test_result = bm.tetrahedron_grad_lambda_3d(cell, node, localface)
        np.testing.assert_almost_equal(result, bm.to_numpy(test_result),decimal=7, 
                                     err_msg=f" `bm.tetrahedron_grad_lambda_3d` function is not equal to real result in backend {backend}")






if __name__ == "__main__":
    pytest.main(['test_backends.py', '-q'])
    #pytest.main(['test_backends.py::TestBackendInterfaces::test_tetrahedron_grad_lambda_3d', '-q'])
