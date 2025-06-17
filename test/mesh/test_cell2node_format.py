from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from fealpy.mesh import HexahedronMesh
from fealpy.sparse import CSRTensor, COOTensor

from cell2node_format_data import *

import numpy as np
import pytest

@pytest.mark.parametrize("backend", ['numpy','jax','pytorch'])
def test_cell2node(backend): 
    bm.set_backend(backend)
    mesh = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=1, ny=2, nz=1)
    A_csr = mesh.cell_to_node(format='csr')
    A_coo = mesh.cell_to_node(format='coo')
    assert isinstance(A_csr, CSRTensor)
    assert isinstance(A_coo, COOTensor)
    np.testing.assert_allclose(bm.to_numpy(A_csr.toarray()), A_cell2node)
    np.testing.assert_allclose(bm.to_numpy(A_coo.toarray()), A_cell2node)

@pytest.mark.parametrize("backend", ['numpy','jax','pytorch'])
def test_face2node(backend): 
    bm.set_backend(backend)
    mesh = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=1, ny=2, nz=1)
    A_csr = mesh.face_to_node(format='csr')
    A_coo = mesh.face_to_node(format='coo')
    assert isinstance(A_csr, CSRTensor)
    assert isinstance(A_coo, COOTensor)
    np.testing.assert_allclose(bm.to_numpy(A_csr.toarray()), A_face2node)
    np.testing.assert_allclose(bm.to_numpy(A_coo.toarray()), A_face2node)

@pytest.mark.parametrize("backend", ['numpy','jax','pytorch'])
def test_edge2node(backend): 
    bm.set_backend(backend)
    mesh = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=1, ny=2, nz=1)
    A_csr = mesh.edge_to_node(format='csr')
    A_coo = mesh.edge_to_node(format='coo')
    assert isinstance(A_csr, CSRTensor)
    assert isinstance(A_coo, COOTensor)
    np.testing.assert_allclose(bm.to_numpy(A_csr.toarray()), A_edge2node)
    np.testing.assert_allclose(bm.to_numpy(A_coo.toarray()), A_edge2node)

@pytest.mark.parametrize("backend", ['numpy','jax','pytorch'])
def test_node2node(backend): 
    bm.set_backend(backend)
    mesh = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=1, ny=2, nz=1)
    A_csr = mesh.node_to_node()
    A_coo = mesh.node_to_node(format='coo')
    A_array = mesh.node_to_node(format='array')
    assert isinstance(A_csr, CSRTensor)
    assert isinstance(A_coo, COOTensor)
    assert isinstance(A_array, TensorLike)
    np.testing.assert_allclose(bm.to_numpy(A_csr.toarray()).astype(int), A_node2node)
    np.testing.assert_allclose(bm.to_numpy(A_coo.toarray()).astype(int), A_node2node)
    np.testing.assert_allclose(bm.to_numpy(A_array).astype(int), A_node2node)

@pytest.mark.parametrize("backend", ['numpy','jax','pytorch'])
def test_node2edge(backend): 
    bm.set_backend(backend)
    mesh = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=1, ny=2, nz=1)
    A_csr = mesh.node_to_edge()
    A_coo = mesh.node_to_edge(format='coo')
    A_array = mesh.node_to_edge(format='array')
    assert isinstance(A_csr, CSRTensor)
    assert isinstance(A_coo, COOTensor)
    assert isinstance(A_array, TensorLike)
    np.testing.assert_allclose(bm.to_numpy(A_csr.toarray()).astype(int), A_node2edge)
    np.testing.assert_allclose(bm.to_numpy(A_coo.toarray()).astype(int), A_node2edge)
    np.testing.assert_allclose(bm.to_numpy(A_array).astype(int), A_node2edge)

@pytest.mark.parametrize("backend", ['numpy','jax','pytorch'])
def test_node2cell(backend): 
    bm.set_backend(backend)
    mesh = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=1, ny=2, nz=1)
    A_csr = mesh.node_to_cell()
    A_coo = mesh.node_to_cell(format='coo')
    A_array = mesh.node_to_cell(format='array')
    assert isinstance(A_csr, CSRTensor)
    assert isinstance(A_coo, COOTensor)
    assert isinstance(A_array, TensorLike)
    np.testing.assert_allclose(bm.to_numpy(A_csr.toarray()).astype(int), A_node2cell)
    np.testing.assert_allclose(bm.to_numpy(A_coo.toarray()).astype(int), A_node2cell)
    np.testing.assert_allclose(bm.to_numpy(A_array).astype(int), A_node2cell)

@pytest.mark.parametrize("backend", ['numpy','jax','pytorch'])
def test_node2face(backend): 
    bm.set_backend(backend)
    mesh = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=1, ny=2, nz=1)
    A_csr = mesh.node_to_face()
    A_coo = mesh.node_to_face(format='coo')
    A_array = mesh.node_to_face(format='array')
    assert isinstance(A_csr, CSRTensor)
    assert isinstance(A_coo, COOTensor)
    assert isinstance(A_array, TensorLike)
    np.testing.assert_allclose(bm.to_numpy(A_csr.toarray()).astype(int), A_node2face)
    np.testing.assert_allclose(bm.to_numpy(A_coo.toarray()).astype(int), A_node2face)
    np.testing.assert_allclose(bm.to_numpy(A_array).astype(int), A_node2face)
