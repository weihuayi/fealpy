import pytest
import numpy as np
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh, QuadrangleMesh, TetrahedronMesh, HexahedronMesh
from fealpy.old.mesh import TriangleMesh as OTriM
from fealpy.old.mesh import QuadrangleMesh as OQuadM
from fealpy.old.mesh import TetrahedronMesh as OTetM
from fealpy.old.mesh import HexahedronMesh as OHexM
from fealpy.utils import timer

@pytest.mark.parametrize("device", ['cpu', 'cuda'])
@pytest.mark.parametrize("backend", ['pytorch'])
@pytest.mark.parametrize("mesh_type", ["triangle"])
def test_mesh(backend, device, mesh_type): 
    bm.set_backend(backend)
    
    # 只在非 numpy 后端设置默认设备
    if backend != 'numpy':
        bm.set_default_device(device)

    # 调用 create_mesh_data 根据 mesh_type 生成对应的网格数据
    
    mesh_data = create_mesh_data(20, mesh_type, backend, device)
    old_mesh = mesh_data['old']
    new_mesh = mesh_data['new']
    tmr = timer()
    next(tmr)
    
    new_node2node = new_mesh.node_to_node()
    new_node2edge = new_mesh.node_to_edge()
    new_node2cell = new_mesh.node_to_cell()
    new_node2face = new_mesh.node_to_face()
    
    tmr.send(f"新接口node方法:{device} and {backend}")
  
    old_node2node = old_mesh.ds.node_to_node()
    old_node2edge = old_mesh.ds.node_to_edge()
    old_node2cell = old_mesh.ds.node_to_cell()
    old_node2face = old_mesh.ds.node_to_face()
    
    tmr.send(f"旧接口node方法:cpu and numpy")

    next(tmr)
    new_node2node_scipy = new_node2node.to_scipy()
    new_node2edge_scipy = new_node2edge.to_scipy()
    new_node2cell_scipy = new_node2cell.to_scipy()
    new_node2face_scipy = new_node2face.to_scipy()
    
    np.testing.assert_allclose(new_node2node_scipy.toarray(), old_node2node.toarray(), rtol=1e-5)
    np.testing.assert_allclose(new_node2edge_scipy.toarray(), old_node2edge.toarray(), rtol=1e-5)
    np.testing.assert_allclose(new_node2cell_scipy.toarray(), old_node2cell.toarray(), rtol=1e-5)
    np.testing.assert_allclose(new_node2face_scipy.toarray(), old_node2face.toarray(), rtol=1e-5)

# 创建 mesh_data 的函数，传入网格类型来生成不同类型的网格
def create_mesh_data(n, mesh_type):
    # tmr = timer()
    # next(tmr)
    if mesh_type == "triangle":
        new_mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=n, ny=n)
        old_mesh = OTriM.from_box(box=[0, 1, 0, 1], nx=n, ny=n)
    elif mesh_type == "quad":
        new_mesh = QuadrangleMesh.from_box(box=[0, 1, 0, 1], nx=n, ny=n)
        old_mesh = OQuadM.from_box(box=[0, 1, 0, 1], nx=n, ny=n)
    elif mesh_type == "tet":
        new_mesh = TetrahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=n, ny=n, nz=n)
        #tmr.send(f"新接口网格生成时间, {device} and {backend}")

        old_mesh = OTetM.from_box(box=[0, 1, 0, 1, 0, 1], nx=n, ny=n, nz=n)
        #tmr.send(f"旧接口网格生成时间, cpu and numpy")

    elif mesh_type == "hex":
        new_mesh = HexahedronMesh.from_box(box=[0, 1, 0, 1, 0, 1], nx=n, ny=n, nz=n)
        old_mesh = OHexM.from_box(box=[0, 1, 0, 1, 0, 1], nx=n, ny=n, nz=n)
    else:
        raise ValueError(f"Unsupported mesh type: {mesh_type}")
    # next(tmr)
    return {'new': new_mesh, 'old': old_mesh}

if __name__ == "__main__":
    pytest.main(["-s", "./test_sparse_mesh_data_structure.py", "-k", "cuda"])
    pytest.main(["-s", "./test_sparse_mesh_data_structure.py", "-k", "cpu"])

