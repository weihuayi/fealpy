
import pytest
import numpy as np
import taichi as ti
from fealpy.ti.mesh import TriangleMesh

@pytest.fixture
def setup_mesh():
    return TriangleMesh.from_box(domain=[0, 1, 0, 1], nx=1, ny=1)  

def test_init_triangle_mesh(setup_mesh):
    mesh = setup_mesh

    print(mesh.node)
    print(mesh.cell)
    print(mesh.edge)
    print(mesh.face)
    assert mesh.node is not None
    assert mesh.cell is not None
    assert mesh.TD == 2
    assert isinstance(mesh.localEdge, ti.FieldsBuilder)
    assert isinstance(mesh.localFace, ti.FieldsBuilder)

def test_quadrature_formula_cell(setup_mesh):
    mesh = setup_mesh
    quad = mesh.quadrature_formula(index=2, etype='cell')
    assert isinstance(quad, TriangleQuadrature)
    assert quad.order == 2  # 假设TriangleQuadrature初始化时设置了阶数

def test_quadrature_formula_edge(setup_mesh):
    mesh = setup_mesh
    quad = mesh.quadrature_formula(index=1, etype='edge')
    assert isinstance(quad, GaussLegendreQuadrature)
    assert quad.order == 1  # 假设GaussLegendreQuadrature初始化时设置了阶数

def test_from_box():
    mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=10, ny=10)
    assert isinstance(mesh, TriangleMesh)
    assert mesh.node.shape[0] == 11 * 11  # 确认节点数量正确
    assert mesh.cell.shape[0] == 10 * 10 * 2  # 确认单元数量正确


if __name__ == "__main__":
    test_init_triangle_mesh(setup_mesh)





