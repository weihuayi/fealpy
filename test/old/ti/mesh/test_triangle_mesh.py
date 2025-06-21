
import pytest
import taichi as ti

import fealpy.ti.numpy as tnp
from fealpy.ti.numpy.testing import assert_array_equal

from fealpy.ti.mesh import TriangleMesh
from fealpy.ti.mesh.quadrature import TriangleQuadrature, GaussLegendreQuadrature

ti.init(arch=ti.cuda)

@pytest.fixture
def setup_mesh():
    return TriangleMesh.from_box(box=[0, 1, 0, 1], nx=1, ny=1)  

def test_init_triangle_mesh(setup_mesh):
    mesh = setup_mesh
    face2cell = mesh.face_to_cell()
    result = tnp.array(
        [[1, 1, 2, 2],
         [0, 0, 1, 1],
         [0, 1, 0, 0],
         [1, 1, 1, 1],
         [0, 0, 2, 2]], dtype=tnp.i64)

    error_flag = ti.field(ti.i32, shape=())
    assert_array_equal(face2cell, result, error_flag) 
    assert error_flag[None] == 0

    face = mesh.entity('face')
    result = tnp.array(
        [[1, 0],
         [0, 2],
         [3, 0],
         [3, 1],
         [2, 3]], dtype=tnp.i64)
    assert_array_equal(face, result, error_flag) 
    assert error_flag[None] == 0

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
