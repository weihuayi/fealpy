from typing import Union

import numpy as np
import taichi as ti

from .. import logger
from .. import to_taichi_field

from .quadrature import TriangleQuadrature 

@ti.data_oriented
class TriangleMeshDataStructure(MeshDS):
    def __init__(self, NN: int, cell: ti.template()):
        # 使用 NumPy 数组初始化字段
        self.NN = NN
        self.cell = cell
        self.localEdge = to_taichi_field(np.array([(1, 2), (2, 0), (0, 1)], dtype=np.int32))
        self.localFace = to_taichi_field(np.array([(1, 2), (2, 0), (0, 1)], dtype=np.int32))
        self.ccw = to_taichi_field(np.array([0, 1, 2], dtype=np.int32))
        self.localCell = to_taichi_field(np.array([
            (0, 1, 2),
            (1, 2, 0),
            (2, 0, 1)], dtype=np.int32))


    # 访问或操作数据的示例方法
    @ti.kernel
    def print_data(self):
        for i in range(3):
            print(f'ccw[{i}] = {self.ccw[i]}')
            for j in range(2):
                print(f'localEdge[{i}, {j}] = {self.localEdge[i, j]}')
                print(f'localFace[{i}, {j}] = {self.localFace[i, j]}')
            for j in range(3):
                print(f'localCell[{i}, {j}] = {self.localCell[i, j]}')

@ti.data_oriented
class TriangleMesh():
    def __init__(self, node: ti.types.field, cell: ti.types.field):
        self.node = node 
        NN = node.shape[0]
        self.ds = TriangleMeshDataStructure(NN, cell) 

        self.ftype = self.node.dtype
        self.itype = self.ds.cell.dtype
        self.p = 1  # 平面三角形

        self.celldata = {}
        self.nodedata = {}
        self.edgedata = {}
        self.facedata = self.edgedata
        self.meshdata = {}

    def top_dimension(self):
        return 2

    def geo_dimension(self):
        return self.node.shape[1]

    def quadrature_formula(self, index: int, etype='cell'):
        if etype in ('cell', 2):
            return TriangleQuadrature(index)
        elif etype in ('face', 'edge', 1):


    integrator = quadrature_formula
            

    def view(self, name='Window Title', res=(640, 360), fps_limit=200, pos = (150, 150)): 
        GD = self.geo_dimension()
        window = ti.ui.Window(
                name=name, res=res, fps_limit=fps_limit, pos=pos)

        if GD == 2:
            canvas = window.get_canvas()
            canvas.set_background_color(color)
            canvas.triangles(vertices, color, indices, per_vertex_color)
        else:
            pass

    @classmethod
    def from_torch_mesh(cls, mesh):
        pass

    @classmethod
    def from_numpy_mesh(cls, mesh): 
        node = to_taichi_field(mesh.entity('node'))
        cell = to_taichi_field(mesh.entity('cell'))
        return cls(node, cell)


    @classmethod
    def from_box(cls, box=[0, 1, 0, 1], nx=10, ny=10, threshold=None):
        from fealpy.mesh import TriangleMesh
        mesh = TriangleMesh.from_box(box=box, nx=nx, ny=ny, threshold=threshold)
        return cls.from_numpy_mesh(mesh)
