from typing import Union

import numpy as np
import taichi as ti

from .. import logger
from .. import numpy as np 

from .utils import EntityName, Entity
from .quadrature import TriangleQuadrature 
from .mesh_base import MeshDS

@ti.data_oriented
class TriangleMesh(MeshDS):
    def __init__(self, node: Entity, cell: Entity):
        super().__init__(node.shape[0], 2)
        self.node = node 
        self.cell = cell
        self.localEdge = tnp.array([(1, 2), (2, 0), (0, 1)], dtype=ti.i8)
        self.localFace = tnp.array([(1, 2), (2, 0), (0, 1)], dtype=ti.i8)
        self.ccw = tnp.array([0, 1, 2], dtype=ti.i8)
        self.localCell = tnp.array([
            (0, 1, 2),
            (1, 2, 0),
            (2, 0, 1)], dtype=ti.i8)

        self.p = 1  # 平面三角形
        self.celldata = {}
        self.nodedata = {}
        self.edgedata = {}
        self.facedata = self.edgedata
        self.meshdata = {}

    def quadrature_formula(self, index: int, etype='cell'):
        if etype in ('cell', 2):
            return TriangleQuadrature(index)
        elif etype in ('face', 'edge', 1):
            return None

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
