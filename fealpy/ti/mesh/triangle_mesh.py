from typing import Union

import numpy as np
import taichi as ti

from .. import logger
from .. import numpy as tnp  # the numpy-like interface in taichi 

from .utils import EntityName, Entity
from .quadrature import TriangleQuadrature, GaussLegendreQuadrature
from .mesh_base import SimplexMesh 

@ti.data_oriented
class TriangleMesh(SimplexMesh):
    def __init__(self, node: Entity, cell: Entity):

        TD = 2 # the topology dimension
        super().__init__(node.shape[0], TD)
        self.node = node 
        self.cell = cell
        kwargs = {'dtype': cell.dtype}
        self.localEdge = tnp.field([(1, 2), (2, 0), (0, 1)], **kwargs)
        self.localFace = tnp.field([(1, 2), (2, 0), (0, 1)], **kwargs)
        self.ccw = tnp.field([0, 1, 2], **kwargs)
        self.localCell = tnp.field([
            (0, 1, 2),
            (1, 2, 0),
            (2, 0, 1)], **kwargs)
        self.construct()

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
            return GaussLegendreQuadrature(index) 

    quadrature_rule = quadrature_formula


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
        node = tnp.array(mesh.entity('node'))
        cell = tnp.array(mesh.entity('cell'))
        return cls(node, cell)


    @classmethod
    def from_box(cls, box=[0, 1, 0, 1], nx=10, ny=10, threshold=None):
        from fealpy.mesh import TriangleMesh
        mesh = TriangleMesh.from_box(box=box, nx=nx, ny=ny, threshold=threshold)
        return cls.from_numpy_mesh(mesh)
