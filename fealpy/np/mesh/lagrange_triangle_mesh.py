from typing import Optional, Union, List,Tuple

import numpy as np
from numpy.typing import NDArray

from .mesh_base import _S
from .lagrange_mesh import LagrangeMesh
from .triangle_mesh import TriangleMesh
from .quadrature import Quadrature

from .. import logger
from . import functional as F
from .mesh_base import HomogeneousMesh, estr2dim

Index = Union[NDArray, int, slice]
_dtype = np.dtype
_S = slice(None)


class LagrangeTriangleMesh(LagrangeMesh):
    def __init__(self, node: NDArray, cell: NDArray, p=1, surface=None):
        super().__init__(TD=2)
        mesh = TriangleMesh(node, cell)
        NN = mesh.number_of_nodes()

        self.meshtype = 'ltri'

        self.p = p
        self.surface = surface

        self.node = mesh.interpolation_points(p)

        if surface is not None:
            self.node,_ = surface.project(self.node)
        cell = mesh.cell_to_ipoint(p)

        self.nodedata = {}
        self.edgedata = {}
        self.celldata = {}
        self.meshdata = {}
    
    @classmethod
    def from_triangle_mesh(cls, mesh, p):
        node = mesh.entity("node")
        cell = mesh.entity("cell")
        return cls(node, cell, p=p)
 
    def vtk_cell_type(self, etype='cell'):
        """
        @berif  返回网格单元对应的 vtk类型。
        """
        if etype in {'cell', 2}:
            VTK_LAGRANGE_TRIANGLE = 69
            return VTK_LAGRANGE_TRIANGLE 
        elif etype in {'face', 'edge', 1}:
            VTK_LAGRANGE_CURVE = 68
            return VTK_LAGRANGE_CURVE

    def to_vtk(self, etype='cell', index=np.s_[:], fname=None):
        """
        Parameters
        ----------

        @berif 把网格转化为 VTK 的格式
        """
        from .vtk_extent import vtk_cell_index, write_to_vtu

        node = self.entity('node')
        GD = self.geo_dimension()
        if GD == 2:
            node = np.concatenate((node, np.zeros((node.shape[0], 1), dtype=self.ftype)), axis=1)

        #cell = self.entity(etype)[index]
        cell = self.entity(etype, index)
        cellType = self.vtk_cell_type(etype)
        idx = vtk_cell_index(self.p, cellType)
        NV = cell.shape[-1]

        cell = np.r_['1', np.zeros((len(cell), 1), dtype=cell.dtype), cell[:, idx]]
        cell[:, 0] = NV

        NC = len(cell)
        if fname is None:
            return node, cell.flatten(), cellType, NC 
        else:
            print("Writting to vtk...")
            write_to_vtu(fname, node, NC, cellType, cell.flatten(),
                    nodedata=self.nodedata,
                    celldata=self.celldata)
