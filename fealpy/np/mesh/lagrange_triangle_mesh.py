from typing import Optional, Union, List,Tuple

import numpy as np
from numpy.typing import NDArray

from .mesh_base import _S
from .lagrange_mesh import LagrangeMesh
from .quadrature import Quadrature

from .. import logger
from . import functional as F
from .mesh_base import HomogeneousMesh, estr2dim

Index = Union[NDArray, int, slice]
_dtype = np.dtype
_S = slice(None)


class LagrangeTriangleMesh(LagrangeMesh):
    def __init__(self, node: NDArray, cell: NDArray, p=1, surface=None,
            construct=False):
        super().__init__(TD=2)

        kwargs = {'dtype': cell.dtype}
        self.p = p
        self.node = node
        self.cell = cell
        self.surface = surface

        self.localEdge = self.generate_local_lagrange_edges(p) 
        self.localFace = self.localEdge
        self.ccw  = np.array([0, 1, 2], **kwargs)
        
        self.localCell = np.array([
            (0, 1, 2),
            (1, 2, 0),
            (2, 0, 1)], **kwargs)

        if construct:
            self.construct()

        self.meshtype = 'ltri'

        self.nodedata = {}
        self.edgedata = {}
        self.celldata = {}
        self.meshdata = {}


    def construct(self):
        pass

    def generate_local_lagrange_edges(self, p: int) -> NDArray:
        """
        Generate the local edges for Lagrange elements of order p.
        """
        TD = self.top_dimension()
        multiIndex = F.multi_index_matrix(p, TD)

        localEdge = np.zeros((3, p+1), dtype=np.int_)
        localEdge[2, :], = np.where(multiIndex[:, 2] == 0)
        localEdge[1, -1::-1], = np.where(multiIndex[:, 1] == 0)
        localEdge[0, :],  = np.where(multiIndex[:, 0] == 0)

        return localEdge

    
    @classmethod
    def from_triangle_mesh(cls, mesh, p, surface=None):
        node = mesh.interpolation_points(p)
        cell = mesh.cell_to_ipoint(p)
        if surface is not None:
            node, _ = surface.project(node)

        lmesh = cls(node, cell, p=p, construct=False)

        lmesh.edge2cell = mesh.edge2cell # (NF, 4)
        lmesh.cell2edge = mesh.cell_to_edge()
        lmesh.edge  = mesh.edge_to_ipoint(p)
        return lmesh 
    
    def cell_area(self, q=None, index=np.s_[:]):
        """
        Calculate the area of a cell.
        """
        p = self.p
        q = p if q is None else q
        GD = self.geo_dimension()

        qf = self.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        J = self.jacobi_matrix(bcs, index=index)
        n = np.cross(J[..., 0], J[..., 1], axis=-1)
        if GD == 3:
            n = np.sqrt(np.sum(n**2, axis=-1))
        a = np.einsum('i, ij->j', ws, n)/2.0
        return a

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
        from fealpy.mesh.vtk_extent import vtk_cell_index, write_to_vtu

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
