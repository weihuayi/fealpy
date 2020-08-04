import numpy as np
from types import ModuleType
from ..quadrature import TriangleQuadrature
from .mesh_tools import unique_row, find_node, find_entity, show_mesh_2d

from .Mesh2d import Mesh2d, Mesh2dDataStructure
from .TriangleMesh import TriangleMesh
from ..functionspace.femdof import CPLFEMDof2d


class HighOrderTriangleMesh(Mesh2d):
    def __init__(self, node, cell, p=1):

        mesh = TriangleMesh(node, cell) 
        dof = CPLFEMDof2d(mesh, p)

        self.node = dof.interpolation_points()
        self.ds = HighOrderTriangleMeshDataStructure(dof)

        self.meshtype = 'hotri'
        self.ftype = mesh.ftype
        self.itype = mesh.itype
        self.nodedata = {}
        self.celldata = {}

    def vtk_cell_type(self, etype='cell'):

        if etype in {'cell', 2}:
            VTK_LAGRANGE_TRIANGLE = 69
            return VTK_LAGRANGE_TRIANGLE 
        elif etype in {'face', 'edge', 1}:
            VTK_LAGRANGE_CURVE = 68
            return VTK_LAGRANGE_CURVE

    def to_vtk(self, etype='cell', index=np.s_[:]):
        """
        Parameters
        ----------

        Notes
        -----
        把网格转化为 VTK 的格式
        """
        node = self.entity('node')
        GD = self.geo_dimension()
        if GD == 2:
            node = np.concatenate((node, np.zeros((node.shape[0], 1), dtype=self.ftype)), axis=1)

        cell = self.entity(etype)[index]
        NV = cell.shape[-1]

        cell = np.r_['1', np.zeros((len(cell), 1), dtype=cell.dtype), cell]
        cell[:, 0] = NV
        celltype = self.vtk_cell_type(etype)

        return node, cell.flatten(), cellType, len(cell)

    def print(self):

        node = self.entity('node')
        print('node:')
        for i, v in enumerate(node):
            print(i, ": ", v)

        edge = self.entity('edge')
        print('edge:')
        for i, e in enumerate(edge):
            print(i, ": ", e)

        cell = self.entity('cell')
        print('cell:')
        for i, c in enumerate(cell):
            print(i, ": ", c)

        edge2cell = self.ds.edge_to_cell()
        print('edge2cell:')
        for i, ec in enumerate(edge2cell):
            print(i, ": ", ec)



class HighOrderTriangleMeshDataStructure(Mesh2dDataStructure):
    def __init__(self, dof):
        self.cell = dof.cell_to_dof()
        self.edge = dof.edge_to_dof()
        self.edge2cell = dof.mesh.ds.edge_to_cell()

        self.NN = dof.number_of_global_dofs() 
        self.NE = len(self.edge)
        self.NC = len(self.cell)

        self.V = dof.number_of_local_dofs() 
        self.E = 3
