import numpy as np

from .triangle_mesh import TriangleMesh
from .mesh_base import Mesh, Plotable


class LagrangeTriangleMeshDataStructure():
    def __init__(self, NN, cell):
        self.NN = NN
        self.TD = 2
        self.cell = cell

    def number_of_cells(self):
        return self.cell.shape[0]

class LagrangeTriangleMesh(Mesh):

    def __init__(self, node, cell, surface=None, p=1):

        mesh = TriangleMesh(node, cell)
        NN = mesh.number_of_nodes()

        self.ftype = node.dtype
        self.itype = cell.dtype
        self.meshtype = 'ltri'

        self.GD = node.shape[1]

        self.p = p
        self.surface = surface

        self.node = mesh.interpolation_points(p)

        if surface is not None:
            self.node, _ = surface.project(self.node)
        cell = mesh.cell_to_ipoint(p)

        NN = self.node.shape[0]
        self.ds = LagrangeTriangleMeshDataStructure(NN, cell)
        self.ds.edge = mesh.edge_to_ipoint(p=p)

        self.nodedata = {}
        self.edgedata = {}
        self.celldata = {}
        self.meshdata = {}
        
        self.edge_bc_to_point = self.bc_to_point
        self.cell_bc_to_point = self.bc_to_point
        self.face_to_ipoint = self.edge_to_ipoint

        self.shape_function = self._shape_function
        self.cell_shape_function = self._shape_function
        self.face_shape_function = self._shape_function
        self.edge_shape_function = self._shape_function

    def ref_cell_measure(self):
        return 0.5

    def ref_face_measure(self):
        return 1.0
 
    def integrator(self, q, etype='cell'):
        """
        @brief 获取不同维度网格实体上的积分公式
        """
        if etype in {'cell', 2}:
            from ..quadrature import TriangleQuadrature
            return TriangleQuadrature(q)
        elif etype in {'edge', 'face', 1}:
            from ..quadrature import GaussLegendreQuadrature
            return GaussLegendreQuadrature(q)

    def entity_measure(self, etype=2, index=np.s_[:]):
        if etype in {'cell', 2}:
            return self.cell_area(index=index)
        elif etype in {'edge', 'face', 1}:
            return self.edge_length(index=index)
        elif etype in {'node', 0}:
            return 0
        else:
            raise ValueError(f"Invalid entity type '{etype}'.")

        
    def sphere_surface_unit_normal(self, index=np.s_[:]):
        """
        @brief 计算单位球面三角形网格中每个面上的单位法线
        """
        assert self.geo_dimension() == 3
        node = self.entity('node')
        cell = self.entity('cell')

        v0 = node[cell[index, 2]] - node[cell[index, 1]]
        v1 = node[cell[index, 0]] - node[cell[index, 2]]
        v2 = node[cell[index, 1]] - node[cell[index, 0]]

        nv = np.cross(v1, v2)
        length = np.linalg.norm(nv, axis=-1, keepdims=True)

        n = nv/length
        return n

    def vtk_cell_type(self, etype='cell'):
        """

        Notes
        -----
            返回网格单元对应的 vtk类型。
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

        Notes
        -----
        把网格转化为 VTK 的格式
        """
        from .vtk_extent import vtk_cell_index, write_to_vtu

        node = self.entity('node')
        GD = self.geo_dimension()
        if GD == 2:
            node = np.concatenate((node, np.zeros((node.shape[0], 1), dtype=self.ftype)), axis=1)

        cell = self.entity(etype)[index]
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
