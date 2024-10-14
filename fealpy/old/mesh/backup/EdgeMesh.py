import numpy as np
from types import ModuleType
from .Mesh1d import Mesh1d

## @defgroup MeshGenerators Meshgeneration algorithms on commonly used domain 
## @defgroup MeshQuality
class EdgeMesh(Mesh1d):
    def __init__(self, node, cell):
        self.node = node
        self.itype = cell.dtype
        self.ftype = node.dtype

        self.meshtype = 'edge'
        
        self.NN = node.shape[0]
        self.GD = node.shape[1]
        self.ds = EdgeMeshDataStructure(self.NN, cell)

        self.nodedata = {}
        self.celldata = {}
        self.meshdata = {}

    def geo_dimension(self):
        return self.GD

    def entity(self, etype='node'):
        if etype in {'node', 'face', 0}:
            return self.node
        elif etype in {'cell', 'edge', 1}:
            return self.ds.cell
    
    def entity_measure(self, etype='cell', index=np.s_[:]):
        if etype in {'cell', 'edge', 1}:
            return self.cell_length()[index] 
        elif etype in {'node', 'face', 0}:
            return 0.0 

    def bc_to_point(self, bc, index=np.s_[:], node=None):
        """

        Notes
        -----
            把重心坐标转换为实际空间坐标
        """
        node = self.node if node is None else node
        cell = self.entity('cell')
        p = np.einsum('...j, ijk->...ik', bc, node[cell[index]])
        return p

    def cell_length(self):
        node = self.entity('node')
        cell = self.entity('cell')

        v = node[cell[:, 0]] - node[cell[:, 1]]
        h = np.sqrt(np.sum(v**2, axis=1))
        return h

    def cell_unit_tangent(self, index=np.s_[:]):
        """
        @brief 计算每个单元的单位切向
        """
        node = self.entity('node')
        cell = self.entity('cell')
        v = node[cell[:, 0]] - node[cell[:, 1]]
        h = np.sqrt(np.sum(v**2, axis=1))
        return v/h[:, None]

    def cell_frame(self):
        """
        @brief 计算每个单元上的标架
        """
        pass

    def add_plot(self, plot, 
            nodecolor='r',
            cellcolor='k', 
            linewidths=1, 
            aspect=None,
            markersize=10,
            box=None,
            disp=None,
            scale=1.0
            ):

        GD = self.geo_dimension()
        import mpl_toolkits.mplot3d as a3
        if isinstance(plot, ModuleType):
            fig = plot.figure()
            if GD == 3:
                axes = fig.add_subplot(111, projection='3d')
            else:
                axes = fig.add_subplot(111)
        else:
            axes = plot

        if (aspect is None) and (GD == 3):
            axes.set_box_aspect((1, 1, 1))
            axes.set_proj_type('ortho')

        if (aspect is None) and (GD == 2):
            axes.set_box_aspect(1)


        node = self.entity('node')
        if GD == 2:
            axes.scatter(node[:, 0], node[:, 1], c=nodecolor, s=markersize)
        else:
            axes.scatter(node[:, 0], node[:, 1], node[:, 2], c=nodecolor, s=markersize)

        cell = self.entity('cell') 

        if box is None:
            if self.geo_dimension() == 2:
                box = np.zeros(4, dtype=np.float64)
            else:
                box = np.zeros(6, dtype=np.float64)

        box[0::2] = np.min(node, axis=0)
        box[1::2] = np.max(node, axis=0)

        axes.set_xlim([box[0], box[1]+0.01])
        axes.set_ylim([box[2]-0.01, box[3]])

        vts = node[cell]
        if GD == 3:
            axes.set_zlim(box[4:6])
            cells = a3.art3d.Line3DCollection(
                   vts,
                   linewidths=linewidths,
                   color=cellcolor)
            return axes.add_collection3d(cells)
        elif GD == 2:
            from matplotlib.collections import LineCollection
            cells = LineCollection(
                    vts,
                    linewidths=linewidths,
                    color=cellcolor)
            return axes.add_collection(cells)

    
    ## @ingroup MeshGenerators
    @classmethod
    def from_triangle_mesh(cls, mesh):
        pass

    ## @ingroup MeshGenerators
    @classmethod
    def from_tetrahedron_mesh(cls, mesh):
        pass

    ## @ingroup MeshGenerators
    @classmethod
    def from_tower(cls):
        node = np.array([
            [-950, 0, 5080], [950, 0, 5080], [-950, 950, 2540], 
            [950, 950, 2540], [950, -950, 2540], [-950, -950, 2540],
            [-2540, 2540, 0], [2540, 2540, 0], [2540, -2540, 0], 
            [-2540, -2540, 0]], dtype=np.float64)
        cell = np.array([
            [0, 1], [3, 0], [1, 2], [1, 5], [0, 4], 
            [1, 3], [1, 4], [0, 2], [0, 5], [2, 5],
            [4, 3], [2, 3], [4, 5], [2, 9], [6, 5], 
            [8, 3], [7, 4], [6, 3], [2, 7], [9, 4],
            [8, 5], [9, 5], [2, 6], [7, 3], [8, 4]], dtype=np.int_)
        mesh = cls(node, cell)

        mesh.meshdata['disp_bc'] = (np.array([6, 7, 8, 9], dtype=np.int_), np.zeros(3))
        mesh.meshdata['force_bc'] = (np.array([0, 1], dtype=np.int_), np.array([0, 900, 0]))

        return mesh 



class EdgeMeshDataStructure():

    def __init__(self, NN, cell):
        self.NN = NN
        self.cell = cell 
        self.NC = len(cell)

    def cell_to_node(self):
        return self.cell

    def node_to_cell(self):
        NN = self.NN
        NC = self.NC
        I = self.cell.flat
        J = np.repeat(range(NC), 2)
        val = np.ones(2*NC, dtype=np.bool_)
        node2edge = csr_matrix((val, (I, J)), shape=(NN, NC))
        return node2edge
