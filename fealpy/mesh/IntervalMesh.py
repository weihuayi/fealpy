import numpy as np
from .mesh_tools import unique_row, find_node, find_entity, show_mesh_2d
from scipy.sparse import csr_matrix
from types import ModuleType

class IntervalMesh():
    def __init__(self, node, cell):
        self.node = node

        self.ds = IntervalMeshDataStructure(len(node), cell)
        self.meshtype = 'interval'

        self.nodedata = {}
        self.celldata = {}

    def number_of_nodes(self):
        return self.ds.NN

    def number_of_cells(self):
        return self.ds.NC

    def number_of_entities_by_dim(self, dim):
        if dim == 1:
            return self.ds.NC
        elif dim == 0:
            return self.ds.NN
        else:
            raise ValueError("`dim` must be 0 or 1!")

    def number_of_entities_by_name(self, entityname):
        if name is 'cell':
            return self.ds.NC
        elif name is 'node':
            return self.ds.NN
        else:
            raise ValueError("`name` must be `cell` or `node`!")

    def geo_dimension(self):
        node = self.node
        if len(node.shape) == 1:
            return 1 
        else:
            return node.shape[-1]

    def top_dimension(self):
        return 1

    def cell_length(self, cellidx=None):
        node = self.node
        cell = self.ds.cell
        if cellidx is None:
            return node[cell[:, 1]] - node[cell[:, 0]]
        else:
            return node[cell[cellidx, 1]] - node[cell[cellidx, 0]]

    def entity_measure_by_dim(self, dim, index=None):
        if dim == 1:
            return self.cell_length(cellidx=index)
        elif dim == 0:
            return 0
        else:
            raise ValueError("`dim` must be 0 or 1!")

    def entity_measure_by_name(self, name, index=None):
        if name is 'cell':
            return self.cell_length(cellidx=index)
        elif name is 'node':
            return 0
        else:
            raise ValueError("`name` must be `cell` or `node`!")

    def entity_barycenter_by_dim(self, dim):
        node = self.node
        cell = self.ds.cell
        if dim == 1:
            return np.sum(node[cell], axis=-1)/2
        elif dim == 0:
            return node

    def entity_barycenter_by_name(self, name):
        node = self.node
        cell = self.ds.cell
        if name is 'cell': 
            return np.sum(node[cell], axis=-1)/2
        elif name is 'node':
            return node
        else:
            raise ValueError("`name` must be `cell` or `node`!")

    def bc_to_points(self, bc)
        node = self.node
        cell = self.ds.cell
        p = np.einsum('...j, ij->...i', bc, node[cell])
        return p 

    def add_plot(self, plot,
            nodecolor='w', cellcolor='k',
            aspect='equal', linewidths=1, markersize=2,  
            showaxis=False):

        if isinstance(plot, ModuleType):
            fig = plot.figure()
            fig.set_facecolor('white')
            axes = fig.gca() 
        else:
            axes = plot
        return show_mesh_2d(axes, self,
                nodecolor=nodecolor, edgecolor=edgecolor,
                cellcolor=cellcolor, aspect=aspect,
                linewidths=linewidths, markersize=markersize,  
                showaxis=showaxis, showcolorbar=showcolorbar, cmap=cmap)

    def find_node(self, axes, node=None,
            index=None, showindex=False,
            color='r', markersize=200, 
            fontsize=24, fontcolor='k'):

        if node is None:
            node = self.node
        find_node(axes, node, 
                index=index, showindex=showindex, 
                color=color, markersize=markersize,
                fontsize=fontsize, fontcolor=fontcolor)


    def find_edge(self, axes, 
            index=None, showindex=False,
            color='g', markersize=400, 
            fontsize=24, fontcolor='k'):

        find_entity(axes, self, entity='edge',
                index=index, showindex=showindex, 
                color=color, markersize=markersize,
                fontsize=fontsize, fontcolor=fontcolor)

    def find_cell(self, axes, 
            index=None, showindex=False,
            color='y', markersize=800, 
            fontsize=24, fontcolor='k'):
        
        find_entity(axes, self, entity='cell',
                index=index, showindex=showindex, 
                color=color, markersize=markersize,
                fontsize=fontsize, fontcolor=fontcolor)


class IntervalMeshDataStructure():
    def __init__(self, N, cell):
        self.N = N
        self.NC = len(cell)
        self.cell = cell
        self.construct()

    def reinit(self, N, cell):
        self.N = N
        self.NC = cell.shape[0]
        self.cell = cell
        self.construct()

    def construct(self):
        N = self.N
        NC = self.NC

        _, i0, j = np.unique(cell.reshape(-1), return_index=True, return_inverse=True)
        self.node2cell = np.zeros((N, 2), dtype=np.int)

        i1 = np.zeros(N, dtype=np.int) 
        i1[j] = np.arange(2*NC)

        self.node2cell[:, 0] = i0//2 
        self.node2cell[:, 1] = i1//2
        self.node2cell[:, 2] = i0%2 
        self.node2cell[:, 3] = i1%2 

    def cell_to_node(self):
        return self.cell

    def cell_to_cell(self):
        NC = self.NC
        node2cell = self.node2cell
        cell2cell = np.zeros((NC, 2), dtype=np.int)
        cell2cell[node2cell[:, 0], node2cell[:, 2]] = node2cell[:, 1]
        cell2cell[node2cell[:, 1], node2cell[:, 3]] = node2cell[:, 0]
        return cell2cell

    def node_to_cell(self):
        return self.node2cell

    def node_to_node(self):
        N = self.N
        NC = self.NC
        cell = self.cell
        I = cell.flatten()
        J = cell[:,[1,0]].flatten()
        val = np.ones((2*NC,), dtype=np.bool)
        node2node = csr_matrix((val, (I, J)), shape=(N, N),dtype=np.bool)
        return node2node

    def boundary_node_flag(self):
        node2cell = self.node2cell
        return node2cell[:, 0] == node2cell[:, 1]

    def boundary_cell_flag(self):
        NC = self.NC
        node2cell = self.node2cell
        isBdCell = np.zeros((NC,), dtype=np.bool)
        isBdNode = self.boundary_node_flag()
        isBdCell[node2cell[isBdNode, 0]] = True
        return isBdCell 
