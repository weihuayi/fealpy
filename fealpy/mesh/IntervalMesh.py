import numpy as np
from .mesh_tools import unique_row, find_node, find_entity, show_mesh_1d
from scipy.sparse import csr_matrix
from types import ModuleType

from ..quadrature import IntervalQuadrature

class IntervalMesh():
    def __init__(self, node, cell):
        self.node = node

        self.ds = IntervalMeshDataStructure(len(node), cell)
        self.meshtype = 'interval'

        self.nodedata = {}
        self.celldata = {}

        self.itype = cell.dtype
        self.ftype = node.dtype


    def integrator(self, k):
        return IntervalQuadrature(k)

    def number_of_nodes(self):
        return self.ds.NN

    def number_of_cells(self):
        return self.ds.NC

    def number_of_entities(self, etype):
        if etype in ['cell', 1]:
            return self.ds.NC
        elif etype in ['node', 0]:
            return self.ds.NN
        else:
            raise ValueError("`dim` must be 0 or 1!")

    def entity(self, etype=1):
        if etype in ['cell', 1]:
            return self.ds.cell
        elif etype in ['node', 0]:
            return self.node
        else:
            raise ValueError("`entitytype` is wrong!")

    def grad_lambda(self):
        node = self.node
        cell = self.ds.cell
        NC = self.number_of_cells()
        v = node[cell[:, 1]] - node[cell[:, 0]]
        dim = self.geo_dimension()
        Dlambda = np.zeros((NC, 2, dim), dtype=np.float)
        if dim == 1:
            Dlambda[:, 0, 0] = -1/v
            Dlambda[:, 1, 0] = 1/v
        else:
            h2 = np.sum(v**2, axis=-1)
            v /=h2.reshape(-1, 1)
            Dlambda[:, 0, :] = -v
            Dlambda[:, 1, :] = v
        return Dlambda

    def geo_dimension(self):
        node = self.node
        if len(node.shape) == 1:
            return 1 
        else:
            return node.shape[-1]

    def top_dimension(self):
        return 1


    def entity_measure(self, etype=1, index=None):
        if etype in ['cell', 1]:
            return self.cell_length(cellidx=index)
        elif etype in ['node', 0]:
            return 0
        else:
            raise ValueError("`etype` is wrong!")

    def entity_barycenter(self, etype=1):
        node = self.node
        cell = self.ds.cell
        if etype in ['cell', 1]:
            return np.sum(node[cell], axis=-1)/2
        elif etype in ['node', 0]:
            return node
        else:
            raise ValueError("`etype` is wrong!")

    def cell_length(self, cellidx=None):
        node = self.node
        cell = self.ds.cell
        if cellidx is None:
            return node[cell[:, 1]] - node[cell[:, 0]]
        else:
            return node[cell[cellidx, 1]] - node[cell[cellidx, 0]]

    def bc_to_point(self, bc):
        node = self.node
        cell = self.ds.cell
        p = np.einsum('...j, ij->...i', bc, node[cell])
        return p

    def uniform_refine(self, n=1):
        for i in range(n):
            NN = self.number_of_nodes()
            NC = self.number_of_cells()
            node = self.entity('node')
            cell = self.entity('cell')
            cell2newNode = np.arange(NN, NN+NC)
            newNode = (node[cell[:,0]] + node[cell[:,1]])/2
            self.node = np.r_['-1', node, newNode] 
            p = np.r_['-1', cell, cell2newNode.reshape(-1,1)] 
            cell = np.r_['0', p[:, [0, 2]], p[:, [2, 1]]] 
            NN = self.node.shape[0]
            self.ds.reinit(NN, cell)




    def add_plot(self, plot,
            nodecolor='k', cellcolor='k',
            aspect='equal', linewidths=1, markersize=20,  
            showaxis=False):

        if isinstance(plot, ModuleType):
            fig = plot.figure()
            fig.set_facecolor('white')
            axes = fig.gca() 
        else:
            axes = plot
        return show_mesh_1d(axes, self,
                nodecolor=nodecolor, cellcolor=cellcolor, aspect=aspect,
                linewidths=linewidths, markersize=markersize,  
                showaxis=showaxis)

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
    def __init__(self, NN, cell):
        self.NN = NN
        self.NC = len(cell)
        self.cell = cell
        self.construct()

    def reinit(self, NN, cell):
        self.NN = NN
        self.NC = cell.shape[0]
        self.cell = cell
        self.construct()

    def construct(self):
        NN = self.NN
        NC = self.NC
        cell = self.cell

        _, i0, j = np.unique(cell.reshape(-1), return_index=True, return_inverse=True)
        self.node2cell = np.zeros((NN, 4), dtype=np.int)

        i1 = np.zeros(NN, dtype=np.int) 
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
        NN = self.NN
        NC = self.NC
        cell = self.cell
        I = cell.flatten()
        J = cell[:,[1,0]].flatten()
        val = np.ones((2*NC,), dtype=np.bool)
        node2node = csr_matrix((val, (I, J)), shape=(NN, NN),dtype=np.bool)
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

    def boundary_node_index(self):
        isBdPoint = self.boundary_node_flag()
        idx, = np.nonzero(isBdPoint)
        return idx 

    def boundary_cell_index(self):
        isBdCell = self.boundary_cell_flag()
        idx, = np.nonzero(isBdCell)
        return idx 
