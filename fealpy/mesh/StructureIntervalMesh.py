import numpy as np
from scipy.sparse import diags
from .mesh_tools import find_node, find_entity, show_mesh_1d
from types import ModuleType


class StructureIntervalMesh(object):

    """结构化的一维网格

    [x_0, x_1, ...., x_N]
    """

    def __init__(self, I, nx=2, itype=np.int_, ftype=np.float64):

        self.I = I
        self.meshtype="interval"
        self.hx = (I[1] - I[0])/nx
        self.NC = nx
        self.NN = self.NC + 1

        self.ds = StructureIntervalMeshDataStructure(nx+1, nx)

        self.itype = itype
        self.ftype = ftype

    def entity(self, etype):
        if etype in {'cell', 1}:
            NN = self.NN
            NC = self.NC
            cell = np.zeros((NC, 2), dtype=np.int)
            cell[:, 0] = range(NC)
            cell[:, 1] = range(1, NN)
            return cell
        elif etype in {'node', 0}:
            return self.node
        else:
            raise ValueError("`entitytype` is wrong!")

    @property
    def node(self):
        node = np.linspace(self.I[0], self.I[1], self.NN)
        return node.reshape(-1, 1)

    def number_of_nodes(self):
        return self.NN

    def number_of_cells(self):
        return self.NC

    def geo_dimension(self):
        return 1

    def laplace_operator(self):
        hx = self.hx
        cx = 1/(self.hx**2)
        NN = self.number_of_nodes()
        k = np.arange(NN)

        A = diags([2*cx], [0], shape=(NN, NN), format='coo')

        val = np.broadcast_to(-cx, (NN-1, ))
        I = k[1:]
        J = k[0:-1]
        A += coo_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += coo_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)
        return A.tocsr()

    def interpolation(self, f):
        return f(node)

    def index(self):
        NN = self.NN
        index = [ '$x_{'+str(i)+'}$' for i in range(NN)]
        return index

    def add_plot(
            self, plot,
            nodecolor='r', cellcolor='k',
            aspect='equal', linewidths=1,
            markersize=20,  showaxis=False):

        if isinstance(plot, ModuleType):
            fig = plot.figure()
            fig.set_facecolor('white')
            axes = fig.gca()
        else:
            axes = plot
        return show_mesh_1d(
                axes, self,
                nodecolor=nodecolor, cellcolor=cellcolor, aspect=aspect,
                linewidths=linewidths, markersize=markersize,
                showaxis=showaxis)

    def find_node(
            self, axes, node=None,
            index=None, showindex=False,
            color='r', markersize=20,
            fontsize=15, fontcolor='r', multiindex=None):

        if node is None:
            node = self.entity('node')
        find_node(
                axes, node,
                index=index, showindex=showindex,
                color=color, markersize=markersize,
                fontsize=fontsize, fontcolor=fontcolor, multiindex=multiindex)

    def find_edge(
            self, axes,
            index=None, showindex=False,
            color='g', markersize=400,
            fontsize=24, fontcolor='k'):

        find_entity(
                axes, self, entity='edge',
                index=index, showindex=showindex,
                color=color, markersize=markersize,
                fontsize=fontsize, fontcolor=fontcolor)

    def find_cell(
            self, axes,
            index=None, showindex=False,
            color='y', markersize=800,
            fontsize=24, fontcolor='k'):
        find_entity(
                axes, self, entity='cell',
                index=index, showindex=showindex,
                color=color, markersize=markersize,
                fontsize=fontsize, fontcolor=fontcolor)


class StructureIntervalMeshDataStructure():
    def __init__(self, NN, NC):
        self.nx = NC
        self.NN = NN
        self.NC = NC

    def reinit(self, NN, NC):
        self.nx = NC
        self.NN = NN
        self.NC = NC

    def boundary_node_flag(self):
        NN = self.NN
        isBdNode = np.zeros(NN, dtype=np.bool)
        isBdNode[[0, -1]] = True
        return isBdNode

    def boundary_cell_flag(self):
        NC = self.NC
        isBdCell = np.zeros((NC,), dtype=np.bool)
        isBdCell[[0, -1]] = True
        return isBdCell

    def boundary_node_index(self):
        isBdNode = self.boundary_node_flag()
        idx, = np.nonzero(isBdNode)
        return idx

    def boundary_cell_index(self):
        isBdCell = self.boundary_cell_flag()
        idx, = np.nonzero(isBdCell)
        return idx
