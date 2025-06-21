import numpy as np
from scipy.sparse import diags, coo_matrix
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

    def uniform_refine(self, n=1, returnim=False):
        if returnim:
            nodeImatrix = []
        for i in range(n):
            print('nx1', self.ds.nx)
            nx = 2*self.ds.nx
            self.ds = StructureIntervalMeshDataStructure(nx+1, nx)
            self.hx = (self.I[1] - self.I[0])/nx
            self.NC = nx
            self.NN = self.NC+1

            if returnim:
                A = self.interpolation_matrix()
                nodeImatrix.append(A)

        if returnim:
            return nodeImatrix



    def interpolation_matrix(self):
        """
        @brief 加密一次生成的矩阵 
        """
        nx = self.ds.nx
        NNH = nx//2 + 1
        NNh = self.number_of_nodes()

        I = np.arange(0, NNh, 2)
        J = np.arange(NNH)
        data = np.broadcast_to(1, (len(J),))
        A = coo_matrix((data, (I, J)), shape=(NNh, NNH))

        I = np.arange(1, NNh, 2)
        J = np.arange(NNH-1)
        data = np.broadcast_to(1/2, (len(J),))
        data = np.ones(NNH-1, dtype=np.float64)/2
        A += coo_matrix((data, (I, J)), shape=(NNh, NNH))

        J = np.arange(1, NNH)
        A += coo_matrix((data, (I, J)), shape=(NNh, NNH))

        A = A.tocsr()
        return A


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
            raise ValueError("`etype` is wrong!")

    def entity_barycenter(self, etype):
        if etype in {'node', 0}:
            return self.node 
        elif etype in {'cell', 1}:
            x = self.node
            return (x[1:] + x[0:-1])/2.0
        else:
            raise ValueError("`etype` is wrong!")

    @property
    def node(self):
        node = np.linspace(self.I[0], self.I[1], self.NN)
        return node

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

    def wave_equation(self, r, theta):
        n0 = self.NC -1
        A0 = diags([1+2*r**2*theta, -r**2*theta, -r**2*theta], 
                [0, 1, -1], shape=(n0, n0), format='csr')
        A1 = diags([2 - 2*r**2*(1-2*theta), r**2*(1-2*theta), r**2*(1-2*theta)], 
                [0, 1, -1], shape=(n0, n0), format='csr')
        A2 = diags([-1 - 2*r**2*theta, r**2*theta, r**2*theta], 
                [0, 1, -1], shape=(n0, n0), format='csr')

        return A0, A1, A2

    def function(self, etype='node', dtype=None):
        """
        @brief 返回一个定义在节点或者单元上的数组，元素取值为 0
        """
        nx = self.ds.nx
        dtype = self.ftype if dtype is None else dtype

        if etype in {'node', 0}:
            uh = np.zeros(nx+1, dtype=dtype)
        elif etype in {'cell', 1}:
            uh = np.zeros(nx, dtype=dtype)
        return uh
    
    def interpolation(self, f, etype='node'):
        x = self.entity_barycenter(etype)
        return f(x)

    def error(self, h, u, uh):
        """
        @brief 计算真解在网格点处与数值解的误差
        
        @param[in] u 
        @param[in] uh
        """
        
        node = self.node
        uI = u(node)
        e = uI - uh
        
        emax = np.max(np.abs(e))
        e0 = np.sqrt(h*np.sum(e**2))
        
        de = e[1:] - e[0:-1]
        e1 = np.sqrt(np.sum(de**2)/h + e0**2)
        return emax, e0, e1

    def index(self):
        NN = self.NN
        index = [ '$x_{'+str(i)+'}$' for i in range(NN)]
        return index

    def show_function(self, plot, uh):
        if isinstance(plot, ModuleType):
            fig = plot.figure()
            fig.set_facecolor('white')
            axes = fig.gca()
        else:
            axes = plot
        node = self.node.flat
        line = axes.plot(node, uh)
        return line

    def show_animation(self, fig, axes, box, forward, fname='test.mp4',
            init=None, fargs=None,
            frames=1000,  lw=2, interval=50):

        import matplotlib.animation as animation

        line, = axes.plot([], [], lw=lw)
        axes.set_xlim(box[0], box[1])
        axes.set_ylim(box[2], box[3])
        x = self.node

        def init_func():
            if callable(init):
                init()
            return line

        def func(n, *fargs):
            uh, t = forward(n)
            line.set_data((x, uh))
            s = "frame=%05d, time=%0.8f"%(n, t)
            print(s)
            axes.set_title(s)
            return line

        ani = animation.FuncAnimation(fig, func, frames=frames,
                init_func=init_func,
                interval=interval)
        ani.save(fname)
        
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
        isBdNode = np.zeros(NN, dtype=np.bool_)
        isBdNode[[0, -1]] = True
        return isBdNode

    def boundary_cell_flag(self):
        NC = self.NC
        isBdCell = np.zeros((NC,), dtype=np.bool_)
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
