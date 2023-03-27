import numpy as np
import warnings
from scipy.sparse import csr_matrix, coo_matrix, diags, spdiags
from .mesh_tools import find_node, find_entity, show_mesh_1d
from types import ModuleType
from typing import Tuple 

from .StructureMesh1dDataStructure import StructureMesh1dDataStructure


class UniformMesh1d():
    """
    @brief A class for representing a uniformly partitioned one-dimensional mesh.
    """
    def __init__(self, 
            extent: Tuple[int, int],
            h: float = 1.0,
            origin: float = 0.0,
            itype: type = np.int_,
            ftype: type = np.float64):
        """
        @brief Initialize the mesh.

        @param[in] extent A tuple representing the range of the mesh in the x direction.
        @param[in] h: Mesh step size.
        @param[in] origin Coordinate of the starting point.
        @param[in] itype Integer type to be used, default: np.int_.
        @param[in] ftype Floating point type to be used, default: np.float64.

        @note The extent parameter defines the index range in the x direction.
              We can define an index range starting from 0, e.g., [0, 10],
              or starting from a non-zero value, e.g., [2, 12]. The flexibility
              in the index range is mainly for handling different scenarios
              and data subsets, such as:
              - Subgrids
              - Parallel computing
              - Data cropping
              - Handling irregular data

        @see UniformMesh2d, UniformMesh3d

        @example
        from fealpy.mesh import UniformMesh1d

        I = [0, 1]
        h = 0.1
        nx = int((I[1] - I[0])/h)
        mesh = UniformMesh1d([0, nx], h=h, origin=I[0])

        """

        self.extent = extent
        self.h = h
        self.origin = origin

        self.nx = extent[1] - extent[0]
        self.NC = self.nx
        self.NN = self.NC + 1

        self.itype = itype
        self.ftype = ftype

        # Data structure for finite element computation
        self.ds: StructureMesh1dDataStructure = StructureMesh1dDataStructure(self.nx, itype=itype)

    def geo_dimension(self):
        """
        @brief Get the geometry dimension of the mesh.
        
        @return The geometry dimension (1 for 1D mesh).
        """
        return 1

    def top_dimension(self):
        """
        @brief Get the topological dimension of the mesh.
        
        @return The topological dimension (1 for 1D mesh).
        """
        return 1

    def number_of_nodes(self):
        """
        @brief Get the number of nodes in the mesh.

        @return The number of nodes.
        """
        return self.NN

    def number_of_cells(self):
        """
        @brief Get the number of cells in the mesh.

        @return The number of cells.
        """
        return self.NC

    def uniform_refine(self, n=1, returnim=False):
        """
        @brief Perform a uniform refinement of the mesh.

        @param[in] n Number of refinements to perform (default: 1).
        @param[in] returnim Boolean flag to return the interpolation matrix (default: False).

        @return If returnim is True, a list of interpolation matrices is returned.
        """
        if returnim:
            nodeImatrix = []
        for i in range(n):
            self.extent = [i * 2 for i in self.extent]
            self.h = self.h/2
            self.nx = self.extent[1] - self.extent[0]
            self.NC = self.nx
            self.NN = self.NC + 1

            if returnim:
                A = self.interpolation_matrix()
                nodeImatrix.append(A)

        if returnim:
            return nodeImatrix

    @property
    def node(self):
        """
        @brief Get the coordinates of the nodes in the mesh.

        @return A NumPy array of shape (NN, ) containing the coordinates of the nodes.

        @details This function calculates the coordinates of the nodes in the mesh based on the
                 mesh's origin, step size, and the number of cells in the x directions.
                 It returns a NumPy array with the coordinates of each node.
        """
        GD = self.geo_dimension()
        nx = self.nx
        node = np.linspace(self.origin, self.origin + nx * self.h, nx+1)
        return node

    def entity(self, etype):
        """
        @brief Get the entity (either cell or node) based on the given entity type.

        @param[in] etype The type of entity, either 'cell', 1, 'node', 'face' or 0.

        @return The cell or node array based on the input entity type.

        @throws ValueError if the given etype is invalid.
        """
        if etype in {'cell', 1}:
            NN = self.NN
            NC = self.NC
            cell = np.zeros((NC, 2), dtype=np.int)
            cell[:, 0] = range(NC)
            cell[:, 1] = range(1, NN)
            return cell
        elif etype in {'node', 'face', 0}:
            return self.node
        else:
            raise ValueError("`etype` is wrong!")

    def entity_barycenter(self, etype):
        """
        @brief Calculate the barycenter of the specified entity.

        @param[in] etype The type of entity, either 'cell', 1, 'node', 'face' or 0.

        @return The barycenter of the given entity type.

        @throws ValueError if the given etype is invalid.
        """
        GD = self.geo_dimension()
        nx = self.nx
        if etype in {'cell', 1}:
            box = [self.origin + self.h/2, self.origin + self.h/2 + (nx - 1) * self.h]
            bc = np.linspace(box[0], box[1], nx)
            return bc
        elif etype in {'node', 0}:
            return self.node
        else:
            raise ValueError('the entity type `{}` is not correct!'.format(etype))

    def function(self, etype='node', dtype=None, ex=0):
        """
        @brief Returns an array defined on nodes or cells with all elements set to 0.

        @param[in] etype The type of entity, either 'node' or 'cell' (default: 'node').
        @param[in] dtype Data type for the array (default: None, which means self.ftype will be used).
        @param[in] ex Non-negative integer to extend the discrete function outward by a certain width (default: 0).

        @return An array with all elements set to 0, defined on nodes or cells.

        @throws ValueError if the given etype is invalid.
        """
        nx = self.nx
        dtype = self.ftype if dtype is None else dtype
        if etype in {'node', 'face', 0}:
            uh = np.zeros(nx + 1, dtype=dtype)
        elif etype in {'cell', 1}:
            uh = np.zeros(nx, dtype=dtype)
        else:
            raise ValueError('the entity `{}` is not correct!'.format(entity))
        return uh

    def interpolation(self, f, intertype='node'):
        """
        This function is deprecated and will be removed in a future version.
        Please use the interpolate() instead.
        """
        warnings.warn("The interpolation() is deprecated and will be removed in a future version. "
                      "Please use the interpolate() instead.", DeprecationWarning)
        nx = self.nx
        node = self.node
        if intertype == 'node':
            F = f(node)
        elif intertype == 'cell':
            bc = self.entity_barycenter('cell')
            F = f(bc)
        return F

    def interpolate(self, f, intertype='node'):
        """
        @brief Interpolate the given function f on the mesh based on the specified interpolation type.

        @param[in] f The function to be interpolated on the mesh.
        @param[in] intertype The type of interpolation, either 'node' or 'cell' (default: 'node').

        @return The interpolated values of the function f on the mesh nodes or cell barycenters.

        @throws ValueError if the given intertype is invalid.
        """
        nx = self.nx
        node = self.node
        if intertype in {'node', 'face', 0}:
            F = f(node)
        elif intertype in {'cell', 1}:
            bc = self.entity_barycenter('cell')
            F = f(bc)
        return F

    def error(self, u, uh, errortype='all'):
        """
        @brief Compute the error between the true solution and the numerical solution.

        @param[in] u The true solution as a function.
        @param[in] uh The numerical solution as an array.
        @param[in] errortype The error type, which can be 'all', 'max', 'L2' or 'H1'
        """

        h = self.h
        node = self.node
        uI = u(node)
        e = uI - uh

        if errortype == 'all':
            emax = np.max(np.abs(e))
            e0 = np.sqrt(h * np.sum(e ** 2))

            de = e[1:] - e[0:-1]
            e1 = np.sqrt(np.sum(de ** 2) / h + e0 ** 2)
            return emax, e0, e1
        elif errortype == 'max':
            emax = np.max(np.abs(e))
            return emax
        elif errortype == 'L2':
            e0 = np.sqrt(h * np.sum(e ** 2))
            return e0
        elif errortype == 'H1':
            e0 = np.sqrt(h * np.sum(e ** 2))
            de = e[1:] - e[0:-1]
            e1 = np.sqrt(np.sum(de ** 2) / h + e0 ** 2)
            return e1

    def elliptic_operator(self, d=1, c=None, r=None):
        """
        @brief Assemble the finite difference matrix for a general elliptic operator.

        The elliptic operator has the form: -d(x) * u'' + c(x) * u' + r(x) * u.

        @param[in] d The diffusion coefficient, default is 1.
        @param[in] c The convection coefficient, default is None.
        @param[in] r The reaction coefficient, default is None.
        """

        h = self.h
        NN = self.number_of_nodes()
        k = np.arange(NN)
        node = self.node

        if callable(d):
            d = d(node)
        
        if c is None:
            c = np.zeros(NN)
        elif callable(c):
            c = c(node)
        
        if r is None:
            r = np.zeros(NN)
        elif callable(r):
            r = r(node)

        # Assemble diffusion term
        cx = d / (h ** 2)
        A = diags([2 * cx], [0], shape=(NN, NN), format='csr')

        val = np.broadcast_to(-cx, (NN - 1,))
        I = k[1:]
        J = k[:-1]
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        # Assemble convection term
        cc = c / (2 * h)
        val_c = np.broadcast_to(cc[:-1], (NN - 1,))
        A += csr_matrix((val_c, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A -= csr_matrix((val_c, (J, I)), shape=(NN, NN), dtype=self.ftype)

        # Assemble reaction term
        A += diags([r], [0], shape=(NN, NN), format='csr')

        return A

    def laplace_operator(self):
        """
        @brief Assemble the finite difference matrix for the Laplace operator u''.

        @note Note that boundary conditions are not handled in this function.

        """
        h = self.h
        cx = 1/(h**2)
        NN = self.number_of_nodes()
        k = np.arange(NN)

        A = diags([2*cx], [0], shape=(NN, NN), format='csr')

        val = np.broadcast_to(-cx, (NN-1, ))
        I = k[1:]
        J = k[0:-1]
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)
        return A

    def apply_dirichlet_bc(self, uh, A, f):
        """
        @brief 组装 u_xx 对应的有限差分矩阵，考虑了 Dirichlet 边界
        """
        NN = self.number_of_nodes()
        isBdNode = np.zeros(NN, dtype=np.bool_)
        isBdNode[[0,-1]] = True

        f -= A@uh
        f[isBdNode] = uh[isBdNode]
    
        bdIdx = np.zeros(A.shape[0], dtype=np.int_)
        bdIdx[isBdNode] = 1
        D0 = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
        D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
        A = D0@A@D0 + D1
        return A, f


    def wave_equation(self, r, theta):
        n0 = self.NC -1
        A0 = diags([1+2*r**2*theta, -r**2*theta, -r**2*theta], 
                [0, 1, -1], shape=(n0, n0), format='csr')
        A1 = diags([2 - 2*r**2*(1-2*theta), r**2*(1-2*theta), r**2*(1-2*theta)], 
                [0, 1, -1], shape=(n0, n0), format='csr')
        A2 = diags([-1 - 2*r**2*theta, r**2*theta, r**2*theta], 
                [0, 1, -1], shape=(n0, n0), format='csr')

        return A0, A1, A2

    def show_function(self, plot, uh):
        """
        @brief 画出定义在网格上的离散函数
        """
        if isinstance(plot, ModuleType):
            fig = plot.figure()
            fig.set_facecolor('white')
            axes = fig.gca()
        else:
            axes = plot
        node = self.node
        line = axes.plot(node, uh)
        return line

    def show_animation(self, fig, axes, box, forward, fname='test.mp4',
                       init=None, fargs=None,
                       frames=1000, lw=2, interval=50):

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
            s = "frame=%05d, time=%0.8f" % (n, t)
            print(s)
            axes.set_title(s)
            return line

        ani = animation.FuncAnimation(fig, func, frames=frames,
                                      init_func=init_func,
                                      interval=interval)
        ani.save(fname)

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
            color='r', markersize=100,
            fontsize=20, fontcolor='k'):

        if node is None:
            node = self.node

        find_node(axes, node,
                index=index, showindex=showindex,
                color=color, markersize=markersize,
                fontsize=fontsize, fontcolor=fontcolor)

    def find_cell(self, axes,
            index=None, showindex=False,
            color='g', markersize=150,
            fontsize=24, fontcolor='g'):

        find_entity(axes, self, entity='cell',
                index=index, showindex=showindex,
                color=color, markersize=markersize,
                fontsize=fontsize, fontcolor=fontcolor)
