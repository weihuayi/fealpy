import numpy as np
import warnings
from scipy.sparse import csr_matrix, coo_matrix, diags, spdiags
from types import ModuleType
from typing import Tuple 

# 这个数据结构为有限元接口服务
from ..quadrature import GaussLegendreQuadrature
from .StructureMesh1dDataStructure import StructureMesh1dDataStructure
from .mesh_tools import find_node, find_entity, show_mesh_1d


## @defgroup FEMInterface 
## @defgroup FDMInterface
## @defgroup GeneralInterface
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
        @brief Initialize the 1D uniform mesh.

        @param[in] extent: A tuple representing the range of the mesh in the x direction.
        @param[in] h: Mesh step size.
        @param[in] origin: Coordinate of the starting point.
        @param[in] itype: Integer type to be used, default: np.int_.
        @param[in] ftype: Floating point type to be used, default: np.float64.

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
        # Mesh properties
        self.extent = extent
        self.h = h
        self.origin = origin

        # Mesh dimensions
        self.nx = extent[1] - extent[0]
        self.NC = self.nx
        self.NN = self.NC + 1

        self.itype = itype
        self.ftype = ftype

        # Data structure for finite element computation
        self.ds: StructureMesh1dDataStructure = StructureMesh1dDataStructure(self.nx, itype=itype)


    ## @ingroup GeneralInterface
    def number_of_nodes(self):
        """
        @brief Get the number of nodes in the mesh.

        @return The number of nodes.
        """
        return self.NN

    ## @ingroup GeneralInterface
    def number_of_edges(self):
        """
        @brief Get the number of nodes in the mesh.

        @note `edge` is the 1D entity.

       return The number of edges.

        """
        return self.NC

    ## @ingroup GeneralInterface
    def number_of_faces(self):
        """
        @brief Get the number of nodes in the mesh.

        @note `face` is the 0D entity

        @return The number of faces.

        """
        return self.NN

    ## @ingroup GeneralInterface
    def number_of_cells(self):
        """
        @brief Get the number of cells in the mesh.

        @return The number of cells.
        """
        return self.NC

    ## @ingroup GeneralInterface
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
                A = self.interpolation_matrix() #TODO: 实现这个功能
                nodeImatrix.append(A)

        if returnim:
            return nodeImatrix

    ## @ingroup GeneralInterface
    def cell_length(self):
        """
        @brief 返回单元的长度，注意这里只返回一个值（因为所有单元长度相同）
        """
        return self.h

    ## @ingroup GeneralInterface
    def cell_location(self, ps):
        """
        @brief 给定一组点，确定所有点所在的单元
        @todo 如果 ps 取最右边的端点应该会出问题！

        """
        hx = self.h
        nx = self.ds.nx

        v = ps - self.origin
        n0 = v//hx

        return n0.astype('int64')

    ## @ingroup GeneralInterface
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

    ## @ingroup GeneralInterface
    def show_animation(self, fig, axes, box, forward, fname='test.mp4',
                       init=None, fargs=None,
                       frames=1000, lw=2, interval=50):
        """
        @brief
        """
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

    ## @ingroup GeneralInterface
    def add_plot(self, plot,
            nodecolor='k', cellcolor='k',
            aspect='equal', linewidths=1, markersize=20,
            showaxis=False):
        """
        @brief
        """

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

    ## @ingroup GeneralInterface
    def find_node(self, axes, node=None,
            index=None, showindex=False,
            color='r', markersize=100,
            fontsize=20, fontcolor='k'):
        """
        @brief
        """

        if node is None:
            node = self.node

        find_node(axes, node,
                index=index, showindex=showindex,
                color=color, markersize=markersize,
                fontsize=fontsize, fontcolor=fontcolor)

    ## @ingroup GeneralInterface
    def find_cell(self, axes,
            index=None, showindex=False,
            color='g', markersize=150,
            fontsize=24, fontcolor='g'):
        """
        @brief
        """

        find_entity(axes, self, entity='cell',
                index=index, showindex=showindex,
                color=color, markersize=markersize,
                fontsize=fontsize, fontcolor=fontcolor)

    ## @ingroup GeneralInterface
    def to_vtk_file(self, filename, celldata=None, nodedata=None):
        """
        @brief
        """
        pass

    ## @ingroup FDMInterface
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

    ## @ingroup FDMInterface
    def cell_barycenter(self):
        """
        @brief
        """
        nx = self.nx
        box = [self.origin + self.h/2, self.origin + self.h/2 + (nx - 1) * self.h]
        bc = np.linspace(box[0], box[1], nx)
        return bc

    ## @ingroup FDMInterface
    def function(self, etype='node', dtype=None, ex=0, flat=False):
        """
        @brief Returns an array defined on nodes or cells with all elements set to 0.

        @param[in] etype: The type of entity, either 'node' or 'cell' (default: 'node').
        @param[in] dtype: Data type for the array (default: None, which means self.ftype will be used).
        @param[in] ex: Non-negative integer to extend the discrete function outward by a certain width (default: 0).
        @param[in] flat 是否展开为一维向量，默认不展开

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
            raise ValueError('the entity `{etype}` is not correct!')
        if flat is False:
            return uh
        else:
            return uh.flat

    ## @ingroup FDMInterface
    def value(self, p, f):
        """
        @brief
        """
        pass

    ## @ingroup FDMInterface
    def gradient(self, f, order=1):
        """
        @brief 求网格函数 f 的梯度
        """
        hx = self.h
        fx = np.gradient(f, hx, edge_order=order)
        return fx 
   
    ## @ingroup FDMInterface
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
            bc = self.cell_barycenter('cell')
            F = f(bc)
        return F

    ## @ingroup FDMInterface
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
            bc = self.cell_barycenter('cell')
            F = f(bc)
        return F

    ## @ingroup FDMInterface
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

    ## @ingroup FDMInterface
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

    ## @ingroup FDMInterface
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

    ## @ingroup FDMInterface
    def apply_dirichlet_bc(self, gD, A, f, uh=None):
        """
        @brief 组装 u_xx 对应的有限差分矩阵，考虑了 Dirichlet 边界
        """
        if uh is None:
            uh = self.function('node')

        node = self.node
        isBdNode = self.ds.boundary_node_flag()
        uh[isBdNode]  = gD(node[isBdNode])

        f -= A@uh
        F[isBdNode] = uh[isBdNode]
    
        bdIdx = np.zeros(A.shape[0], dtype=np.int_)
        bdIdx[isBdNode] = 1
        D0 = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
        D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
        A = D0@A@D0 + D1
        return A, f 


    ## @ingroup FDMInterface
    def wave_equation(self, r, theta):
        n0 = self.NC -1
        A0 = diags([1+2*r**2*theta, -r**2*theta, -r**2*theta], 
                [0, 1, -1], shape=(n0, n0), format='csr')
        A1 = diags([2 - 2*r**2*(1-2*theta), r**2*(1-2*theta), r**2*(1-2*theta)], 
                [0, 1, -1], shape=(n0, n0), format='csr')
        A2 = diags([-1 - 2*r**2*theta, r**2*theta, r**2*theta], 
                [0, 1, -1], shape=(n0, n0), format='csr')

        return A0, A1, A2

    ## @ingroup FDMInterface
    def fast_sweeping_method(self, phi0):
        """
        @brief 均匀网格上的 fast sweeping method
        @param[in] phi 是一个离散的水平集函数
        """
        pass

    ## @ingroup FEMInterface
    def geo_dimension(self):
        """
        @brief Get the geometry dimension of the mesh.
        
        @return The geometry dimension (1 for 1D mesh).
        """
        return 1

    ## @ingroup FEMInterface
    def top_dimension(self):
        """
        @brief Get the topological dimension of the mesh.
        
        @return The topological dimension (1 for 1D mesh).
        """
        return 1
   
    ## @ingroup FEMInterface
    def integrator(self, q, etype='cell'):
        return GaussLegendreQuadrature(q)

    ## @ingroup FEMInterface
    def bc_to_point(self, bc, index=np.s_[:]):
        node = self.node if node is None else node
        cell = self.entity('cell', index=index)
        p = np.einsum('...j, ijk->...ik', bc, node[cell[index]])
        return p

    ## @ingroup FEMInterface
    def entity(self, etype, index=np.s_[:]):
        """
        @brief Get the entity (either cell or node) based on the given entity type.

        @param[in] etype The type of entity, either 'cell', 'edge' or 1, 'node', 'face' or 0.

        @return The cell or node array based on the input entity type.

        @throws ValueError if the given etype is invalid.
        """
        if etype in {'cell', 'edge', 1}:
            return self.ds.cell[index]
        elif etype in {'node', 'face', 0}:
            return self.node[index].reshape(-1, 1)
        else:
            raise ValueError(f"The entiry type `{etype}` is not support!")

    ## @ingroup FEMInterface
    def entity_barycenter(self, etype, index=np.s_[:]):
        """
        @brief Calculate the barycenter of the specified entity.

        @param[in] etype The type of entity, either 'cell', 1, 'node', 'face' or 0.

        @return The barycenter of the given entity type.

        @throws ValueError if the given etype is invalid.
        """
        if etype in {'cell', 1}:
            return self.cell_barycenter().reshape(-1, 1)[index]
        elif etype in {'node', 0}:
            return self.node.reshape(-1, 1)[index]
        else:
            raise ValueError(f'the entity type `{etype}` is not correct!')

    ## @ingroup FEMInterface
    def entity_measure(self, etype, index=np.s_[:]):
        if etype in {1, 'cell', 'edge'}:
            NC = self.number_of_cells() if index == np.s_[:] else len(index)
            return np.broadcast_to(self.cell_length(), shape=NC)
        elif etype in {0, 'face', 'node'}:
            return 0
        else:
            raise ValueError(f"The entity type '{etype}` is not correct!")
        pass

    ## @ingroup FEMInterface
    def multi_index_matrix(self, p, etype=1):
        ldof = p+1
        multiIndex = np.zeros((ldof, 2), dtype=np.int_)
        multiIndex[:, 0] = np.arange(p, -1, -1)
        multiIndex[:, 1] = p - multiIndex[:, 0]
        return multiIndex

    ## @ingroup FEMInterface
    def shape_function(self, bc, p=1):
        """
        @brief 
        """
        TD = bc.shape[-1] - 1 
        multiIndex = self.multi_index_matrix(p)
        c = np.arange(1, p+1, dtype=np.int_)
        P = 1.0/np.multiply.accumulate(c)
        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = np.arange(TD+1)
        phi = np.prod(A[..., multiIndex, idx], axis=-1)
        return phi

    ## @ingroup FEMInterface
    def grad_shape_function(self, bc, p=1, index=np.s_[:]):
        TD = self.top_dimension()

        multiIndex = self.multi_index_matrix(p)

        c = np.arange(1, p+1, dtype=self.itype)
        P = 1.0/np.multiply.accumulate(c)

        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., np.newaxis, :] - t.reshape(-1, 1)

        FF = np.einsum('...jk, m->...kjm', A[..., 1:, :], np.ones(p))
        FF[..., range(p), range(p)] = p
        np.cumprod(FF, axis=-2, out=FF)
        F = np.zeros(shape, dtype=self.ftype)
        F[..., 1:, :] = np.sum(np.tril(FF), axis=-1).swapaxes(-1, -2)
        F[..., 1:, :] *= P.reshape(-1, 1)

        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)

        Q = A[..., multiIndex, range(TD+1)]
        M = F[..., multiIndex, range(TD+1)]
        ldof = self.number_of_local_ipoints(p)
        shape = bc.shape[:-1]+(ldof, TD+1)
        R = np.zeros(shape, dtype=self.ftype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            R[..., i] = M[..., i]*np.prod(Q[..., idx], axis=-1)

        Dlambda = self.grad_lambda(index=index)
        gphi = np.einsum('...ij, kjm->...kim', R, Dlambda)
        return gphi #(..., NC, ldof, GD)

    ## @ingroup FEMInterface
    def grad_lambda(self, index=np.s_[:]):
        """
        @brief 计算所有单元上重心坐标函数的导数
        """
        node = self.entity('node')
        cell = self.entity('cell')
        GD = self.geo_dimension()
        NC = self.number_of_cells() if index == np.s_[:] else len(index)
        Dlambda = np.zeros((NC, 2, GD), dtype=self.ftype)
        Dlambda[:, 0, :] = -1/self.h
        Dlambda[:, 1, :] = 1/self.h 
        return Dlambda
   
    ## @ingroup FEMInterface
    def number_of_local_ipoints(self, p, iptype='cell'):
        return p+1
    
    ## @ingroup FEMInterface
    def number_of_global_ipoints(self, p):
        NN = self.number_of_nodes()
        NC = self.number_of_cells()
        return NN + (p-1)*NC

    ## @ingroup FEMInterface
    def interpolation_points(self, p):
        """
        @brief 获取网格上的所有插值点
        """
        node = self.entity('node')
        cell = self.entity('cell')
        if p == 1:
            return node
        if p > 1:
            NN = self.number_of_nodes()
            GD = self.geo_dimension()

            gdof = self.number_of_global_ipoints(p)
            ipoints = np.zeros((gdof, GD), dtype=self.ftype)
            ipoints[:NN, :] = node

            NC = self.number_of_cells()

            w = np.zeros((p-1, 2), dtype=np.float64)
            w[:, 0] = np.arange(p-1, 0, -1)/p
            w[:, 1] = w[-1::-1, 0]
            ipoints[NN:NN+(p-1)*NC, :] = np.einsum('ij, ...jm->...im', w, node[cell,:]).reshape(-1, GD)

    ## @ingroup FEMInterface
    def node_to_ipoint(self, p):
        NN = self.number_of_nodes()
        return np.arange(NN)

    ## @ingroup FEMInterface
    def edge_to_ipoint(self, p):
        return self.cell_to_ipoint(p)

    ## @ingroup FEMInterface
    def face_to_ipoint(self, p):
        NN = self.number_of_nodes()
        return np.arange(NN)

    ## @ingroup FEMInterface
    def cell_to_ipoint(self, p):
        NN = self.number_of_nodes()
        NC = self.number_of_cells()

        cell = self.entity('cell')
        cell2ipoints = np.zeros((NC, p+1), dtype=np.itype)
        cell2ipoints[:, [0, -1]] = cell 
        if p > 1:
            cell2ipoints[:, 1:-1] = NN + np.arange(NC*(p-1)).reshape(NC, p-1)
        return cell2ipoints

