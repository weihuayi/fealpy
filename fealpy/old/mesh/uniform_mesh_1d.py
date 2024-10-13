import numpy as np
import warnings

from typing import Callable, Union, Tuple, List
from scipy.sparse import csr_matrix, coo_matrix, diags, spdiags, spmatrix
from types import ModuleType

from .mesh_base import Mesh, Plotable
from .mesh_data_structure import StructureMesh1dDataStructure, HomogeneousMeshDS

# 这个数据结构为有限元接口服务
from ..quadrature import GaussLegendreQuadrature

## @defgroup FEMInterface 
## @defgroup FDMInterface
## @defgroup GeneralInterface
class UniformMesh1d(Mesh, Plotable):
    """
    @brief    A class for representing a uniformly partitioned one-dimensional mesh.
    """
    def __init__(self, 
            extent: List[np.int_],
            h: float = 1.0,
            origin: float = 0.0,
            itype: type = np.int_,
            ftype: type = np.float64):
        """
        @brief        Initialize the 1D uniform mesh.

        @param[in]    extent: A tuple representing the range of the mesh in the x direction.
        @param[in]    h: Mesh step size.
        @param[in]    origin: Coordinate of the starting point.
        @param[in]    itype: Integer type to be used, default: np.int_.
        @param[in]    ftype: Floating point type to be used, default: np.float64.

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
        self.type = "U1D"
        # Data structure for finite element computation
        self.ds: StructureMesh1dDataStructure = StructureMesh1dDataStructure(self.nx, itype=itype)

    ## @ingroup GeneralInterface
    def uniform_refine(self, n=1, returnim=False):
        """
        @brief: Perform a uniform refinement of the mesh.
        @param[in] n: Number of refinements to perform (default: 1).
        @param[in] returnim: Boolean flag to return the interpolation matrix (default: False).
        @return: If returnim is True, a list of interpolation matrices is returned.
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

        # Data structure for finite element computation
        self.ds: StructureMesh1dDataStructure = StructureMesh1dDataStructure(self.nx, itype=self.itype)

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
    def show_function(self, plot, uh, box=None):
        """
        @brief 画出定义在网格上的离散函数
        """
        if isinstance(plot, ModuleType):
            fig = plot.figure()
            fig.set_facecolor('white')
            axes = fig.gca()
        else:
            axes = plot

        # 设置 x 轴和 y 轴的显示范围
        if box is not None:
            axes.set_xlim(box[0], box[1])
            axes.set_ylim(box[2], box[3])

        node = self.node
        line = axes.plot(node, uh)
        return line


    ## @ingroup GeneralInterface
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from typing import Optional, Callable, Any, Tuple
    def show_animation(self, 
                fig: Figure, 
                axes: Axes, 
                box: Tuple[float, float, float, float], 
                advance: Callable[[int, Any], Tuple[Any, float]], 
                fname: str = 'test.mp4',
                init: Optional[Callable[..., Any]] = None, 
                fargs: Optional[Tuple] = None,
                frames: int = 1000, 
                interval: int = 50,
                **kwargs) -> None:
        """
        在一维一致网格中生成一个动画

        @param fig: matplotlib的Figure对象，用于绘制动画
        @param axes: matplotlib的Axes对象，用于设置坐标轴和画图
        @param box: 一个四元组，分别表示x轴的最小值、最大值和y轴的最小值、最大值
        @param advance: 一个函数，接受当前帧序号（和可选的其他参数），返回当前时间步的解和时间
        @param fname: 字符串，保存的视频文件的名称，默认为'test.mp4'
        @param init: 一个可选的函数，用于初始化线的数据，返回初始化的数据，默认为None
        @param fargs: 一个可选的元组，包含传递给init和advance函数的额外参数，默认为None
        @param frames: 整数，动画的帧数，默认为1000
        @param interval: 整数，动画中每帧之间的间隔（以毫秒为单位），默认为50
        @param kwargs: 其他的可选关键字参数，例如线的宽度（lw）、线的样式（linestyle）、线条上标记点的样式（marker）、线的颜色（color）等

        @return: None
        """
        import matplotlib.animation as animation
        
        x = self.node
        line, = axes.plot([], [], **kwargs)

        if init is not None:
            if fargs is not None:
                init_data = init(*fargs)
            else:
                init_data = init()
        else:
            init_data, _ = advance(0)
        line.set_data(x, init_data)  

        axes.set_xlim(box[0], box[1])
        axes.set_ylim(box[2], box[3])

        def func(n, *fargs):
            uh, t = advance(n, *fargs)
            line.set_data(x, uh) 
            s = "frame=%05d, time=%0.8f" % (n, t)
            print(s)
            axes.set_title(s)
            return line

        ani = animation.FuncAnimation(fig, func, frames=frames, interval=interval)
        ani.save(fname)



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
        @brief: Get the coordinates of the nodes in the mesh.
        @return: A NumPy array of shape (NN, ) containing the coordinates of the nodes.
        @details: This function calculates the coordinates of the nodes in the mesh based on the
                 mesh's origin, step size, and the number of cells in the x directions.
                 It returns a NumPy array with the coordinates of each node.

        """
        GD = self.geo_dimension()
        nx = self.nx
        node = np.linspace(self.origin, self.origin + nx * self.h, nx+1)
        return node

    ## @ingroup FDMInterface
    def edge_barycenter(self):
        """
        @brief
        Note: 一维中，edge 和 cell 相同
        """
        nx = self.nx
        box = [self.origin + self.h/2, self.origin + self.h/2 + (nx - 1) * self.h]
        bc = np.linspace(box[0], box[1], nx)
        return bc

    ## @ingroup FDMInterface
    def face_barycenter(self):
        """
        @brief
        """
        return self.node

    ## @ingroup FDMInterface
    def cell_barycenter(self):
        """
        @brief
        Note: 一维中，edge 和 cell 相同
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
        @brief                 Interpolate the given function f on the mesh based on the specified interpolation type.

        @param[in] f           The function to be interpolated on the mesh.
        @param[in] intertype   The type of interpolation, either 'node' or 'cell' (default: 'node').

        @return                The interpolated values of the function f on the mesh nodes or cell barycenters.

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
    def error(self, u: Callable, uh: np.ndarray, errortype: str = 'all') -> Union[np.float64, Tuple[np.float64, np.float64, np.float64]]:
        """
        计算真实解和数值解之间的误差

        @param[in] u: 真实解的函数
        @param[in] uh: 数值解的数组
        @param[in] errortype: 误差类型，可以是'all'、'max'、'L2' 或 'H1'
        @return 如果errortype为'all'，则返回一个包含最大误差、L2误差和H1误差的元组；
                如果errortype为'max'，则返回最大误差；
                如果errortype为'L2'，则返回L2误差；
                如果errortype为'H1'，则返回H1误差
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
    def matrix_operator(self,diag,val_1,val_2,I,J,NN):
        A = diags([diag],[0],shape=(NN,NN),dtype=self.ftype)
        A += csr_matrix((val_1, (I, J)), shape=(NN,NN), dtype=self.ftype)
        A += csr_matrix((val_2, (J, I)), shape=(NN,NN), dtype=self.ftype)
        
        return A
    ## @ingroup FDMInterface
    def elliptic_operator(self, d=1, c=None, r=None):
        """
        @brief 对于一般的椭圆算子组装有限差分矩阵

        椭圆算子的形式: -d(x) * u'' + c(x) * u' + r(x) * u.

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
            c = np.zeros(NN-1)
        elif callable(c):
            c = c(node)
        
        if r is None:
            r = np.zeros(NN)
        elif callable(r):
            r = r(node)

        I = k[1:]
        J = k[:-1]

        # Assemble diffusion term
        cx = d / (h ** 2)
        val = np.broadcast_to(-cx, (NN - 1,))

        A = self.matrix_operator(2*cx,val,val,I,J,NN)
        
        # Assemble convection term
        cc = c / (2 * h)
        val_c = np.broadcast_to(cc, (NN - 1,))

        A += self.matrix_operator(r,-val_c,val_c,I,J,NN)


        return A

    ## @ingroup FDMInterface
    def laplace_operator(self) -> csr_matrix:
        """
        @brief 组装 Laplace 算子 ∆u 对应的有限差分离散矩阵

        @note 并未处理边界条件
        """
        h = self.h
        cx = 1/(h**2)
        NN = self.number_of_nodes()
        k = np.arange(NN)

        I = k[1:]
        J = k[0:-1]

        val = np.broadcast_to(-cx, (NN-1, ))
        A = self.matrix_operator(2*cx,val,val,I,J,NN)

        return A


    ## @ingroup FDMInterface
    def cdr_operator(self) -> csr_matrix: 
        """
        @brief： 组装 C D R 算子对应的有限差分离散矩阵

        @note 并未处理边界条件
        """
        h = self.h
        cx = 5 / (self.h ** 2)
        NN = self.number_of_nodes()
        k = np.arange(NN)

        I = k[1:]
        J = k[0:-1]

        val = np.broadcast_to(cx, (NN-1,))
        A = self.matrix_operator(-2*cx,val,val,I,J,NN)

        cy = 1 / (2 * h)
        val2 = np.broadcast_to(cy, (NN-1,))
        A += self.matrix_operator(0.001,-val2,val2,I,J,NN)

        return A

    ## @ingroup FDMInterface
    def apply_dirichlet_bc(self, 
        gD: Callable[[np.ndarray], np.ndarray], 
        A: spmatrix, 
        f: np.ndarray, 
        uh: Union[np.ndarray, np.flatiter, None] = None, 
        threshold: Optional[Union[int, Callable[[np.ndarray], np.ndarray]]] = None) -> Tuple[np.ndarray, np.ndarray]:

        """
        应用 Dirichlet 边界条件，并更新给定的矩阵 A 和向量 f

        参数：
        gD : Callable[[np.ndarray, np.float64], np.ndarray]
            描述 Dirichlet 边界条件的函数
            这个函数接收两个参数，一个是网格节点的坐标（numpy 数组）
            另一个是时间 t（浮点数），但在定义 gD 的时候已经把它，例如（t + tau）"硬编码"（即预先设定）进去了
            并返回一个 numpy 数组，数组中的值是在给定的网格节点和时间 t 上的 Dirichlet 边界条件的值

        A : spmatrix
            需要更新的矩阵。此函数将直接修改这个矩阵来应用 Dirichlet 边界条件

        f : np.ndarray
            需要更新的向量。此函数将直接修改这个向量来应用 Dirichlet 边界条件

        uh : Union[np.ndarray, np.flatiter, None] = None
            表示网格上的函数值的 numpy 数组。
            如果提供了此参数，则此函数将直接修改这个数组以应用 Dirichlet 边界条件
            如果此参数为 None（默认），则此函数将创建一个新的网格函数数组

        threshold : Optional[Union[int, Callable[[np.ndarray], np.ndarray]]]
            用于确定哪些网格节点应用 Dirichlet 边界条件
            如果 threshold 是 None（默认），则应用 Dirichlet 边界条件到所有边界节点上
            如果 threshold 是一个整数，则只将 Dirichlet 边界条件应用于具有该索引的节点上
            如果 threshold 是一个函数，则将该函数应用于网格节点的坐标，并将 Dirichlet 边界条件应用于该函数返回 True 的所有节点上

        返回：
        A : np.ndarray
            更新后的矩阵

        f : np.ndarray
            更新后的向量
        """
        if uh is None:
            uh = self.function('node')

        node = self.node
        if threshold is None:
            isBdNode = self.ds.boundary_node_flag()
            index = isBdNode
        elif isinstance(threshold, int):
            index = threshold
        elif callable(threshold):
            index = threshold(node)
        else:
            raise ValueError(f"Invalid threshold: {threshold}")

        uh[index]  = gD(node[index])

        f -= A@uh
        f[index] = uh[index]
    
        bdIdx = np.zeros(A.shape[0], dtype=np.int_)
        bdIdx[index] = 1
        D0 = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
        D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
        A = D0@A@D0 + D1
        return A, f 


    ## @ingroup FDMInterface
    def apply_neumann_bc(self, gN, A, f, uh=None, threshold=None):
        """
        @brief 考虑 Neumann 边界，只有右边界点为 Neumann 边界条件

        TODO:错误！
        """
        if uh is None:
            uh = self.function('node')

        node = self.node
        if threshold is None:
            isBdNode = self.ds.boundary_node_flag()
            index = isBdNode
        elif isinstance(threshold, int):
            index = threshold
        elif callable(threshold):
            index = threshold(node)
        else:
            index = self.ds.boundary_node_flag()

        # uh[index]  = gN(node[index])

        # f -= A@uh
        # f[index] = uh[index]
        f[index] = gN(node[index])
    
        bdIdx = np.zeros(A.shape[0], dtype=np.int_)
        bdIdx[index] = 1
        D0 = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
        bdIdx1 = np.zeros(A.shape[0], dtype=np.float64)
        bdIdx1[index] = 1 / self.h
        bdIdx2 = np.zeros(A.shape[0]-1, dtype=float)
        bdIdx2[-1] = -1 / self.h
        D1 = spdiags(bdIdx1, 0, A.shape[0], A.shape[0])
        D1 += spdiags(bdIdx2, -1, A.shape[0], A.shape[0])
        A = D0@A@D0 + D1
        return A, f


    ## @ingroup FDMInterface
    def update_dirichlet_bc(self, 
        gD: Callable[[np.ndarray], np.ndarray], 
        uh: np.ndarray, 
        threshold: Optional[Union[int, Callable[[np.ndarray], np.float64]]] = None) -> None:
        """
        更新网格函数 uh 的 Dirichlet 边界值

        参数：
        gD : Callable[[np.ndarray], np.ndarray]
            描述 Dirichlet 边界条件的函数
            这个函数接收两个参数，一个是网格节点的坐标（numpy 数组）
            另一个是时间 t（浮点数），但在定义 gD 的时候已经把它，例如（t + tau）"硬编码"（即预先设定）进去了
            并返回一个 numpy 数组，数组中的值是在给定的网格节点和时间 t 上的 Dirichlet 边界条件的值

        uh : np.ndarray
            表示网格上的函数值的 numpy 数组
            此函数将更新这个数组的部分元素，以应用 Dirichlet 边界条件

        threshold : Optional[Union[int, Callable[[np.ndarray], np.ndarray]]]
            用于确定哪些网格节点应用 Dirichlet 边界条件
            如果 threshold 是 None（默认），则应用 Dirichlet 边界条件到所有边界节点上
            如果 threshold 是一个整数，则只将 Dirichlet 边界条件应用于具有该索引的节点上
            如果 threshold 是一个函数，则将该函数应用于网格节点的坐标，并将 Dirichlet 边界条件应用于该函数返回 True 的所有节点上

        返回：
        None。这个函数直接修改传入的 uh 数组，而不返回任何值
        """
        node = self.node
        if threshold is None:
            isBdNode = self.ds.boundary_node_flag()
            uh[isBdNode]  = gD(node[isBdNode])
        elif isinstance(threshold, int):
            uh[threshold] = gD(node[threshold])
        elif callable(threshold):
            isBdNode = threshold(node)
            uh[isBdNode]  = gD(node[isBdNode])

    def parabolic_operator_forward(self, tau):
        """
        @brief 生成抛物方程的向前差分迭代矩阵

        @param[in] tau float, 当前时间步长
        """
        r = tau/self.h**2 
        if r > 0.5:
            raise ValueError(f"The r: {r} should be smaller than 0.5")

        NN = self.number_of_nodes()
        k = np.arange(NN)

        I = k[1:]
        J = k[0:-1]

        val = np.broadcast_to(r, (NN-1, ))
        A = self.matrix_operator(1-2*r,val,val,I,J,NN)

        return A

    def parabolic_operator_backward(self, tau):
        """
        @brief 生成抛物方程的向后差分迭代矩阵

        @param[in] tau float, 当前时间步长
        """
        r = tau/self.h**2 

        NN = self.number_of_nodes()
        k = np.arange(NN)

        I = k[1:]
        J = k[0:-1] 

        val = np.broadcast_to(-r, (NN-1, ))
        A = self.matrix_operator(1+2*r,val,val,I,J,NN)


        return A

    def parabolic_operator_crank_nicholson(self, tau):
        """
        @brief 生成抛物方程的 CN 差分格式的迭代矩阵

        @param[in] tau float, 当前时间步长
        """
        r = tau/self.h**2 

        NN = self.number_of_nodes()
        k = np.arange(NN)

        I = k[1:]
        J = k[0:-1]

        val = np.broadcast_to(-r/2, (NN-1, ))
        A = self.matrix_operator(1+r,val,val,I,J,NN)

        val = np.broadcast_to(r/2, (NN-1, ))
        B = self.matrix_operator(1-r,val,val,I,J,NN)

        return A, B


    ## @ingroup FDMInterface
    def wave_operator_explicit(self, tau: float, a: float = 1.0):
        """
        @brief 生成波动方程的显格式离散矩阵

        @param[in] tau float, 时间步长
        @param[in] a float, 波速，默认值为 1

        @return 离散矩阵 A
        """
        r = a * tau / self.h 

        NN = self.number_of_nodes()
        k = np.arange(NN)

        I = k[1:]
        J = k[0:-1]

        val = np.broadcast_to(r**2, (NN-1, ))
        A = self.matrix_operator(2-2*r**2,val,val,I,J,NN)


        return A


    ## @ingroup FDMInterface
    def wave_operator_implicit(self, tau: float, a: float = 1.0, theta: float = 0.25):
        """
        @brief 生成波动方程的隐格式离散矩阵

        @param[in] tau float, 时间步长
        @param[in] a float, 波速，默认值为 1
        @param[in] theta float, 时间离散格式参数，默认值为 0.25

        @return 三个离散矩阵 A0, A1, A2，分别对应于不同的时间步
        """
        r = a * tau / self.h 

        NN = self.number_of_nodes()
        k = np.arange(NN)

        I = k[1:]
        J = k[0:-1]
        
        val = np.broadcast_to(- r**2 * theta, (NN-1, ))
        A0 = self.matrix_operator(1 + 2 * r**2 * theta,val,val,I,J,NN)
        
        val = np.broadcast_to(r**2 * (1 - 2 * theta), (NN-1, ))
        A1 = self.matrix_operator(2 - 2 * r**2 * (1 - 2 * theta),val,val,I,J,NN)

        val = np.broadcast_to(r**2 * theta, (NN-1, ))
        A2 = self.matrix_operator(- 1 - 2 * r**2 * theta,val,val,I,J,NN)

        return A0, A1, A2


    ## @ingroup FDMInterface
    def wave_operator(self, tau: float, a: float = 1.0, theta: float = 0.5):
        """
        @brief 生成波动方程的离散矩阵

        @param[in] tau float, 时间步长
        @param[in] a float, 波速，默认值为 1
        @param[in] theta float, 时间离散格式参数，默认值为 0.5

        @return 三个离散矩阵 A0, A1, A2，分别对应于不同的时间步
        """
        r = a * tau / self.h 

        NN = self.number_of_nodes()
        k = np.arange(NN)

        I = k[1:]
        J = k[0:-1]

        val = np.broadcast_to(- r**2 * theta, (NN-1, ))
        A0 = self.matrix_operator(1 + 2 * r**2 * theta,val,val,I,J,NN)

        val = np.broadcast_to(r**2 * (1 - 2 * theta), (NN-1, ))
        A1 = self.matrix_operator(2 - 2 * r**2 * (1 - 2 * theta),val,val,I,J,NN)        

        val = np.broadcast_to(r**2 * theta, (NN-1, ))
        A2 = self.matrix_operator(- 1 - 2 * r**2 * theta,val,val,I,J,NN)         

        return A0, A1, A2


    ## @ingroup FDMInterface
    def hyperbolic_operator_explicity_upwind(self, tau, a=1):
        """
        @brief 双曲方程的显式迎风格式
        """
        r = a*tau/self.h

        if r > 1.0:
            raise ValueError(f"The r: {r} should be smaller than 1.0")

        NN = self.number_of_nodes()
        k = np.arange(NN)

        I = k[1:]
        J = k[0:-1]
        
        if a > 0:
            val_0 = np.broadcast_to(r, (NN-1, ))
            val_1 = np.broadcast_to(0, (NN-1, ))
            A = self.matrix_operator(1-r,val_0,val_1,I,J,NN)

            return A
        else:
            val_0 = np.broadcast_to(-r, (NN-1, ))
            val_1 = np.broadcast_to(r, (NN-1, ))
            B = self.matrix_operator(1+r,val_0,val_1,I,J,NN)

            return B


    def hyperbolic_operator_central_upwind(self, tau, a=1):
        """
        @brief 双曲方程的中心差分格式

        @param[in] tau float, 当前时间步长
        NOTE 该格式不稳定
        """
        r = a*tau/self.h
        
        NN = self.number_of_nodes()
        k = np.arange(NN)

        I = k[1:]
        J = k[0:-1]

        val0 = np.broadcast_to(-r/2, (NN-1, ))
        val1 = np.broadcast_to(r/2, (NN-1, ))
        A = self.matrix_operator(1,val1,val0,I,J,NN)

        return A


    def hyperbolic_operator_explicity_upwind_with_viscous(self, tau, a=1):
        """
        @brief 双曲方程带粘性项的显式迎风格式 
        """
        r = a*tau/self.h
    
        if r > 1.0:
            raise ValueError(f"The r: {r} should be smaller than 1.0")
    
        NN = self.number_of_nodes()
        k = np.arange(NN)
    
        I = k[1:]
        J = k[0:-1]  

        val0 = np.broadcast_to(0, (NN-1, ))
        val1 = np.broadcast_to(r, (NN-1, ))
        A = self.matrix_operator(1-r,val1,val0,I,J,NN)

        return A


    def hyperbolic_operator_explicity_lax_friedrichs(self, tau, a=1):
        """
        @brief 积分守恒型 lax_friedrichs 格式
        """
        r = a*tau/self.h
    
        if r > 1.0:
            raise ValueError(f"The r: {r} should be smaller than 1.0")

        NN = self.number_of_nodes()
        k = np.arange(NN)

        I = k[1:]
        J = k[0:-1]

        val0 = np.broadcast_to(1/2*(1 - r), (NN-1, ))
        val1 = np.broadcast_to(1/2*(1 + r), (NN-1, ))

        A = self.matrix_operator(0,val1,val0,I,J,NN)
    
        return A


    def hyperbolic_operator_implicity_upwind(self, a, tau):
        """
        @brief 隐式迎风格式
        """
        r = a*tau/self.h

        NN = self.number_of_nodes()
        k = np.arange(NN)

        I = k[1:]
        J = k[0:-1]
        
        if a > 0:
            val0 = np.broadcast_to(0, (NN-1, ))
            val1 = np.broadcast_to(-r, (NN-1, ))
            A = self.matrix_operator(1+r,val0,val1,I,J,NN)

            return A
        else:
            val0 = np.broadcast_to(r, (NN-1, ))
            val1 = np.broadcast_to(0, (NN-1, ))
            B = self.matrix_operator(1-r,val0,val1,I,J,NN)

            return B

    def hyperbolic_operator_implicity_center(self, a, tau):
        """
        @brief 隐式中心格式
        """
        r = a*tau/self.h
    
        NN = self.number_of_nodes()
        k = np.arange(NN)

        I = k[1:]
        J = k[0:-1]

        val = np.broadcast_to(r/2, (NN-1, ))
        A = self.matrix_operator(1,val,-val,I,J,NN)

        return A
    def hyperbolic_operator_leap_frog(self, a, tau):
        """
        @brief 蛙跳格式
        """
        r = a*tau/self.h
    
        if r > 1.0:
            raise ValueError(f"The r: {r} should be smaller than 1.0")
    
        NN = self.number_of_nodes()
        k = np.arange(NN)
        I = k[1:]
        J = k[0:-1]

        val = np.broadcast_to(-r, (NN-1, ))
        A = self.matrix_operator(0,-val,val,I,J,NN)
       
        return A

    def hyperbolic_operator_lax_wendroff(self, a, tau):
        """
        @brief Lax-Wendroff 格式
        """
        r = a*tau/self.h
    
        if r > 1.0:
            raise ValueError(f"The r: {r} should be smaller than 1.0")

        NN = self.number_of_nodes()
        k = np.arange(NN)

        I = k[1:]
        J = k[0:-1]

        val0 = np.broadcast_to(-r/2 + r**2/2, (NN-1, ))
        val1 = np.broadcast_to(r/2 + r**2/2 , (NN-1, ))
        A=self.matrix_operator(1 - r**2,val1,val0,I,J,NN)
    
        return A


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
        @brief: Get the entity (either cell or node) based on the given entity type.
        @param[in] etype: The type of entity, either 'cell', 'edge' or 1, 'node', 'face' or 0.
        @return: The cell or node array based on the input entity type.

        @throws ValueError if the given etype is invalid.
        """
        if etype in {'cell', 'edge', 1}:
            return self.ds.cell[index]
        elif etype in {'node', 'face', 0}:
            return self.node[index].reshape(-1, 1)
        else:
            raise ValueError("The entiry type `{etype}` is not support!")

    ## @ingroup FEMInterface
    def entity_barycenter(self, etype, index=np.s_[:]):
        """
        print("node:", node)
        @brief: Calculate the barycenter of the specified entity.
        @param[in] etype: The type of entity, either 'cell', 'edge', 1, 'node', 'face' 0.
        @return: The barycenter of the given entity type.
        @throws ValueError if the given etype is invalid.
        """
        if etype in {'edge', 'cell', 1}:
            return self.cell_barycenter().reshape(-1, 1)[index]
        elif etype in {'node', 'face', 0}:
            return self.node.reshape(-1, 1)[index]
        else:
            raise ValueError('the entity type `{etype}` is not correct!')

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

# 调用画图接口
UniformMesh1d.set_ploter('1d')
