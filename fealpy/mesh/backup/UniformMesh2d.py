import numpy as np
import warnings

from matplotlib.pyplot import Figure
from matplotlib.axes import Axes
from scipy.sparse import coo_matrix, csr_matrix, diags, spdiags, spmatrix
from types import ModuleType
from typing import Optional, Tuple, Callable, Any, Union, List
from .Mesh2d import Mesh2d

# 这个数据接口为有限元服务
from .StructureMesh2dDataStructure import StructureMesh2dDataStructure
from ..quadrature import TensorProductQuadrature, GaussLegendreQuadrature
from ..geometry import project

## @defgroup FEMInterface
## @defgroup FDMInterface
## @defgroup GeneralInterface
class UniformMesh2d(Mesh2d):
    """
    @brief A class for representing a two-dimensional structured mesh with uniform discretization in both x and y directions.
    """
    def __init__(self,
            extent: Tuple[int, int, int, int],
            h: Tuple[float, float] = (1.0, 1.0),
            origin: Tuple[float, float] = (0.0, 0.0),
            itype: type = np.int_,
            ftype: type = np.float64):
        """
        @brief Initialize the 2D uniform mesh.

        @param[in] extent: A tuple representing the range of the mesh in the x and y directions.
        @param[in] h: A tuple representing the mesh step sizes in the x and y directions, default: (1.0, 1.0).
        @param[in] origin: A tuple representing the coordinates of the starting point, default: (0.0, 0.0).
        @param[in] itype: Integer type to be used, default: np.int_.
        @param[in] ftype: Floating point type to be used, default: np.float64.

        @note The extent parameter defines the index range in the x and y directions.
              We can define an index range starting from 0, e.g., [0, 10, 0, 10],
              or starting from a non-zero value, e.g., [2, 12, 3, 13]. The flexibility
              in the index range is mainly for handling different scenarios
              and data subsets, such as:
              - Subgrids
              - Parallel computing
              - Data cropping
              - Handling irregular data

        @example
        from fealpy.mesh import UniformMesh2d

        I = [0, 1, 0, 1]
        h = (0.1, 0.1)
        nx = int((I[1] - I[0])/h[0])
        ny = int((I[3] - I[2])/h[1])
        mesh = UniformMesh2d([0, nx, 0, ny], h=h, origin=(I[0], I[2]))

        """
        super().__init__()
        # Mesh properties
        self.extent: Tuple[int, int, int, int] = extent
        self.h: Tuple[float, float] = h
        self.origin: Tuple[float, float] = origin

        # Mesh dimensions
        self.nx: int = self.extent[1] - self.extent[0]
        self.ny: int = self.extent[3] - self.extent[2]
        self.NN: int = (self.nx + 1) * (self.ny + 1)
        self.NC: int = self.nx * self.ny

        self.itype: type = itype
        self.ftype: type = ftype

        self.meshtype = 'UniformMesh2d'

        # Data structure for finite element computation
        self.ds: StructureMesh2dDataStructure = StructureMesh2dDataStructure(self.nx, self.ny, itype)

    def geo_dimension(self):
        """
        @brief
        """
        return 2

    ## @ingroup GeneralInterface
    def uniform_refine(self, n=1, surface=None, interface=None, returnim=False):
        """
        @brief Perform uniform refinement on the mesh.

        @param[in] n The number of refinement iterations to perform, default: 1.
        @param[in] returnim If True, returns a list of interpolation matrices for each refinement iteration, default: False.

        @return A list of interpolation matrices if returnim is True, otherwise None.

        @details This function performs n iterations of uniform refinement on the mesh.
                 For each iteration, it updates the mesh extent, step size, number of cells, and number of nodes,
                 as well as the data structure. If returnim is True, it also calculates and returns the
                 interpolation matrices for each iteration.

        """
        for i in range(n):
            self.extent = [i*2 for i in self.extent]
            self.h = [h/2.0 for h in self.h]
            self.nx = self.extent[1] - self.extent[0]
            self.ny = self.extent[3] - self.extent[2]

            self.NC = self.nx * self.ny
            self.NN = (self.nx + 1) * (self.ny + 1)
            self.ds = StructureMesh2dDataStructure(self.nx, self.ny, itype=self.itype)

    ## @ingroup GeneralInterface
    def cell_area(self):
        """
        @brief 返回单元的面积，注意这里只返回一个值（因为所有单元面积相同）
        """
        return self.h[0]*self.h[1]

    ## @ingroup GeneralInterface
    def edge_length(self, index=np.s_[:]):
        """
        @brief 返回边长，注意这里返回两个值，一个 x 方向，一个 y 方向
        """
        return self.h[0], self.h[1]

    ## @ingroup GeneralInterface
    def cell_location(self, p):
        """
        @brief 给定一组点，确定所有点所在的单元

        """
        hx = self.h[0]
        hy = self.h[1]
        nx = self.ds.nx
        ny = self.ds.ny

        v = p - np.array(self.origin, dtype=self.ftype)
        n0 = v[..., 0]//hx
        n1 = v[..., 1]//hy

        return n0.astype('int64'), n1.astype('int64')

    ## @ingroup GeneralInterface
    def show_function(self, plot, uh, aspect=[1, 1, 1], cmap='rainbow'):
            
        """
        @brief    显示一个定义在网格节点上的函数
        @param    uh 网格节点上的函数值(二维数组)
        """
        if isinstance(plot, ModuleType):
            fig = plot.figure()
            axes = fig.add_subplot(111, projection='3d')
        else:
            axes = plot

        axes.set_box_aspect(aspect)
        axes.set_proj_type('ortho')

        node = self.node # 获取二维节点上的网格坐标
        return axes.plot_surface(node[..., 0], node[..., 1], uh, cmap=cmap)

    ## @ingroup GeneralInterface
    def show_animation(self, fig: Figure, axes: Axes, box: List[float], 
                    advance: Callable[[int, Any], Tuple[np.ndarray, float]], 
                    fname: str = 'test.mp4',
                    init: Optional[Callable] = None, 
                    fargs: Optional[Callable] = None, 
                    frames: np.int_ = 1000, 
                    interval: np.int_ = 50, 
                    plot_type: str = 'imshow',
                    cmap='rainbow') -> None:
        """
        @brief 生成求解过程动画并保存为指定文件名的视频文件

        @param fig         : plt.Figure | matplotlib 图形对象
        @param axes        : Union[plt.Axes, Axes3D] | matplotlib 坐标轴对象
        @param box         : list      | 定义图像显示范围
        @param advance     : Callable   | 用于更新求解过程的函数
        @param plot_type   : str, 可选  | 默认值为 'imshow',要显示的绘图类型('imshow', 'surface', 'contourf')
        @param fname       : str, 可选  | 输出动画文件的名称，默认值为 'test.mp4'
        @param init        : Optional[Callable] | 初始化函数（可选）
        @param fargs       : Optional[Tuple]    | 传递给 advance 函数的参数（可选）
        @param frames      : int        | 动画的总帧数，默认为 1000
        @param interval    : int        | 帧之间的时间间隔，默认为 50 毫秒
        """
        # 创建动画所需的类和函数
        import matplotlib.animation as animation
        # 绘制颜色条的类
        import matplotlib.colorbar as colorbar

        # 初始化二维网格数据
        uh, t = advance(0) 
        if plot_type == 'imshow':
            data = axes.imshow(uh, cmap=cmap, vmin=box[4], vmax=box[5], 
                    extent=box[0:4], interpolation='bicubic')
        elif plot_type == 'surface':
            X = self.node[..., 0]
            Y = self.node[..., 1]
            data = axes.plot_surface(X, Y, uh, linewidth=0, cmap=cmap, vmin=box[4],
                    vmax=box[5], rstride=1, cstride=1)
            axes.set_xlim(box[0], box[1])
            axes.set_ylim(box[2], box[3])
            axes.set_zlim(box[4], box[5])
        elif plot_type == 'contourf':
            X = self.node[..., 0]
            Y = self.node[..., 1]
            data = axes.contourf(X, Y, uh, cmap=cmap, vmin=box[4], vmax=box[5], 
                    interpolation='bicubic')
            # data 的值在每一帧更新时都会发生改变 颜色条会根据这些更改自动更新
            # 后续的代码中无需对颜色条进行额外的更新操作
            cbar = fig.colorbar(data, ax=axes)
        
        # 根据当前帧序号计算数值解，更新图像对象的数值数组，显示当前帧序号和时刻
        def func(n, *fargs):
            # 声明 data 为非局部变量 这样在 func 函数内部对 data 进行的修改会影响到外部的 data 变量
            nonlocal data
            # 计算当前时刻的数值解并返回，uh 是数值解，t 是当前时刻
            uh, t = advance(n, *fargs)
            if plot_type == 'imshow':
                # 更新 data 对象的数值数组。导致图像的颜色根据新的数值解 uh 更新
                data.set_array(uh)
                # 设置坐标轴的长宽比。'equal' 选项使得 x 轴和 y 轴的单位尺寸相等
                axes.set_aspect('equal')
            elif plot_type == 'surface':
                axes.clear()  # 清除当前帧的图像
                data = axes.plot_surface(X, Y, uh, cmap=cmap, vmin=box[4],
                        vmax=box[5])
                axes.set_xlim(box[0], box[1])
                axes.set_ylim(box[2], box[3])
                axes.set_zlim(box[4], box[5])
            elif plot_type == 'contourf':
                # 使用 contourf 时，每次更新图像时都会生成一个新的等高线填充层
                # data.collections 保存了所有已经生成的等高线填充层 
                # 更新图像时 需要将旧的等高线填充层从图形中移除 以免遮挡住新的等高线填充层
                for coll in data.collections:
                    axes.collections.remove(coll)
                data = axes.contourf(X, Y, uh, cmap=cmap, vmin=box[4],
                        vmax=box[5])
                axes.set_aspect('equal')

            # 创建一个格式化的字符串，显示当前帧序号 n 和当前时刻 t
            s = "frame=%05d, time=%0.8f"%(n, t)
            print(s)
            # 将格式化的字符串设置为坐标轴的标题
            axes.set_title(s)
            return data

        # 创建一个 funcanimation 对象，它将 fig 作为画布，func 作为帧更新函数，
        # init_func 作为初始化函数，用于在动画开始之前设置图像的初始状态，
        # fargs 作为一个元组，包含要传递给 func 函数的额外参数，frames 为帧数
        # 并设置动画间隔时间
        ani = animation.FuncAnimation(fig, func, init_func=init, fargs=fargs, 
                                      frames=frames, interval=interval)
        ani.save('{}_{}'.format(plot_type, fname))


    def show_animation_vtk(self):
        import vtkmodules.vtkInteractionStyle
        import vtkmodules.vtkRenderingOpenGL2
        from vtkmodules.vtkCommonColor import vtkNamedColors
        from vtkmodules.vtkCommonDataModel import vtkImageData
        from vtkmodules.vtkCommonCore import vtkLookupTable
        from vtkmodules.vtkFiltersGeometry import vtkImageDataGeometryFilter
        from vtkmodules.vtkRenderingCore import (
            vtkActor,
            vtkDataSetMapper,
            vtkRenderWindow,
            vtkRenderWindowInteractor,
            vtkRenderer
        )
        from vtk.util import numpy_support


    ## @ingroup GeneralInterface
    def to_vtk_file(self, filename, celldata=None, nodedata=None):
        """
        @brief
        """
        from pyevtk.hl import gridToVTK

        nx = self.ds.nx
        ny = self.ds.ny
        box = [self.origin[0], self.origin[0] + nx*self.h[0],
               self.origin[1], self.origin[1] + ny*self.h[1]]

        x = np.linspace(box[0], box[1], nx+1)
        y = np.linspace(box[2], box[3], ny+1)
        z = np.zeros(1)
        gridToVTK(filename, x, y, z, cellData=celldata, pointData=nodedata)

        return filename

    ## @ingroup FDMInterface
    @property
    def node(self):
        """
        @brief Get the coordinates of the nodes in the mesh.

        @return A NumPy array of shape (nx+1, ny+1, 2) containing the coordinates of the nodes.

        @details This function calculates the coordinates of the nodes in the mesh based on the
                 mesh's origin, step size, and the number of cells in the x and y directions.
                 It returns a NumPy array with the coordinates of each node.

        """
        GD = self.geo_dimension()
        nx = self.nx
        ny = self.ny
        box = [self.origin[0], self.origin[0] + nx * self.h[0],
               self.origin[1], self.origin[1] + ny * self.h[1]]
        node = np.zeros((nx + 1, ny + 1, GD), dtype=self.ftype)
        node[..., 0], node[..., 1] = np.mgrid[
                                     box[0]:box[1]:(nx + 1)*1j,
                                     box[2]:box[3]:(ny + 1)*1j]
        return node

    ## @ingroup FDMInterface
    def cell_barycenter(self):
        """
        @brief
        """
        GD = self.geo_dimension()
        nx = self.nx
        ny = self.ny
        box = [self.origin[0] + self.h[0]/2, self.origin[0] + self.h[0]/2 + (nx-1)*self.h[0],
               self.origin[1] + self.h[1]/2, self.origin[1] + self.h[1]/2 + (ny-1)*self.h[1]]
        bc = np.zeros((nx, ny, 2), dtype=self.ftype)
        bc[..., 0], bc[..., 1] = np.mgrid[
                                 box[0]:box[1]:nx*1j,
                                 box[2]:box[3]:ny*1j]
        return bc

    ## @ingroup FDMInterface
    def edge_barycenter(self):
        """
        @brief
        """
        bcx = self.edgex_barycenter()
        bcy = self.edgey_barycenter()
        return bcx, bcy

    ## @ingroup FDMInterface
    def edgex_barycenter(self):
        """
        @brief
        """
        GD = self.geo_dimension()
        nx = self.nx
        ny = self.ny
        box = [self.origin[0] + self.h[0]/2, self.origin[0] + self.h[0]/2 + (nx-1)*self.h[0],
               self.origin[1],               self.origin[1] + ny*self.h[1]]
        bc = np.zeros((nx, ny+1, 2), dtype=self.ftype)
        bc[..., 0], bc[..., 1] = np.mgrid[
                                 box[0]:box[1]:nx*1j,
                                 box[2]:box[3]:(ny+1)*1j]
        return bc

    ## @ingroup FDMInterface
    def edgey_barycenter(self):
        """
        @breif
        """
        GD = self.geo_dimension()
        nx = self.nx
        ny = self.ny
        box = [self.origin[0],               self.origin[0] + nx*self.h[0],
               self.origin[1] + self.h[1]/2, self.origin[1] + self.h[1]/2 + (ny-1)*self.h[1]]
        bc = np.zeros((nx+1, ny, 2), dtype=self.ftype)
        bc[..., 0], bc[..., 1] = np.mgrid[
                                 box[0]:box[1]:(nx+1)*1j,
                                 box[2]:box[3]:ny*1j]
        return bc

    ## @ingroup FDMInterface
    def function(self, etype='node', dim=None, dtype=None, ex=0):
        """
        @brief: Return a discrete function (array) defined on nodes, mesh edges, or mesh cells with elements set to 0.
        @param etype: The entity type can be 'node', 'edge', 'face', 'edgex', 'edgey', 'cell', or their corresponding numeric values.
        @type etype: str or int
        @param dim: The dimension of the discrete function, default: None.
        @type dim: int, optional
        @param dtype: The data type of the discrete function, default: None.
        @type dtype: data-type, optional
        @param ex: A non-negative integer to extend the discrete function by a certain width outward, default: 0.
        @type ex: int, optional

        @return: The discrete function (array) with elements set to 0.
        @rtype: numpy.ndarray
        @raise ValueError: If the given entity type is invalid.
        """

        nx = self.nx
        ny = self.ny
        dtype = self.ftype if dtype is None else dtype
        if etype in {'node', 0}:
            if dim is None:
                uh = np.zeros((nx+1+2*ex, ny+1+2*ex), dtype=dtype)
            else:
                uh = np.zeros((nx+1+2*ex, ny+1+2*ex, dim), dtype=dtype)

        elif etype in {'edge','face', 1}:
            ex = np.zeros((nx, ny+1), dtype=dtype)
            ey = np.zeros((nx+1, ny), dtype=dtype)
            uh = (ex, ey)
        elif etype in {'edgex'}:
            uh = np.zeros((nx, ny+1), dtype=dtype)
        elif etype in {'edgey'}:
            uh = np.zeros((nx+1, ny), dtype=dtype)
        elif etype in {'cell', 2}:
            uh = np.zeros((nx+2*ex, ny+2*ex), dtype=dtype)
        else:
            raise ValueError(f'the entity `{etype}` is not correct!')

        return uh

    ## @ingroup FDMInterface
    def gradient(self, f, order=1):
        """
        @brief 求网格函数 f 的梯度
        """
        hx = self.h[0]
        hy = self.h[1]
        fx, fy = np.gradient(f, hx, hy, edge_order=order)
        return fx, fy

    ## @ingroup FDMInterface
    def divergence(self, f_x, f_y, order=1):
        """
        @brief 求向量网格函数 (fx, fy) 的散度
        """

        hx = self.h[0]
        hy = self.h[1]
        f_xx,f_xy = np.gradient(f_x, hx, edge_order=order)
        f_yx,f_yy = np.gradient(f_y, hy, edge_order=order)
        return f_xx + f_yy

    ## @ingroup FDMInterface
    def laplace(self, f, order=1):
        hx = self.h[0]
        hy = self.h[1]
        fx, fy = np.gradient(f, hx, hy, edge_order=order)
        fxx,fxy = np.gradient(fx, hx, edge_order=order)
        fyx,fyy = np.gradient(fy, hy, edge_order=order)
        return fxx + fyy

    ## @ingroup FDMInterface
    def value(self, p, f):
        """
        @brief Compute the values of a function at given non-grid points based on known values at grid nodes.

        @param[in] p A NumPy array of shape (N, 2) containing the coordinates of the points where the function values are sought.
        @param[in] f A NumPy array of shape (nx+1, ny+1) containing the known function values at the grid nodes.

        @return A NumPy array of shape (N,) containing the function values at the given points.
        @todo check
        """
        nx = self.ds.nx
        ny = self.ds.ny
        box = [self.origin[0], self.origin[0] + nx * self.h[0],
               self.origin[1], self.origin[1] + ny * self.h[1]]

        hx = self.h[0]
        hy = self.h[1]

        i, j = self.cell_location(p)
        np.clip(i, 0, nx - 1, out=i)
        np.clip(j, 0, ny - 1, out=j)
        x0 = i * hx + box[0]
        y0 = j * hy + box[2]
        a = (p[..., 0] - x0) / hx
        b = (p[..., 1] - y0) / hy
        F = f[i, j] * (1-a) * (1-b)  + f[i + 1, j] * a * (1-b) \
            + f[i, j + 1] * (1-a) * b + f[i + 1, j + 1] * a * b
        return F

    ## @ingroup FDMInterface
    def interpolation(self, f, intertype='node'):
        """
        This function is deprecated and will be removed in a future version.
        Please use the interpolate() instead.
        """
        warnings.warn("The interpolation() is deprecated and will be removed in a future version. "
                      "Please use the interpolate() instead.", DeprecationWarning)
        nx = self.ds.nx
        ny = self.ds.ny
        node = self.node
        if intertype == 'node':
            F = f(node)
        elif intertype == 'edge':
            xbc, ybc = self.entity_barycenter('edge')
            F = f(xbc), f(ybc)
        elif intertype == 'edgex':
            xbc = self.entity_barycenter('edgex')
            F = f(xbc)
        elif intertype == 'edgey':
            ybc = self.entity_barycenter('edgey')
            F = f(ybc)
        elif intertype == 'cell':
            bc = self.entity_barycenter('cell')
            F = f(bc)
        return F

    ## @ingroup FDMInterface
    def interpolate(self, f, intertype='node'):
        """
        """
        nx = self.ds.nx
        ny = self.ds.ny
        node = self.node
        if intertype in {'node', 0}:
            F = f(node)
        elif intertype in {'edge', 'face', 1}:
            xbc, ybc = self.entity_barycenter('edge')
            F = f(xbc), f(ybc)
        elif intertype in {'edgex'}:
            xbc = self.entity_barycenter('edgex')
            F = f(xbc)
        elif intertype in {'edgey'}:
            ybc = self.entity_barycenter('edgey')
            F = f(ybc)
        elif intertype in {'cell'}:
            bc = self.entity_barycenter('cell')
            F = f(bc)
        return F

    ## @ingroup FDMInterface
    def error(self, u: Callable, uh: np.ndarray, errortype: str = 'all') -> Union[float, Tuple[np.float64, np.float64, np.float64]]:
        """
        计算真实解和数值解之间的误差。

        @param[in] u: 真实解的函数。
        @param[in] uh: 数值解的二维数组。
        @param[in] errortype: 误差类型，可以是'all'、'max'、'L2' 或 'l2'。
        @return 如果errortype为'all'，则返回一个包含最大误差、L2误差和l2误差的元组；
                如果errortype为'max'，则返回最大误差；
                如果errortype为'L2'，则返回L2误差；
                如果errortype为'l2'，则返回l2误差。
        """
        assert (uh.shape[0] == self.nx+1) and (uh.shape[1] == self.ny+1)
        hx = self.h[0]
        hy = self.h[1]
        nx = self.nx
        ny = self.ny
        node = self.node
        uI = u(node)
        e = uI - uh

        if errortype == 'all':
            emax = np.max(np.abs(e))
            e0 = np.sqrt(hx * hy * np.sum(e ** 2))
            el2 = np.sqrt(1 / ((nx - 1) * (ny - 1)) * np.sum(e ** 2))

            return emax, e0, el2
        elif errortype == 'max':
            emax = np.max(np.abs(e))
            return emax
        elif errortype == 'L2':
            e0 = np.sqrt(hx * hy * np.sum(e ** 2))
            return e0
        elif errortype == 'l2':
            el2 = np.sqrt(1 / ((nx - 1) * (ny - 1)) * np.sum(e ** 2))
            return el2

    ## @ingroup FDMInterface
    def elliptic_operator(self, d=2, c=None, r=None):
        """
        @brief Assemble the finite difference matrix for a general elliptic operator.
        """
        pass

    ## @ingroup FDMInterface
    def laplace_operator(self):
        """
        @brief Construct the discrete Laplace operator on a Cartesian grid

        Generate the corresponding discrete Laplace operator matrix based on
        the partition information of the x and y directions in the class.

        @note Both the x and y directions are uniformly partitioned, but the step sizes
        can be different.

        @return Returns a scipy.sparse.csr_matrix representing the discrete Laplace operator.
        """

        n0 = self.ds.nx + 1
        n1 = self.ds.ny + 1
        cx = 1/(self.h[0]**2)
        cy = 1/(self.h[1]**2)
        NN = self.number_of_nodes()
        k = np.arange(NN).reshape(n0, n1)

        A = diags([2*(cx+cy)], [0], shape=(NN, NN), format='csr')

        val = np.broadcast_to(-cx, (NN-n1, ))
        I = k[1:, :].flat
        J = k[0:-1, :].flat
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(-cy, (NN-n0, ))
        I = k[:, 1:].flat
        J = k[:, 0:-1].flat
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        return A

    ## @ingroup FDMInterface
    def apply_dirichlet_bc(self, gD: Callable[[np.ndarray], np.ndarray], A: spmatrix, 
                           f: np.ndarray, uh: Optional[np.ndarray] = None) -> Tuple[spmatrix, np.ndarray]:
        """
        @brief: 组装 \\Delta u 对应的有限差分矩阵，考虑了 Dirichlet 边界和向量型函数
        
        @param[in] gD  表示 Dirichlet 边界值函数
        @param[in] A  (NN, NN), 稀疏矩阵
        @param[in] f  可能是一维数组（标量型右端项）或二维数组（向量型右端项）
        @param[in, optional] uh  默认为 None，表示网格函数，如果为 None 则创建一个新的网格函数

        @return Tuple[spmatrix, np.ndarray], 返回处理后的稀疏矩阵 A 和数组 f
        """        
        if uh is None:
            uh = self.function('node').reshape(-1)
        else:
            uh = uh.reshape(-1) # 展开为一维数组 TODO:向量型函数

        f = f.reshape(-1, ) # 展开为一维数组 TODO：向量型右端
        
        node = self.entity('node')
        isBdNode = self.ds.boundary_node_flag()
        uh[isBdNode]  = gD(node[isBdNode])
        f -= A@uh
        f[isBdNode] = uh[isBdNode]

        bdIdx = np.zeros(A.shape[0], dtype=self.itype)
        bdIdx[isBdNode] = 1
        D0 = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
        D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
        A = D0@A@D0 + D1
        return A, f 

    ## @ingroup FDMInterface
    def update_dirichlet_bc(self, gD: Callable[[np.ndarray], Any], uh: np.ndarray) -> None:
        """
        @brief 更新网格函数 uh 的 Dirichlet 边界值
        @todo 考虑向量型函数
        @param[in] gD Callable[[np.ndarray], Any], 表示 Dirichlet 边界值函数
        @param[in, out] uh np.ndarray, 表示网格函数，将被更新为具有正确的 Dirichlet 边界值
        """
        node = self.node
        isBdNode = self.ds.boundary_node_flag().reshape(uh.shape)
        uh[isBdNode] = gD(node[isBdNode, :])

    ## @ingroup FDMInterface 
    def parabolic_operator_forward(self, tau):
        """
        @brief 生成抛物方程的向前差分迭代矩阵

        @param[in] tau float, 当前时间步长
        """
        rx = tau/self.h[0]**2
        ry = tau/self.h[1]**2
        if rx + ry > 0.5:
            raise ValueError(f"The rx+ry: {rx+ry} should be smaller than 0.5")

        NN = self.number_of_nodes()
        n0 = self.nx + 1
        n1 = self.ny + 1
        k = np.arange(NN).reshape(n0, n1)

        A = diags([1 - 2 * rx - 2 * ry], [0], shape=(NN, NN), format='csr')

        val = np.broadcast_to(rx, (NN - n1,))
        I = k[1:, :].flat
        J = k[0:-1, :].flat
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(ry, (NN - n0, ))
        I = k[:, 1:].flat
        J = k[:, 0:-1].flat
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        return A

    ## @ingroup FDMInterface 
    def parabolic_operator_backward(self, tau):
        """
        @brief 生成抛物方程的向后差分迭代矩阵

        @param[in] tau float, 当前时间步长
        """
        rx = tau/self.h[0]**2
        ry = tau/self.h[1]**2
        if rx + ry > 1.5:
            raise ValueError(f"The sum rx + ry: {rx + ry} should be smaller than 0.5")

        NN = self.number_of_nodes()
        n0 = self.nx + 1
        n1 = self.ny + 1
        k = np.arange(NN).reshape(n0, n1)

        A = diags([1 + 2 * rx + 2 * ry], [0], shape=(NN, NN), format='csr')

        val = np.broadcast_to(-rx, (NN - n1,))
        I = k[1:, :].flat
        J = k[0:-1, :].flat
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(-ry, (NN - n0, ))
        I = k[:, 1:].flat
        J = k[:, 0:-1].flat
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)
        return A

    def parabolic_operator_crank_nicholson(self, tau):
        """
        @brief 生成抛物方程的 CN 差分格式的迭代矩阵

        @param[in] tau float, 当前时间步长
        """
        rx = tau/self.h[0]**2
        ry = tau/self.h[1]**2
        if rx + ry > 1.5:
            raise ValueError(f"The sum rx + ry: {rx + ry} should be smaller than 1.5")

        NN = self.number_of_nodes()
        n0 = self.nx + 1
        n1 = self.ny + 1
        k = np.arange(NN).reshape(n0, n1)

        A = diags([1 + rx + ry], [0], shape=(NN, NN), format='csr')

        val = np.broadcast_to(-rx/2, (NN-n1, ))
        I = k[1:, :].flat
        J = k[0:-1, :].flat
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(-ry/2, (NN-n0, ))
        I = k[:, 1:].flat
        J = k[:, 0:-1].flat
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        B = diags([1 - rx - ry], [0], shape=(NN, NN), format='csr')

        val = np.broadcast_to(rx/2, (NN-n1, ))
        I = k[1:, :].flat
        J = k[0:-1, :].flat
        B += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        B += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(ry/2, (NN-n0, ))
        I = k[:, 1:].flat
        J = k[:, 0:-1].flat
        B += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        B += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        return A, B

        A1 += csr_matrix((val, (J.flat, I.flat)), shape=(NN, NN), dtype=self.ftype)


    ## @ingroup FDMInterface
    def wave_operator_explicity(self, tau, a=1): 
        """
        @brief 用显格式求解波动方程
        """
        rx = a*tau/self.h[0]
        ry = a*tau/self.h[1]

        NN = self.number_of_nodes()
        n0 = self.nx + 1
        n1 = self.ny + 1
        k = np.arange(NN).reshape(n0, n1)

        A = diags([2*(1 -  rx**2 - ry**2)], [0], shape=(NN, NN), format='csr')

        val = np.broadcast_to(rx**2, (NN - n1,))
        I = k[1:, :].flat
        J = k[0:-1, :].flat
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(ry**2, (NN - n0, ))
        I = k[:, 1:].flat
        J = k[:, 0:-1].flat
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        return A


    ## @ingroup FDMInterface
    def wave_operator_theta(self, tau, a=1, theta=0.5):
        """
        @brief 生成波动方程的离散矩阵
        """
        rx = a*tau/self.h[0]
        ry = a*tau/self.h[1]

        NN = self.number_of_nodes()
        n0 = self.nx + 1
        n1 = self.ny + 1
        k = np.arange(NN).reshape(n0, n1)

        A0 = diags([1 + 2*rx**2*theta + 2*ry**2*theta], [0], shape=(NN, NN), format='csr')
        A1 = diags([2*(1 - (rx**2 + ry**2)*(1 - 2*theta))], [0], shape=(NN, NN), format='csr')
        A2 = diags([-(1 + 2*rx**2*theta + 2*ry**2*theta)], [0], shape=(NN, NN), format='csr')

        val = np.broadcast_to(-rx**2, (NN-n1, ))
        I = k[1:, :].flat
        J = k[0:-1, :].flat
        A0 += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A0 += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(-ry**2, (NN-n0, ))
        I = k[:, 1:].flat
        J = k[:, 0:-1].flat
        A0 += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A0 += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(rx**2*(1 - 2*theta), (NN-n1, ))
        I = k[1:, :].flat
        J = k[0:-1, :].flat
        A1 += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A1 += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(ry**2*(1 - 2*theta), (NN-n1, ))
        I = k[:, 1:].flat
        J = k[:, 0:-1].flat
        A1 += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A1 += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)
        
        val = np.broadcast_to(rx**2*theta, (NN-n1, ))
        I = k[1:, :].flat
        J = k[0:-1, :].flat
        A2 += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A2 += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(ry**2*theta, (NN-n0, ))
        I = k[:, 1:].flat
        J = k[:, 0:-1].flat
        A2 += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A2 += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        return A0, A1, A2

    ## @ingroup FDMInterface
    def fast_sweeping_method(self, phi0):
        """
        @brief 均匀网格上的 fast sweeping method
        @param[in] phi 是一个离散的水平集函数

        @note 注意，我们这里假设 x 和 y 方向剖分的段数相等
        """
        m = 2
        nx = self.ds.nx
        ny = self.ds.ny
        k = nx/ny
        ns = ny
        h = self.h[0]

        phi = self.function(ex=1)
        isNearNode = self.function(dtype=np.bool_, ex=1)

        # 把水平集函数转化为离散的网格函数
        node = self.entity('node')
        phi[1:-1, 1:-1] = phi0
        sign = np.sign(phi[1:-1, 1:-1])

        # 标记界面附近的点
        isNearNode[1:-1, 1:-1] = np.abs(phi[1:-1, 1:-1]) < 2*h
        lsfun = UniformMesh2dFunction(self, phi[1:-1, 1:-1])
        _, d = lsfun.project(node[isNearNode[1:-1, 1:-1]])
        phi[isNearNode] = np.abs(d) #界面附近的点用精确值
        phi[~isNearNode] = m  # 其它点用一个比较大的值

        a = np.zeros(ns+1, dtype=np.float64)
        b = np.zeros(ns+1, dtype=np.float64)
        c = np.zeros(ns+1, dtype=np.float64)
        d = np.zeros(int(k*ns+1), dtype=np.float64)
        e = np.zeros(int(k*ns+1), dtype=np.float64)
        f = np.zeros(int(k*ns+1), dtype=np.float64)

        n = 0
        for i in range(1, int(k*ns+2)):
            a[:] = np.minimum(phi[i-1, 1:-1], phi[i+1, 1:-1])
            b[:] = np.minimum(phi[i, 0:ns+1], phi[i, 2:])
            flag = np.abs(a-b) >= h
            c[flag] = np.minimum(a[flag], b[flag]) + h
            c[~flag] = (a[~flag] + b[~flag] + np.sqrt(2*h*h - (a[~flag] - b[~flag])**2))/2
            phi[i, 1:-1] = np.minimum(c, phi[i, 1:-1])
            n += 1

        for i in range(int(k*ns+1), 0, -1):
            a[:] = np.minimum(phi[i-1, 1:-1], phi[i+1, 1:-1])
            b[:] = np.minimum(phi[i, 0:ns+1], phi[i, 2:])
            flag = np.abs(a-b) >= h
            c[flag] = np.minimum(a[flag], b[flag]) + h
            c[~flag] = (a[~flag] + b[~flag] + np.sqrt(2*h*h - (a[~flag] - b[~flag])**2))/2
            phi[i, 1:-1] = np.minimum(c, phi[i, 1:-1])
            n += 1

        for j in range(1, ns+2):
            d[:] = np.minimum(phi[0:int(k*ns+1), j], phi[2:, j])
            e[:] = np.minimum(phi[1:-1, j-1], phi[1:-1, j+1])
            flag = np.abs(d-e) >= h
            f[flag] = np.minimum(d[flag], e[flag]) + h
            f[~flag] = (d[~flag] + e[~flag] + np.sqrt(2*h*h - (d[~flag] - e[~flag])**2))/2
            phi[1:-1, j] = np.minimum(f, phi[1:-1, j])
            n += 1

        for j in range(ns+1, 0, -1):
            d[:] = np.minimum(phi[0:int(k*ns+1), j], phi[2:, j])
            e[:] = np.minimum(phi[1:-1, j-1], phi[1:-1, j+1])
            flag = np.abs(d-e) >= h
            f[flag] = np.minimum(d[flag], e[flag]) + h
            f[~flag] = (d[~flag] + e[~flag] + np.sqrt(2*h*h - (d[~flag] - e[~flag])**2))/2
            phi[1:-1, j] = np.minimum(f, phi[1:-1, j])
            n += 1

        return sign*phi[1:-1, 1:-1]

    ## @ingroup FEMInterface
    def geo_dimension(self):
        """
        @brief Get the geometry dimension of the mesh.

        @return The geometry dimension (2 for 2D mesh).
        """
        return 2

    ## @ingroup FEMInterface
    def top_dimension(self):
        """
        @brief Get the topological dimension of the mesh.

        @return The topological dimension (2 for 2D mesh).
        """
        return 2

    ## @ingroup FEMInterface
    def integrator(self, q, etype='cell'):
        qf = GaussLegendreQuadrature(q)
        if etype in {'cell', 2}:
            return TensorProductQuadrature((qf, qf)) 
        elif etype in {'edge', 'face', 1}:
            return qf 

    ## @ingroup FEMInterface
    def bc_to_point(self, bc, index=np.s_[:]):
        pass

    ## @ingroup FEMInterface
    def entity(self, etype):
        """
        @brief Get the entity (either cell or node) based on the given entity type.

        @param[in] etype The type of entity, either 'cell', 2, 'edge', 'face' or 1, `node', 0.

        @return The cell, edeg, face, or node array based on the input entity type.

        @throws ValueError if the given etype is invalid.
        """
        if etype in {'cell', 2}:
            return self.ds.cell
        elif etype in {'edge', 'face', 1}:
            return self.ds.edge
        elif etype in {'node', 0}:
            return self.node.reshape(-1, 2)
        else:
            raise ValueError("`etype` is wrong!")

    ## @ingroup FEMInterface
    def entity_barycenter(self, etype, index=np.s_[:]):
        """
        @brief: Get the entity (cell, {face, edge}, or  node) based on the given entity type.
        @param[in] etype: The type of entity can be 'cell', 2, 'face', 'edge',
        1, 'node', or 0.
        @return: The cell or node array based on the input entity type.
        @throws ValueError if the given etype is invalid.
        """
        if etype in {'cell', 2}:
            return self.cell_barycenter().reshape(-1, 2)
        elif etype in {'edge', 'face', 1}:
            bcx, bcy = self.edge_barycenter()
            return np.concatenate((bcx.reshape(-1, 2), bcy.reshape(-1, 2)), axis=0)
        elif etype in {'node', 0}:
            return self.node.reshape(-1, 2)
        else:
            raise ValueError('the entity type `{etype}` is not correct!')

    ## @ingroup FEMInterface
    #def entity_measure(self, etype):
        #"""
        #@brief
        #"""
        #pass

    ## @ingroup FEMInterface
    def multi_index_matrix(self, p, etype=1):
        pass

    ## @ingroup FEMInterface
    def shape_function(self, bc, p=1):
        pass

    ## @ingroup FEMInterface
    def grad_shape_function(self, bc, p=1):
        pass

    ## @ingroup FEMInterface
    def number_of_local_ipoints(self, p, iptype='cell'):
        pass

    ## @ingroup FEMInterface
    def number_of_global_ipoints(self, p):
        pass

    ## @ingroup FEMInterface
    def interpolation_points(self, p):
        pass

    ## @ingroup FEMInterface
    def node_to_ipoint(self, p):
        pass

    ## @ingroup FEMInterface
    def edge_to_ipoint(self, p):
        pass

    ## @ingroup FEMInterface
    def face_to_ipoint(self, p):
        pass

    ## @ingroup FEMInterface
    def cell_to_ipoint(self, p):
        pass

    def t2sidx(self):
        """
        @brief 已知结构三角形网格点的值，将其排列到结构四边形网格上
        @example a[s2tidx] = uh
        """
        snx = self.ds.nx
        sny = self.ds.ny
        idx1= np.arange(0,sny+1,2)+np.arange(0,(sny+1)*(snx+1),2*(sny+1)).reshape(-1,1)
        idx1 = idx1.flatten()
        a = np.array([1,sny+1,sny+2])
        b = np.arange(0,sny,2).reshape(-1,1)
        c = np.append(a+b,[(sny+1)*2-1])
        e = np.arange(0,(sny+1)*snx,2*(sny+1)).reshape(-1,1)
        idx2 = (e+c).flatten()
        idx3  = np.arange((sny+1)*snx+1,(snx+1)*(sny+1),2)
        idx = np.r_[idx1,idx2,idx3]
        return idx

    def  s2tidx(self):
        """
        @brief 已知结构四边形网格点的值，将其排列到结构三角形网格上
        @example a[s2tidx] = uh
        """
        tnx = int(self.ds.nx/2)
        tny = int(self.ds.ny/2)
        a = np.arange(tny+1)
        b = 3*np.arange(tny).reshape(-1,1)+(tnx+1)*(tny+1)
        idx1 = np.zeros((tnx+1,2*tny+1))#sny+1
        idx1[:,0::2] = a+np.arange(tnx+1).reshape(-1,1)*(tny+1)
        idx1[:,1::2] = b.flatten()+np.arange(tnx+1).reshape(-1,1)*(2*tny+1+tny)
        idx1[-1,1::2] = np.arange((2*tnx+1)*(2*tny+1)-tny,(2*tnx+1)*(2*tny+1))
        c = np.array([(tnx+1)*(tny+1)+1,(tnx+1)*(tny+1)+2])
        d = np.arange(tny)*3
        d = 3*np.arange(tny).reshape(-1,1)+c
        e = np.append(d.flatten(),[d.flatten()[-1]+1])
        idx2 = np.arange(tnx).reshape(-1,1)*(2*tny+1+tny)+e

        idx = np.c_[idx1[:tnx],idx2]
        idx = np.append(idx.flatten(),[idx1[-1,:]])
        idx = idx.astype(int)
        return idx

    def data_edge_to_cell(self, Ex, Ey):
        """
        @brief 把定义在边上的数组转换到单元上
        """
        dx = self.function(etype='cell')
        dy = self.function(etype='cell')

        dx[:] = (Ex[:, :-1] + Ex[:, 1:])/2.0
        dy[:] = (Ey[:-1, :] + Ey[1:, :])/2.0

        return dx, dy

    def function_remap(self, tmesh, p=2):
        """
        @brief 获取结构三角形和结构四边形网格上的自由度映射关系

        @example
        phi.flat[idxMap] = uh
        """
        nx = self.ds.nx
        ny = self.ds.ny

        NN = self.number_of_nodes()
        idxMap = np.arange(NN)

        idx = np.arange(0, nx*(ny+1), 2*(ny+1)).reshape(-1, 1) + np.arange(0,
                ny+1, 2)
        idxMap[idx] = range(tmesh.number_of_nodes())

        return idxMap


class UniformMesh2dFunction():
    def __init__(self, mesh, f):
        self.mesh = mesh # (nx+1, ny+1)
        self.f = f   # (nx+1, ny+1)
        self.fx, self.fy = mesh.gradient(f)

    def __call__(self, p):
        mesh = self.mesh
        F = mesh.value(p, self.f)
        return F

    def value(self, p):
        mesh = self.mesh
        F = mesh.value(p, self.f)
        return F

    def gradient(self, p):
        mesh = self.mesh
        fx = self.fx
        fy = self.fy
        gf = np.zeros_like(p)
        gf[..., 0] = mesh.value(p, fx)
        gf[..., 1] = mesh.value(p, fy)
        return gf

    def project(self, p):
        """
        @brief 把曲线附近的点投影到曲线上
        """
        p, d = project(self, p, maxit=200, tol=1e-8, returnd=True)
        return p, d
