from matplotlib.projections import Axes3D
import numpy as np
import warnings

from matplotlib.pyplot import Figure
from matplotlib.axes import Axes
from scipy.sparse import coo_matrix, csr_matrix, diags, spdiags, spmatrix
from types import ModuleType
from typing import Optional, Tuple, Callable, Any, Union, List

from .mesh_base import Mesh, Plotable

# 这个数据接口为有限元服务
from .mesh_data_structure import StructureMesh2dDataStructure
from ..quadrature import TensorProductQuadrature, GaussLegendreQuadrature
from ..geometry import project, find_cut_point, msign


## @defgroup FEMInterface
## @defgroup FDMInterface
## @defgroup GeneralInterface
class UniformMesh2d(Mesh, Plotable):
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
        self.type = 'U2d'

        # Data structure for finite element computation
        self.ds: StructureMesh2dDataStructure = StructureMesh2dDataStructure(self.nx, self.ny, itype=itype)

        self.face_to_ipoint = self.edge_to_ipoint

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
            self.extent = [i * 2 for i in self.extent]
            self.h = [h / 2.0 for h in self.h]
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
        #return self.h[0] * self.h[1]
        return np.full(self.NC, self.h[0] * self.h[1])

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
        v = np.real(p - np.array(self.origin, dtype=p.dtype))
        n0 = v[..., 0] // hx
        n1 = v[..., 1] // hy

        return n0.astype('int64'), n1.astype('int64')

    ## @ingroup GeneralInterface
    def point_to_bc(self, p):

        x = p[..., 0]
        y = p[..., 1]

        bc_x_ = np.real((x - self.origin[0]) / self.h[0]) % 1
        bc_y_ = np.real((y - self.origin[1]) / self.h[1]) % 1
        bc_x = np.array([[bc_x_, 1 - bc_x_]], dtype=np.float64)
        bc_y = np.array([[bc_y_, 1 - bc_y_]], dtype=np.float64)
        val = (bc_x, bc_y)

        return val

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

        node = self.node  # 获取二维节点上的网格坐标
        if uh.ndim == 1:
            uh = uh.reshape(self.nx+1, self.ny+1)
        return axes.plot_surface(node[..., 0], node[..., 1], uh, cmap=cmap)

    ## @ingroup GeneralInterface
    def show_animation(self,
                       fig: Figure,
                       axes: Union[Axes, Axes3D],
                       box: List[float],
                       advance: Callable[[int], Tuple[np.ndarray, float]],
                       fname: str = 'test.mp4',
                       init: Optional[Callable] = None,
                       fargs: Optional[Callable] = None,
                       frames: int = 1000,
                       interval: int = 50,
                       plot_type: str = 'imshow',
                       cmap='rainbow') -> None:
        """
        @brief 生成求解过程动画并保存为指定文件名的视频文件

        @param fig         : plt.Figure | matplotlib 图形对象
        @param axes        : Union[Axes, Axes3D] | matplotlib 坐标轴对象
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
        from matplotlib.contour import QuadContourSet

        # 初始化二维网格数据
        uh, _ = advance(0)
        if isinstance(axes, Axes) and plot_type == 'imshow':
            data = axes.imshow(uh, cmap=cmap, vmin=box[4], vmax=box[5],
                               extent=box[0:4], interpolation='bicubic')
        elif isinstance(axes, Axes3D) and plot_type == 'surface':
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
            data = axes.contourf(X, Y, uh, cmap=cmap, vmin=box[4], vmax=box[5])
            # data 的值在每一帧更新时都会发生改变 颜色条会根据这些更改自动更新
            # 后续的代码中无需对颜色条进行额外的更新操作
            # cbar = fig.colorbar(data, ax=axes)

        def func(n, *fargs):  # 根据当前帧序号计算数值解，更新图像对象的数值数组，显示当前帧序号和时刻
            nonlocal data  # 声明 data 为非局部变量 这样在 func 函数内部对 data 进行的修改会影响到外部的 data 变量
            uh, t = advance(n, *fargs)  # 计算当前时刻的数值解并返回，uh 是数值解，t 是当前时刻

            if data is None:
                if isinstance(axes, Axes) and plot_type == 'imshow':
                    data = axes.imshow(uh, cmap=cmap, vmin=box[4], vmax=box[5],
                                       extent=box[0:4], interpolation='bicubic')
                elif isinstance(axes, Axes3D) and plot_type == 'surface':
                    data = axes.plot_surface(X, Y, uh, cmap=cmap, vmin=box[4],
                                             vmax=box[5], rstride=1, cstride=1)
                elif plot_type == 'contourf':
                    data = axes.contourf(X, Y, uh, cmap=cmap, vmin=box[4], vmax=box[5])

            if isinstance(axes, Axes) and plot_type == 'imshow':
                data.set_array(uh)  # 更新 data 对象的数值数组。导致图像的颜色根据新的数值解 uh 更新
                axes.set_aspect('equal')  # 设置坐标轴的长宽比。'equal' 选项使得 x 轴和 y 轴的单位尺寸相等

            elif isinstance(axes, Axes3D) and plot_type == 'surface':
                axes.clear()  # 清除当前帧的图像
                data = axes.plot_surface(X, Y, uh, cmap=cmap, vmin=box[4], vmax=box[5])
                axes.set_xlim(box[0], box[1])
                axes.set_ylim(box[2], box[3])
                axes.set_zlim(box[4], box[5])
            elif plot_type == 'contourf':
                # 使用 contourf 时，每次更新图像时都会生成一个新的等高线填充层
                # data.collections 保存了所有已经生成的等高线填充层
                # 更新图像时 需要将旧的等高线填充层从图形中移除 以免遮挡住新的等高线填充层
                if data is not None:
                    if isinstance(data, QuadContourSet):
                        for coll in data.collections:
                            if coll in axes.collections:
                                coll.remove()
                data = axes.contourf(X, Y, uh, cmap=cmap, vmin=box[4], vmax=box[5])
                axes.set_aspect('equal')

            s = "frame=%05d, time=%0.8f" % (n, t)  # 创建一个格式化的字符串，显示当前帧序号 n 和当前时刻 t
            print(s)
            axes.set_title(s)  # 将格式化的字符串设置为坐标轴的标题
            return data

        # 创建一个 funcanimation 对象
        # fig 作为画布，func 作为帧更新函数
        # init_func 作为初始化函数，用于在动画开始之前设置图像的初始状态
        # fargs 作为一个元组，包含要传递给 func 函数的额外参数
        # frames 为帧数，interval 为动画间隔时间
        ani = animation.FuncAnimation(fig, func, init_func=init, fargs=fargs, frames=frames, interval=interval)
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
        box = [self.origin[0], self.origin[0] + nx * self.h[0],
               self.origin[1], self.origin[1] + ny * self.h[1]]
        
        x = np.linspace(box[0], box[1], nx + 1)
        y = np.linspace(box[2], box[3], ny + 1)
        z = np.zeros(1)
        gridToVTK(filename, x, y, z, cellData=celldata, pointData=nodedata)

        return filename

    def to_vtk(self, filename, celldata=None, nodedata=None):
        """
        @brief: Converts the mesh data to a VTK structured grid format and writes to a VTS file
        """
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk

        # 网格参数
        nx, ny = self.ds.nx, self.ds.ny
        h = self.h
        origin = self.origin

        # 创建坐标点
        x = np.linspace(origin[0], origin[0] + nx * h[0], nx + 1)
        y = np.linspace(origin[1], origin[1] + ny * h[1], ny + 1)
        z = np.zeros(1)

        # 按 y,x 顺序重新组织坐标数组
        yx_x = np.repeat(x, ny + 1)
        yx_y = np.tile(y, nx + 1)
        yx_z = np.zeros_like(yx_x)

        # 创建 VTK 网格对象
        rectGrid = vtk.vtkStructuredGrid()
        rectGrid.SetDimensions(ny + 1, nx + 1, 1)

        # 创建点
        points = vtk.vtkPoints()
        for i in range(len(yx_x)):
            points.InsertNextPoint(yx_x[i], yx_y[i], yx_z[i])
        rectGrid.SetPoints(points)

        # 添加节点数据
        if nodedata is not None:
            for name, data in nodedata.items():
                data_array = numpy_to_vtk(data, deep=True)
                data_array.SetName(name)
                rectGrid.GetPointData().AddArray(data_array)

        # 添加单元格数据
        if celldata is not None:
            for name, data in celldata.items():
                data_array = numpy_to_vtk(data, deep=True)
                data_array.SetName(name)
                rectGrid.GetCellData().AddArray(data_array)

        # 写入 VTK 文件
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetInputData(rectGrid)
        writer.SetFileName(filename)
        writer.Write()

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
                                     box[0]:box[1]:(nx + 1) * 1j,
                                     box[2]:box[3]:(ny + 1) * 1j]
        return node

    ## @ingroup FDMInterface
    def cell_barycenter(self):
        """
        @brief
        """
        GD = self.geo_dimension()
        nx = self.nx
        ny = self.ny
        box = [self.origin[0] + self.h[0] / 2, self.origin[0] + self.h[0] / 2 + (nx - 1) * self.h[0],
               self.origin[1] + self.h[1] / 2, self.origin[1] + self.h[1] / 2 + (ny - 1) * self.h[1]]
        bc = np.zeros((nx, ny, 2), dtype=self.ftype)
        bc[..., 0], bc[..., 1] = np.mgrid[
                                 box[0]:box[1]:nx * 1j,
                                 box[2]:box[3]:ny * 1j]
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
        box = [self.origin[0] + self.h[0] / 2, self.origin[0] + self.h[0] / 2 + (nx - 1) * self.h[0],
               self.origin[1], self.origin[1] + ny * self.h[1]]
        bc = np.zeros((nx, ny + 1, 2), dtype=self.ftype)
        bc[..., 0], bc[..., 1] = np.mgrid[
                                 box[0]:box[1]:nx * 1j,
                                 box[2]:box[3]:(ny + 1) * 1j]
        return bc

    ## @ingroup FDMInterface
    def edgey_barycenter(self):
        """
        @breif
        """
        GD = self.geo_dimension()
        nx = self.nx
        ny = self.ny
        box = [self.origin[0], self.origin[0] + nx * self.h[0],
               self.origin[1] + self.h[1] / 2, self.origin[1] + self.h[1] / 2 + (ny - 1) * self.h[1]]
        bc = np.zeros((nx + 1, ny, 2), dtype=self.ftype)
        bc[..., 0], bc[..., 1] = np.mgrid[
                                 box[0]:box[1]:(nx + 1) * 1j,
                                 box[2]:box[3]:ny * 1j]
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
                uh = np.zeros((nx + 1 + 2 * ex, ny + 1 + 2 * ex), dtype=dtype)
            else:
                uh = np.zeros((nx + 1 + 2 * ex, ny + 1 + 2 * ex, dim), dtype=dtype)

        elif etype in {'edge', 'face', 1}:
            ex = np.zeros((nx, ny + 1), dtype=dtype)
            ey = np.zeros((nx + 1, ny), dtype=dtype)
            uh = (ex, ey)
        elif etype in {'edgex'}:
            uh = np.zeros((nx, ny + 1), dtype=dtype)
        elif etype in {'edgey'}:
            uh = np.zeros((nx + 1, ny), dtype=dtype)
        elif etype in {'cell', 2}:
            uh = np.zeros((nx + 2 * ex, ny + 2 * ex), dtype=dtype)
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
        f_xx, f_xy = np.gradient(f_x, hx, edge_order=order)
        f_yx, f_yy = np.gradient(f_y, hy, edge_order=order)
        return f_xx + f_yy

    ## @ingroup FDMInterface
    def laplace(self, f, order=1):
        hx = self.h[0]
        hy = self.h[1]
        fx, fy = np.gradient(f, hx, hy, edge_order=order)
        fxx, fxy = np.gradient(fx, hx, edge_order=order)
        fyx, fyy = np.gradient(fy, hy, edge_order=order)
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
        F = f[i, j] * (1 - a) * (1 - b) + f[i + 1, j] * a * (1 - b) \
            + f[i, j + 1] * (1 - a) * b + f[i + 1, j + 1] * a * b
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
    def error(self,
              u: Callable,
              uh: np.ndarray,
              errortype: str = 'all') -> Union[float, Tuple[np.float64, np.float64, np.float64]]:

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
        assert (uh.shape[0] == self.nx + 1) and (uh.shape[1] == self.ny + 1)
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
        cx = 1 / (self.h[0] ** 2)
        cy = 1 / (self.h[1] ** 2)
        NN = self.number_of_nodes()
        K = np.arange(NN).reshape(n0, n1)

        A = diags([2 * (cx + cy)], [0], shape=(NN, NN), format='csr')

        val = np.broadcast_to(-cx, (NN - n1,))
        I = K[1:, :].flat
        J = K[0:-1, :].flat
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(-cy, (NN - n0,))
        I = K[:, 1:].flat
        J = K[:, 0:-1].flat
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        return A

    ## @ingroup FDMInterface
    def apply_dirichlet_bc(self,
                           gD: Callable[[np.ndarray], np.ndarray],
                           A: spmatrix,
                           f: np.ndarray,
                           uh: Optional[np.ndarray] = None,
                           threshold: Optional[Union[int, Callable[[np.ndarray], np.ndarray]]] = None) -> Tuple[
        spmatrix, np.ndarray]:

        """
        @brief: 组装 \\Delta u 对应的有限差分矩阵，考虑了 Dirichlet 边界和向量型函数

        @param[in] gD  表示 Dirichlet 边界值函数
        @param[in] A  (NN, NN), 稀疏矩阵
        @param[in] f  可能是一维数组（标量型右端项）或二维数组（向量型右端项）
        @param[in, optional] uh  默认为 None,表示网格函数,如果为 None 则创建一个新的网格函数
        @param[in, optional] threshold 用于确定哪些网格节点应用 Dirichlet 边界条件

        @return Tuple[spmatrix, np.ndarray], 返回处理后的稀疏矩阵 A 和数组 f
        """
        if uh is None:
            uh = self.function('node').reshape(-1)
        else:
            uh = uh.reshape(-1)  # 展开为一维数组 TODO:向量型函数

        f = f.reshape(-1, )  # 展开为一维数组 TODO：向量型右端

        node = self.entity('node')
        # isBdNode = self.ds.boundary_node_flag()
        if threshold is None:
            isBdNode = self.ds.boundary_node_flag()
        elif isinstance(threshold, int):
            isBdNode = (np.arange(node.shape[0]) == threshold)
        elif callable(threshold):
            isBdNode = threshold(node)
        else:
            raise ValueError(f"Invalid threshold: {threshold}")

        uh[isBdNode] = gD(node[isBdNode])
        f -= A @ uh
        f[isBdNode] = uh[isBdNode]

        bdIdx = np.zeros(A.shape[0], dtype=self.itype)
        bdIdx[isBdNode] = 1
        D0 = spdiags(1 - bdIdx, 0, A.shape[0], A.shape[0])
        D1 = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
        A = D0 @ A @ D0 + D1

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
        rx = tau / self.h[0] ** 2
        ry = tau / self.h[1] ** 2
        if rx + ry > 0.5:
            raise ValueError(f"The rx+ry: {rx + ry} should be smaller than 0.5")

        NN = self.number_of_nodes()
        n0 = self.nx + 1
        n1 = self.ny + 1
        K = np.arange(NN).reshape(n0, n1)

        A = diags([1 - 2 * rx - 2 * ry], 0, shape=(NN, NN), format='csr')

        val = np.broadcast_to(rx, (NN - n1,))
        I = K[1:, :].flat
        J = K[0:-1, :].flat
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(ry, (NN - n0,))
        I = K[:, 1:].flat
        J = K[:, 0:-1].flat
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        return A

    ## @ingroup FDMInterface
    def parabolic_operator_backward(self, tau):
        """
        @brief 生成抛物方程的向后差分迭代矩阵

        @param[in] tau float, 当前时间步长
        """
        rx = tau / self.h[0] ** 2
        ry = tau / self.h[1] ** 2

        NN = self.number_of_nodes()
        n0 = self.nx + 1
        n1 = self.ny + 1
        K = np.arange(NN).reshape(n0, n1)

        A = diags([1 + 2 * rx + 2 * ry], 0, shape=(NN, NN), format='csr')

        val = np.broadcast_to(-rx, (NN - n1,))
        I = K[1:, :].flat
        J = K[0:-1, :].flat
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(-ry, (NN - n0,))
        I = K[:, 1:].flat
        J = K[:, 0:-1].flat
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)
        return A

    def parabolic_operator_crank_nicholson(self, tau):
        """
        @brief 生成抛物方程的 CN 差分格式的迭代矩阵

        @param[in] tau float, 当前时间步长
        """
        rx = tau / self.h[0] ** 2
        ry = tau / self.h[1] ** 2
        if rx + ry > 1.5:
            raise ValueError(f"The sum rx + ry: {rx + ry} should be smaller than 1.5")

        NN = self.number_of_nodes()
        n0 = self.nx + 1
        n1 = self.ny + 1
        K = np.arange(NN).reshape(n0, n1)

        A = diags([1 + rx + ry], [0], shape=(NN, NN), format='csr')

        val = np.broadcast_to(-rx / 2, (NN - n1,))
        I = K[1:, :].flat
        J = K[0:-1, :].flat
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(-ry / 2, (NN - n0,))
        I = K[:, 1:].flat
        J = K[:, 0:-1].flat
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        B = diags([1 - rx - ry], [0], shape=(NN, NN), format='csr')

        val = np.broadcast_to(rx / 2, (NN - n1,))
        I = K[1:, :].flat
        J = K[0:-1, :].flat
        B += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        B += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(ry / 2, (NN - n0,))
        I = K[:, 1:].flat
        J = K[:, 0:-1].flat
        B += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        B += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        return A, B


    ## @ingroup FDMInterface
    def wave_operator_explicit(self, tau: float, a: float = 1.0):
        """
        @brief 生成波动方程的显格式离散矩阵

        @param[in] tau float, 时间步长
        @param[in] a float, 波速，默认值为 1

        @return 离散矩阵 A
        """
        rx = a * tau / self.h[0]
        ry = a * tau / self.h[1]

        NN = self.number_of_nodes()
        n0 = self.nx + 1
        n1 = self.ny + 1
        K = np.arange(NN).reshape(n0, n1)

        A = diags([2 * (1 - rx ** 2 - ry ** 2)], 0, shape=(NN, NN), format='csr')

        val = np.broadcast_to(rx ** 2, (NN - n1,))
        I = K[1:, :].flat
        J = K[0:-1, :].flat
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(ry ** 2, (NN - n0,))
        I = K[:, 1:].flat
        J = K[:, 0:-1].flat
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        return A

    ## @ingroup FDMInterface
    def wave_operator_implicit(self, tau, a=1, theta=0.25):
        """
        @brief 生成波动方程的隐格式离散矩阵

        @param[in] tau float, 时间步长
        @param[in] a float, 波速，默认值为 1
        @param[in] theta float, 时间离散格式参数，默认值为 0.25

        @return 三个离散矩阵 A0, A1, A2，分别对应于不同的时间步
        """
        rx = a * tau / self.h[0]
        ry = a * tau / self.h[1]

        NN = self.number_of_nodes()
        n0 = self.nx + 1
        n1 = self.ny + 1
        K = np.arange(NN).reshape(n0, n1)

        A0 = diags([1 + 2 * rx ** 2 * theta + 2 * ry ** 2 * theta], 0, shape=(NN, NN), format='csr')
        A1 = diags([2 * (1 - (rx ** 2 + ry ** 2) * (1 - 2 * theta))], 0, shape=(NN, NN), format='csr')
        A2 = diags([-(1 + 2 * rx ** 2 * theta + 2 * ry ** 2 * theta)], 0, shape=(NN, NN), format='csr')

        val = np.broadcast_to(-rx ** 2 * theta, (NN - n1,))
        I = K[1:, :].flat
        J = K[0:-1, :].flat
        A0 += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A0 += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(-ry ** 2 * theta, (NN - n0,))
        I = K[:, 1:].flat
        J = K[:, 0:-1].flat
        A0 += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A0 += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(rx ** 2 * (1 - 2 * theta), (NN - n1,))
        I = K[1:, :].flat
        J = K[0:-1, :].flat
        A1 += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A1 += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(ry ** 2 * (1 - 2 * theta), (NN - n0,))
        I = K[:, 1:].flat
        J = K[:, 0:-1].flat
        A1 += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A1 += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(rx ** 2 * theta, (NN - n1,))
        I = K[1:, :].flat
        J = K[0:-1, :].flat
        A2 += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A2 += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(ry ** 2 * theta, (NN - n0,))
        I = K[:, 1:].flat
        J = K[:, 0:-1].flat
        A2 += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A2 += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        return A0, A1, A2

    ## @ingroup FDMInterface
    def wave_operator_explicity(self, tau, a=1):
        """
        @brief 用显格式求解波动方程
        """
        rx = a * tau / self.h[0]
        ry = a * tau / self.h[1]

        NN = self.number_of_nodes()
        n0 = self.nx + 1
        n1 = self.ny + 1
        K = np.arange(NN).reshape(n0, n1)

        A = diags([2 * (1 - rx ** 2 - ry ** 2)], [0], shape=(NN, NN), format='csr')

        val = np.broadcast_to(rx ** 2, (NN - n1,))
        I = K[1:, :].flat
        J = K[0:-1, :].flat
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(ry ** 2, (NN - n0,))
        I = K[:, 1:].flat
        J = K[:, 0:-1].flat
        A += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        return A

    ## @ingroup FDMInterface
    def wave_operator_theta(self, tau, a=1, theta=0.5):
        """
        @brief 生成波动方程的离散矩阵
        """
        rx = a * tau / self.h[0]
        ry = a * tau / self.h[1]

        NN = self.number_of_nodes()
        n0 = self.nx + 1
        n1 = self.ny + 1
        K = np.arange(NN).reshape(n0, n1)

        A0 = diags([1 + 2 * rx ** 2 * theta + 2 * ry ** 2 * theta], [0], shape=(NN, NN), format='csr')
        A1 = diags([2 * (1 - (rx ** 2 + ry ** 2) * (1 - 2 * theta))], [0], shape=(NN, NN), format='csr')
        A2 = diags([-(1 + 2 * rx ** 2 * theta + 2 * ry ** 2 * theta)], [0], shape=(NN, NN), format='csr')

        val = np.broadcast_to(-rx ** 2, (NN - n1,))
        I = K[1:, :].flat
        J = K[0:-1, :].flat
        A0 += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A0 += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(-ry ** 2, (NN - n0,))
        I = K[:, 1:].flat
        J = K[:, 0:-1].flat
        A0 += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A0 += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(rx ** 2 * (1 - 2 * theta), (NN - n1,))
        I = K[1:, :].flat
        J = K[0:-1, :].flat
        A1 += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A1 += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(ry ** 2 * (1 - 2 * theta), (NN - n1,))
        I = K[:, 1:].flat
        J = K[:, 0:-1].flat
        A1 += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A1 += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(rx ** 2 * theta, (NN - n1,))
        I = K[1:, :].flat
        J = K[0:-1, :].flat
        A2 += csr_matrix((val, (I, J)), shape=(NN, NN), dtype=self.ftype)
        A2 += csr_matrix((val, (J, I)), shape=(NN, NN), dtype=self.ftype)

        val = np.broadcast_to(ry ** 2 * theta, (NN - n0,))
        I = K[:, 1:].flat
        J = K[:, 0:-1].flat
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
        k = nx / ny
        ns = ny
        h = self.h[0]

        phi = self.function(ex=1)
        isNearNode = self.function(dtype=np.bool_, ex=1)

        # 把水平集函数转化为离散的网格函数
        node = self.entity('node')
        phi[1:-1, 1:-1] = phi0
        sign = np.sign(phi[1:-1, 1:-1])

        # 标记界面附近的点
        isNearNode[1:-1, 1:-1] = np.abs(phi[1:-1, 1:-1]) < 2 * h
        lsfun = UniformMesh2dFunction(self, phi[1:-1, 1:-1])
        _, d = lsfun.project(node[isNearNode[1:-1, 1:-1]])
        phi[isNearNode] = np.abs(d)  # 界面附近的点用精确值
        phi[~isNearNode] = m  # 其它点用一个比较大的值

        a = np.zeros(ns + 1, dtype=np.float64)
        b = np.zeros(ns + 1, dtype=np.float64)
        c = np.zeros(ns + 1, dtype=np.float64)
        d = np.zeros(int(k * ns + 1), dtype=np.float64)
        e = np.zeros(int(k * ns + 1), dtype=np.float64)
        f = np.zeros(int(k * ns + 1), dtype=np.float64)

        n = 0
        for i in range(1, int(k * ns + 2)):
            a[:] = np.minimum(phi[i - 1, 1:-1], phi[i + 1, 1:-1])
            b[:] = np.minimum(phi[i, 0:ns + 1], phi[i, 2:])
            flag = np.abs(a - b) >= h
            c[flag] = np.minimum(a[flag], b[flag]) + h
            c[~flag] = (a[~flag] + b[~flag] + np.sqrt(2 * h * h - (a[~flag] - b[~flag]) ** 2)) / 2
            phi[i, 1:-1] = np.minimum(c, phi[i, 1:-1])
            n += 1

        for i in range(int(k * ns + 1), 0, -1):
            a[:] = np.minimum(phi[i - 1, 1:-1], phi[i + 1, 1:-1])
            b[:] = np.minimum(phi[i, 0:ns + 1], phi[i, 2:])
            flag = np.abs(a - b) >= h
            c[flag] = np.minimum(a[flag], b[flag]) + h
            c[~flag] = (a[~flag] + b[~flag] + np.sqrt(2 * h * h - (a[~flag] - b[~flag]) ** 2)) / 2
            phi[i, 1:-1] = np.minimum(c, phi[i, 1:-1])
            n += 1

        for j in range(1, ns + 2):
            d[:] = np.minimum(phi[0:int(k * ns + 1), j], phi[2:, j])
            e[:] = np.minimum(phi[1:-1, j - 1], phi[1:-1, j + 1])
            flag = np.abs(d - e) >= h
            f[flag] = np.minimum(d[flag], e[flag]) + h
            f[~flag] = (d[~flag] + e[~flag] + np.sqrt(2 * h * h - (d[~flag] - e[~flag]) ** 2)) / 2
            phi[1:-1, j] = np.minimum(f, phi[1:-1, j])
            n += 1

        for j in range(ns + 1, 0, -1):
            d[:] = np.minimum(phi[0:int(k * ns + 1), j], phi[2:, j])
            e[:] = np.minimum(phi[1:-1, j - 1], phi[1:-1, j + 1])
            flag = np.abs(d - e) >= h
            f[flag] = np.minimum(d[flag], e[flag]) + h
            f[~flag] = (d[~flag] + e[~flag] + np.sqrt(2 * h * h - (d[~flag] - e[~flag]) ** 2)) / 2
            phi[1:-1, j] = np.minimum(f, phi[1:-1, j])
            n += 1

        return sign * phi[1:-1, 1:-1]

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
        """
        @brief 把积分点变换到实际网格实体上的笛卡尔坐标点
        """
        node = self.entity('node')
        if isinstance(bc, tuple):
            assert len(bc) == 2
            cell = self.entity('cell')[index]

            bc0 = bc[0].reshape(-1, 2) # (NQ0, 2)
            bc1 = bc[1].reshape(-1, 2) # (NQ1, 2)
            bc = np.einsum('im, jn->ijmn', bc0, bc1).reshape(-1, 4) # (NQ0, NQ1, 2, 2)

            # node[cell].shape == (NC, 4, 2)
            # bc.shape == (NQ, 4)
            p = np.einsum('...j, cjk->...ck', bc, node[cell[:]]) # (NQ, NC, 2)
            if p.shape[0] == 1: # 如果只有一个积分点
                p = p.reshape(-1, 2)
        else:
            edge = self.entity('edge')[index]
            p = np.einsum('...j, ejk->...ek', bc, node[edge]) # (NQ, NE, 2)
        return p

    ## @ingroup FEMInterface
    def entity(self, etype, index=np.s_[:]):
        """
        @brief Get the entity (either cell or node) based on the given entity type.

        @param[in] etype The type of entity, either 'cell', 2, 'edge', 'face' or 1, `node', 0.

        @return The cell, edeg, face, or node array based on the input entity type.

        @throws ValueError if the given etype is invalid.
        """
        if etype in {'cell', 2}:
            return self.ds.cell[index, ...]
        elif etype in {'edge', 'face', 1}:
            return self.ds.edge[index, ...]
        elif etype in {'node', 0}:
            return self.node.reshape(-1, 2)[index, ...]
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
    def entity_measure(self, etype=2, index=np.s_[:]):
        """
        @brief
        """
        TD = self.top_dimension()
        if etype in {'cell', TD}:
            return self.cell_area()
        elif etype in {'edge', 'face', TD - 1}:
            return self.edge_length(index=index)
        elif etype in {'node', 0}:
            return np.array(0.0, self.ftype)
        raise ValueError(f"Invalid entity type: {type(etype).__name__}.")

    ## @ingroup FEMInterface
    def prolongation_matrix(self, p0:int, p1:int):
        """
        @brief 生成从 p0 元到 p1 元的延拓矩阵，假定 0 < p0 < p1
        """

        assert 0 < p0 < p1

        TD = self.top_dimension()

        gdof1 = self.number_of_global_ipoints(p1)
        gdof0 = self.number_of_global_ipoints(p0)

        # 1. 网格节点上的插值点 
        NN = self.number_of_nodes()
        I = range(NN)
        J = range(NN)
        V = np.ones(NN, dtype=self.ftype)
        P = coo_matrix((V, (I, J)), shape=(gdof1, gdof0))

        # 2. 网格边内部的插值点 
        NE = self.number_of_edges()
        # p1 元在边上插值点对应的重心坐标
        bcs = self.multi_index_matrix(p1, 1)/p1 
        # p0 元基函数在 p1 元对应的边内部插值点处的函数值
        phi = self.edge_shape_function(bcs[1:-1], p=p0) # (ldof1 - 2, ldof0)  

        e2p1 = self.edge_to_ipoint(p1)[:, 1:-1]
        e2p0 = self.edge_to_ipoint(p0)
        shape = (NE, ) + phi.shape

        I = np.broadcast_to(e2p1[:, :, None], shape=shape).flat
        J = np.broadcast_to(e2p0[:, None, :], shape=shape).flat
        V = np.broadcast_to( phi[None, :, :], shape=shape).flat

        P += coo_matrix((V, (I, J)), shape=(gdof1, gdof0))

        # 3. 单元内部的插值点
        NC = self.number_of_cells()
        # p1 元在单元上对应插值点的重心坐标
        bcs = self.multi_index_matrix(p1, 1)/p1
        # p0 元基函数在 p1 元对应的单元内部插值点处的函数值
        phi = self.cell_shape_function((bcs[1:-1], bcs[1:-1]), p=p0) #
        c2p1 = self.cell_to_ipoint(p1).reshape(NC, p1+1, p1+1)[:, 1:-1, 1:-1]
        c2p1 = c2p1.reshape(NC, -1)
        c2p0 = self.cell_to_ipoint(p0)

        shape = (NC, ) + phi.shape

        I = np.broadcast_to(c2p1[:, :, None], shape=shape).flat
        J = np.broadcast_to(c2p0[:, None, :], shape=shape).flat
        V = np.broadcast_to( phi[None, :, :], shape=shape).flat

        P += coo_matrix((V, (I, J)), shape=(gdof1, gdof0))

        return P.tocsr()

    ## @ingroup FEMInterface
    def shape_function(self, bc, p=1):
        """
        @brief 四边形单元上的形函数
        """
        assert isinstance(bc, tuple)
        GD = len(bc)
        phi = [self._shape_function(val, p=p) for val in bc]
        ldof = (p+1)**GD
        return np.einsum('im, jn->ijmn', phi[0], phi[1]).reshape(-1, ldof)


    ## @ingroup FEMInterface
    def grad_shape_function(self, bc, p=1, variables='x', index=np.s_[:]):
        """
        @brief  四边形单元形函数的导数

        @note 计算单元形函数关于参考单元变量 u=(xi, eta) 或者实际变量 x 梯度。

        bc 是一个长度为 2 的 tuple

        bc[i] 是一个一维积分公式的重心坐标数组

        这里假设 bc[0] == bc[1] == ... = bc[TD-1]
        """
        assert isinstance(bc, tuple) and len(bc) == 2

        Dlambda = np.array([-1, 1], dtype=self.ftype)

        phi0 = self._shape_function(bc[0], p=p)
        R0 = self._grad_shape_function(bc[0], p=p)
        gphi0 = np.einsum('...ij, j->...i', R0, Dlambda) # (..., ldof)

        phi1 = self._shape_function(bc[1], p=p)
        R1 = self._grad_shape_function(bc[1], p=p)
        gphi1 = np.einsum('...ij, j->...i', R1, Dlambda) # (..., ldof)

        n = phi0.shape[0]*phi1.shape[0] # 张量积分点的个数
        ldof = phi0.shape[-1]*phi1.shape[-1]

        shape = (n, ldof, 2)
        gphi = np.zeros(shape, dtype=self.ftype)

        gphi[..., 0] = np.einsum('im, kn->ikmn', gphi0, phi1).reshape(-1, ldof)
        gphi[..., 1] = np.einsum('im, kn->ikmn', phi0, gphi1).reshape(-1, ldof)

        if variables == 'u':
            return gphi
        elif variables == 'x':
            J = self.jacobi_matrix(bc, index=index)
            G = self.first_fundamental_form(J)
            G = np.linalg.inv(G)
            gphi = np.einsum('...ikm, ...imn, ...ln->...ilk', J, G, gphi)
            return gphi

    def jacobi_matrix(self, bc, index=np.s_[:]):
        """
        @brief 计算参考单元 (xi, eta) 到实际 Lagrange 四边形(x) 之间映射的 Jacobi 矩阵。

        x(xi, eta) = phi_0 x_0 + phi_1 x_1 + ... + phi_{ldof-1} x_{ldof-1}
        """
        node = self.entity('node')
        cell = self.entity('cell', index=index)
        gphi = self.grad_shape_function(bc, p=1, variables='u', index=index)
        J = np.einsum( 'cim, ...in->...cmn', node[cell[:]], gphi)
        return J

    def first_fundamental_form(self, J):
        """
        @brief 由 Jacobi 矩阵计算第一基本形式。
        """
        TD = J.shape[-1]

        shape = J.shape[0:-2] + (TD, TD)
        G = np.zeros(shape, dtype=self.ftype)
        for i in range(TD):
            G[..., i, i] = np.einsum('...d, ...d->...', J[..., i], J[..., i])
            for j in range(i+1, TD):
                G[..., i, j] = np.einsum('...d, ...d->...', J[..., i], J[..., j])
                G[..., j, i] = G[..., i, j]
        return G

    ## @ingroup FEMInterface
    def number_of_local_ipoints(self, p, iptype='cell'):
        if iptype in {'cell', 2}:
            return (p+1)*(p+1)
        elif iptype in {'face', 'edge',  1}:
            return p + 1
        elif iptype in {'node', 0}:
            return 1

    ## @ingroup FEMInterface
    def number_of_global_ipoints(self, p):
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()
        return NN + (p-1)*NE + (p-1)*(p-1)*NC

    ## @ingroup FEMInterface
    def interpolation_points(self, p, index=np.s_[:]):
        """
        @brief 获取四边形网格上所有 p 次插值点
        """
        cell = self.entity('cell')
        node = self.entity('node')
        if p == 1:
            return node

        NN = self.number_of_nodes()
        GD = self.geo_dimension()

        gdof = self.number_of_global_ipoints(p)
        ipoints = np.zeros((gdof, GD), dtype=self.ftype)
        ipoints[:NN, :] = node

        NE = self.number_of_edges()

        edge = self.entity('edge')

        multiIndex = self.multi_index_matrix(p, 1)
        w = multiIndex[1:-1, :]/p
        ipoints[NN:NN+(p-1)*NE, :] = np.einsum('ij, ...jm->...im', w,
                node[edge,:]).reshape(-1, GD)

        w = np.einsum('im, jn->ijmn', w, w).reshape(-1, 4)
        ipoints[NN+(p-1)*NE:, :] = np.einsum('ij, kj...->ki...', w,
                node[cell[:]]).reshape(-1, GD)

        return ipoints

    ## @ingroup FEMInterface
    def node_to_ipoint(self, p):
        pass

    ## @ingroup FEMInterface
    def face_to_ipoint(self, p):
        pass

    ## @ingroup FEMInterface
    def cell_to_ipoint(self, p, index=np.s_[:]):
        """
        @brief 获取单元上的双 p 次插值点
        """

        cell = self.entity('cell')

        if p==1:
            return cell[index] # 先排 y 方向，再排 x 方向

        edge2cell = self.ds.edge_to_cell()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()

        cell2ipoint = np.zeros((NC, (p+1)*(p+1)), dtype=self.itype)
        c2p= cell2ipoint.reshape((NC, p+1, p+1))

        e2p = self.edge_to_ipoint(p)
        flag = edge2cell[:, 2] == 0
        c2p[edge2cell[flag, 0], :, 0] = e2p[flag]
        flag = edge2cell[:, 2] == 1
        c2p[edge2cell[flag, 0], -1, :] = e2p[flag]
        flag = edge2cell[:, 2] == 2
        c2p[edge2cell[flag, 0], :, -1] = e2p[flag, -1::-1]
        flag = edge2cell[:, 2] == 3
        c2p[edge2cell[flag, 0], 0, :] = e2p[flag, -1::-1]


        iflag = edge2cell[:, 0] != edge2cell[:, 1]
        flag = iflag & (edge2cell[:, 3] == 0)
        c2p[edge2cell[flag, 1], :, 0] = e2p[flag, -1::-1]
        flag = iflag & (edge2cell[:, 3] == 1)
        c2p[edge2cell[flag, 1], -1, :] = e2p[flag, -1::-1]
        flag = iflag & (edge2cell[:, 3] == 2)
        c2p[edge2cell[flag, 1], :, -1] = e2p[flag]
        flag = iflag & (edge2cell[:, 3] == 3)
        c2p[edge2cell[flag, 1], 0, :] = e2p[flag]

        c2p[:, 1:-1, 1:-1] = NN + NE*(p-1) + np.arange(NC*(p-1)*(p-1)).reshape(NC, p-1, p-1)

        return cell2ipoint[index]

    def t2sidx(self):
        """
        @brief 已知结构三角形网格点的值，将其排列到结构四边形网格上
        @example a[s2tidx] = uh
        """
        snx = self.ds.nx
        sny = self.ds.ny
        idx1 = np.arange(0, sny + 1, 2) + np.arange(0, (sny + 1) * (snx + 1), 2 * (sny + 1)).reshape(-1, 1)
        idx1 = idx1.flatten()
        a = np.array([1, sny + 1, sny + 2])
        b = np.arange(0, sny, 2).reshape(-1, 1)
        c = np.append(a + b, [(sny + 1) * 2 - 1])
        e = np.arange(0, (sny + 1) * snx, 2 * (sny + 1)).reshape(-1, 1)
        idx2 = (e + c).flatten()
        idx3 = np.arange((sny + 1) * snx + 1, (snx + 1) * (sny + 1), 2)
        idx = np.r_[idx1, idx2, idx3]
        return idx

    def s2tidx(self):
        """
        @brief 已知结构四边形网格点的值，将其排列到结构三角形网格上
        @example a[s2tidx] = uh
        """
        tnx = int(self.ds.nx / 2)
        tny = int(self.ds.ny / 2)
        a = np.arange(tny + 1)
        b = 3 * np.arange(tny).reshape(-1, 1) + (tnx + 1) * (tny + 1)
        idx1 = np.zeros((tnx + 1, 2 * tny + 1))  # sny+1
        idx1[:, 0::2] = a + np.arange(tnx + 1).reshape(-1, 1) * (tny + 1)
        idx1[:, 1::2] = b.flatten() + np.arange(tnx + 1).reshape(-1, 1) * (2 * tny + 1 + tny)
        idx1[-1, 1::2] = np.arange((2 * tnx + 1) * (2 * tny + 1) - tny, (2 * tnx + 1) * (2 * tny + 1))
        c = np.array([(tnx + 1) * (tny + 1) + 1, (tnx + 1) * (tny + 1) + 2])
        d = np.arange(tny) * 3
        d = 3 * np.arange(tny).reshape(-1, 1) + c
        e = np.append(d.flatten(), [d.flatten()[-1]+ 1])
        idx2 = np.arange(tnx).reshape(-1, 1) * (2 * tny + 1 + tny) + e

        idx = np.c_[idx1[:tnx], idx2]
        idx = np.append(idx.flatten(), [idx1[-1, :]])
        idx = idx.astype(int)
        return idx

    def data_edge_to_cell(self, Ex, Ey):
        """
        @brief 把定义在边上的数组转换到单元上
        """
        dx = self.function(etype='cell')
        dy = self.function(etype='cell')

        dx[:] = (Ex[:, :-1] + Ex[:, 1:]) / 2.0
        dy[:] = (Ey[:-1, :] + Ey[1:, :]) / 2.0

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

        idx = np.arange(0, nx * (ny + 1), 2 * (ny + 1)).reshape(-1, 1) + np.arange(0,
                                                                                   ny + 1, 2)
        idxMap[idx] = range(tmesh.number_of_nodes())

        return idxMap

    def is_cut_cell(self, phi):
        """
        @brief
        """
        phiSign = msign(phi)
        cell = self.entity('cell')
        isCutCell = np.abs(np.sum(phiSign[cell], axis=1)) < 3
        return isCutCell

    def compute_cut_point(self, phi):
        """
        """
        node = self.entity('node')
        edge = self.entity('edge')
        phiSign = msign(phi)
        isCutEdge = phiSign[edge[:, 0]] * phiSign[edge[:, 1]] < 0
        A = node[edge[isCutEdge, 0]]
        B = node[edge[isCutEdge, 1]]

        interface = UniformMesh2dFunction(self, phi)
        cutNode = find_cut_point(interface, A, B)
        return cutNode

    def find_interface_node(self, phi):
        N = self.number_of_nodes()
        h = min(self.h)

        node = self.entity('node')
        cell = self.entity('cell')[:, [0, 2, 3, 1]]
        phiValue = phi(node)
        phiValue[np.abs(phiValue) < 0.1*h**2] = 0.0
        phiSign = np.sign(phiValue)

        # 寻找 cut 点
        edge = self.entity('edge')
        isCutEdge = phiSign[edge[:, 0]] * phiSign[edge[:, 1]] < 0
        e0 = node[edge[isCutEdge, 0]]
        e1 = node[edge[isCutEdge, 1]]
        cutNode = find_cut_point(phi, e0, e1)
        ncut = cutNode.shape[0]

        # 界面单元与界面节点
        isInterfaceCell = self.is_cut_cell(phiValue)
        isInterfaceNode = np.zeros(N, dtype=np.bool_)
        isInterfaceNode[cell[isInterfaceCell, :]] = True

        # 需要处理的特殊单元，即界面经过两对顶点的单元
        isSpecialCell = (np.sum(np.abs(phiSign[cell]), axis=1) == 2) \
                        & (np.sum(phiSign[cell], axis=1) == 0)
        scell = cell[isSpecialCell, :]

        # 构建辅助点，即单元重心
        auxNode = (node[scell[:, 0], :] + node[scell[:, 2], :]) / 2
        naux = auxNode.shape[0]

        # 拼接界面节点
        interfaceNode = np.concatenate(
            (node[isInterfaceNode, :], cutNode, auxNode),
            axis=0)

        return interfaceNode, isInterfaceNode, isInterfaceCell, ncut, naux


UniformMesh2d.set_ploter('2d')


class UniformMesh2dFunction():
    def __init__(self, mesh, f):
        self.mesh = mesh  # (nx+1, ny+1)
        self.f = f  # (nx+1, ny+1)
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
