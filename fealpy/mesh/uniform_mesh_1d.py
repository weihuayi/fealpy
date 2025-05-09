from typing import Union, Tuple, List
from types import ModuleType

from .mesh_base import StructuredMesh, TensorMesh
from .plot import Plotable

from builtins import int , float
from ..typing import TensorLike, Index, _S, Union, Tuple
from .utils import entitymethod, estr2dim
from ..backend import backend_manager as bm

class UniformMesh1d(StructuredMesh, TensorMesh, Plotable):
    """
    @brief    A class for representing a uniformly partitioned one-dimensional mesh.
    """
    def __init__(self, extent: Tuple[int, int] = (0, 1),
            h: Union[float, Tuple[float], List[float]] = 1.0,
            origin: float = 0.0,
            itype= None, ftype= None, device=None):
        """
        @brief        Initialize the 1D uniform mesh.
        Parameters:
            extent: Defines the number of cells in the mesh divisions.
            h: Mesh step size.
            origin: Coordinate of the starting point.
            itype: Integer type to be used, default: int32.
            ftype: Floating point type to be used, default: float64.

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
        if itype is None:
            itype = bm.int32
        if ftype is None:
            ftype = bm.float64
        super().__init__(TD=1, itype=itype, ftype=ftype)

        self.device = device

        if isinstance(h, float):
            h = (h, )

        # Mesh properties
        self.extent = extent
        self.h = h
        self.origin = origin
        # self.extent = bm.array(extent, dtype=itype, device=device)
        # self.h = bm.array(h, dtype=ftype, device=device) 
        # self.origin = bm.array(origin, dtype=ftype, device=device)
        self.shape = (self.extent[1] - self.extent[0], )

        # Mesh dimensions
        self.nx = self.extent[1] - self.extent[0]
        self.NC = self.nx
        self.NE = self.NC
        self.NC = self.NC
        self.NN = self.NC + 1
        self.NF = self.NN

        # Mesh datas
        self.nodedata = {}
        self.edgedata = {}
        self.facedata = self.edgedata
        self.celldata = {}
        self.meshdata = {}

        self.meshtype = 'UniformMesh1d'

    def interpolate(self, u, etype=0, keepdims=False) -> TensorLike:
        """
        Compute the interpolation of a function u on the mesh.

        Parameters:
            u: The function to be interpolated.
            etype: The type of entity on which to interpolate.

        Example:
        ```
            from fealpy.mesh import UniformMesh1d
            mesh = UniformMesh1d(extent=[0, 10], h=0.1, origin=0.0)
            u = mesh.interpolate(lambda x: x**2)
            print(u)
        ```
        """
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        if etype == 0:
            node = self.entity('node')
            return u(node)
        else:
            raise ValueError(f"Unsupported entity type: {etype}")

    def linear_index_map(self, etype: Union[int, str]=0):
        """
        Build and return the tensor mapping multi-dimensional 
        indices to linear indices.
        """
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        if etype == 0:
            return bm.arange(self.NN, dtype=self.itype, device=self.device)
        elif etype == 1:
            return bm.arange(self.NC, dtype=self.itype, device=self.device)

    def linear_to_multi_index(self, linear_indices):
        """
        Given linear indices, return the corresponding multi-dimensional indices.
        """
        pass


    # 实体生成方法
    @entitymethod(0)
    def _get_node(self) -> TensorLike:
        """
        @berif Generate the nodes in a structured mesh.
        """
        device = self.device

        GD = 1
        nx = self.nx
        node = bm.linspace(self.origin, self.origin + nx * self.h[0], nx+1, dtype=self.ftype, device=device)
        return node.reshape(-1, 1)

    #未测试
    @entitymethod(1) 
    def _get_edge(self) -> TensorLike:
        """
        @berif Generate the edges in a structured mesh.
        """
        device = self.device

        nx = self.nx  
        NN = self.NN  
        NE = self.NE  

        idx = bm.arange(NN, dtype=self.itype, device=device)  
        edge = bm.zeros((NE, 2), dtype=self.itype, device=device)

        edge = bm.set_at(edge, (slice(None), 0), idx[:-1])  
        edge = bm.set_at(edge, (slice(None), 1), idx[1:])   
        
        return edge
    
    @entitymethod(2)#未测试
    def _get_cell(self) -> TensorLike:
        """
        @berif Generate the cells in a structured mesh.
        """
        return self._get_edge()
    
    # 实体拓扑
    def number_of_nodes_of_cells(self):
        return 2

    def number_of_edges_of_cells(self):
        return 1

    def number_of_faces_of_cells(self):
        return 1

    def boundary_node_flag(self):
        """ Determine if a point is a boundary point.
        """
        device = self.device
        isBdNode = bm.zeros((self.NN,), dtype=bm.bool, device=device)
        isBdNode = bm.set_at(isBdNode, [0, -1], True)
        
        return isBdNode

    def boundary_cell_flag(self):
        """
        @brief Determine boundary cells in the 1D structure mesh.

        @return isBdCell : np.array, dtype=np.bool_
            An array of booleans where True indicates a boundary cell.
        """
        device = self.device

        NC = self.number_of_cells()
        isBdCell = bm.zeros((NC,), dtype=bm.bool, device=device)
        isBdCell = bm.set_at(isBdCell, [0, -1], True)
        return isBdCell
    
    def boundary_edge_flag(self):
        """
        @brief Determine if an edge is a boundary edge.
        """
        return self.boundary_cell_flag()

#################################### 实体几何 #############################################
    def entity_measure(self, etype: Union[int, str], index: Index = _S) -> TensorLike:
        """
        @brief Get the measure of the entities of the specified type.
        """
        device = self.device
        NE = self.NE
        NC = self.NC

        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        NC = self.number_of_cells()
        if etype == 0:
            return bm.tensor(0, dtype=self.ftype, device=device)
        elif etype == 1:
            temp1 = bm.zeros((NE, ), dtype=self.ftype, device=self.device)
            temp2 = temp1 + self.h
            return temp2.reshape(-1)
        elif etype == 2:
            temp1 = bm.zeros((NC, ), dtype=self.ftype, device=self.device)
            temp2 = temp1 + self.h
            return temp2.reshape(-1)
        else:
            raise ValueError(f"Unsupported entity or top-dimension: {etype}")
        
    def cell_barycenter(self)-> TensorLike:
        """
        @brief
        Note: 一维中, edge 和 cell 相同
        """
        nx = self.nx
        box = [self.origin + self.h/2, self.origin + self.h/2 + (nx - 1) * self.h]
        bc = bm.linspace(box[0], box[1], nx)
        return bc
    
    def edge_barycenter(self)-> TensorLike:
        """
        @brief Calculate the coordinates range for the edge centers.
        """
        return self.cell_barycenter()
    
    def bc_to_point(self, bcs: Union[Tuple, TensorLike], index=_S):
        node = self.node
        cell = self.entity('cell', index=index)
        p = bm.einsum('...j, ijk->...ik', bcs, node[cell[index]])
        return p

    def cell_location(self, ps):
        """
        @brief 给定一组点，确定所有点所在的单元
        """
        h = self.h

        v = ps - self.origin
        n0 = v//h

        return n0.astype('int64')


    def point_to_bc(self, points):
        device = self.device

        bc_x_ = ((points - self.origin) / self.h[0]) % 1
        bc_x = bm.array([[bc_x_, 1 - bc_x_]], dtype=bm.float64, device=device)
    
        return bc_x


#################################### 插值点 #############################################
    ## @ingroup FEMInterface
    def interpolation_points(self, p: int, index: Index=_S) -> TensorLike:
        """
        @brief 获取网格上的所有插值点
        """
        device = self.device

        if p <= 0:
            raise ValueError("p must be a integer larger than 0.")
        if p == 1:
            return self.entity('node', index=index)

        node = self.entity('node')
        cell = self.entity('cell')

        NN = self.number_of_nodes()
        GD = self.geo_dimension()

        gdof = self.number_of_global_ipoints(p)
        ipoints = bm.zeros((gdof, GD), dtype=self.ftype, device=device)
        ipoints[:NN, :] = node

        NC = self.number_of_cells()

        w = bm.zeros((p-1, 2), dtype=bm.float64, device=device)
        w[:, 0] = bm.arange(p-1, 0, -1)/p
        w[:, 1] = bm.flip(w[:, 0], axis=0)
        ipoints[NN:NN+(p-1)*NC, :] = bm.einsum('ij, ...jm->...im', w, node[cell,:]).reshape(-1, GD)

        return ipoints[index]

    def node_to_ipoint(self, p):
        device = self.device
        NN = self.number_of_nodes()
        return bm.arange(NN, dtype=self.itype, device=device)

    def edge_to_ipoint(self, p):
        return self.cell_to_ipoint(p)

    def face_to_ipoint(self, p):
        device = self.device
        NN = self.number_of_nodes()
        return bm.arange(NN, dtype=self.itype, device=device)

    def cell_to_ipoint(self, p: int, index: Index=_S) -> TensorLike:
        device = self.device

        NN = self.number_of_nodes()
        NC = self.number_of_cells()

        cell = self.entity('cell')
        cell2ipoints = bm.zeros((NC, p+1), dtype=self.itype, device=device)
        cell2ipoints[:, [0, -1]] = cell 
        if p > 1:
            cell2ipoints[:, 1:-1] = NN + bm.arange(NC*(p-1), bm.arange(NN, dtype=bm.int64, device=device)).reshape(NC, p-1)
        return cell2ipoints[index]

    def quadrature_formula(self, q: int, etype:Union[int, str]='cell'):
        """
        @brief Get the quadrature formula for numerical integration.
        """
        from ..quadrature import GaussLegendreQuadrature, TensorProductQuadrature
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        qf = GaussLegendreQuadrature(q, dtype=self.ftype, device=self.device)
        if etype == 2:
            return TensorProductQuadrature((qf, qf))
        elif etype == 1:
            return qf
        else:
            raise ValueError(f"entity type: {etype} is wrong!")

    def uniform_refine(self, n: int=1):
        """
        @brief Uniformly refine the 1D structured mesh.

        Note:
        The clear method is used at the end to clear the cache of entities. 
        This is necessary because the entities remain the same as before refinement due to caching.
        Structured meshes have their own entity generation methods, so the cache needs to be manually cleared.
        Unstructured meshes do not require this because they do not have entity generation methods.
        """
        for i in range(n):
            self.extent = 2*self.extent 
            self.h = self.h/2
            self.nx = self.extent[1] - self.extent[0]
            self.NC = self.nx
            self.NE = self.NC
            self.NC = self.NC
            self.NN = self.NC + 1

        self.clear()

    # 其他方法
    def quadrature_formula(self, q: int, etype:Union[int, str]='cell'):
        """
        @brief Get the quadrature formula for numerical integration.
        """
        from ..quadrature import GaussLegendreQuadrature
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        qf = GaussLegendreQuadrature(q, dtype=self.ftype, device=self.device)
        return qf        


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

UniformMesh1d.set_ploter('1d')
