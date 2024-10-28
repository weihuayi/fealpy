import numpy as np
from typing import Union
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, coo_matrix
import inspect

from fealpy.common import ranges

from .mesh_base import Mesh, Plotable
from .mesh_data_structure import Mesh2dDataStructure, ArrRedirector


class PolygonMesh(Mesh, Plotable):
    """
    @brief Polygon mesh type.
    """
    ds: "PolygonMeshDataStructure"

    def __init__(self, node: NDArray, cell: NDArray, cellLocation=None, topdata=None):
        self.node = node
        if cellLocation is None:
            if isinstance(cell, list):
                cellLocation = np.zeros(len(cell)+1, dtype=np.int_)
                for i in range(len(cell)):
                    cellLocation[i+1] = cellLocation[i]+len(cell[i])
                cell = np.concatenate(cell)
            elif isinstance(cell, np.ndarray) and len(cell.shape)== 2:
                NC = cell.shape[0]
                NV = cell.shape[1]
                cell = cell.reshape(-1)
                cellLocation = np.arange(0, (NC+1)*NV, NV)
            else:
                raise ValueError("Miss `cellLocation` array!")

        self.ds = PolygonMeshDataStructure(node.shape[0], cell, cellLocation,
                topdata=None)
        self.meshtype = 'polygon'
        self.itype = cell.dtype
        self.ftype = node.dtype

        self.celldata = {}
        self.nodedata = {}
        self.edgedata = {}
        self.facedata = self.edgedata
        self.meshdata = {}


    def integrator(self, q, etype='cell', qtype='legendre'):
        """
        @brief 获取不同维度网格实体上的积分公式
        """
        if etype in {'cell', 2}:
            from ..quadrature import TriangleQuadrature
            return TriangleQuadrature(q)
        elif etype in {'edge', 'face', 1}:
            if qtype in {'legendre'}:
                from ..quadrature import GaussLegendreQuadrature
                return GaussLegendreQuadrature(q)
            elif qtype in {'lobatto'}:
                from ..quadrature import GaussLobattoQuadrature
                return GaussLobattoQuadrature(q)

    def entity_barycenter(self, etype: Union[int, str]='cell', index=np.s_[:]):
        node = self.entity('node')
        GD = self.geo_dimension()

        if etype in {'cell', 2}:
            cell2node = self.ds.cell_to_node(return_sparse=True)
            NV = self.ds.number_of_vertices_of_cells().reshape(-1, 1)
            bc = cell2node*node/NV
        elif etype in {'edge', 'face', 1}:
            edge = self.ds.edge
            bc = np.mean(node[edge, :], axis=1).reshape(-1, GD)
        elif etype in {'node', 0}:
            bc = node
        return bc

    def entity_measure(self, etype=2, index=np.s_[:]):
        if etype in {'cell', 2}:
            return self.cell_area(index=index)
        elif etype in {'edge', 'face', 1}:
            return self.edge_length(index=index)
        elif etype in {'node', 0}:
            return 0
        else:
            raise ValueError(f"Invalid entity type '{etype}'.")

    def cell_area(self, index=np.s_[:]):
        """
        @brief 根据散度定理计算多边形的面积
        @note 请注意下面的计算方式不方便实现部分单元面积的计算
        """
        NC = self.number_of_cells()
        node = self.entity('node')
        edge = self.entity('edge')
        edge2cell = self.ds.edge_to_cell()

        t = self.edge_tangent()
        val = t[:, 1]*node[edge[:, 0], 0] - t[:, 0]*node[edge[:, 0], 1]

        a = np.zeros(NC, dtype=self.ftype)
        np.add.at(a, edge2cell[:, 0], val)

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        np.add.at(a, edge2cell[isInEdge, 1], -val[isInEdge])

        a /= 2.0

        return a[index]

    def bc_to_point(self, bc: NDArray, etype: Union[int, str]='cell',
                    index=np.s_[:]) -> NDArray:
        if etype in {'cell', 2}:
            raise NotImplementedError("cell_bc_to_point has not been implemented"
                                      "for polygon mesh.")
        else:
            return self.edge_bc_to_point(bcs=bc, index=index)

    def edge_bc_to_point(self, bcs: NDArray, index=np.s_[:]):
        """
        @brief 给出边上的重心坐标，返回其对应的插值点
        """
        node = self.entity('node')
        edge = self.entity('edge')
        ps = np.einsum('ij, kjm->ikm', bcs, node[edge[index]])
        return ps

    face_bc_to_point = edge_bc_to_point

    def number_of_global_ipoints(self, p: int) -> int:
        """
        @brief 插值点总数
        """
        gdof = self.number_of_nodes()
        if p > 1:
            NE = self.number_of_edges()
            NC = self.number_of_cells()
            gdof += NE*(p-1) + NC*(p-1)*p//2
        return gdof

    def number_of_local_ipoints(self,
            p: int, iptype: Union[int, str]='all') -> Union[NDArray, int]:
        """
        @brief 获取局部插值点的个数
        """
        if iptype in {'all'}:
            NV = self.ds.number_of_vertices_of_cells()
            ldof = NV + (p-1)*NV + (p-1)*p//2
            return ldof
        elif iptype in {'cell', 2}:
            return (p-1)*p//2
        elif iptype in {'edge', 'face', 1}:
            return (p+1)
        elif iptype in {'node', 0}:
            return 1

    def cell_to_ipoint(self, p: int, index=np.s_[:]) -> NDArray:
        """
        @brief
        """
        cell = self.entity('cell')
        if p == 1:
            return cell[index]
        else:
            NC = self.number_of_cells()
            ldof = self.number_of_local_ipoints(p, iptype='all')

            location = np.zeros(NC+1, dtype=self.itype)
            location[1:] = np.add.accumulate(ldof)

            cell2ipoint = np.zeros(location[-1], dtype=self.itype)

            edge2ipoint = self.edge_to_ipoint(p)
            edge2cell = self.ds.edge_to_cell()

            idx = location[edge2cell[:, [0]]] + edge2cell[:, [2]]*p + np.arange(p)
            cell2ipoint[idx] = edge2ipoint[:, 0:p]

            isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
            idx = (location[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]*p).reshape(-1, 1) + np.arange(p)
            cell2ipoint[idx] = edge2ipoint[isInEdge, p:0:-1]

            NN = self.number_of_nodes()
            NV = self.ds.number_of_vertices_of_cells()
            NE = self.number_of_edges()
            cdof = self.number_of_local_ipoints(p, iptype='cell')
            idx = (location[:-1] + NV*p).reshape(-1, 1) + np.arange(cdof)
            cell2ipoint[idx] = NN + NE*(p-1) + np.arange(NC*cdof).reshape(NC, cdof)
            return np.hsplit(cell2ipoint, location[1:-1])[index]

    def edge_to_ipoint(self, p: int, index=np.s_[:]) -> NDArray:
        """
        @brief 获取网格边与插值点的对应关系

        """
        if isinstance(index, slice) and index == slice(None):
            NE = self.number_of_edges()
            index = np.arange(NE)
        elif isinstance(index, np.ndarray) and (index.dtype == np.bool_):
            index, = np.nonzero(index)
            NE = len(index)
        elif isinstance(index, list) and (type(index[0]) is np.bool_):
            index, = np.nonzero(index)
            NE = len(index)
        else:
            NE = len(index)

        NN = self.number_of_nodes()

        edge = self.entity('edge', index=index)
        edge2ipoints = np.zeros((NE, p+1), dtype=self.itype)
        edge2ipoints[:, [0, -1]] = edge
        if p > 1:
            idx = NN + np.arange(p-1)
            edge2ipoints[:, 1:-1] =  (p-1)*index[:, None] + idx
        return edge2ipoints

    face_to_ipoint = edge_to_ipoint
    
    def edge_normal(self, index=np.s_[:]):
        """
        @brief 计算二维网格中每条边上单位法线
        """
        assert self.geo_dimension() == 2
        v = self.edge_tangent(index=index)
        w = np.array([(0,-1),(1,0)])
        return v@w

    def edge_unit_normal(self, index=np.s_[:]):
        """
        @brief 计算二维网格中每条边上单位法线
        """
        assert self.geo_dimension() == 2
        v = self.edge_unit_tangent(index=index)
        w = np.array([(0,-1),(1,0)])
        return v@w

    face_normal = edge_normal
    face_unit_normal = edge_unit_normal


    def interpolation_points(self, p: int,
            index=np.s_[:], scale: float=0.3):
        """
        @brief 获取多边形网格上的插值点
        
        @TODO: 边上可以取不同的插值点，lagrange, lobatto, legendre
        """
        node = self.entity('node')

        if p == 1:
            return node

        gdof = self.number_of_global_ipoints(p)

        GD = self.geo_dimension()
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NC = self.number_of_cells()
        start = 0
        ipoint = np.zeros((gdof, GD), dtype=self.ftype)
        ipoint[start:NN, :] = node

        start += NN

        edge = self.entity('edge')
        qf = self.integrator(p+1, etype='edge', qtype='lobatto')
        bcs = qf.quadpts[1:-1, :]
        ipoint[start:NN+(p-1)*NE, :] = np.einsum('ij, ...jm->...im', bcs, node[edge, :]).reshape(-1, GD)
        start += (p-1)*NE

        if p == 2:
            ipoint[start:] = self.entity_barycenter('cell')
            return ipoint

        h = np.sqrt(self.cell_area())[:, None]*scale
        bc = self.entity_barycenter('cell')
        t = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3)/2]], dtype=self.ftype)
        t -= np.array([0.5, np.sqrt(3)/6.0], dtype=self.ftype)

        tri = np.zeros((NC, 3, GD), dtype=self.ftype)
        tri[:, 0, :] = bc + t[0]*h
        tri[:, 1, :] = bc + t[1]*h
        tri[:, 2, :] = bc + t[2]*h

        bcs = self.multi_index_matrix(p-2, 2)/(p-2)
        ipoint[start:] = np.einsum('ij, ...jm->...im', bcs, tri).reshape(-1, GD)
        return ipoint

    def shape_function(self, bc: NDArray, p: int) -> NDArray:
        raise NotImplementedError

    def grad_shape_function(self, bc: NDArray, p: int, index=np.s_[:]) -> NDArray:
        raise NotImplementedError

    def uniform_refine(self, n: int=1) -> None:
        raise NotImplementedError

    def integral(self, u, q=3, celltype=False):
        """
        @brief 多边形网格上的数值积分

        @param[in] u 被积函数, 需要两个参数 (x, index)
        @param[in] q 积分公式编号
        """
        node = self.entity('node')
        edge = self.entity('edge')
        edge2cell = self.ds.edge_to_cell()
        NC = self.number_of_cells()

        bcs, ws = self.integrator(q).get_quadrature_points_and_weights()

        bc = self.entity_barycenter('cell')
        tri = [bc[edge2cell[:, 0]], node[edge[:, 0]], node[edge[:, 1]]]

        v1 = node[edge[:, 0]] - bc[edge2cell[:, 0]]
        v2 = node[edge[:, 1]] - bc[edge2cell[:, 0]]
        a = np.cross(v1, v2)/2.0

        pp = np.einsum('ij, jkm->ikm', bcs, tri, optimize=True)
        val = u(pp, edge2cell[:, 0])

        shape = (NC, ) + val.shape[2:]
        e = np.zeros(shape, dtype=np.float64)

        ee = np.einsum('i, ij..., j->j...', ws, val, a, optimize=True)
        np.add.at(e, edge2cell[:, 0], ee)

        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        if np.sum(isInEdge) > 0:
            tri = [
                    bc[edge2cell[isInEdge, 1]],
                    node[edge[isInEdge, 1]],
                    node[edge[isInEdge, 0]]
                    ]
            v1 = node[edge[isInEdge, 1]] - bc[edge2cell[isInEdge, 1]]
            v2 = node[edge[isInEdge, 0]] - bc[edge2cell[isInEdge, 1]]
            a = np.cross(v1, v2)/2.0

            pp = np.einsum('ij, jkm->ikm', bcs, tri, optimize=True)
            val = u(pp, edge2cell[isInEdge, 1])
            ee = np.einsum('i, ij..., j->j...', ws, val, a, optimize=True)
            np.add.at(e, edge2cell[isInEdge, 1], ee)

        if celltype is True:
            return e
        else:
            return e.sum(axis=0)

    def error(self, u, v, q=3, celltype=False, power=2):
        """
        @brief 在当前多边形网格上计算误差 \int |u - v|^power dx

        @param[in] u 函数
        @param[in] v 函数
        @param[in] q 积分公式编号
        """

        nu = len(inspect.signature(u).parameters)
        nv = len(inspect.signature(v).parameters)

        assert 1 <= nu <= 2
        assert 1 <= nv <= 2

        if (nu == 1) and (nv == 2):
            def efun(x, index):
                return np.abs(u(x) - v(x, index))**power
        elif (nu == 2) and (nv == 2):
            def efun(x, index):
                return np.abs(u(x, index) - v(x, index))**power
        elif (nu == 1) and (nv == 1):
            def efun(x, index):
                return np.abs(u(x) - v(x))**power
        else:
            def efun(x, index):
                return np.abs(u(x, index) - v(x))**power

        e = self.integral(efun, q, celltype=celltype)
        if isinstance(e, np.ndarray):
            n = len(e.shape) - 1
            if n > 0:
                for i in range(n):
                    e = e.sum(axis=-1)
        if celltype == False:
            e = np.power(np.sum(e), 1/power)
        else:
            e = np.power(np.sum(e, axis=tuple(range(1, len(e.shape)))), 1/power)
        return e

    @classmethod
    def show_cvem_dofs(cls, p=5):
        """
        @brief 显示多边形单元上的自由度
        """
        import matplotlib.pyplot as plt
        mesh = cls.from_one_pentagon()

        fig = plt.figure()
        for i in range(p):
            axes = fig.add_subplot(1, p, i+1)
            mesh.add_plot(axes, cellcolor=[0.5, 0.9, 0.45], edgecolor='k')
            ips = mesh.interpolation_points(p=i+1)
            mesh.find_node(axes, node=ips)
            axes.set_title(f'$k={i+1}$')

        plt.show()

    @classmethod
    def from_unit_square(cls, nx=10, ny=10, threshold=None):
        """
        Generate a polygon mesh for a unit square.

        @param nx Number of divisions along the x-axis (default: 10)
        @param ny Number of divisions along the y-axis (default: 10)
        @param threshold Optional function to filter cells based on their barycenter coordinates (default: None)
        @return TriangleMesh instance
        """
        return cls.from_box(box=[0, 1, 0, 1], nx=nx, ny=ny, threshold=threshold)

    @classmethod
    def from_box(cls, box=[0, 1, 0, 1], nx=10, ny=10, threshold=None):
        """
        @brief Generate a polygon mesh for a box domain
        """
        from .triangle_mesh import TriangleMesh
        tmesh = TriangleMesh.from_box(box, nx=nx, ny=ny, threshold=threshold)
        mesh = cls.from_triangle_mesh_by_dual(tmesh)
        return mesh

    @classmethod
    def from_quadtree(cls, qtree):
        """
        @brief 从四叉树生成多边形网格
        """
        isRootCell = qtree.is_root_cell()

        if np.all(isRootCell):
            NC = qtree.number_of_cells()

            node = qtree.entity('node')
            cell = qtree.entity('cell')

            pcell = cell.reshape(-1)
            pcellLocation = np.arange(0, 4*(NC+1), 4)

            return cls(node, pcell, pcellLocation)
        else:
            NN = qtree.number_of_nodes()
            NE = qtree.number_of_edges()
            NC = qtree.number_of_cells()

            cell = self.entity('cell')
            edge = self.entity('edge')
            edge2cell = self.ds.edge_to_cell()
            cell2cell = self.ds.cell_to_cell()
            cell2edge = self.ds.cell_to_edge()

            parent = qtree.parent
            child = qtree.child

            isLeafCell = qtree.is_leaf_cell()
            isLeafEdge = isLeafCell[edge2cell[:, 0]] & isLeafCell[edge2cell[:, 1]]

            pedge2cell = edge2cell[isLeafEdge, :]
            pedge = edge[isLeafEdge, :]

            isRootCell = qtree.is_root_cell()
            isLevelBdEdge =  (pedge2cell[:, 0] == pedge2cell[:, 1])

            # Find the index of all boundary edges on each tree level
            pedgeIdx, = np.nonzero(isLevelBdEdge)
            while len(pedgeIdx) > 0:
                cellIdx = pedge2cell[pedgeIdx, 1]
                localIdx = pedge2cell[pedgeIdx, 3]

                parentCellIdx = parent[cellIdx, 0]

                neighborCellIdx = cell2cell[parentCellIdx, localIdx]

                isFound = isLeafCell[neighborCellIdx] | isRootCell[neighborCellIdx]
                pedge2cell[pedgeIdx[isFound], 1] = neighborCellIdx[isFound]

                edgeIdx = cell2edge[parentCellIdx, localIdx]

                isCase = (edge2cell[edgeIdx, 0] != parentCellIdx) & isFound
                pedge2cell[pedgeIdx[isCase], 3] = edge2cell[edgeIdx[isCase], 2]

                isCase = (edge2cell[edgeIdx, 0] == parentCellIdx) & isFound
                pedge2cell[pedgeIdx[isCase], 3] = edge2cell[edgeIdx[isCase], 3]

                isSpecial = isFound & (parentCellIdx == neighborCellIdx)
                pedge2cell[pedgeIdx[isSpecial], 1] =  pedge2cell[pedgeIdx[isSpecial], 0]
                pedge2cell[pedgeIdx[isSpecial], 3] =  pedge2cell[pedgeIdx[isSpecial], 2]

                pedgeIdx = pedgeIdx[~isFound]
                pedge2cell[pedgeIdx, 1] = parentCellIdx[~isFound]


            PNC = isLeafCell.sum()
            cellIdxMap = np.zeros(NC, dtype=qtree.itype)
            cellIdxMap[isLeafCell] = np.arange(PNC)
            cellIdxInvMap, = np.nonzero(isLeafCell)

            pedge2cell[:, 0:2] = cellIdxMap[pedge2cell[:, 0:2]]

            # 计算每个叶子四边形单元的每条边上有几条叶子边
            # 因为叶子单元的边不一定是叶子边
            isInPEdge = (pedge2cell[:, 0] != pedge2cell[:, 1])
            cornerLocation = np.zeros((PNC, 5), dtype=qtree.itype)
            np.add.at(cornerLocation.ravel(), 5*pedge2cell[:, 0] + pedge2cell[:, 2] + 1, 1)
            np.add.at(cornerLocation.ravel(), 5*pedge2cell[isInPEdge, 1] + pedge2cell[isInPEdge, 3] + 1, 1)
            cornerLocation = cornerLocation.cumsum(axis=1)


            pcellLocation = np.zeros(PNC+1, dtype=qtree.itype)
            pcellLocation[1:] = cornerLocation[:, 4].cumsum()
            pcell = np.zeros(pcellLocation[-1], dtype=qtree.itype)
            cornerLocation += pcellLocation[:-1].reshape(-1, 1)
            pcell[cornerLocation[:, 0:-1]] = cell[isLeafCell, :]

            PNE = pedge.shape[0]
            val = np.ones(PNE, dtype=np.bool_)
            p2pe = coo_matrix(
                    (val, (pedge[:,0], range(PNE))),
                    shape=(NN, PNE), dtype=np.bool_)
            p2pe += coo_matrix(
                    (val, (pedge[:,1], range(PNE))),
                    shape=(NN, PNE), dtype=np.bool_)
            p2pe = p2pe.tocsr()
            NES = np.asarray(p2pe.sum(axis=1)).reshape(-1)
            isPast = np.zeros(PNE, dtype=np.bool_)
            for i in range(4):
                currentIdx = cornerLocation[:, i]
                endIdx = cornerLocation[:, i+1]
                cellIdx = np.arange(PNC)
                while True:
                    isNotOK = ((currentIdx + 1) < endIdx)
                    currentIdx = currentIdx[isNotOK]
                    endIdx = endIdx[isNotOK]
                    cellIdx = cellIdx[isNotOK]
                    if len(currentIdx) == 0:
                        break
                    nodeIdx = pcell[currentIdx]
                    _, J = p2pe[nodeIdx].nonzero()
                    isEdge = (pedge2cell[J, 1] == np.repeat(cellIdx, NES[nodeIdx])) \
                            & (pedge2cell[J, 3] == i) & (~isPast[J])
                    isPast[J[isEdge]] = True
                    pcell[currentIdx + 1] = pedge[J[isEdge], 0]
                    currentIdx += 1

            return cls(node,  pcell, pcellLocation)

    @classmethod
    def from_one_triangle(cls, meshtype='iso'):
        if meshtype == 'equ':
            node = np.array([
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.5, np.sqrt(3)/2]], dtype=np.float64)
        elif meshtype =='iso':
            node = np.array([
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0]], dtype=np.float64)
        cell = np.array([[0, 1, 2]],dtype=np.int64)
        return cls(node, cell)

    @classmethod
    def from_one_square(cls):
        node = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]],dtype=np.float64)
        cell = np.array([[0, 1, 2, 3]], dtype=np.int64)
        return cls(node, cell)

    @classmethod
    def from_triangle_mesh_by_dual(cls, mesh, bc=True):
        """
        @brief 生成三角形网格的对偶网格，目前默认用三角形的重心做为对偶网格的顶点

        @param mesh
        @param bc bool 如果为真，则对偶网格点为三角形单元重心; 否则为三角形单元外心
        """
        from .triangle_mesh import TriangleMeshWithInfinityNode

        mesh = TriangleMeshWithInfinityNode(mesh, bc=bc)
        pnode, pcell, pcellLocation = mesh.to_polygonmesh()
        return cls(pnode, pcell, pcellLocation)

    @classmethod
    def from_one_pentagon(cls):
        pi = np.pi
        node = np.array([
            (0.0, 0.0),
            (np.cos(2/5*pi), -np.sin(2/5*pi)),
            (np.cos(2/5*pi)+1, -np.sin(2/5*pi)),
            ( 2*np.cos(1/5*pi), 0.0),
            (np.cos(1/5*pi), np.sin(1/5*pi))],dtype=np.float64)
        cell = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        cellLocation = np.array([0, 5], dtype=np.int64)
        return cls(node, cell, cellLocation)

    @classmethod
    def from_one_hexagon(cls):
        node = np.array([
            [0.0, 0.0],
            [1/2, -np.sqrt(3)/2],
            [3/2, -np.sqrt(3)/2],
            [2.0, 0.0],
            [3/2, np.sqrt(3)/2],
            [1/2, np.sqrt(3)/2]], dtype=np.float64)
        cell = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
        cellLocation = np.array([0, 6], dtype=np.int64)
        return cls(node, cell ,cellLocation)

    @classmethod
    def from_mixed_polygon(cls):
        """
        @brief 生成一个包含多种类型多边形的网格，方便测试相关的程序
        """
        pass

    @classmethod
    def from_mesh(cls, mesh: Mesh):
        """
        @brief 把一个由同一类型单元组成网格转化为多边形网格的格式
        """
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        NC = mesh.number_of_cells()
        NV = cell.shape[1]
        cellLocation = np.arange(0, (NC+1)*NV, NV)
        return cls(node, cell.reshape(-1), cellLocation)

    @classmethod
    def distorted_concave_rhombic_quadrilaterals_mesh(cls, box=[0, 1, 0, 1], nx=10, ny=10, ratio=0.618):
        """
        @brief 虚单元网格，矩形内部包含一个菱形，两者共用左下和右上的节点

        @param box 网格所占区域
        @param nx 沿 x 轴方向剖分段数
        @param ny 沿 y 轴方向剖分段数
        @param ratio 矩形内部菱形的大小比例
        """
        from .quadrangle_mesh import QuadrangleMesh
        from .uniform_mesh_2d import UniformMesh2d

        hx = (box[1] - box[0]) / nx
        hy = (box[3] - box[2]) / ny

        mesh0 = UniformMesh2d([0, nx, 0, ny], h=(hx, hy), origin=(box[0], box[2]))
        node0 = mesh0.entity("node")
        cell0 = mesh0.entity("cell")[:, [0, 2, 3, 1]]
        mesh = QuadrangleMesh(node0, cell0)

        edge = mesh.entity("edge")
        node = mesh.entity("node")
        cell = mesh.entity("cell")
        NC = mesh.number_of_cells()
        NN = mesh.number_of_nodes()

        node_append1 = node[cell[:, 3]] * (1-ratio) + node[cell[:, 1]] * ratio
        node_append2 = node[cell[:, 3]] * ratio + node[cell[:, 1]] * (1-ratio)
        new_node = np.vstack((node, node_append1, node_append2))

        cell = np.tile(cell, (3, 1))
        idx1 = np.arange(NN, NN + NC)
        idx2 = np.arange(NN + NC, NN + 2 * NC)
        cell[0:NC, 3] = idx1
        cell[NC:2 * NC, 1] = idx1
        cell[NC:2 * NC, 3] = idx2
        cell[2 * NC:3 * NC, 1] = idx2

        return cls(new_node, cell)


    @classmethod
    def nonconvex_octagonal_mesh(cls, box=[0, 1, 0, 1], nx=10, ny=10):
        """
        @brief 虚单元网格，矩形网格的每条内部边上加一个点后形成的八边形网格

        @param box 网格所占区域
        @param nx 沿 x 轴方向剖分段数
        @param ny 沿 y 轴方向剖分段数
        """
        from .quadrangle_mesh import QuadrangleMesh
        from .uniform_mesh_2d import UniformMesh2d

        hx = (box[1] - box[0]) / nx
        hy = (box[3] - box[2]) / ny
        NN = (nx + 1) * (ny + 1)

        mesh0 = UniformMesh2d([0, nx, 0, ny], h=(hx, hy), origin=(box[0], box[2]))
        node0 = mesh0.entity("node")
        cell0 = mesh0.entity("cell")[:, [0, 2, 3, 1]]
        mesh = QuadrangleMesh(node0, cell0)

        edge = mesh.entity("edge")
        node = mesh.entity("node")
        cell = mesh.entity("cell")
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        cell2edge = mesh.ds.cell_to_edge()
        isbdedge = mesh.ds.boundary_edge_flag()
        isbdcell = mesh.ds.boundary_cell_flag()

        nie = np.sum(~isbdedge)
        hx = 1 / nx
        hy = 1 / ny
        newnode = np.zeros((NN + nie, 2), dtype=np.float_)
        newnode[:NN] = node
        newnode[NN:] = 0.5 * node[edge[~isbdedge, 0]] + 0.5 * node[edge[~isbdedge, 1]]
        newnode[NN: NN + (nx - 1) * ny] = newnode[NN: NN + (nx - 1) * ny] + np.array([[0.2 * hx, 0.1 * hy]])
        newnode[NN + (nx - 1) * ny:] = newnode[NN + (nx - 1) * ny:] + np.array([[0.1 * hx, 0.2 * hy]])

        edge2newnode = -np.ones(NE, dtype=np.int_)
        edge2newnode[~isbdedge] = np.arange(NN, newnode.shape[0])
        newcell = np.zeros((NC, 8), dtype=np.int_)
        newcell[:, ::2] = cell
        newcell[:, 1::2] = edge2newnode[cell2edge]

        flag = newcell > -1
        num = np.zeros(NC + 1, dtype=np.int_)
        num[1:] = np.sum(flag, axis=-1)
        newcell = newcell[flag]
        cellLocation = np.cumsum(num)

        return cls(newnode, newcell, cellLocation)

    @classmethod
    def hybrid_polygon_mesh(cls, box=[0, 1, 0, 1], nx=4, ny=4, ratio=0.9):
        """
        @brief  虚单元网格，混合多边形

        @param box 网格所占区域
        @param nx 沿 x 轴方向剖分段数
        @param ny 沿 y 轴方向剖分段数
        """
        from fealpy.mesh.quadrangle_mesh import QuadrangleMesh
        from fealpy.mesh.uniform_mesh_2d import UniformMesh2d

        hx = (box[1] - box[0]) / nx
        hy = (box[3] - box[2]) / ny
        NN = (nx + 1) * (ny + 1)

        mesh0 = UniformMesh2d([0, nx, 0, ny], h=(hx, hy), origin=(box[0], box[2]))
        node0 = mesh0.entity("node")
        cell0 = mesh0.entity("cell")[:, [0, 2, 3, 1]]
        mesh = QuadrangleMesh(node0, cell0)

        edge = mesh.entity("edge")
        node = mesh.entity("node")
        cell = mesh.entity("cell")
        NE = mesh.number_of_edges()
        NC = mesh.number_of_cells()

        cell2edge = mesh.ds.cell_to_edge()
        isbdedge = mesh.ds.boundary_edge_flag()
        isbdcell = mesh.ds.boundary_cell_flag()

        new_num_edge = NE + 8 * NC
        hx = 1 / nx
        hy = 1 / ny
        newnode = np.zeros((NN + new_num_edge, 2), dtype=np.float_)
        newnode[:NN] = node
        newnode[NN:NN + NE] = 0.5 * node[edge[:, 0]] + 0.5 * node[edge[:, 1]]

        newnode[NN + NE:NN + NE + NC] = 0.75 * node[cell[:, 0]] + 0.25 * node[cell[:, 2]]
        newnode[NN + NE + NC:NN + NE + 2 * NC] = 0.25 * node[cell[:, 0]] + 0.75 * node[cell[:, 2]]
        newnode[NN + NE + 2 * NC:NN + NE + 3 * NC] = 0.75 * node[cell[:, 1]] + 0.25 * node[cell[:, 3]]
        newnode[NN + NE + 3 * NC:NN + NE + 4 * NC] = 0.25 * node[cell[:, 1]] + 0.75 * node[cell[:, 3]]

        newnode[NN + NE + 4 * NC:NN + NE + 5 * NC] = 0.5 * node[cell[:, 0]] + 0.5 * newnode[NN + NE:NN + NE + NC]
        newnode[NN + NE + 5 * NC:NN + NE + 6 * NC] = 0.5 * node[cell[:, 1]] + 0.5 * newnode[
                                                                                    NN + NE + 2 * NC:NN + NE + 3 * NC]
        newnode[NN + NE + 6 * NC:NN + NE + 7 * NC] = (1 - ratio) * (
                    0.5 * node[cell[:, 0]] + 0.5 * node[cell[:, 3]]) + ratio * newnode[
                                                                               NN + NE + 3 * NC:NN + NE + 4 * NC]
        newnode[NN + NE + 7 * NC:NN + NE + 8 * NC] = (1 - ratio) * (
                    0.5 * node[cell[:, 1]] + 0.5 * node[cell[:, 2]]) + ratio * newnode[NN + NE + NC:NN + NE + 2 * NC]

        edge2newnode = np.arange(NN, NN + NE)
        newcell = np.zeros((NC, 42), dtype=np.int_)
        # newcell[:, ::2] = cell
        # newcell[:, 1::2] = edge2newnode[cell2edge]
        newcell[:, [0, 24]] = cell[:, 0][:, np.newaxis]
        newcell[:, [4, 6]] = cell[:, 1][:, np.newaxis]
        newcell[:, [11, 13]] = cell[:, 2][:, np.newaxis]
        newcell[:, [15, 20]] = cell[:, 3][:, np.newaxis]

        newcell[:, [1, 3, 29]] = edge2newnode[cell2edge[:, 0]][:, np.newaxis]
        newcell[:, [7, 10, 33]] = edge2newnode[cell2edge[:, 1]][:, np.newaxis]
        newcell[:, [14]] = edge2newnode[cell2edge[:, 2]][:, np.newaxis]
        newcell[:, [21, 23, 36]] = edge2newnode[cell2edge[:, 3]][:, np.newaxis]

        newcell[:, [26, 27, 37]] = np.arange(NN + NE, NN + NE + NC)[:, np.newaxis]
        newcell[:, [18, 40]] = np.arange(NN + NE + NC, NN + NE + 2 * NC)[:, np.newaxis]
        newcell[:, [8, 31, 32]] = np.arange(NN + NE + 2 * NC, NN + NE + 3 * NC)[:, np.newaxis]
        newcell[:, [17, 41]] = np.arange(NN + NE + 3 * NC, NN + NE + 4 * NC)[:, np.newaxis]
        newcell[:, [2, 25, 28]] = np.arange(NN + NE + 4 * NC, NN + NE + 5 * NC)[:, np.newaxis]
        newcell[:, [5, 9, 30]] = np.arange(NN + NE + 5 * NC, NN + NE + 6 * NC)[:, np.newaxis]
        newcell[:, [16, 22, 35, 38]] = np.arange(NN + NE + 6 * NC, NN + NE + 7 * NC)[:, np.newaxis]
        newcell[:, [12, 19, 34, 39]] = np.arange(NN + NE + 7 * NC, NN + NE + 8 * NC)[:, np.newaxis]

        # local_location = np.array([3, 6, 10, 13, 20, 23, 27, 32, 38, 42])
        local_location = np.array([3, 3, 4, 3, 7, 3, 4, 5, 6, 4])
        temp_location = np.tile(local_location, NC)
        num = np.zeros(temp_location.shape[0] + 1, dtype=np.int_)
        num[1:] = temp_location
        cellLocation = np.cumsum(num)

        return cls(newnode, newcell.reshape(-1), cellLocation)

    def cell_area(self, index=None):
        #TODO: 3D Case
        NC = self.number_of_cells()
        node = self.node
        edge = self.ds.edge
        edge2cell = self.ds.edge2cell
        isInEdge = (edge2cell[:, 0] != edge2cell[:, 1])
        w = np.array([[0, -1], [1, 0]], dtype=self.itype)
        v= (node[edge[:, 1], :] - node[edge[:, 0], :])@w
        val = np.sum(v*node[edge[:, 0], :], axis=1)
        a = np.bincount(edge2cell[:, 0], weights=val, minlength=NC)
        a+= np.bincount(edge2cell[isInEdge, 1], weights=-val[isInEdge], minlength=NC)
        a /=2
        return a


    @classmethod
    def interfacemesh_generator(cls, box, nx, ny, phi):
        """
        @brief

        @param
        @param
        @param
        """
        from scipy.spatial import Delaunay
        from fealpy.mesh.uniform_mesh_2d import UniformMesh2d

        hx = (box[1] - box[0]) / nx
        hy = (box[3] - box[2]) / ny

        mesh = UniformMesh2d((0, nx, 0, ny), ((box[1] - box[0]) / nx, (box[3] - box[2]) / ny), (box[0], box[2]))

        interfaceNode, isInterfaceNode, isInterfaceCell, ncut, naux = mesh.find_interface_node(phi)

        N = mesh.number_of_nodes()
        cell = mesh.entity('cell')[:, [0, 2, 3, 1]]
        node = mesh.entity('node')

        dt = Delaunay(interfaceNode)
        tri = dt.simplices
        NI = np.sum(isInterfaceNode)
        isUnnecessaryCell = (np.sum(tri < NI, axis=1) == 3)
        tri = tri[~isUnnecessaryCell, :]

        interfaceNodeIdx = np.zeros(interfaceNode.shape[0], dtype=np.int)
        interfaceNodeIdx[:NI], = np.nonzero(isInterfaceNode)
        interfaceNodeIdx[NI:NI + ncut] = N + np.arange(ncut)
        interfaceNodeIdx[NI + ncut:] = N + ncut + np.arange(naux)
        tri = interfaceNodeIdx[tri]

        NS = np.sum(~isInterfaceCell)
        NT = tri.shape[0]
        pnode = np.concatenate((node, interfaceNode[NI:]), axis=0)
        pcell = np.zeros(NS * 4 + NT * 3, dtype=np.int)
        pcellLocation = np.zeros(NS + NT + 1, dtype=np.int)

        sview = pcell[:4 * NS].reshape(NS, 4)
        sview[:] = cell[~isInterfaceCell, :]

        tview = pcell[4 * NS:].reshape(NT, 3)
        tview[:] = tri
        pcellLocation[:NS] = range(0, 4 * NS, 4)
        pcellLocation[NS:-1] = range(4 * NS, 4 * NS + 3 * NT, 3)
        pcellLocation[-1] = 4 * NS + 3 * NT
        pmesh = cls(pnode, pcell, pcellLocation)

        return pmesh
    def is_boundary_edge(self, threshold=None):
        isbdedge = self.ds.boundary_edge_flag()
        if threshold is not None:
            bc = self.entity_barycenter('edge')
            isbdedge = threshold(bc) 
        return isbdedge


PolygonMesh.set_ploter('polygon2d')


class PolygonMeshDataStructure(Mesh2dDataStructure):
    # Variables
    face = ArrRedirector('edge')
    edge2cell: NDArray

    # Constants
    TD: int = 2

    def __init__(self, NN: int, cell: NDArray, cellLocation: NDArray, topdata=None):
        self.NN = NN
        self._cell = cell
        self.cellLocation = cellLocation
        self.itype = cell.dtype

        if topdata is None:
            self.construct()
        else:
            self.edge = topdata[0]
            self.edge2cell = topdata[1]

    def reinit(self, NN: int, cell: NDArray, cellLocation: NDArray):
        self.NN = NN
        self._cell = cell
        self.itype = cell.dtype
        self.cellLocation = cellLocation
        self.construct()

    def number_of_cells(self) -> int:
        return self.cellLocation.shape[0] - 1

    def number_of_vertices_of_cells(self):
        return self.cellLocation[1:] - self.cellLocation[0:-1]

    number_of_edges_of_cells = number_of_vertices_of_cells
    number_of_faces_of_cells = number_of_vertices_of_cells

    def total_edge(self) -> NDArray:
        totalEdge = np.zeros((self._cell.shape[0], 2), dtype=self.itype)
        totalEdge[:, 0] = self._cell
        totalEdge[:-1, 1] = self._cell[1:]
        totalEdge[self.cellLocation[1:] - 1, 1] = self._cell[self.cellLocation[:-1]]
        return totalEdge

    total_face = total_edge

    def construct(self):
        """
        @brief 构建多边形网格实体之间的邻接关系矩阵
        """
        totalEdge = self.total_edge()
        _, i0, j = np.unique(np.sort(totalEdge, axis=1),
                return_index=True,
                return_inverse=True,
                axis=0)
        NE = i0.shape[0]
        self.edge2cell = np.zeros((NE, 4), dtype=self.itype)

        i1 = np.zeros(NE, dtype=self.itype)
        i1[j] = np.arange(len(self._cell))

        self.edge = totalEdge[i0]

        NV = self.number_of_vertices_of_cells()
        NC = self.number_of_cells()
        cellIdx = np.repeat(range(NC), NV)

        localIdx = ranges(NV)

        self.edge2cell[:, 0] = cellIdx[i0]
        self.edge2cell[:, 1] = cellIdx[i1]
        self.edge2cell[:, 2] = localIdx[i0]
        self.edge2cell[:, 3] = localIdx[i1]
        self.cell2edge = j

    @property
    def cell(self):
        return np.hsplit(self._cell, self.cellLocation[1:-1])

    ### cell ###

    def cell_to_node(self, return_sparse=False):
        """
        @brief 单元到节点的拓扑关系，默认返回稀疏矩阵
        @note 当获取单元实体时，请使用 `mesh.entity('cell')` 接口
        """

        if return_sparse:
            NN = self.number_of_nodes()
            NC = self.number_of_cells()

            NV = self.number_of_vertices_of_cells()
            I = np.repeat(range(NC), NV)
            J = self._cell

            val = np.ones(len(self._cell), dtype=np.bool_)
            cell2node = csr_matrix((val, (I, J)), shape=(NC, NN), dtype=np.bool_)
            return cell2node
        else:
            return self.cell

    def face_to_edge(self) -> NDArray:
        NE = self.number_of_edges()
        return np.arange(NE).reshape(-1, 1)

    def cell_to_edge(self, return_sparse=False) -> NDArray:
        """
        @brief 获取网格单元与网格边的邻接关系
        """
        if return_sparse:
            NE = self.number_of_edges()
            NC = self.number_of_cells()
            J = np.arange(NE)
            val = np.ones((NE,), dtype=np.bool_)
            cell2edge  = coo_matrix((val, (self.edge2cell[:, 0], J)), shape=(NC, NE), dtype=np.bool_)
            cell2edge += coo_matrix((val, (self.edge2cell[:, 1], J)), shape=(NC, NE), dtype=np.bool_)
            return cell2edge.tocsr()
        else:
            return np.hsplit(self.cell2edge, self.cellLocation[1:-1])

    cell_to_face = cell_to_edge

    def edge_to_cell(self, return_sparse=False):
        """
        @brief 获取网格边与网格单元之间的邻接关系
        """
        if return_sparse:
            NE = self.number_of_edges()
            NC = self.number_of_cells()
            val = np.ones(NE, dtype=np.bool_)
            edge2cell  = coo_matrix((val, (range(NE), self.edge2cell[:, 0])), shape=(NE, NC), dtype=np.bool_)
            edge2cell += coo_matrix((val, (range(NE), self.edge2cell[:, 1])), shape=(NE, NC), dtype=np.bool_)
            return edge2cell.tocsr()
        else:
            return self.edge2cell

    face_to_cell = edge_to_cell

    def cell_to_edge_sign(self, return_sparse=True):
        NC = self.number_of_cells()
        NE = self.number_of_edges()
        edge2cell = self.edge2cell
        cellLocation = self.cellLocation
        if return_sparse:
            val = np.ones((NE,), dtype=np.bool_)
            cell2edgeSign = csr_matrix((val, (edge2cell[:,0], range(NE))), shape=(NC,NE), dtype=np.bool_)
            return cell2edgeSign
        else:
            cell2edgeSign = np.zeros(self._cell.shape[0], dtype=self.itype)
            isInEdge = edge2cell[:, 0] != edge2cell[:, 1]
            cell2edgeSign[cellLocation[edge2cell[:, 0]] + edge2cell[:, 2]] = 1
            cell2edgeSign[cellLocation[edge2cell[isInEdge, 1]] + edge2cell[isInEdge, 3]] = -1

            return cell2edgeSign
