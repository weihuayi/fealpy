from typing import Union, Optional
from numpy.typing import NDArray
import numpy as np

from ..mesh_data_structure import MeshDataStructure


class Mesh():
    """
    @brief The base class for mesh.

    """
    ds: MeshDataStructure
    node: NDArray
    itype: np.dtype
    ftype: np.dtype

    ### General Interfaces ###

    def number_of_nodes(self) -> int:
        return self.ds.NN

    def number_of_edges(self) -> int:
        return self.ds.number_of_edges()

    def number_of_faces(self) -> int:
        return self.ds.number_of_faces()

    def number_of_cells(self) -> int:
        return self.ds.number_of_cells()

    def number_of_nodes_of_cells(self) -> int:
        """Number of nodes in a cell"""
        return self.ds.number_of_vertices_of_cells()

    def number_of_edges_of_cells(self) -> int:
        """Number of edges in a cell"""
        return self.ds.number_of_edges_of_cells()

    def number_of_faces_of_cells(self) -> int:
        """Number of faces in a cell"""
        return self.ds.number_of_faces_of_cells()

    number_of_vertices_of_cells = number_of_nodes_of_cells

    def geo_dimension(self) -> int:
        """
        @brief Get geometry dimension of the mesh.
        """
        return self.node.shape[-1]

    def top_dimension(self) -> int:
        """
        @brief Get topology dimension of the mesh.
        """
        return self.ds.TD

    @staticmethod
    def multi_index_matrix(p: int, etype: int) -> NDArray:
        """
        @brief 获取 p 次的多重指标矩阵

        @param[in] p 正整数

        @return multiIndex  ndarray with shape (ldof, TD+1)
        """
        if etype == 3:
            ldof = (p+1)*(p+2)*(p+3)//6
            idx = np.arange(1, ldof)
            idx0 = (3*idx + np.sqrt(81*idx*idx - 1/3)/3)**(1/3)
            idx0 = np.floor(idx0 + 1/idx0/3 - 1 + 1e-4) # a+b+c
            idx1 = idx - idx0*(idx0 + 1)*(idx0 + 2)/6
            idx2 = np.floor((-1 + np.sqrt(1 + 8*idx1))/2) # b+c
            multiIndex = np.zeros((ldof, 4), dtype=np.int_)
            multiIndex[1:, 3] = idx1 - idx2*(idx2 + 1)/2
            multiIndex[1:, 2] = idx2 - multiIndex[1:, 3]
            multiIndex[1:, 1] = idx0 - idx2
            multiIndex[:, 0] = p - np.sum(multiIndex[:, 1:], axis=1)
            return multiIndex
        elif etype == 2:
            ldof = (p+1)*(p+2)//2
            idx = np.arange(0, ldof)
            idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
            multiIndex = np.zeros((ldof, 3), dtype=np.int_)
            multiIndex[:,2] = idx - idx0*(idx0 + 1)/2
            multiIndex[:,1] = idx0 - multiIndex[:,2]
            multiIndex[:,0] = p - multiIndex[:, 1] - multiIndex[:, 2]
            return multiIndex
        elif etype == 1:
            ldof = p+1
            multiIndex = np.zeros((ldof, 2), dtype=np.int_)
            multiIndex[:, 0] = np.arange(p, -1, -1)
            multiIndex[:, 1] = p - multiIndex[:, 0]
            return multiIndex

    def _shape_function(self, bc: NDArray, p: int =1, mi: NDArray=None):
        """
        @brief

        @param[in] bc
        """
        if p == 1:
            return bc
        TD = bc.shape[-1] - 1
        if mi is None:
            mi = self.multi_index_matrix(p, etype=TD)
        c = np.arange(1, p+1, dtype=np.int_)
        P = 1.0/np.multiply.accumulate(c)
        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=self.ftype)
        A[..., 1:, :] = p*bc[..., None, :] - t.reshape(-1, 1)
        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)
        idx = np.arange(TD+1)
        phi = np.prod(A[..., mi, idx], axis=-1)
        return phi


    def _grad_shape_function(self, bc: NDArray, p: int =1, mi: NDArray=None) -> NDArray:
        """
        @brief 计算形状为 (..., TD+1) 的重心坐标数组 bc 中, 每一个重心坐标处的 p 次 Lagrange 形函数值关于该重心坐标的梯度。
        """
        TD = bc.shape[-1] - 1
        if mi is None:
            mi = self.multi_index_matrix(p, etype=TD)
        ldof = mi.shape[0] # p 次 Lagrange 形函数的个数

        c = np.arange(1, p+1)
        P = 1.0/np.multiply.accumulate(c)

        t = np.arange(0, p)
        shape = bc.shape[:-1]+(p+1, TD+1)
        A = np.ones(shape, dtype=bc.dtype)
        A[..., 1:, :] = p*bc[..., None, :] - t.reshape(-1, 1)

        FF = np.einsum('...jk, m->...kjm', A[..., 1:, :], np.ones(p))
        FF[..., range(p), range(p)] = p
        np.cumprod(FF, axis=-2, out=FF)
        F = np.zeros(shape, dtype=bc.dtype)
        F[..., 1:, :] = np.sum(np.tril(FF), axis=-1).swapaxes(-1, -2)
        F[..., 1:, :] *= P.reshape(-1, 1)

        np.cumprod(A, axis=-2, out=A)
        A[..., 1:, :] *= P.reshape(-1, 1)

        Q = A[..., mi, range(TD+1)]
        M = F[..., mi, range(TD+1)]

        shape = bc.shape[:-1]+(ldof, TD+1)
        R = np.zeros(shape, dtype=bc.dtype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            R[..., i] = M[..., i]*np.prod(Q[..., idx], axis=-1)
        return R # (..., ldof, TD+1)

    def _tensor_shape_function(self, bc: tuple, p: int =1, mi: NDArray=None):
        """
        @brief 四边形和六面体参考单元上的形函数

        @param[in] bc
        """
        TD = len(bc)
        phi = self._shape_function(bc[0], p) # 线上的形函数
        if TD == 2: # 面上的形函数
            phi = np.einsum('im, jn->ijmn', phi, phi)
            shape = phi.shape[:-2] + (-1, )
            phi = phi.reshape(shape) # 展平自由度
            shape = (-1, 1) + phi.shape[-1:] # 增加一个单元轴，方便广播运算
            phi = phi.reshape(shape) # 展平积分点
        elif TD == 3: # 体上的形函数
            phi = np.einsum('il, jm, kn->ijklmn', phi, phi, phi)
            shape = phi.shape[:-3] + (-1, )
            phi = phi.reshape(shape) # 展平自由度
            shape = (-1, 1) + phi.shape[-1:] # 增加一个单元轴，方便广播运算
            phi = phi.reshape(shape) # 展平积分点
        return phi 

    def _grad_tensor_shape_function(self, bc: tuple, p: int=1, index=np.s_[:]):
        """

        Notes
        -----
        计算单元形函数关于参考单元变量 u=(xi, eta) 或者实际变量 x 梯度。

        bc 是一个长度为 TD 的 tuple

        bc[i] 是一个一维积分公式的重心坐标数组

        这里假设 bc[0] == bc[1] == ... = bc[TD-1]

        """
        p = self.p if p is None else p
        TD = len(bc)
        Dlambda = np.array([[-1], [1]], dtype=self.ftype)

        # 一维基函数值
        # (NQ, p+1)
        phi = self._shape_function(bc[0], p)  

        # 关于**一维变量重心坐标**的导数
        # lambda_0 = 1 - xi
        # lambda_1 = xi
        # (NQ, ldof, 2) 
        R = self._grad_shape_function(bc[0], p)  

        # 关于**一维变量**的导数
        gphi = np.einsum('...ij, jn->...in', R, Dlambda) # (..., ldof, 1)

        if TD == 2:
            gphi0 = np.einsum('imt, kn->ikmn', gphi, phi)
            gphi1 = np.einsum('kn, imt->kinm', phi, gphi)
            n = gphi0.shape[0]*gphi0.shape[1]
            shape = (n, (p+1)*(p+1), TD)
            gphi = np.zeros(shape, dtype=self.ftype)
            gphi[..., 0].flat = gphi0.flat
            gphi[..., 1].flat = gphi1.flat
        elif TD == 3:
            gphi0 = np.einsum('imt, kn->ikmn', gphi, phi)
            gphi1 = np.einsum('kn, imt->kinm', phi, gphi)
            n = gphi0.shape[0]*gphi0.shape[1]
            shape = (n, (p+1)*(p+1), TD)
            gphi = np.zeros(shape, dtype=self.ftype)
            gphi[..., 0].flat = gphi0.flat
            gphi[..., 1].flat = gphi1.flat

        return gphi[..., None, :, :] #(..., 1, ldof, TD) 增加一个单元轴

    def _bernstein_shape_function(self, bc: NDArray, p: int=1, mi: NDArray=None):
        """
        @brief
        """
        TD = bc.shape[1]-1
        if mi is None:
            mi = self.multi_index_matrix(p, TD)
        ldof = mi.shape[0]

        shape = bc.shape[:-1]+(p+1, TD+1)
        B = np.ones(shape, dtype=bc.dtype)
        B[..., 1:, :] = bc[..., None, :]
        B = np.cumprod(B, axis=1)

        P = np.arange(p+1)
        P[0] = 1
        P = np.cumprod(P)
        B /= P[:, None]

        # B : (NQ, p+1, TD+1)
        # B[:, multiIndex, np.arange(TD+1).reshape(1, -1)]: (NQ, ldof, TD+1)
        phi = P[-1]*np.prod(B[:, mi, np.arange(TD+1).reshape(1, -1)], axis=-1)

        return phi

    def _grad_bernstein_shape_function(self, bc: NDArray, p: int=1, mi:
            NDArray=None):
        """
        @brief
        """

        TD = bc.shape[1]-1
        if mi is None:
            mi = self.multi_index_matrix(p, TD)
        ldof = mi.shape[0]

        shape = bc.shape[:-1] + (p+1, TD+1)
        B = np.ones(shape, dtype=bc.dtype)
        B[..., 1:, :] = bc[..., None, :]
        B = np.cumprod(B, axis=1)

        P = np.arange(p+1)
        P[0] = 1
        P = np.cumprod(P)
        B /= P[:, None]

        F = np.zeros(B.shape, dtype=bc.dtype)
        F[:, 1:] = B[:, :-1]

        shape = bc.shape[:-1]+(ldof, TD+1)
        R = np.zeros(shape, dtype=bc.dtype)
        for i in range(TD+1):
            idx = list(range(TD+1))
            idx.remove(i)
            idx = np.array(idx, dtype=np.int_)
            R[..., i] = np.prod(B[..., mi[:, idx], idx.reshape(1, -1)],
                    axis=-1)*F[..., mi[:, i], [i]]

        return P[-1]*R

    def shape_function(self, bc, p=1) -> NDArray:
        """
        @brief The cell shape function.
        """
        raise NotImplementedError

    def grad_shape_function(self, bc, p=1, variables='x', index=np.s_[:]):
        """
        @brief The gradient of the cell shape function.
        """
        raise NotImplementedError

    def uniform_refine(self, n: int=1) -> None:
        """
        @brief Refine the whole mesh uniformly for `n` times.
        """
        raise NotImplementedError


    def integrator(self, k: int, etype: Union[int, str]):
        """
        @brief Get the integration formula on a mesh entity of different dimensions.
        """
        raise NotImplementedError

    def bc_to_point(self, bc: NDArray, index=np.s_[:]) -> NDArray:
        """
        @brief Convert barycenter coordinate points to cartesian coordinate points\
               on mesh entities.

        @param bc: Barycenter coordinate points array, with shape (NQ, NVC), where\
                   NVC is the number of nodes in each entity.
        @param etype: Specify the type of entities on which the coordinates be converted.
        @param index: Index to slice entities.

        @note: To get the correct result, the order of bc must match the order of nodes\
               in the entity.

        @return: Cartesian coordinate points array, with shape (NQ, GD).
        """
        node = self.entity('node')
        TD = bc.shape[-1] - 1
        entity = self.entity(TD, index=index)
        p = np.einsum('...j, ijk -> ...ik', bc, node[entity])
        return p

    def number_of_entities(self, etype: Union[int, str], index=np.s_[:]) -> int:
        raise NotImplementedError

    def entity(self, etype: Union[int, str], index=np.s_[:]) -> NDArray:
        """
        @brief Get entities.

        @param etype: Type of entities. Accept dimension or name.
        @param index: Index for entities.

        @return: A tensor representing the entities in this mesh.
        """
        TD = self.top_dimension()
        GD = self.geo_dimension()
        if etype in {'cell', TD}:
            return self.ds.cell[index]
        elif etype in {'edge', 1}:
            return self.ds.edge[index]
        elif etype in {'node', 0}:
            return self.node.reshape(-1, self.geo_dimension())[index]
        elif etype in {'face', TD-1}: # Try 'face' in the last
            return self.ds.face[index]
        raise ValueError(f"Invalid etype '{etype}'.")

    def entity_barycenter(self, etype: Union[int, str], index=np.s_[:]) -> NDArray:
        """
        @brief Calculate barycenters of entities.
        """
        node = self.entity('node')
        TD = self.ds.TD
        if etype in {'cell', TD}:
            cell = self.ds.cell
            return np.sum(node[cell[index], :], axis=1) / cell.shape[1]
        elif etype in {'edge', 1}:
            edge = self.ds.edge
            return np.sum(node[edge[index], :], axis=1) / edge.shape[1]
        elif etype in {'node', 0}:
            return node[index]
        elif etype in {'face', TD-1}: # Try 'face' in the last
            face = self.ds.face
            return np.sum(node[face[index], :], axis=1) / face.shape[1]
        raise ValueError(f"Invalid entity type '{etype}'.")

    def entity_measure(self, etype: Union[int, str], index=np.s_[:]) -> NDArray:
        """
        @brief Calculate measurements of entities.
        """
        raise NotImplementedError


    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell') -> int:
        """
        @brief Return the number of p-order interpolation points in a single entity.
        """
        raise NotImplementedError

    def number_of_global_ipoints(self, p: int) -> int:
        """
        @brief Return the number of all p-order interpolation points.
        """
        raise NotImplementedError

    def interpolation_points(self, p: int) -> NDArray:
        """
        @brief Get all the p-order interpolation points in the mesh.
        """
        raise NotImplementedError

    def node_to_ipoint(self, p: int, index=np.s_[:]) -> NDArray:
        return np.arange(self.number_of_nodes())[index]

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

    def edge_length(self, index=np.s_[:], node: Optional[NDArray]=None):
        """
        @brief Calculate the length of each edge.

        @param index: int, NDArray or slice.
        @param node: NDArray, optional. Use the nodes of the mesh if not provided.

        @return: An array with shape (NE, ).
        """
        node = self.entity('node') if node is None else node
        edge = self.entity('edge', index=index)
        v = node[edge[:, 1]] - node[edge[:, 0]]
        return np.linalg.norm(v, axis=1)

    def edge_tangent(self, index=np.s_[:], node: Optional[NDArray]=None):
        """
        @brief Calculate the tangent vector of each edge.

        @param index: int, NDArray or slice.
        @param node: NDArray, optional. Use the nodes of the mesh if not provided.

        @return: An array with shape (NE, GD).
        """
        node = self.entity('node') if node is None else node
        edge = self.entity('edge', index)
        v = node[edge[:, 1], :] - node[edge[:, 0], :]
        return v

    def edge_unit_tangent(self, index=np.s_[:], node: Optional[NDArray]=None):
        """
        @brief Calculate the tangent vector with unit length of each edge.\
               See `Mesh.edge_tangent`.
        """
        node = self.entity('node') if node is None else node
        edge = self.entity('edge', index=index)
        v = node[edge[:, 1], :] - node[edge[:, 0], :]
        length = np.sqrt(np.square(v).sum(axis=1))
        return v/length.reshape(-1, 1)

    def cell_normal(self, index=np.s_[:], node: Optional[NDArray]=None):
        """
        @brief 计算网格单元的外法线方向，适用于三维空间中单元拓扑维数为 2 的情况，
        比如三维空间中的三角形或四边形网格.
        """
        node = self.entity('node') if node is None else node
        cell = self.entity('cell', index=index)
        v1 = node[cell[:, 1]] - node[cell[:, 0]]
        v2 = node[cell[:, 2]] - node[cell[:, 1]]
        normal = np.cross(v1, v2)
        return normal

    def integral(self, f, q=3, celltype=False):
        """
        @brief 在网格中数值积分一个函数
        """
        GD = self.geo_dimension()
        qf = self.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = self.bc_to_point(bcs)

        if callable(f):
            if not hasattr(f, 'coordtype'):
                f = f(ps)
            else:
                if f.coordtype == 'cartesian':
                    f = f(ps)
                elif f.coordtype == 'barycentric':
                    f = f(bcs)
        cm = self.entity_measure('cell')

        if isinstance(f, (int, float)): #  u 为标量常函数
            e = f*cm
        elif isinstance(f, np.ndarray):
            if f.shape == (GD, ): # 常向量函数
                e = cm[:, None]*f
            elif f.shape == (GD, GD):
                e = cm[:, None, None]*f
            else:
                e = np.einsum('q, qc..., c->c...', ws, f, cm)
        else:
            raise ValueError(f"Unsupported type of return value: {f.__class__.__name__}.")

        if celltype:
            return e
        else:
            return np.sum(e)

    def error(self, u, v, q=3, power=2, celltype=False, integrator=None):
        """
        @brief Calculate the error between two functions.
        """
        GD = self.geo_dimension()

        qf = self.integrator(q, etype='cell') if integrator is None else integrator
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = self.bc_to_point(bcs)

        if callable(u):
            if not hasattr(u, 'coordtype'):
                u = u(ps)
            else:
                if u.coordtype == 'cartesian':
                    u = u(ps)
                elif u.coordtype == 'barycentric':
                    u = u(bcs)

        if callable(v):
            if not hasattr(v, 'coordtype'):
                v = v(ps)
            else:
                if v.coordtype == 'cartesian':
                    v = v(ps)
                elif v.coordtype == 'barycentric':
                    v = v(bcs)

        if u.shape[-1] == 1:
           u = u[..., 0]

        if v.shape[-1] == 1:
           v = v[..., 0]

        cm = self.entity_measure('cell')
        NC = self.number_of_cells()
        if v.shape[-1] == NC:
            v = np.swapaxes(v, 1, -1)
        f = np.power(np.abs(u - v), power)
        if len(f.shape) == 1: 
            f = f[:, None]

        if isinstance(f, (int, float)): # f为标量常函数
            e = f*cm
        elif isinstance(f, np.ndarray):
            if f.shape == (GD, ): # 常向量函数
                e = cm[:, None]*f
            elif f.shape == (GD, GD):
                e = cm[:, None, None]*f
            else:
                e = np.einsum('q, qc..., c->c...', ws, f, cm)

        if celltype is False:
            e = np.power(np.sum(e), 1/power)
        else:
            e = np.power(np.sum(e, axis=tuple(range(1, len(e.shape)))), 1/power)
        return e # float or (NC, )

    def paraview(self, file_name = "temp.vtu",
            background_color='1.0, 1.0, 1.0',
            show_type='Surface With Edges',
            ):
        """
        @brief 调用 ParaView 进行可视化

        @param[in] file_name str 网格子类可以设置不同的 vtk 文件后缀名
        @param[in] show_type str
        """
        import subprocess
        import os
        # 尝试找到pvpython的路径
        try:
            pvpython_path = subprocess.check_output(['which', 'pvpython']).decode().strip()
            # 确保路径不为空
            if not pvpython_path:
                raise Exception("pvpython path is empty.")
        except subprocess.CalledProcessError as e:
            print("pvpython was not found. Please make sure ParaView is installed.")
            print("On Ubuntu, you can install ParaView using the following commands:")
            print("sudo apt-get update")
            print("sudo apt-get install paraview python3-paraview")
            print("\nAdditionally, you may need to set the PYTHONPATH environment variable to include the path to the ParaView Python modules.")
            print("You can do this by adding the following line to your .bashrc file:")
            print("export PYTHONPATH=$PYTHONPATH:/usr/lib/python3/dist-packages")
            return  # 退出函数

        # 将网格数据转换为VTU文件
        fname = "/tmp/" + file_name
        self.to_vtk(fname=fname)

        # 获取当前文件的绝对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # 构建load_vtk.py的相对路径
        # 假设当前文件在 fealpy/mesh/mesh_base/mesh.py
        load_vtk_path = os.path.join(current_dir, '..', '..', 'plotter',
                'paraview_plotting.py')

        command = [
            pvpython_path, load_vtk_path, fname,
            '--show_type', show_type,
        ]

        # 移除 None 参数
        command = [str(arg) for arg in command if arg is not None]

        # 调用 pvpython 执行画图脚本，并传递参数
        subprocess.run(command)
        os.remove(fname)

    def vtkview(self, showedge=True,
            background_color=(0.3, 0.2, 0.1), 
            edge_color=(0, 0, 0),
            edge_width=1.5,
            window_size=(800, 800)):
        """
        """
        import vtk
        import vtk.util.numpy_support as vnp

        NC = self.number_of_cells()
        GD = self.geo_dimension()

        node = self.entity('node')
        if GD == 2:
            node = np.concatenate((node, np.zeros((node.shape[0], 1), dtype=self.ftype)), axis=1)

        cell = self.entity('cell')
        cellType = self.vtk_cell_type('cell')
        NV = cell.shape[-1]

        cell = np.r_['1', np.zeros((len(cell), 1), dtype=cell.dtype), cell]
        cell[:, 0] = NV
        cell = cell.astype(np.int64)

        points = vtk.vtkPoints()
        points.SetData(vnp.numpy_to_vtk(node))

        cells = vtk.vtkCellArray()
        cells.SetCells(NC, vnp.numpy_to_vtkIdTypeArray(cell))

        mesh =vtk.vtkUnstructuredGrid() 
        mesh.SetPoints(points)
        mesh.SetCells(cellType, cells)

        # 创建一个映射器和 actor
        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputData(mesh)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        if showedge:
            # 创建显示边缘的管线
            edges = vtk.vtkExtractEdges()
            edges.SetInputData(mesh)

            edgesMapper = vtk.vtkPolyDataMapper()
            edgesMapper.SetInputConnection(edges.GetOutputPort())

            edgesActor = vtk.vtkActor()
            edgesActor.SetMapper(edgesMapper)
            edgesActor.GetProperty().SetColor(edge_color)  # 黑色边缘
            edgesActor.GetProperty().SetLineWidth(edge_width)  # 设置线宽


        # 创建渲染器、渲染窗口和渲染窗口交互器
        renderer = vtk.vtkRenderer()
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.AddRenderer(renderer)
        renderWindow.SetSize(window_size)
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        # 添加 actor 到渲染器
        renderer.AddActor(actor)
        if showedge:
            renderer.AddActor(edgesActor)
        renderer.SetBackground(background_color)  # 背景颜色

        # 开始交互
        renderWindow.Render()
        renderWindowInteractor.Start()
