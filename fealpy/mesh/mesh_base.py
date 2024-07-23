
from typing import (
    Union, Optional, Dict, Sequence, overload, Callable, Tuple, Any
)

from ..typing import (
    TensorLike, Index, EntityName, _S,
    _int_func
)
from .. import logger
from ..backend import backend_manager as fealpy
from ..quadrature import Quadrature
from .utils import estr2dim, edim2entity, MeshMeta


##################################################
### Mesh Data Structure Base
##################################################
# NOTE: MeshDS provides a storage for mesh entities and all topological methods.

class MeshDS(metaclass=MeshMeta):
    _STORAGE_ATTR = ['cell', 'face', 'edge', 'node']
    cell: TensorLike
    face: TensorLike
    edge: TensorLike
    node: TensorLike
    face2cell: TensorLike
    cell2edge: TensorLike
    localEdge: TensorLike # only for homogeneous mesh
    localFace: TensorLike # only for homogeneous mesh
    localFace2Edge: TensorLike

    def __init__(self, TD: int) -> None:
        assert hasattr(self, '_entity_dim_method_name_map')
        self._entity_storage: Dict[int, TensorLike] = {}
        self._entity_factory: Dict[int, Callable] = {
            k: getattr(self, self._entity_dim_method_name_map[k])
            for k in self._entity_dim_method_name_map
        }
        self.TD = TD

    @overload
    def __getattr__(self, name: EntityName) -> TensorLike: ...
    def __getattr__(self, name: str):
        if name in self._STORAGE_ATTR:
            etype_dim = estr2dim(self, name)
            return edim2entity(self._entity_storage, self._entity_factory, etype_dim)
        else:
            return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self._STORAGE_ATTR:
            if not hasattr(self, '_entity_storage'):
                raise RuntimeError('please call super().__init__() before setting attributes.')
            etype_dim = estr2dim(self, name)
            self._entity_storage[etype_dim] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        if name in self._STORAGE_ATTR:
            del self._entity_storage[estr2dim(self, name)]
        else:
            super().__delattr__(name)

    ### cuda
    # def to(self, device: Union[_device, str, None]=None, non_blocking=False):
    #     for edim in self._entity_storage.keys():
    #         entity = self._entity_storage[edim]
    #         self._entity_storage[edim] = entity.to(device, non_blocking=non_blocking)
    #     for attr in self.__dict__:
    #         value = self.__dict__[attr]
    #         if isinstance(value, Tensor):
    #             self.__dict__[attr] = value.to(device, non_blocking=non_blocking)
    #     return self

    ### properties
    def top_dimension(self) -> int: return self.TD
    @property
    def itype(self) -> Any: return self.cell.dtype
    @property
    def device(self) -> Any: return self.cell.device
    def storage(self) -> Dict[int, TensorLike]:
        return self._entity_storage

    ### counters
    def count(self, etype: Union[int, str]) -> int:
        """Return the number of entities of the given type."""
        entity = self.entity(etype)

        if entity is None:
            logger.info(f'count: entity {etype} is not found and 0 is returned.')
            return 0

        if hasattr(entity, 'location'):
            return entity.location.size(0) - 1
        else:
            return entity.size(0)

    def number_of_nodes(self): return self.count('node')
    def number_of_edges(self): return self.count('edge')
    def number_of_faces(self): return self.count('face')
    def number_of_cells(self): return self.count('cell')

    def _nv_entity(self, etype: Union[int, str]) -> TensorLike:
        entity = self.entity(etype)
        if hasattr(entity, 'location'):
            loc = entity.location
            return loc[1:] - loc[:-1]
        else:
            return fealpy.tensor((entity.shape[-1],), dtype=self.itype)

    def number_of_vertices_of_cells(self): return self._nv_entity('cell')
    def number_of_vertices_of_faces(self): return self._nv_entity('face')
    def number_of_vertices_of_edges(self): return self._nv_entity('edge')
    number_of_nodes_of_cells = number_of_vertices_of_cells
    number_of_edges_of_cells: _int_func = lambda self: self.localEdge.shape[0]
    number_of_faces_of_cells: _int_func = lambda self: self.localFace.shape[0]

    def entity(self, etype: Union[int, str], index: Optional[Index]=None) -> TensorLike:
        """Get entities in mesh structure.

        Parameters:
            index (int | slice | Tensor): The index of the entity.\n
            etype (int | str): The topological dimension of the entity, or name\
            'cell' | 'face' | 'edge' | 'node'.\n
            index (int | slice | Tensor): The index of the entity.

        Returns:
            Tensor: Entity or the default value. Returns None if not found.
        """
        if isinstance(etype, str):
            etype = estr2dim(self, etype)
        return edim2entity(self.storage(), self._entity_factory, etype, index)

    ### topology
    def cell_to_node(self, index: Optional[Index]=None) -> TensorLike:
        return self.entity('cell', index)

    def face_to_node(self, index: Optional[Index]=None) -> TensorLike:
        return self.entity('face', index)

    def edge_to_node(self, index: Optional[Index]=None) -> TensorLike:
        return self.entity('edge', index)

    def cell_to_edge(self, index: Index=_S) -> TensorLike:
        if not hasattr(self, 'cell2edge'):
            raise RuntimeError('Please call construct() first or make sure the cell2edge'
                               'has been constructed.')
        return self.cell2edge[index]

    def face_to_edge(self, index: Index=_S):
        assert self.TD == 3
        cell2edge = self.cell2edge
        face2cell = self.face2cell
        localFace2edge = self.localFace2edge
        face2edge = cell2edge[
            face2cell[:, [0]],
            localFace2edge[face2cell[:, 2]]
        ]
        return face2edge[index]

    def cell_to_face(self, index: Index=_S) -> TensorLike:
        NC = self.number_of_cells()
        NF = self.number_of_faces()
        NFC = self.number_of_faces_of_cells()

        face2cell = self.face2cell
        dtype = self.itype

        cell2face = fealpy.zeros((NC, NFC), dtype=dtype)
        arange_tensor = fealpy.arange(0, NF, dtype=dtype)

        assert cell2face.dtype == arange_tensor.dtype, f"Data type mismatch: cell2face is {cell2face.dtype}, arange_tensor is {arange_tensor.dtype}"

        cell2face[face2cell[:, 0], face2cell[:, 2]] = arange_tensor
        cell2face[face2cell[:, 1], face2cell[:, 3]] = arange_tensor
        return cell2face[index]

    def face_to_cell(self, index: Index=_S) -> TensorLike:
        if not hasattr(self, 'face2cell'):
            raise RuntimeError('Please call construct() first or make sure the face2cell'
                               'has been constructed.')
        face2cell = self.face2cell[index]
        return face2cell

    ### boundary
    def boundary_node_flag(self) -> TensorLike:
        """Return a boolean tensor indicating the boundary nodes.

        Returns:
            Tensor: boundary node flag.
        """
        NN = self.number_of_nodes()
        bd_face_flag = self.boundary_face_flag()
        kwargs = {'dtype': bd_face_flag.dtype, 'device': bd_face_flag.device}
        bd_face2node = self.entity('face', index=bd_face_flag)
        bd_node_flag = fealpy.zeros((NN,), **kwargs)
        bd_node_flag[bd_face2node.ravel()] = True
        return bd_node_flag

    def boundary_face_flag(self) -> TensorLike:
        """Return a boolean tensor indicating the boundary faces.

        Returns:
            Tensor: boundary face flag.
        """
        return self.face2cell[:, 0] == self.face2cell[:, 1]

    def boundary_cell_flag(self) -> TensorLike:
        """Return a boolean tensor indicating the boundary cells.

        Returns:
            Tensor: boundary cell flag.
        """
        NC = self.number_of_cells()
        bd_face_flag = self.boundary_face_flag()
        kwargs = {'dtype': bd_face_flag.dtype, 'device': bd_face_flag.device}
        bd_face2cell = self.face2cell[bd_face_flag, 0]
        bd_cell_flag = fealpy.zeros((NC,), **kwargs)
        bd_cell_flag[bd_face2cell.ravel()] = True
        return bd_cell_flag

    def boundary_node_index(self): return self.boundary_node_flag().nonzero().ravel()
    # TODO: finish this:
    # def boundary_edge_index(self): return self.boundary_edge_flag().nonzero().ravel()
    def boundary_face_index(self): return self.boundary_face_flag().nonzero().ravel()
    def boundary_cell_index(self): return self.boundary_cell_flag().nonzero().ravel()

    ### Homogeneous Mesh ###
    def is_homogeneous(self, etype: Union[int, str]='cell') -> bool:
        """Return True if the mesh entity is homogeneous.

        Returns:
            bool: Homogeneous indicator.
        """
        entity = self.entity(etype)
        if entity is None:
            raise RuntimeError(f'{etype} is not found.')
        return entity.ndim == 2

    def total_face(self) -> TensorLike:
        cell = self.entity(self.TD)
        local_face = self.localFace
        NVF = local_face.shape[-1]
        total_face = cell[..., local_face].reshape(-1, NVF)
        return total_face

    def total_edge(self) -> TensorLike:
        cell = self.entity(self.TD)
        local_edge = self.localEdge
        NVE = local_edge.shape[-1]
        total_edge = cell[..., local_edge].reshape(-1, NVE)
        return total_edge

    def construct(self):
        if not self.is_homogeneous():
            raise RuntimeError('Can not construct for a non-homogeneous mesh.')

        NC = self.number_of_cells()
        NFC = self.number_of_faces_of_cells()

        totalFace = self.total_face()
        _, i0_np, j_np = fealpy.unique(
            fealpy.sort(totalFace, dim=1)[0],
            return_index=True,
            return_inverse=True,
            axis=0
        )
        self.face = totalFace[i0_np, :] # this also adds the edge in 2-d meshes
        NF = i0_np.shape[0]

        i1_np = fealpy.zeros(NF, dtype=i0_np.dtype)
        i1_np[j_np] = fealpy.arange(0, NFC*NC, dtype=i0_np.dtype)

        self.cell2face = j_np.reshape(NC, NFC)

        face2cell_np = fealpy.stack([i0_np//NFC, i1_np//NFC, i0_np%NFC, i1_np%NFC], axis=-1)
        self.face2cell = face2cell_np

        if self.TD == 3:
            NEC = self.number_of_edges_of_cells()

            total_edge = self.total_edge()
            _, i2, j = fealpy.unique(
                fealpy.sort(total_edge, dim=1)[0],
                return_index=True,
                return_inverse=True,
                axis=0
            )
            self.edge = total_edge[i2, :]
            self.cell2edge = j.reshape(NC, NEC)

        elif self.TD == 2:
            self.edge2cell = self.face2cell
            self.cell2edge = self.cell2face

        logger.info(f"Mesh toplogy relation constructed, with {NF} faces, "
                    f"on device {self.device}")


##################################################
### Mesh Base
##################################################

class Mesh(MeshDS):
    @property
    def ftype(self) -> Any:
        node = self.entity(0)
        if node is None:
            raise RuntimeError('Can not get the float type as the node '
                               'has not been assigned.')
        return node.dtype

    def geo_dimension(self) -> int:
        node = self.entity(0)
        if node is None:
            raise RuntimeError('Can not get the geometrical dimension as the node '
                               'has not been assigned.')
        return node.shape[-1]

    GD = property(geo_dimension)

    def multi_index_matrix(self, p: int, etype: int) -> TensorLike:
        return fealpy.multi_index_matrix(p, etype, dtype=self.itype)

    def entity_barycenter(self, etype: Union[int, str], index: Optional[Index]=None) -> TensorLike:
        """Get the barycenter of the entity.

        Parameters:
            etype (int | str): The topology dimension of the entity, or name
                'cell' | 'face' | 'edge' | 'node'. Returns sliced node if 'node'.
            index (int | slice | Tensor): The index of the entity.

        Returns:
            Tensor: A 2-d tensor containing barycenters of the entity.
        """
        # if etype in ('node', 0):
        #     return self.node if index is None else self.node[index]

        # node = self.node
        # if isinstance(etype, str):
        #     etype = estr2dim(self, etype)
        # etn = edim2node(self, etype, index, dtype=node.dtype)
        # return F.entity_barycenter(etn, node) # TODO: finish this
        raise NotImplementedError

    def edge_length(self, index: Index=_S, out=None) -> TensorLike:
        """Calculate the length of the edges.

        Parameters:
            index (int | slice | Tensor, optional): Index of edges.
            out (Tensor, optional): The output tensor. Defaults to None.

        Returns:
            Tensor[NE,]: Length of edges, shaped [NE,].
        """
        edge = self.entity(1, index=index)
        return fealpy.edge_length(edge, self.node, out=out)

    def edge_normal(self, index: Index=_S, normalize: bool=False, out=None) -> TensorLike:
        """Calculate the normal of the edges.

        Parameters:
            index (int | slice | Tensor, optional): Index of edges.\n
            normalize (bool, optional): _description_. Defaults to False.\n
            out (Tensor, optional): _description_. Defaults to None.

        Returns:
            Tensor[NE, GD]: _description_
        """
        edge = self.entity(1, index=index)
        return fealpy.edge_normal(edge, self.node, normalize=normalize, out=out)

    def edge_unit_normal(self, index: Index=_S, out=None) -> TensorLike:
        """Calculate the unit normal of the edges.
        Equivalent to `edge_normal(index=index, unit=True)`.
        """
        return self.edge_normal(index=index, normalize=True, out=out)

    def edge_tangent(self, index: Index=_S, normalize: bool=False, out=None) -> TensorLike:
        """Calculate the tangent of the edges.

        Parameters:
            index (Index, optional): _description_. Defaults to _S.\n
            normalize (bool, optional): _description_. Defaults to False.\n
            out (TensorLike, optional): _description_. Defaults to None.

        Returns:
            TensorLike[NE, GD]: _description_
        """
        edge = self.entity(1, index=index)
        return fealpy.edge_tengent(edge, self.node, normalize=normalize, out=out)

    def cell_normal(self, index: Index=_S, node: Optional[TensorLike]=None) -> TensorLike:
        """
        @brief 计算网格单元的外法线方向，适用于三维空间中单元拓扑维数为 2 的情况，
        比如三维空间中的三角形或四边形网格.
        """
        node = self.entity('node') if node is None else node
        cell = self.entity('cell', index=index)
        v1 = node[cell[:, 1]] - node[cell[:, 0]]
        v2 = node[cell[:, 2]] - node[cell[:, 1]]
        normal = fealpy.cross(v1, v2)
        return normal

    def quadrature_formula(self, q: int, etype: Union[int, str]='cell', qtype: str='legendre') -> Quadrature:
        """Get the quadrature points and weights.

        Parameters:
            q (int): The index of the quadrature points.
            etype (int | str, optional): The topology dimension of the entity to\
            generate the quadrature points on. Defaults to 'cell'.

        Returns:
            Quadrature: Object for quadrature points and weights.
        """
        raise NotImplementedError

    def integrator(self, q: int, etype: Union[int, str]='cell', qtype: str='legendre') -> Quadrature:
        logger.warning("The `integrator` is deprecated and will be removed after 3.0. "
                       "Use `quadrature_formula` instead.")
        return self.quadrature_formula(q, etype, qtype)

    # ipoints
    def edge_to_ipoint(self, p: int, index: Index=_S) -> TensorLike:
        """Get the relationship between edges and integration points."""
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        edges = self.edge[index]
        kwargs = {'dtype': edges.dtype, 'device': self.device}
        indices = fealpy.arange(0, NE, **kwargs)[index]
        return fealpy.cat([
            edges[:, 0].reshape(-1, 1),
            (p-1) * indices.reshape(-1, 1) + fealpy.arange(0, p-1, **kwargs) + NN,
            edges[:, 1].reshape(-1, 1),
        ], dim=-1)

    # shape function
    def shape_function(self, bc: TensorLike, p: int=1, *, index: Index=_S,
                       variable: str='u', mi: Optional[TensorLike]=None) -> TensorLike:
        """Shape function value on the given bc points, in shape (..., ldof).

        Parameters:
            bc (Tensor): The bc points, in shape (NQ, bc).\n
            p (int, optional): The order of the shape function. Defaults to 1.\n
            index (int | slice | Tensor, optional): The index of the cell.\n
            variable (str, optional): The variable name. Defaults to 'u'.\n
            mi (Tensor, optional): The multi-index matrix. Defaults to None.

        Returns:
            Tensor: The shape function value with shape (NQ, ldof). The shape will\
            be (1, NQ, ldof) if `variable == 'x'`.
        """
        raise NotImplementedError(f"shape function is not supported by {self.__class__.__name__}")

    def grad_shape_function(self, bc: TensorLike, p: int=1, *, index: Index=_S,
                            variable: str='u', mi: Optional[TensorLike]=None) -> TensorLike:
        """Gradient of shape function on the given bc points, in shape (..., ldof, bc).

        Parameters:
            bc (Tensor): The bc points, in shape (NQ, bc).\n
            p (int, optional): The order of the shape function. Defaults to 1.\n
            index (int | slice | Tensor, optional): The index of the cell.\n
            variable (str, optional): The variable name. Defaults to 'u'.\n
            mi (Tensor, optional): The multi-index matrix. Defaults to None.

        Returns:
            Tensor: The shape function value with shape (NQ, ldof, bc). The shape will\
            be (NC, NQ, ldof, GD) if `variable == 'x'`.
        """
        raise NotImplementedError(f"grad shape function is not supported by {self.__class__.__name__}")

    def hess_shape_function(self, bc: TensorLike, p: int=1, *, index: Index=_S,
                            variable: str='u', mi: Optional[TensorLike]=None) -> TensorLike:
        raise NotImplementedError(f"hess shape function is not supported by {self.__class__.__name__}")

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
        elif fealpy.is_tensor(f):
            if f.shape == (GD, ): # 常向量函数
                e = cm[:, None]*f
            elif f.shape == (GD, GD):
                e = cm[:, None, None]*f
            else:
                e = fealpy.einsum('q, qc..., c->c...', ws, f, cm)
        else:
            raise ValueError(f"Unsupported type of return value: {f.__class__.__name__}.")

        if celltype:
            return e
        else:
            return fealpy.sum(e)

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
        #if v.shape[-1] == NC:
        #    v = np.swapaxes(v, 1, -1)
        f = fealpy.power(fealpy.abs(u - v), power)
        if len(f.shape) == 1:
            f = f[:, None]

        if isinstance(f, (int, float)): # f为标量常函数
            e = f*cm
        elif fealpy.is_tensor(f):
            if f.shape == (GD, ): # 常向量函数
                e = cm[:, None]*f
            elif f.shape == (GD, GD):
                e = cm[:, None, None]*f
            else:
                e = fealpy.einsum('q, qc..., c->c...', ws, f, cm)

        if celltype is False:
            e = fealpy.power(fealpy.sum(e), 1/power)
        else:
            e = fealpy.power(fealpy.sum(e, axis=tuple(range(1, len(e.shape)))), 1/power)
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
        import numpy as np
        import vtk
        import vtk.util.numpy_support as vnp

        NC = self.number_of_cells()
        GD = self.geo_dimension()

        node = self.entity('node')
        if GD == 2:
            node = np.concatenate(
                (node, np.zeros((node.shape[0], 1), dtype=self.ftype)),
                axis=1
            )

        cell = self.entity('cell')
        cellType = self.vtk_cell_type('cell')
        NV = cell.shape[-1]

        cell = np.r_['1', np.zeros((len(cell), 1), dtype=cell.dtype), cell]
        cell[:, 0] = NV
        cell = cell.astype(fealpy.int64)

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


class HomogeneousMesh(Mesh):
    # entity
    def entity_barycenter(self, etype: Union[int, str], index: Optional[Index]=None) -> TensorLike:
        node = self.entity('node')
        if etype in ('node', 0):
            return node if index is None else node[index]
        entity = self.entity(etype, index)
        return fealpy.barycenter(entity, node)

    def bc_to_point(self, bcs: Union[TensorLike, Sequence[TensorLike]],
                    etype: Union[int, str]='cell', index: Index=_S) -> TensorLike:
        """Convert barycenter coordinate points to cartesian coordinate points
        on mesh entities.
        """
        node = self.entity('node')
        entity = self.entity(etype, index)
        order = getattr(entity, 'bc_order', None)
        return fealpy.bc_to_points(bcs, node, entity, order)

    ### ipoints
    def interpolation_points(self, p: int, index: Index=_S) -> TensorLike:
        raise NotImplementedError

    def cell_to_ipoint(self, p: int, index: Index=_S) -> TensorLike:
        raise NotImplementedError

    def face_to_ipoint(self, p: int, index: Index=_S) -> TensorLike:
        raise NotImplementedError


class SimplexMesh(HomogeneousMesh):
    # ipoints
    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell'):
        if isinstance(iptype, str):
            iptype = estr2dim(self, iptype)
        return fealpy.simplex_ldof(p, iptype)

    def number_of_global_ipoints(self, p: int):
        nums = [self.entity(i).shape[0] for i in range(self.TD+1)]
        return fealpy.simplex_gdof(p, nums)

    # shape function
    def grad_lambda(self, index: Index=_S) -> TensorLike:
        raise NotImplementedError

    def shape_function(self, bc: TensorLike, p: int=1, *, index: Index=_S,
                       variable: str='u', mi: Optional[TensorLike]=None) -> TensorLike:
        TD = bc.shape[-1] - 1
        if mi is None:
            mi = fealpy.multi_index_matrix(p, TD, dtype=self.itype)
        phi = fealpy.simplex_shape_function(bc, p, mi)
        if variable == 'u':
            return phi
        elif variable == 'x':
            return phi.unsqueeze_(0)
        else:
            raise ValueError("Variable type is expected to be 'u' or 'x', "
                             f"but got '{variable}'.")

    def grad_shape_function(self, bc: TensorLike, p: int=1, *, index: Index=_S,
                            variable: str='u', mi: Optional[TensorLike]=None) -> TensorLike:
        TD = bc.shape[-1] - 1
        if mi is None:
            mi = fealpy.multi_index_matrix(p, TD, dtype=self.itype)
        R = fealpy.simplex_grad_shape_function(bc, p, mi) # (NQ, ldof, bc)
        if variable == 'u':
            return R
        elif variable == 'x':
            Dlambda = self.grad_lambda(index=index)
            gphi = fealpy.einsum('...bm, qjb -> ...qjm', Dlambda, R) # (NC, NQ, ldof, dim)
            # NOTE: the subscript 'q': NQ, 'm': dim, 'j': ldof, 'b': bc, '...': cell
            return gphi
        else:
            raise ValueError("Variable type is expected to be 'u' or 'x', "
                             f"but got '{variable}'.")


class TensorMesh(HomogeneousMesh):
    # ipoints
    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell') -> int:
        if isinstance(iptype, str):
            iptype = estr2dim(self, iptype)
        return fealpy.tensor_ldof(p, iptype)

    def number_of_global_ipoints(self, p: int) -> int:
        nums = [self.entity(i).shape[0] for i in range(self.TD+1)]
        return fealpy.tensor_gdof(p, nums)

    # shape function
    def grad_lambda(self, index: Index=_S) -> TensorLike:
        raise NotImplementedError

    def shape_function(self, bc: Tuple[TensorLike], p: int=1, *, index: Index=_S,
                       variable: str='u', mi: Optional[TensorLike]=None) -> TensorLike:
        pass

    def grad_shape_function(self, bc: Tuple[TensorLike], p: int=1, *, index: Index=_S,
                            variable: str='u', mi: Optional[TensorLike]=None) -> TensorLike:
        pass


class StructuredMesh(HomogeneousMesh):
    pass

    # NOTE: Here are some examples for entity factories:
    # implement them in subclasses if necessary.

    # @entitymethod
    # def _node(self, index: Index=_S):
    #     raise NotImplementedError

    # @entitymethod
    # def _edge(self, index: Index=_S):
    #     raise NotImplementedError

    # @entitymethod
    # def _face(self, index: Index=_S):
    #     raise NotImplementedError

    # @entitymethod
    # def _cell(self, index: Index=_S):
    #     raise NotImplementedError
