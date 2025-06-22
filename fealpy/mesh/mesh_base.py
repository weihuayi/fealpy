
from typing import Union, Optional, Sequence, Tuple, Any

from ..backend import backend_manager as bm
from ..typing import TensorLike, Index, _S
from .. import logger
from ..quadrature import Quadrature
from .mesh_data_structure import MeshDS
from .utils import (
    estr2dim, simplex_gdof, simplex_ldof, tensor_gdof, tensor_ldof
)


##################################################
### Mesh Base
##################################################

class Mesh(MeshDS):
    """
    Base class for all mesh types in FEALPy.
    
    This class provides fundamental mesh operations and properties that are common
    across different mesh types, including geometric calculations, shape functions,
    and visualization capabilities.
    
    Attributes:
        GD : int
            Geometric dimension of the mesh (property)
        itype : dtype
            Integer type used for mesh indices
        ftype : dtype  
            Floating point type used for mesh coordinates
        device : str
            Device where mesh data is stored ('cpu' or 'cuda')
    """
    def geo_dimension(self) -> int:
        """Get the geometric dimension of the mesh.
        
        Returns:
            int: The geometric dimension (such as 2 for 2D, 3 for 3D)
            
        Raises:
            RuntimeError: If nodes are not assigned
        """
        node = self.entity(0)
        if node is None:
            raise RuntimeError('Can not get the geometrical dimension as the node '
                               'has not been assigned.')
        return node.shape[-1]

    GD = property(geo_dimension)

    def multi_index_matrix(self, p: int, etype: int, dtype=None, device=None) -> TensorLike:
        """Generate multi-index matrix for given order and entity type.
        
        Parameters:
            p : int
                Polynomial order
            etype : int
                Entity type (0=node, 1=edge, etc.)
            dtype : dtype, optional
                Data type for the matrix
            device : str, optional
                Device to store the matrix
                
        Returns:
            TensorLike: The multi-index matrix
        """
        dtype = self.itype if dtype is None else dtype
        device = self.device if device is None else device
        return bm.device_put(bm.multi_index_matrix(p, etype, dtype=dtype), device=device)

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
            index (int | slice | Tensor, optional): Index of edges. Defaults to _S, _S means all.
            out (Tensor, optional): The output tensor. Defaults to None.

        Returns:
            Tensor, shape = (NE, ).
        """
        edge = self.entity(1, index=index)
        return bm.edge_length(edge, self.node, out=out)

    def edge_normal(self, index: Index=_S, unit: bool=False, out=None) -> TensorLike:
        """Calculate the normal of the edges.

        Parameters:
            index (int | slice | Tensor, optional): Index of edges.
            unit (bool, optional): If unit=True, it means to calculate the unit normal vector.
            out (Tensor, optional): The output tensor.

        Returns:
            Tensor, shape = (NE, GD).
        """
        edge = self.entity(1, index=index)
        return bm.edge_normal(edge, self.node, unit=unit, out=out)

    def edge_unit_normal(self, index: Index=_S, out=None) -> TensorLike:
        """Calculate the unit normal of the edges.
        Equivalent to `edge_normal(index=index, unit=True)`.

        Parameters:
            index (int | slice | Tensor, optional): Index of edges.Defaults to _S, _S means all.
            out (Tensor, optional): Defaults to None.

        Returns:
            Tensor, shape = (NE, GD).
        """
        return self.edge_normal(index=index, unit=True, out=out)

    def edge_tangent(self, index: Index=_S, unit: bool=False, out=None) -> TensorLike:
        """Calculate the tangent of the edges.

        Parameters:
            index (Index, optional): Defaults to _S, _S means all.
            unit (bool, optional): If unit=True, it means to calculate the unit normal vector.
            out (Tensor, optional): The output tensor.
        Returns:
            Tensor, shape = (NE, GD).
        """
        edge = self.entity(1, index=index)
        return bm.edge_tangent(edge, self.node, unit=unit, out=out)

    def cell_normal(self, index: Index=_S, node: Optional[TensorLike]=None) -> TensorLike:
        """Calculate normals of cells (for 2D surfaces in 3D space).
        
        Parameters:
            index : Index, optional
                Indices of cells to compute
            node : TensorLike, optional
                Custom node coordinates
                
        Returns:
            TensorLike: Cell normals (NC, 3).
        """
        node = self.entity('node') if node is None else node
        cell = self.entity('cell', index=index)
        v1 = node[cell[:, 1]] - node[cell[:, 0]]
        v2 = node[cell[:, 2]] - node[cell[:, 1]]
        normal = bm.cross(v1, v2)
        return normal

    def quadrature_formula(self, q: int, etype: Union[int, str]='cell', qtype: str='legendre') -> Quadrature:
        """Get the quadrature points and weights.

        Parameters:
            q (int): The index of the quadrature points.
            etype (int | str, optional): The topology dimension of the entity to
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
        """Map edges to integration points.
        
        Parameters:
            p : int
                The order of the shape function.
            index : Index, optional
                Edge indices to include
                
        Returns:
            TensorLike: Mapping matrix
        """
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        edges = self.edge[index]
        # kwargs = {'dtype': edges.dtype}
        kwargs = bm.context(edges)
        indices = bm.arange(NE, **kwargs)[index]
        return bm.concatenate([
            edges[:, 0].reshape(-1, 1),
            (p-1) * indices.reshape(-1, 1) + bm.arange(0, p-1, **kwargs) + NN,
            edges[:, 1].reshape(-1, 1),
        ], axis=-1)

    # shape function
    def shape_function(self, bcs: TensorLike, p: int=1, *, index: Index=_S,
                       variables: str='u', mi: Optional[TensorLike]=None) -> TensorLike:
        """Shape function value on the given bc points, in shape (..., ldof).

        Parameters:
            bcs (Tensor): The bc points, in shape (NQ, bc).
            p (int, optional): The order of the shape function. Defaults to 1.
            index (int | slice | Tensor, optional): The index of the cell.
            variables (str, optional): The variables name. Defaults to 'u'.
            mi (Tensor, optional): The multi-index matrix. Defaults to None.

        Returns:
            Tensor: The shape function value with shape (NQ, ldof). The shape will
            be (1, NQ, ldof) if `variables == 'x'`.
        """
        raise NotImplementedError(f"shape function is not supported by {self.__class__.__name__}")

    def grad_shape_function(self, bcs: TensorLike, p: int=1, *, index: Index=_S,
                            variables: str='u', mi: Optional[TensorLike]=None) -> TensorLike:
        """Gradient of shape function on the given bc points, in shape (..., ldof, bc).

        Parameters:
            bcs (Tensor): The bc points, in shape (NQ, bc).
            p (int, optional): The order of the shape function. Defaults to 1.
            index (int | slice | Tensor, optional): The index of the cell.
            variables (str, optional): The variables name. Defaults to 'u'.
            mi (Tensor, optional): The multi-index matrix. Defaults to None.

        Returns:
            Tensor: The shape function value with shape (NQ, ldof, bc). The shape will
            be (NC, NQ, ldof, GD) if `variables == 'x'`.
        """
        raise NotImplementedError(f"grad shape function is not supported by {self.__class__.__name__}")

    def hess_shape_function(self, bcs: TensorLike, p: int=1, *, index: Index=_S,
                            variables: str='u', mi: Optional[TensorLike]=None) -> TensorLike:
        raise NotImplementedError(f"hess shape function is not supported by {self.__class__.__name__}")

    # tools
    def paraview(self, file_name = "temp.vtu",
            background_color='1.0, 1.0, 1.0',
            show_type='Surface With Edges',
            ):
        """Visualize mesh using ParaView.
        
        Parameters:
            file_name : str, default="temp.vtu"
                Output VTK filename
            background_color : str, default='1.0, 1.0, 1.0'
                Background color in RGB
            show_type : str, default='Surface With Edges'
                Visualization style
            
        Notes:
            Requires ParaView to be installed. On Ubuntu, install with:
                sudo apt-get install paraview python3-paraview
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
            print("\nAdditionally, you may need to set the PYTHONPATH environment variables to include the path to the ParaView Python modules.")
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

    def vtkview(self, etype='cell', showedge=True,
            background_color: Tuple[float, float, float]=(0.3, 0.2, 0.1),
            edge_color: Tuple[float, float, float]=(0, 0, 0),
            edge_width=1.5,
            window_size: Tuple[int, int]=(800, 800)):
        """Visualize mesh using VTK.
        
        Parameters:
            etype : str, default='cell'
                Entity type to visualize
            showedge : bool, default=True
                Whether to show edges
            background_color : Tuple[float, float, float], default=(0.3, 0.2, 0.1)
                Background color in RGB
            edge_color : Tuple[float, float, float], default=(0, 0, 0)
                Edge color in RGB
            edge_width : float, default=1.5
                Edge line width
            window_size : Tuple[int, int], default=(800, 800)
                Window dimensions
        """
        import numpy as np
        import vtk
        from vtkmodules.util import numpy_support as vnp

        NC = self.number_of_cells()
        GD = self.geo_dimension()

        node = bm.to_numpy(self.entity('node'))
        if GD == 2:
            node = np.concatenate(
                (node, np.zeros((node.shape[0], 1), dtype=self.ftype)),
                axis=1
            )

        cell = bm.to_numpy(self.entity(etype))
        cellType = self.vtk_cell_type(etype)
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
        
        #自适应标记工具 
    def mark(eta, theta, method='L2'):
        isMarked = bm.zeros(len(eta), dtype=bm.bool)
        if method == 'MAX':
            # isMarked[eta > theta*bm.max(eta)] = True
            isMarked = bm.set_at(isMarked,(eta > theta*bm.max(eta)),True)
        elif method == 'COARSEN':
            # isMarked[eta < theta*bm.max(eta)] = True
            isMarked = bm.set_at(isMarked,(eta < theta*bm.max(eta)),True)
        elif method == 'L2':
            eta = eta**2
            
            idx = bm.flip(bm.argsort(eta))
            # idx = bm.argsort(eta)[-1::-1]
        
            x = bm.cumsum(eta[idx], axis=0)
            # isMarked[idx[x < theta*x[-1]]] = True
            # isMarked[idx[0]] = True
            isMarked = bm.set_at(isMarked,(idx[x < theta*x[-1]]),True)
            isMarked = bm.set_at(isMarked,(idx[0]),True)

        else:
            raise ValueError("I have not code the method")
        return isMarked 

class HomogeneousMesh(Mesh):
    """Base class for homogeneous meshes where all elements have same topology.
    
    This class extends Mesh with implementations specific to homogeneous meshes,
    providing common operations like barycenter calculation, point conversion,
    and numerical integration that work for any homogeneous mesh type.
    """
    # entity
    def entity_barycenter(self, etype: Union[int, str], index: Optional[Index]=None) -> TensorLike:
        """Calculate barycenters of mesh entities.
        
        Parameters:
            etype : int | str
                Entity type (dimension or name like 'cell', 'face')
            index : Index, optional
                Indices of specific entities to compute
                
        Returns:
            TensorLike: Barycenter coordinates (N, GD)
        """
        node = self.entity('node')
        if etype in ('node', 0):
            return node if index is None else node[index]
        entity = self.entity(etype, index)
        return bm.barycenter(entity, node)

    def bc_to_point(self, bcs: Union[TensorLike, Sequence[TensorLike]], index: Index=_S) -> TensorLike:
        """Convert barycentric coordinates to Cartesian coordinates.
        
        Parameters:
            bcs : TensorLike | Sequence[TensorLike]
                Barycentric coordinates (either tensor or sequence of tensors)
            index : Index, optional
                Entity indices to compute points for
                
        Returns:
            TensorLike: Cartesian coordinates of points
            
        Raises:
            TypeError: If bcs has invalid type
        """
        if isinstance(bcs, Sequence): # tensor type
            etype = len(bcs)
        elif isinstance(bcs, TensorLike): # simplex type
            etype = bcs.shape[-1] - 1
        else:
            raise TypeError("bcs is expected to be a tensor or sequence of tensor, "
                            f"but got {type(bcs).__name__}")

        node = self.entity('node')
        entity = self.entity(etype, index)

        return bm.bc_to_points(bcs, node, entity)

    # ipoints
    def interpolation_points(self, p: int, index: Index=_S) -> TensorLike:
        """Get interpolation points of order p.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

    def cell_to_ipoint(self, p: int, index: Index=_S) -> TensorLike:
        raise NotImplementedError

    def face_to_ipoint(self, p: int, index: Index=_S) -> TensorLike:
        raise NotImplementedError

    # tools
    def integral(self, f, q=3, celltype=False) -> TensorLike:
        """Numerically integrate a function over the mesh.
        
        Parameters:
            f : callable | scalar | tensor
                Function to integrate (can be callable or constant)
            q : int, default=3
                Quadrature order
            celltype : bool, default=False
                Whether to return cell-wise results
                
        Returns:
            TensorLike: Integral value (scalar if celltype=False, per-cell if True)
            
        Raises:
            ValueError: For unsupported function return types
        """
        GD = self.geo_dimension()
        qf = self.integrator(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = self.bc_to_point(bcs)

        if callable(f):
            if getattr(f, 'coordtype', None) == 'barycentric':
                f = f(bcs)
            else:
                f = f(ps)

        cm = self.entity_measure('cell')

        if isinstance(f, (int, float)): #  u 为标量常函数
            e = f*cm
        elif bm.is_tensor(f):
            if f.shape == (GD, ): # 常向量函数
                e = cm[:, None]*f
            elif f.shape == (GD, GD):
                e = cm[:, None, None]*f
            else:
                e = bm.einsum('q, cq..., c -> c...', ws, f, cm)
        else:
            raise ValueError(f"Unsupported type of return value: {f.__class__.__name__}.")

        if celltype:
            return e
        else:
            return bm.sum(e)

    def error(self, u, v, q=3, power=2, celltype=False) -> TensorLike:
        """Calculate error between two functions.
        
        Parameters:
            u : callable | array-like
                Reference function/values
            v : callable | array-like  
                Comparison function/values
            q : int, default=3
                Quadrature order
            power : int, default=2
                Power for error norm (L^power norm)
            celltype : bool, default=False
                Whether to return cell-wise errors
                
        Returns:
            TensorLike: Error measure (scalar or per-cell)
        """
        GD = self.geo_dimension()

        qf = self.quadrature_formula(q, etype='cell')
        bcs, ws = qf.get_quadrature_points_and_weights()
        ps = self.bc_to_point(bcs)

        if callable(u):
            if getattr(u, 'coordtype', None) == 'barycentric':
                u = u(bcs)
            else:
                u = u(ps)

        if callable(v):
            if getattr(v, 'coordtype', None) == 'barycentric':
                v = v(bcs)
            else:
                v = v(ps)
        
        if u.ndim == 2 and v.ndim == 3 and v.shape[-1] == 1:  # 处理一维梯度计算形状不一致问题
            u = u.reshape(u.shape[0], u.shape[1], 1)

        cm = self.entity_measure('cell')
        NC = self.number_of_cells()
        #if v.shape[-1] == NC:
        #    v = bm.swapaxes(v, 0, -1)
        #f = bm.power(bm.abs(u - v), power)
        f = bm.abs(u - v)**power
        if len(f.shape) == 1:
            f = f[:, None]

        if isinstance(f, (int, float)): # f为标量常函数
            e = f*cm
        elif bm.is_tensor(f):
            if f.shape == (GD, ): # 常向量函数
                e = cm[:, None]*f
            elif f.shape == (GD, GD):
                e = cm[:, None, None]*f
            else:
                e = bm.einsum('q, cq..., c -> c...', ws, f, cm)

        if celltype is False:
            #e = bm.power(bm.sum(e), 1/power)
            e = bm.sum(e)**(1/power)
        else:
            e = bm.pow(bm.sum(e, axis=tuple(range(1, len(e.shape)))), 1/power)
        return e # float or (NC, )


class SimplexMesh(HomogeneousMesh):
    """Mesh class for simplex elements (triangles, tetrahedrons, etc.).
    
    Provides specialized implementations for simplex meshes including shape functions
    and interpolation point handling specific to simplex geometry.
    """
    # ipoints
    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell'):
        """Get number of local interpolation points.
        
        Parameters:
            p : int
                Polynomial order
            iptype : int | str, default='cell'
                Entity type
                
        Returns:
            int: Number of local interpolation points
        """
        if isinstance(iptype, str):
            iptype = estr2dim(self, iptype)
        return simplex_ldof(p, iptype)

    def number_of_global_ipoints(self, p: int):
        """Get number of global interpolation points.
        
        Parameters:
            p : int
                Polynomial order
                
        Returns:
            int: Number of global interpolation points
        """
        nums = [self.entity(i).shape[0] for i in range(self.TD+1)]
        return simplex_gdof(p, nums)

    # shape function
    def grad_lambda(self, index: Index=_S) -> TensorLike:
        raise NotImplementedError

    def shape_function(self, bcs: TensorLike, p: int=1, *, index: Index=_S,
                       mi: Optional[TensorLike]=None) -> TensorLike:
        """Evaluate simplex shape functions at given points.
        
        Parameters:
            bcs : TensorLike
                Barycentric coordinates
            p : int, default=1
                Polynomial order
            index : Index, optional
                Cell indices
            mi : TensorLike, optional
                Multi-index matrix
                
        Returns:
            TensorLike: Shape function values
        """
        TD = bcs.shape[-1] - 1
        if mi is None:
            mi = bm.multi_index_matrix(p, TD, dtype=self.itype)
        phi = bm.simplex_shape_function(bcs, p, mi)
        return phi
    
    face_shape_function = shape_function
    edge_shape_function = shape_function

    def grad_shape_function(self, bcs: TensorLike, p: int=1, *, index: Index=_S,
                            variables: str='u', mi: Optional[TensorLike]=None) -> TensorLike:
        """Evaluate gradient of simplex shape functions.
        
        Parameters:
            bcs : TensorLike
                Barycentric coordinates
            p : int, default=1
                Polynomial order
            index : Index, optional
                Cell indices
            variables : str, default='u'
                Variable space ('u' or 'x')
            mi : TensorLike, optional
                Multi-index matrix
                
        Returns:
            TensorLike: Gradient values
            
        Raises:
            ValueError: For invalid variables type
        """
        TD = bcs.shape[-1] - 1
        if mi is None:
            mi = bm.multi_index_matrix(p, TD, dtype=self.itype)
        R = bm.simplex_grad_shape_function(bcs, p, mi) # (NQ, ldof, bc)
        
        if variables == 'u':
            return R
        elif variables == 'x':
            Dlambda = self.grad_lambda(index=index, TD=TD)
            gphi = bm.einsum('...bm, qjb -> ...qjm', Dlambda, R) # (NC, NQ, ldof, dim)
            # NOTE: the subscript 'q': NQ, 'm': dim, 'j': ldof, 'b': bc, '...': cell
            return gphi
        else:
            raise ValueError("Variables type is expected to be 'u' or 'x', "
                             f"but got '{variables}'.")
    
    def hess_shape_function(self, bcs: TensorLike, p: int=1, *, index: Index=_S,
                            variables: str='u', mi: Optional[TensorLike]=None) -> TensorLike:
        """Evaluate Hessian (second derivatives) of simplex shape functions.
    
        Computes the second derivatives of shape functions either in reference 
        ('u') or physical ('x') coordinates.

        Parameters:
            bcs : TensorLike
                Barycentric coordinates of evaluation points with shape (NQ, TD+1)
            p : int, default=1
                Polynomial order of shape functions
            index : Index, optional
                Indices of cells to evaluate on (default all cells)
            variables : str, default='u'
                Coordinate space for derivatives:
                - 'u': Reference element coordinates
                - 'x': Physical coordinates
            mi : TensorLike, optional
                Precomputed multi-index matrix for efficiency

        Returns:
            TensorLike: 
                - If variables='u': Hessian in reference space with shape (NQ, ldof, bc, bc)
                - If variables='x': Hessian in physical space with shape (NC, NQ, ldof, GD, GD)

        Raises:
            ValueError: If invalid variables type is provided

        Notes:
            The physical space Hessian is computed using chain rule:
            H_phys = J^-T @ H_ref @ J^-1
            where J is the Jacobian matrix of the transformation
        """
        TD = bcs.shape[1] - 1
        if mi is None:
            mi = bm.multi_index_matrix(p, TD, dtype=self.itype)
        H = bm.simplex_hess_shape_function(bcs, p, mi)
        if variables == 'x':
            Dlambda = self.grad_lambda(index=index, TD=TD) # (NC, NQ, ldof, dim) 
            Hphi = bm.einsum('...inm, knj, kml  -> k...ijl', H, Dlambda, Dlambda)
            return Hphi
        elif variables == 'u':
            return H

class TensorMesh(HomogeneousMesh):
    # ipoints
    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell') -> int:
        """Get number of local interpolation points.
        
        Parameters:
            p : int
                Polynomial order
            iptype : int | str, default='cell'
                Entity type
                
        Returns:
            int: Number of local interpolation points
        """
        if isinstance(iptype, str):
            iptype = estr2dim(self, iptype)
        return tensor_ldof(p, iptype)

    def number_of_global_ipoints(self, p: int) -> int:
        """Get number of global interpolation points.
        
        Parameters:
            p : int
                Polynomial order
                
        Returns:
            int: Number of global interpolation points
        """
        nums = [self.entity(i).shape[0] for i in range(self.TD+1)]
        return tensor_gdof(p, nums)

    def bc_to_point(self, bc, index=None):
        """Convert barycentric coordinates to Cartesian coordinates.
        
        Parameters:
            bcs : TensorLike | Sequence[TensorLike]
                Barycentric coordinates (either tensor or sequence of tensors)
            index : Index, optional
                Entity indices to compute points for
                
        Returns:
            TensorLike: Cartesian coordinates of points
            
        Raises:
            TypeError: If bcs has invalid type
        """
        
        node = self.entity('node')
        if isinstance(bc, tuple) and len(bc) == 3:
            cell = self.entity('cell', index)

            bc0 = bc[0].reshape(-1, 2) # (NQ0, 2)
            bc1 = bc[1].reshape(-1, 2) # (NQ1, 2)
            bc2 = bc[2].reshape(-1, 2) # (NQ2, 2)
            bc = bm.einsum('im, jn, ko->ijkmno', bc0, bc1, bc2).reshape(-1, 8) # (NQ0, NQ1, 2, 2, 2)

            p = bm.einsum('qj, cjk->cqk', bc, node[cell[:, [0, 4, 3, 7, 1, 5, 2, 6]]]) # (NC, NQ, 3)

        elif isinstance(bc, tuple) and len(bc) == 2:
            face = self.entity(2, index=index)

            bc0 = bc[0].reshape(-1, 2) # (NQ0, 2)
            bc1 = bc[1].reshape(-1, 2) # (NQ1, 2)
            bc = bm.einsum('im, jn->ijmn', bc0, bc1).reshape(-1, 4) # (NQ0, NQ1, 2, 2)

            p = bm.einsum('qj, cjk->cqk', bc, node[face[:, [0, 3, 1, 2]]]) # (NC, NQ, 2)
        else:
            edge = self.entity('edge', index=index)
            p = bm.einsum('qj, ejk->eqk', bc[0], node[edge]) # (NE, NQ, 2)
        return p

    edge_bc_to_point = bc_to_point
    face_bc_to_point = bc_to_point
    cell_bc_to_point = bc_to_point

    # shape function
    def grad_lambda(self, index: Index=_S) -> TensorLike:
        raise NotImplementedError

    def shape_function(self, bcs: Tuple[TensorLike], p: int=1, *, index: Index=_S,
                       variables: str='u', mi: Optional[TensorLike]=None) -> TensorLike:
        """Evaluate simplex shape functions at given points.
        
        Parameters:
            bcs : TensorLike
                Barycentric coordinates
            p : int, default=1
                Polynomial order
            index : Index, optional
                Cell indices
            variables : str, default='u'
                Variable space ('u' or 'x')
            mi : TensorLike, optional
                Multi-index matrix
                
        Returns:
            TensorLike: Shape function values
        """
        if mi is None:
            mi = bm.multi_index_matrix(p, 1, dtype=self.itype)
        raw_phi = [bm.simplex_shape_function(bc, p, mi) for bc in bcs]
        phi = bm.tensorprod(*raw_phi)
        if variables == 'u':
            return phi
        elif variables == 'x':
            return phi[None, ...]
        else:
            raise ValueError("Variables type is expected to be 'u' or 'x', "
                             f"but got '{variables}'.")
    
    face_shape_function = shape_function
    edge_shape_function = shape_function

    def grad_shape_function(self, bcs: Tuple[TensorLike], p: int=1, *, index: Index=_S,
                            variables: str='u', mi: Optional[TensorLike]=None) -> TensorLike:
        """Evaluate gradient of simplex shape functions.
        
        Parameters:
            bcs : TensorLike
                Barycentric coordinates
            p : int, default=1
                Polynomial order
            index : Index, optional
                Cell indices
            variables : str, default='u'
                Variable space ('u' or 'x')
            mi : TensorLike, optional
                Multi-index matrix
                
        Returns:
            TensorLike: Gradient values
            
        Raises:
            ValueError: For invalid variables type
        """
        assert isinstance(bcs, tuple)
        TD = len(bcs)
        Dlambda = bm.array([-1, 1], dtype=self.ftype, device=bm.get_device(bcs[0]))
        phi = bm.simplex_shape_function(bcs[0], p=p)
        R = bm.simplex_grad_shape_function(bcs[0], p=p)
        dphi = bm.einsum('...ij, j->...i', R, Dlambda)

        n = phi.shape[0]**TD
        ldof = phi.shape[-1]**TD
        shape = (n, ldof, TD)
        gphi = bm.zeros(shape, dtype=self.ftype, device=bm.get_device(bcs[0]))

        if TD == 3:
            gphi0 = bm.einsum('im, jn, ko->ijkmno', dphi, 
                              phi, phi).reshape(-1, ldof, 1)
            gphi1 = bm.einsum('im, jn, ko->ijkmno', phi, dphi,
                              phi).reshape(-1, ldof, 1)
            gphi2 = bm.einsum('im, jn, ko->ijkmno', phi, phi,
                                     dphi).reshape(-1, ldof, 1)
            gphi = bm.concatenate((gphi0, gphi1, gphi2), axis=-1)
            if variables == 'x':
                J = self.jacobi_matrix(bcs, index=index)
                J = bm.linalg.inv(J)
                gphi = bm.einsum('cqmn, qlm -> cqln', J, gphi)

                return gphi
        elif TD == 2:
            gphi0 = bm.einsum('im, jn -> ijmn', dphi, phi).reshape(-1, ldof, 1)
            gphi1 = bm.einsum('im, jn -> ijmn', phi, dphi).reshape(-1, ldof, 1)
            gphi = bm.concatenate((gphi0, gphi1), axis=-1)              # (NQ, ldof, GD)
            if variables == 'x':
                J = self.jacobi_matrix(bcs, index=index)                # (NC, NQ, GD, GD)
                G = self.first_fundamental_form(J)                      # (NC, NQ, GD, GD)
                G = bm.linalg.inv(G)
                gphi = bm.einsum('cqkm, cqmn, qln -> cqlk', J, G, gphi) # (NC, NQ, ldof, GD)

                return gphi
            
        return gphi

    def quad_to_ipoint(self, p, index=None):
        """Generate global indices for quadrilateral face interpolation points.
    
        Constructs the global numbering of interpolation points on quadrilateral faces
        for a given polynomial order p, handling edge orientation and interior points.

        Parameters:
            p : int
                Polynomial order of the interpolation points
            index : Index, optional
                Indices of specific faces to compute (default: all faces)

        Returns:
            TensorLike: 
                Integer array of shape (NF, (p+1)^2) containing global indices of 
                interpolation points for each face, where NF is number of faces.
                Points are ordered following tensor-product ordering.

        Notes:
            1. For each quadrilateral face, interpolation points consist of:
            - Vertex nodes
            - Edge nodes (p-1 points per edge)
            - Interior points ((p-1)^2 points)
            2. Edge points are numbered consistently with edge orientation
            3. Interior points are numbered sequentially after edge points
        """
        NN = self.number_of_nodes()
        NE = self.number_of_edges()
        NF = self.number_of_faces()
        edge = self.entity('edge')
        face = self.entity('face')
        face2edge = self.face_to_edge()
        edge2ipoint = self.edge_to_ipoint(p)

        mi = bm.repeat(bm.arange(p+1, device=bm.get_device(edge)), p+1).reshape(-1, p+1)
        multiIndex0 = mi.flatten().reshape(-1, 1);
        multiIndex1 = mi.T.flatten().reshape(-1, 1);
        multiIndex = bm.concatenate([multiIndex0, multiIndex1], axis=1)

        dofidx = [0 for i in range(4)] 
        dofidx[0], = bm.nonzero(multiIndex[:, 1]==0)
        dofidx[1], = bm.nonzero(multiIndex[:, 0]==p)
        dofidx[2], = bm.nonzero(multiIndex[:, 1]==p)
        dofidx[3], = bm.nonzero(multiIndex[:, 0]==0)
        

        face2ipoint = bm.zeros([NF, (p+1)**2], dtype=self.itype, device=bm.get_device(edge))
        localEdge = bm.array([[0, 1], [1, 2], [3, 2], [0, 3]], 
                            dtype=self.itype, device=bm.get_device(edge))
        for i in range(4):
            ge = face2edge[:, i]
            idx = bm.nonzero(face[:, localEdge[i, 0]] != edge[ge, 0])[0]

            face2ipoint = bm.set_at(face2ipoint, (slice(None), dofidx[i]), 
                                edge2ipoint[ge])
            face2ipoint = bm.set_at(face2ipoint, (idx[:, None], dofidx[i]), 
                                bm.flip(edge2ipoint[ge[idx]], axis=1))
            # face2ipoint[:, dofidx[i]] = edge2ipoint[ge]
            # face2ipoint[idx[:, None], dofidx[i]] = bm.flip(edge2ipoint[ge[idx]], axis=1)

        indof = bm.all(multiIndex>0, axis=-1) & bm.all(multiIndex<p, axis=-1)
        face2ipoint = bm.set_at(face2ipoint, (slice(None), indof), 
                    bm.arange(NN + NE * (p - 1), NN + NE * (p - 1) + NF * (p - 1) ** 2, 
                    dtype=self.itype, device=bm.get_device(edge)).reshape(NF, -1))
        # face2ipoint[:, indof] = bm.arange(NN+NE*(p-1), NN+NE*(p-1)+NF*(p-1)**2, 
        #                     dtype=self.itype, device=bm.get_device(edge)).reshape(NF, -1)
        face2ipoint = face2ipoint[index]
        
        return face2ipoint
    

class StructuredMesh(HomogeneousMesh):

    # shape function
    def grad_lambda(self, index: Index=_S) -> TensorLike:
        raise NotImplementedError
    
    @property
    def device(self) -> Any:
        return self._device
    
    @device.setter
    def device(self, value: Any):
        self._device = value

