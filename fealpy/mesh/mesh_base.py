
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
    def geo_dimension(self) -> int:
        node = self.entity(0)
        if node is None:
            raise RuntimeError('Can not get the geometrical dimension as the node '
                               'has not been assigned.')
        return node.shape[-1]

    GD = property(geo_dimension)

    def multi_index_matrix(self, p: int, etype: int, dtype=None, device=None) -> TensorLike:
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
            index (int | slice | Tensor, optional): Index of edges.
            out (Tensor, optional): The output tensor. Defaults to None.

        Returns:
            Tensor[NE,]: Length of edges, shaped [NE,].
        """
        edge = self.entity(1, index=index)
        return bm.edge_length(edge, self.node, out=out)

    def edge_normal(self, index: Index=_S, unit: bool=False, out=None) -> TensorLike:
        """Calculate the normal of the edges.

        Parameters:
            index (int | slice | Tensor, optional): Index of edges.\n
            unit (bool, optional): _description_. Defaults to False.\n
            out (Tensor, optional): _description_. Defaults to None.

        Returns:
            Tensor[NE, GD]: _description_
        """
        edge = self.entity(1, index=index)
        return bm.edge_normal(edge, self.node, unit=unit, out=out)

    def edge_unit_normal(self, index: Index=_S, out=None) -> TensorLike:
        """Calculate the unit normal of the edges.
        Equivalent to `edge_normal(index=index, unit=True)`.
        """
        return self.edge_normal(index=index, unit=True, out=out)

    def edge_tangent(self, index: Index=_S, unit: bool=False, out=None) -> TensorLike:
        """Calculate the tangent of the edges.

        Parameters:
            index (Index, optional): _description_. Defaults to _S.\n
            unit (bool, optional): _description_. Defaults to False.\n
            out (TensorLike, optional): _description_. Defaults to None.

        Returns:
            TensorLike[NE, GD]: _description_
        """
        edge = self.entity(1, index=index)
        return bm.edge_tangent(edge, self.node, unit=unit, out=out)

    def cell_normal(self, index: Index=_S, node: Optional[TensorLike]=None) -> TensorLike:
        """
        @brief 计算网格单元的外法线方向，适用于三维空间中单元拓扑维数为 2 的情况，
        比如三维空间中的三角形或四边形网格.
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
            bcs (Tensor): The bc points, in shape (NQ, bc).\n
            p (int, optional): The order of the shape function. Defaults to 1.\n
            index (int | slice | Tensor, optional): The index of the cell.\n
            variables (str, optional): The variables name. Defaults to 'u'.\n
            mi (Tensor, optional): The multi-index matrix. Defaults to None.

        Returns:
            Tensor: The shape function value with shape (NQ, ldof). The shape will\
            be (1, NQ, ldof) if `variables == 'x'`.
        """
        raise NotImplementedError(f"shape function is not supported by {self.__class__.__name__}")

    def grad_shape_function(self, bcs: TensorLike, p: int=1, *, index: Index=_S,
                            variables: str='u', mi: Optional[TensorLike]=None) -> TensorLike:
        """Gradient of shape function on the given bc points, in shape (..., ldof, bc).

        Parameters:
            bcs (Tensor): The bc points, in shape (NQ, bc).\n
            p (int, optional): The order of the shape function. Defaults to 1.\n
            index (int | slice | Tensor, optional): The index of the cell.\n
            variables (str, optional): The variables name. Defaults to 'u'.\n
            mi (Tensor, optional): The multi-index matrix. Defaults to None.

        Returns:
            Tensor: The shape function value with shape (NQ, ldof, bc). The shape will\
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

    def vtkview(self, showedge=True,
            background_color: Tuple[float, float, float]=(0.3, 0.2, 0.1),
            edge_color: Tuple[float, float, float]=(0, 0, 0),
            edge_width=1.5,
            window_size: Tuple[int, int]=(800, 800)):
        """
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

        cell = bm.to_numpy(self.entity('cell'))
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
    # entity
    def entity_barycenter(self, etype: Union[int, str], index: Optional[Index]=None) -> TensorLike:
        node = self.entity('node')
        if etype in ('node', 0):
            return node if index is None else node[index]
        entity = self.entity(etype, index)
        return bm.barycenter(entity, node)

    def bc_to_point(self, bcs: Union[TensorLike, Sequence[TensorLike]], index: Index=_S) -> TensorLike:
        """Convert barycenter coordinate points to cartesian coordinate points
        on mesh entities.
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
        raise NotImplementedError

    def cell_to_ipoint(self, p: int, index: Index=_S) -> TensorLike:
        raise NotImplementedError

    def face_to_ipoint(self, p: int, index: Index=_S) -> TensorLike:
        raise NotImplementedError

    # tools
    def integral(self, f, q=3, celltype=False) -> TensorLike:
        """
        @brief 在网格中数值积分一个函数
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
        """
        @brief Calculate the error between two functions.
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
            e = bm.power(bm.sum(e, axis=tuple(range(1, len(e.shape)))), 1/power)
        return e # float or (NC, )


class SimplexMesh(HomogeneousMesh):
    # ipoints
    def number_of_local_ipoints(self, p: int, iptype: Union[int, str]='cell'):
        if isinstance(iptype, str):
            iptype = estr2dim(self, iptype)
        return simplex_ldof(p, iptype)

    def number_of_global_ipoints(self, p: int):
        nums = [self.entity(i).shape[0] for i in range(self.TD+1)]
        return simplex_gdof(p, nums)

    # shape function
    def grad_lambda(self, index: Index=_S) -> TensorLike:
        raise NotImplementedError

    def shape_function(self, bcs: TensorLike, p: int=1, *, index: Index=_S,
                       mi: Optional[TensorLike]=None) -> TensorLike:
        TD = bcs.shape[-1] - 1
        if mi is None:
            mi = bm.multi_index_matrix(p, TD, dtype=self.itype)
        phi = bm.simplex_shape_function(bcs, p, mi)
        return phi
    
    face_shape_function = shape_function
    edge_shape_function = shape_function

    def grad_shape_function(self, bcs: TensorLike, p: int=1, *, index: Index=_S,
                            variables: str='u', mi: Optional[TensorLike]=None) -> TensorLike:
        TD = bcs.shape[-1] - 1
        if mi is None:
            mi = bm.multi_index_matrix(p, TD, dtype=self.itype)
        R = bm.simplex_grad_shape_function(bcs, p, mi) # (NQ, ldof, bc)
        
        if variables == 'u':
            return R
        elif variables == 'x':
            Dlambda = self.grad_lambda(index=index)
            gphi = bm.einsum('...bm, qjb -> ...qjm', Dlambda, R) # (NC, NQ, ldof, dim)
            # NOTE: the subscript 'q': NQ, 'm': dim, 'j': ldof, 'b': bc, '...': cell
            return gphi
        else:
            raise ValueError("Variables type is expected to be 'u' or 'x', "
                             f"but got '{variables}'.")
    
    def hess_shape_function(self, bcs: TensorLike, p: int=1, *, index: Index=_S,
                            variables: str='u', mi: Optional[TensorLike]=None) -> TensorLike:
        """
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
        if isinstance(iptype, str):
            iptype = estr2dim(self, iptype)
        return tensor_ldof(p, iptype)

    def number_of_global_ipoints(self, p: int) -> int:
        nums = [self.entity(i).shape[0] for i in range(self.TD+1)]
        return tensor_gdof(p, nums)

    def bc_to_point(self, bc, index=None):
        """
        @brief 把积分点变换到实际网格实体上的笛卡尔坐标点
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
        """
        @brief Generate global indices for interpolation points on each face
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

