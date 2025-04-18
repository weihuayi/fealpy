
from typing import Tuple, Union, Callable,Optional
from math import prod

from ..backend import backend_manager as bm
from ..typing import TensorLike, Size, _S
from .functional import generate_tensor_basis, generate_tensor_grad_basis
from .space import FunctionSpace, _S, Index, Function
from .utils import to_tensor_dof
from fealpy.decorator import barycentric, cartesian


class TensorFunctionSpace(FunctionSpace):
    def __init__(self, scalar_space: FunctionSpace, shape: Tuple[int, ...]) -> None:
        """_summary_

        Parameters:
            scalar_space (FunctionSpace): The scalar space to build tensor space from.\n
            shape (int, ...): Shape of each dof.
                Requires a `-1` be the first or last element to mark the priority
                of the DoF in arrangement.
        """
        self.scalar_space = scalar_space
        self.shape = shape

        if len(shape) < 2:
            raise ValueError('shape must be a tuple of at least two element')

        if shape[0] == -1:
            self.dof_shape = tuple(shape[1:])
            self.dof_priority = False
        elif shape[-1] == -1:
            self.dof_shape = tuple(shape[:-1])
            self.dof_priority = True
        else:
            raise ValueError('`-1` is required as the first or last element')
        
        self.p = self.scalar_space.p

    @property
    def mesh(self):
        return self.scalar_space.mesh

    @property
    def device(self): return self.scalar_space.device
    @property
    def ftype(self): return self.scalar_space.ftype
    @property
    def itype(self): return self.scalar_space.itype

    @property
    def dof_numel(self) -> int:
        return prod(self.dof_shape)

    @property
    def dof_ndim(self) -> int:
        return len(self.dof_shape)

    def number_of_global_dofs(self) -> int:
        return self.dof_numel * self.scalar_space.number_of_global_dofs()

    def number_of_local_dofs(self, doftype='cell') -> int:
        return self.dof_numel * self.scalar_space.number_of_local_dofs(doftype)

    def basis(self, p: TensorLike, index: Index=_S, **kwargs) -> TensorLike:
        phi = self.scalar_space.basis(p, index, **kwargs) # (NC, NQ, ldof)
        return generate_tensor_basis(phi, self.dof_shape, self.dof_priority)
    
    def face_basis(self, p: TensorLike, index: Index=_S, **kwargs) -> TensorLike:
        phi = self.scalar_space.face_basis(p, index, **kwargs)
        return generate_tensor_basis(phi, self.dof_shape, self.dof_priority)


    def grad_basis(self, p: TensorLike, index: Index=_S, **kwargs) -> TensorLike:
        gphi = self.scalar_space.grad_basis(p, index, **kwargs)
        return generate_tensor_grad_basis(gphi, self.dof_shape, self.dof_priority)
    
     
    @barycentric
    def cell_basis_on_face(self, bc: TensorLike, eindex: TensorLike) -> TensorLike:
        result = self.scalar_space.cell_basis_on_face(bc, eindex)
        return generate_tensor_basis(result, self.dof_shape, self.dof_priority)
    
    @barycentric
    def cell_grad_basis_on_face(self, bc: TensorLike, eindex: TensorLike, 
                                isleft = True) -> TensorLike:
        result = self.scalar_space.cell_grad_basis_on_face(bc, eindex, isleft) 
        return generate_tensor_grad_basis(result, self.dof_shape, self.dof_priority)


    def cell_to_dof(self, index: Index=_S) -> TensorLike:
        """Get the cell to dof mapping.

        Returns:
            Tensor: Cell to dof mapping, shaped (NC, ldof*dof_numel).
        """
        return to_tensor_dof(
            self.scalar_space.cell_to_dof(),
            self.dof_numel,
            self.scalar_space.number_of_global_dofs(),
            self.dof_priority
        )[index]

    def face_to_dof(self, index: Index=_S) -> TensorLike:
        """Get the face to dof mapping.

        Returns:
            Tensor: Face to dof mapping, shaped (NF, ldof*dof_numel).
        """
        return to_tensor_dof(
            self.scalar_space.face_to_dof(),
            self.dof_numel,
            self.scalar_space.number_of_global_dofs(),
            self.dof_priority
        )[index]
    
    def edge_to_dof(self, index: Index=_S) -> TensorLike:
        """Get the edge to dof mapping.

        Returns:
            Tensor: Edge to dof mapping, shaped (NE, ldof*dof_numel).
        """
        return to_tensor_dof(
            self.scalar_space.edge_to_dof(),
            self.dof_numel,
            self.scalar_space.number_of_global_dofs(),
            self.dof_priority
        )[index]
    
    def entity_to_dof(self, etype: int, index: Index=_S):
        TD = self.mesh.top_dimension()
        if etype == TD:
            return self.cell_to_dof(index=index)
        elif etype == TD-1:
            return self.face_to_dof(index=index)
        elif etype == 1:
            return self.edge_to_dof(index=index)
        else:
            raise ValueError(f"Unknown entity type: {etype}")

    def interpolation_points(self) -> TensorLike:

        return self.scalar_space.interpolation_points()
    
    def interpolate(self, u: Union[Callable[..., TensorLike], TensorLike], ) -> TensorLike:

        if self.dof_priority:
            uI = self.scalar_space.interpolate(u)
            ndim = len(self.shape)
            uI = bm.swapaxes(uI, ndim-1, ndim-2) 
        else:
            uI = self.scalar_space.interpolate(u)   

        return self.function(uI.reshape(-1))
    
    def is_boundary_dof(self, threshold=None, method='interp') -> TensorLike:
        scalar_space = self.scalar_space

        scalar_gdof = scalar_space.number_of_global_dofs()
        if bm.is_tensor(threshold):
            index = threshold
            if (index.dtype == bm.bool) and (len(index) == self.number_of_global_dofs()):
                return index
            else:
                raise ValueError("len(threshold) must equal tensorspace gdof")
        elif (threshold is None) | (callable(threshold)) :
            scalar_is_bd_dof = scalar_space.is_boundary_dof(threshold, method=method)

            if self.dof_priority:
                is_bd_dof = bm.reshape(scalar_is_bd_dof, (-1,) * self.dof_ndim + (scalar_gdof,))
                is_bd_dof = bm.broadcast_to(is_bd_dof, self.dof_shape + (scalar_gdof,))
            else:
                is_bd_dof = bm.reshape(scalar_is_bd_dof, (scalar_gdof,) + (-1,) * self.dof_ndim)
                is_bd_dof = bm.broadcast_to(is_bd_dof, (scalar_gdof,) + self.dof_shape)
        elif isinstance(threshold, tuple):
            ### 只处理了向量型的tensorspace空间
            assert self.dof_numel == len(threshold)
            scalar_is_bd_dof = [scalar_space.is_boundary_dof(i, method=method) for i in threshold] 
            if self.dof_priority:
                return bm.concatenate(scalar_is_bd_dof)
            else:
                return bm.stack(scalar_is_bd_dof, axis=1).flatten()                        
        else:
            raise ValueError(f"Unknown type of threshold {type(threshold)}")
        return is_bd_dof.reshape(-1)   
   
    def boundary_interpolate(self,
        gd: Union[Callable, int, float, TensorLike],
        uh: Optional[TensorLike]=None,
        *,
        threshold: Union[Callable, TensorLike, None]=None, method=None) -> TensorLike:
        ipoints = self.interpolation_points()
        scalar_space = self.scalar_space
        if uh is None:
            uh = self.function()
        # 根据不同gd类型进行判断
        if isinstance(gd, (int, float)):
            if bm.is_tensor(threshold):
                assert len(threshold) == self.number_of_global_dofs()
                isTensorBDof = threshold
            else : #threshold callable None
                isTensorBDof = self.is_boundary_dof(threshold=threshold, method=method)
            uh[threshold] = gd
            return uh, isTensorBDof

        elif (bm.is_tensor(gd)) or (isinstance(gd, Function)):
            assert len(gd[:]) == self.number_of_global_dofs()
            if bm.is_tensor(threshold):
                assert len(threshold) == self.number_of_global_dofs()
                isTensorBDof = threshold
            else : #threshold callable None
                isTensorBDof = self.is_boundary_dof(threshold=threshold, method=method)
            uh[isTensorBDof] = gd[isTensorBDof]
            return uh, isTensorBDof

        elif callable(gd):
            if bm.is_tensor(threshold):
                assert len(threshold) == self.number_of_global_dofs()
                gd_tensor = gd(ipoints)
                assert gd_tensor.shape[-1] == self.dof_numel
                isTensorBDof = threshold

            elif (threshold is None) | (callable(threshold)):
                isScalarBDof = scalar_space.is_boundary_dof(threshold=threshold, method=method)
                gd_tensor = gd(ipoints[isScalarBDof])
                assert gd_tensor.shape[-1] == self.dof_numel
                isTensorBDof = self.is_boundary_dof(threshold = threshold, method=method)

            elif isinstance(threshold, tuple):
                assert len(threshold) == self.dof_numel
                isScalarBDof = [scalar_space.is_boundary_dof(i, method=method) for i in threshold]
                gd_tensor = [gd(ipoints[isScalarBDof[i]])[..., i] for i in range(self.dof_numel)]
                isTensorBDof = self.is_boundary_dof(threshold=threshold, method=method)
                if self.dof_priority:
                    gd_tensor = bm.concatenate(gd_tensor)
                else:
                    # ! 注意这里的效率没有 dof_priority 的高
                    scalar_gdof = scalar_space.number_of_global_dofs()
                    full_values = bm.zeros((scalar_gdof, self.dof_numel), 
                                            dtype=bm.float64, device=self.device)
                    
                    for i in range(self.dof_numel):
                        indices = bm.where(isScalarBDof[i])[0]
                        full_values = bm.set_at(full_values, (indices, i), gd_tensor[i])
                    
                    gd_tensor = full_values.reshape(-1)
                    gd_tensor = gd_tensor[isTensorBDof]

                    # gd_tensor = bm.stack(gd_tensor, axis=1).flatten()
                    # scalar_gdof = scalar_space.number_of_global_dofs()
                    # # 使用列表推导式重新排列数据
                    # gd_values = []
                    # for i in range(scalar_gdof):
                    #     for j in range(self.dof_numel):
                    #         # 从 gd_tensor[j] 中获取第 i 个标量基函数的值
                    #         gd_values.append(gd_tensor[j][i] if i < len(gd_tensor[j]) else 0.0)
                    # gd_tensor = bm.array(gd_values, device=self.device)
                    # gd_tensor = gd_tensor[isTensorBDof]

                uh[:] = bm.set_at(uh[:], isTensorBDof, gd_tensor)
                return uh, isTensorBDof
            else:
                raise ValueError(f"Unknown type of threshold {type(threshold)}")
        else:
            raise ValueError(f"Unknown type of gd {type(gd)}")
        if self.dof_priority:
            uh[:] = bm.set_at(uh[:], isTensorBDof, gd_tensor.T.reshape(-1))
        else:
            uh[:] = bm.set_at(uh[:], isTensorBDof, gd_tensor.reshape(-1))

        return uh, isTensorBDof

    
    @barycentric
    def value(self, uh: TensorLike, bc: TensorLike, index: Index=_S) -> TensorLike:
        if isinstance(bc, tuple):
            TD = len(bc)
        else :
            TD = bc.shape[-1] - 1
        phi = self.basis(bc, index=index)
        e2dof = self.entity_to_dof(TD, index=index)
        val = bm.einsum('cql..., cl... -> cq...', phi, uh[e2dof, ...])
        return val
    
    @barycentric
    def grad_value(self, uh: TensorLike, bc: TensorLike, index: Index=_S) -> TensorLike:
        if isinstance(bc, tuple):
            TD = len(bc)
        else :
            TD = bc.shape[-1] - 1
        gphi = self.grad_basis(bc, index=index)
        e2dof = self.entity_to_dof(TD, index=index)
        val = bm.einsum('cqlmn..., cl... -> cqmn', gphi, uh[e2dof, ...])
        return val[...]

