
from typing import Tuple, Union, Callable
from math import prod

from ..backend import backend_manager as bm
from ..typing import TensorLike, Size, _S
from .functional import generate_tensor_basis, generate_tensor_grad_basis
from .space import FunctionSpace, _S, Index
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

    def grad_basis(self, p: TensorLike, index: Index=_S, **kwargs) -> TensorLike:
        gphi = self.scalar_space.grad_basis(p, index, **kwargs)
        return generate_tensor_grad_basis(gphi, self.dof_shape, self.dof_priority)

    def cell_to_dof(self) -> TensorLike:
        """Get the cell to dof mapping.

        Returns:
            Tensor: Cell to dof mapping, shaped (NC, ldof*dof_numel).
        """
        return to_tensor_dof(
            self.scalar_space.cell_to_dof(),
            self.dof_numel,
            self.scalar_space.number_of_global_dofs(),
            self.dof_priority
        )

    def face_to_dof(self) -> TensorLike:
        """Get the face to dof mapping.

        Returns:
            Tensor: Face to dof mapping, shaped (NF, ldof*dof_numel).
        """
        return to_tensor_dof(
            self.scalar_space.face_to_dof(),
            self.dof_numel,
            self.scalar_space.number_of_global_dofs(),
            self.dof_priority
        )

    def interpolation_points(self) -> TensorLike:

        return self.scalar_space.interpolation_points()
    
    def interpolate(self, u: Union[Callable[..., TensorLike], TensorLike], ) -> TensorLike:

        if self.dof_priority:
            uI = self.scalar_space.interpolate(u)
            ndim = len(self.shape)
            uI = bm.swapaxes(uI, ndim-1, ndim-2) 
        else:
            uI = self.scalar_space.interpolate(u)   

        return uI.reshape(-1)
    
    def is_boundary_dof(self, threshold=None) -> TensorLike:
        scalar_space = self.scalar_space
        mesh = self.mesh

        scalar_gdof = scalar_space.number_of_global_dofs()

        if threshold is None:
            scalar_is_bd_dof = scalar_space.is_boundary_dof(threshold)

            if self.dof_priority:
                is_bd_dof = bm.reshape(scalar_is_bd_dof, (-1,) * self.dof_ndim + (scalar_gdof,))
                is_bd_dof = bm.broadcast_to(is_bd_dof, self.dof_shape + (scalar_gdof,))
            else:
                is_bd_dof = bm.reshape(scalar_is_bd_dof, (scalar_gdof,) + (-1,) * self.dof_ndim)
                is_bd_dof = bm.broadcast_to(is_bd_dof, (scalar_gdof,) + self.dof_shape)
        else:
            if mesh.geo_dimension() == 2:
                edge_threshold = threshold[0]
                node_threshold = threshold[1] if len(threshold) > 1 else None
                dof_threshold = threshold[2] if len(threshold) > 2 else None

                scalar_is_bd_dof = scalar_space.is_boundary_dof(edge_threshold)

                if node_threshold is not None:
                    node_flags = node_threshold()
                    scalar_is_bd_dof = scalar_is_bd_dof & node_flags
                
                if self.dof_priority:
                    is_bd_dof = bm.reshape(scalar_is_bd_dof, (-1,) * self.dof_ndim + (scalar_gdof,))
                    is_bd_dof = bm.broadcast_to(is_bd_dof, self.dof_shape + (scalar_gdof,))
                else:
                    is_bd_dof = bm.reshape(scalar_is_bd_dof, (scalar_gdof,) + (-1,) * self.dof_ndim)
                    is_bd_dof = bm.broadcast_to(is_bd_dof, (scalar_gdof,) + self.dof_shape)

                if dof_threshold is not None:
                    dof_flags = dof_threshold()
                    is_bd_dof = is_bd_dof & dof_flags

            elif mesh.geo_dimension() == 3:
                face_threshold = threshold[0]
                edge_threshold = threshold[1] if len(threshold) > 2 else None
                node_threshold = threshold[2] if len(threshold) > 3 else None
                dof_threshold = threshold[3] if len(threshold) > 4 else None

                scalar_is_bd_dof = scalar_space.is_boundary_dof(face_threshold)
                    
                if edge_threshold is not None:
                    edge_flags = edge_threshold()
                    scalar_is_bd_dof = scalar_is_bd_dof & edge_flags

                if node_threshold is not None:
                    node_flags = node_threshold()
                    scalar_is_bd_dof = scalar_is_bd_dof & node_flags
                
                if self.dof_priority:
                    is_bd_dof = bm.reshape(scalar_is_bd_dof, (-1,) * self.dof_ndim + (scalar_gdof,))
                    is_bd_dof = bm.broadcast_to(is_bd_dof, self.dof_shape + (scalar_gdof,))
                else:
                    is_bd_dof = bm.reshape(scalar_is_bd_dof, (scalar_gdof,) + (-1,) * self.dof_ndim)
                    is_bd_dof = bm.broadcast_to(is_bd_dof, (scalar_gdof,) + self.dof_shape)

                if dof_threshold is not None:
                    dof_flags = dof_threshold()
                    is_bd_dof = is_bd_dof & dof_flags

        return is_bd_dof.reshape(-1)

    
    def boundary_interpolate(self,
        gD: Union[Callable, int, float, TensorLike],
        uh: TensorLike,
        threshold: Union[Callable, TensorLike, None]=None) -> TensorLike:

        ipoints = self.interpolation_points()
        scalar_space = self.scalar_space
        mesh = self.mesh

        if threshold is None:
            isScalarBDof = scalar_space.is_boundary_dof(threshold=threshold)
            
            if callable(gD):
                gD_scalar = gD(ipoints[isScalarBDof])
            else:
                gD_scalar = gD

            gD_vector = gD_scalar

            isTensorBDof = self.is_boundary_dof(threshold=threshold)
        else:
            if mesh.geo_dimension() == 2:
                edge_threshold = threshold[0]
                node_threshold = threshold[1] if len(threshold) > 1 else None
                dof_threshold = threshold[2] if len(threshold) > 2 else None

                isScalarBDof = scalar_space.is_boundary_dof(threshold=edge_threshold)

                if node_threshold is not None:
                    node_flags = node_threshold()
                    isScalarBDof = isScalarBDof & node_flags

                if callable(gD):
                    gD_scalar = gD(ipoints[isScalarBDof])
                else:
                    gD_scalar = gD

                if dof_threshold is not None:
                    dof_flags = dof_threshold()
                    node_dof_flags = dof_flags[node_flags] 
                    gD_vector = gD_scalar[node_dof_flags] 
                else:
                    gD_vector = gD_scalar

                isTensorBDof = self.is_boundary_dof(threshold=(edge_threshold, 
                                                            node_threshold, 
                                                            dof_threshold))
            elif mesh.geo_dimension() == 3:
                face_threshold = threshold[0]
                edge_threshold = threshold[1] if len(threshold) > 1 else None
                node_threshold = threshold[2] if len(threshold) > 2 else None
                dof_threshold = threshold[3] if len(threshold) > 3 else None

                isScalarBDof = scalar_space.is_boundary_dof(threshold=face_threshold)

                if edge_threshold is not None:
                    edge_flags = edge_threshold()
                    isScalarBDof = isScalarBDof & edge_flags

                if node_threshold is not None:
                    node_flags = node_threshold()
                    isScalarBDof = isScalarBDof & node_flags

                if callable(gD):
                    gD_scalar = gD(ipoints[isScalarBDof])
                else:
                    gD_scalar = gD

                if dof_threshold is not None:
                    dof_flags = dof_threshold()
                    node_dof_flags = dof_flags[node_flags] 
                    gD_vector = gD_scalar[node_dof_flags] 
                else:
                    gD_vector = gD_scalar

                isTensorBDof = self.is_boundary_dof(threshold=(face_threshold,
                                                            edge_threshold, 
                                                            node_threshold, 
                                                            dof_threshold))

        if self.dof_priority:
            uh = bm.set_at(uh, isTensorBDof, gD_vector.T.reshape(-1))
        else:
            uh = bm.set_at(uh, isTensorBDof, gD_vector.reshape(-1))

        return uh, isTensorBDof

    @barycentric
    def value(self, uh: TensorLike, bc: TensorLike, index: Index=_S) -> TensorLike:
        phi = self.basis(bc, index=index)
        c2dof = self.cell_to_dof()[index]
        val = bm.einsum('cql..., cl... -> cq...', phi, uh[c2dof, ...])
        return val
    
    @barycentric
    def grad_value(self, uh: TensorLike, bc: TensorLike, index: Index=_S) -> TensorLike:
        gphi = self.grad_basis(bc, index=index)
        cell2dof = self.cell_to_dof()[index]
        val = bm.einsum('cqlmn..., cl... -> cqmn', gphi, uh[cell2dof, ...])
        return val[...]
