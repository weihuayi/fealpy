
from typing import Union, Any, TypeVar, Generic, Optional, Dict, Type, Literal

import torch
from torch import Tensor, float64

from ..nntyping import S as S_


class PoU():
    """
    @brief Base class for PoU. Provides linear mapping between local and global.\
           The mapping is **independent** in each geometry dimension.
    """
    def __init__(self, dtype, device, **kwargs) -> None:
        self.dtype = dtype
        self.device = device

    def number_of_partitions(self) -> int:
        """Number of partitions."""
        raise NotImplementedError

    def global_to_local(self, p: Tensor, index=S_) -> Tensor:
        """
        @brief Mapping from global to local. Return tensor with shape\
               (#Samples, #Partitions, #Dims)
        """
        raise NotImplementedError

    def local_to_global(self, p: Tensor, index=S_) -> Tensor:
        """Mapping from local to global."""
        raise NotImplementedError

    # NOTE: PoUs are designed to perform linear transform to every partitions.
    # So, gradients are relevant to only partitions and dims.
    def grad_global_to_local(self, index=S_) -> Tensor:
        """
        @brief Gradient of mapping from global to local. Return tensor with shape\
               (#partitions, #dims).
        """
        raise NotImplementedError

    def grad_local_to_global(self, index=S_) -> Tensor:
        """Gradient of mapping from local to global."""
        raise NotImplementedError

    __call__ = global_to_local

    def grad_transform(self, trans: Optional[Tensor]=None):
        trans_new = self.grad_global_to_local()
        if trans is None:
            return trans_new
        else:
            return trans_new @ trans


class CRPoU(PoU):
    def __init__(self, centers: Tensor, radius: Union[Tensor, Any], device, **kwargs) -> None:
        super().__init__(dtype=centers.dtype, device=device)
        self.centers = centers
        if not isinstance(radius, Tensor):
            rdata = torch.tensor(radius, dtype=self.dtype, device=device)
        else:
            rdata = radius.to(device=device).to(dtype=self.dtype)
        self.radius = rdata.expand(centers.shape)

    def global_to_local(self, p: Tensor, index=S_) -> Tensor:
        return (p[:, None, :] - self.centers[None, index, :]) / self.radius[None, index, :]

    def local_to_global(self, p: Tensor, index=S_) -> Tensor:
        return p[:, None, :] * self.radius[None, index, :] + self.centers[None, index, :]

    def grad_global_to_local(self, index=S_) -> Tensor:
        return 1 / self.radius[index, :] # (part, dim)

    def grad_local_to_global(self, index=S_) -> Tensor:
        return self.radius[index, :] # (part, dim)


##################################################
### PoU from Mesh
##################################################

_MT = TypeVar('_MT')
_pou_library: Dict[str, Type["MeshPoU"]] = {}


class MeshPoU(PoU, Generic[_MT]):
    def __new__(cls, mesh: _MT, **kwargs):
        key = mesh.__class__.__name__
        _pou_class = cls._get_pou_class(key)
        return super().__new__(_pou_class)

    @classmethod
    def _assigned(cls, mesh_name: str):
        if mesh_name in _pou_library:
            raise KeyError(f"Mesh '{mesh_name}' has already been assigned to "
                           f"PoU '{_pou_library[mesh_name].__name__}'.")
        _pou_library[mesh_name] = cls

    @classmethod
    def _get_pou_class(cls, mesh_name: str):
        if mesh_name in _pou_library:
            return _pou_library[mesh_name]
        else:
            raise KeyError("No PoU model found.")

    def __init__(self, mesh: _MT, dtype=float64, device=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mesh = mesh
        self.dtype = dtype
        self.device = device

    def sub_to_partition(self):
        """
        @brief Return the relationship between sub-boundaries and partitions.

        @return: `Tensor` with shape (#Subs, 2).
        """
        _bd_flag = self.mesh.ds.boundary_face_flag()
        data = self.mesh.ds.face_to_cell()[~_bd_flag, 0:2]
        return torch.from_numpy(data).to(device=self.device)


class _UniformPoU(MeshPoU):
    def __init__(self, mesh: Any, location: Literal['node', 'cell'],
                 dtype=float64, device=None, **kwargs) -> None:
        super().__init__(mesh, dtype, device, **kwargs)
        if location == 'cell':
            self.cell_ctrs = torch.tensor(
                self.mesh.entity_barycenter('cell'),
                dtype=dtype, device=device
            )
        elif location == 'node':
            self.cell_ctrs = torch.tensor(
                self.mesh.entity('node'),
                dtype=dtype, device=device
            )
        else:
            raise ValueError(f"Invalid location center '{location}'")
        self.cell_rds = torch.tensor(mesh.h, dtype=dtype, device=device) / 2
        self.cell_rds = self.cell_rds.broadcast_to(self.cell_ctrs.shape)
        self.mesh = mesh
        self.location = location

    def number_of_partitions(self) -> int:
        return self.mesh.ds.number_of_cells()

    def global_to_local(self, p: Tensor, index=S_) -> Tensor:
        return (p[:, None, :] - self.cell_ctrs[None, index, :]) / self.cell_rds[None, index, :]

    def local_to_global(self, p: Tensor, index=S_) -> Tensor:
        return p[:, None, :] * self.cell_rds[None, index, :] + self.cell_ctrs[None, index, :]

    def grad_global_to_local(self, index=S_) -> Tensor:
        return (1 / self.cell_rds[index, :])

    def grad_local_to_global(self, index=S_) -> Tensor:
        return self.cell_rds[index, :]

_UniformPoU._assigned('UniformMesh2d')
