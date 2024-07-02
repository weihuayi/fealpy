
from typing import Optional, Dict

import torch

from .. import logger

Tensor = torch.Tensor
_dtype = torch.dtype


##################################################
### Utils
##################################################

def mesh_top_csr(entity: Tensor, num_targets: int, location: Optional[Tensor]=None, *,
                 dtype: Optional[_dtype]=None) -> Tensor:
    r"""CSR format of a mesh topology relaionship matrix."""
    device = entity.device

    if entity.ndim == 1: # for polygon case
        if location is None:
            raise ValueError('location is required for 1D entity (usually for polygon mesh).')
        crow = location
    elif entity.ndim == 2: # for homogeneous case
        crow = torch.arange(
            entity.size(0) + 1, dtype=entity.dtype, device=device
        ).mul_(entity.size(1))
    else:
        raise ValueError('dimension of entity must be 1 or 2.')

    return torch.sparse_csr_tensor(
        crow,
        entity.reshape(-1),
        torch.ones(entity.numel(), dtype=dtype, device=device),
        size=(entity.size(0), num_targets),
        dtype=dtype, device=device
    )


def estr2dim(mesh, estr: str) -> int:
    if estr == 'cell':
        return mesh.top_dimension()
    elif estr == 'face':
        TD = mesh.top_dimension()
        return TD - 1
    elif estr == 'edge':
        return 1
    elif estr == 'node':
        return 0
    else:
        raise KeyError(f'{estr} is not a valid entity name in FEALPy.')


def edim2entity(storage: Dict, factory: Dict, edim: int, index=None):
    r"""Get entity tensor by its top dimension. Returns None if not found."""
    if edim in storage:
        et = storage[edim]
    else:
        if edim in factory:
            et = factory[edim]()
            storage[edim] = et
        else:
            logger.info(f'entity with top-dimension {edim} is not in the storage,'
                        'and no factory is assigned for it,'
                        'therefore a NoneType is returned.')
            return None

    if index is None:
        return et
    else: # TODO: finish this for homogeneous mesh
        return et[index]


def edim2node(mesh, etype_dim: int, index=None, dtype=None) -> Tensor:
    r"""Get the <entiry>_to_node sparse matrix by entity's top dimension."""
    entity = edim2entity(mesh.storage(), mesh._entity_dim_method_map,
                         etype_dim, index)
    location = getattr(entity, 'location', None)
    NN = mesh.count('node')
    if NN <= 0:
        raise RuntimeError('No valid node is found in the mesh.')
    return mesh_top_csr(entity, NN, location, dtype=dtype)
