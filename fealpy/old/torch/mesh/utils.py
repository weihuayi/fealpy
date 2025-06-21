
from typing import Optional, Dict, Callable, TypeVar, Tuple, Any

import torch

from .. import logger

Tensor = torch.Tensor
_dtype = torch.dtype
_Meth = TypeVar('_Meth', bound=Callable)


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
    entity = edim2entity(mesh.storage(), mesh._entity_factory,
                         etype_dim, index)
    location = getattr(entity, 'location', None)
    NN = mesh.count('node')
    if NN <= 0:
        raise RuntimeError('No valid node is found in the mesh.')
    return mesh_top_csr(entity, NN, location, dtype=dtype)


# NOTE: this meta class is used to register the entity factory method.
# The entity factory methods can works in Structured meshes such as
# UniformMesh2d to construct entities like `cell`.

# NOTE: When query a entity, the factory method is called if the entity
# is not found in the storage.
# The result from the factory method is cached in the storage automatically.
# Therefore, the storage is regarded as a cache for structured meshes.

# TODO: This feature does not hinder the unstructured mesh, but wee still need
# to see if it is an over-design or if there is a better way to do this.

class MeshMeta(type):
    def __init__(self, name: str, bases: Tuple[type, ...], dict: Dict[str, Any], /, **kwds: Any):
        if '_entity_dim_method_name_map' in dict:
            raise RuntimeError('_entity_method is a reserved attribute.')
        self._entity_dim_method_name_map = {}

        # NOTE: Look up the functions to build the class, seeing if there are
        # any functions having the `__entity__` attribute which is marked
        # by the entitymethod decorator.
        for name, item in dict.items():
            if callable(item):
                if hasattr(item, '__entity__'):
                    dim = getattr(item, '__entity__')
                    assert isinstance(dim, int)
                    self._entity_dim_method_name_map[dim] = item.__name__

        return type.__init__(self, name, bases, dict, **kwds)


def entitymethod(top_dim: int):
    """A decorator registering the method as an entity factory method.

    Requires that the metaclass is MeshMeta or derived from it.

    Parameters:
        top_dim (int): Topological dimension of the entity.
    """
    def decorator(meth: _Meth) -> _Meth:
        meth.__entity__ = top_dim
        return meth
    return decorator
