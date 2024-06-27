from typing import (
    Union, Optional, Dict, Tuple, Sequence, overload, Callable,
    Literal, TypeVar
)

from math import factorial, comb

import numpy as np
import taichi as ti


from .. import logger
from .. import numpy as tnp
from ..sparse import CSRMatrix

from .utils import Entity, Field

def mesh_top_csr(entity: Entity, shape: Tuple[int, int], copy=False) -> CSRMatrix:
    if entity.ndim == 1:
        if ~hasattr(entity, 'location'):
             raise ValueError('entity.location is required for 1D entity (usually for polygon mesh).')
        indices = entity
        indptr = entity.location
    elif entity.ndim == 2: # for homogeneous case
        M = entity.shape[0]
        N = entity.shape[1]
        indices = ti.field(entity.dtype, shape=(M*N, ))
        indptr = ti.field(entity.dtype, shape=(M+1, ))

        @ti.kernel
        def process_entity():
            for i, j in ti.ndrange(M, N):
                indices[i*N + j] = entity[i, j]
            for i in range(M+1):
                indptr[i] = i*N

        process_entity()
    else:
        raise ValueError('dimension of entity must be 1 or 2.')
    # Notice that the data can be None for CSRMatrix, which intend to save memory
    return CSRMatrix((None, entity, entity.location), shape=shape, copy=copy) 


def multi_index_matrix(p: int, edim: int) -> Field:
    """
    TODO:
    """
    if edim == 3:
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
    elif edim == 2:
        ldof = (p+1)*(p+2)//2
        idx = np.arange(0, ldof)
        idx0 = np.floor((-1 + np.sqrt(1 + 8*idx))/2)
        multiIndex = np.zeros((ldof, 3), dtype=np.int_)
        multiIndex[:,2] = idx - idx0*(idx0 + 1)/2
        multiIndex[:,1] = idx0 - multiIndex[:,2]
        multiIndex[:,0] = p - multiIndex[:, 1] - multiIndex[:, 2]
        return multiIndex
    elif edim == 1:
        ldof = p+1
        multiIndex = np.zeros((ldof, 2), dtype=np.int_)
        multiIndex[:, 0] = np.arange(p, -1, -1)
        multiIndex[:, 1] = p - multiIndex[:, 0]
        return multiIndex

    return tnp.field(multiIndex)


def entity_barycenter(entity: Entity, node: Entity):
    """
    """
    N = entity.shape[0]
    n = entity.shape[1]
    GD = node.shape[1]

    bc = ti.field(dtype=node.dtype, shape=(N, GD)) 
    bc.fill(0.0)

    @ti.kernel
    def compute_barycenter():
        for i in range(N):
            for d in range(GD):
                for j in range(n):
                    bc[i, d] += node[entity[i, j], d] 
                bc[i, d] /= n  
    compute_barycenter()
    return bc

def simplex_ldof(p: int, dim: int) -> int:
    if dim == 0:
        return 1
    return comb(p + dim, dim)

def simplex_gdof(p: int, mesh) -> int:
    coef = 1
    count = mesh.node.shape[0]
    for i in range(1, mesh.TD+1):
        coef = (coef * (p - i)) //i
        count += coef * mesh.entity(i).shape[0]

    return count
