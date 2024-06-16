from typing import (
    Union, Optional, Dict, Sequence, overload, Callable,
    Literal, TypeVar
)

import numpy as np
import taichi as ti


from .. import logger
from ..sparse import CSRMatrix

from .utils import Entity

def mesh_top_csr(entity: Entity, shape: Tuple(int, int), copy=False) -> CSRMatrix:
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

    

