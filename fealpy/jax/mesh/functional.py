
from typing import Optional, Sequence, Union
from itertools import combinations_with_replacement
from functools import reduce
from math import factorial, comb

import numpy as np
import jax
import jax.numpy as jnp

Array = jax.Array


##################################################
### Mesh
##################################################

def multi_index_matrix(p: int, etype: int, *, dtype=None) -> Array:
    r"""Create a multi-index matrix."""
    dtype = dtype or jnp.int_
    sep = np.flip(np.array(
        tuple(combinations_with_replacement(range(p+1), etype)),
        dtype=np.int_
    ), axis=0)
    raw = np.zeros((sep.shape[0], etype+2), dtype=np.int_)
    raw[:, -1] = p
    raw[:, 1:-1] = sep
    return jnp.array(raw[:, 1:] - raw[:, :-1])


##################################################
### Homogeneous Mesh
##################################################

def bc_tensor(bcs: Sequence[Array]):
    num = len(bcs)
    NVC = reduce(lambda x, y: x * y.shape[-1], bcs, 1)
    desp1 = 'mnopq'
    desp2 = 'abcde'
    string = ", ".join([desp1[i]+desp2[i] for i in range(num)])
    string += " -> " + desp1[:num] + desp2[:num]
    return jnp.einsum(string, *bcs).reshape(-1, NVC)


def bc_to_points(bcs: Union[Array, Sequence[Array]], node: Array,
                 entity: Array, order: Optional[Array]) -> Array:
    r"""Barycentric coordinates to cartesian coordinates in homogeneous meshes."""
    if order is not None:
        entity = entity[:, order]
    points = node[entity, :]

    if not isinstance(bcs, Array):
        bcs = bc_tensor(bcs)
    return jnp.einsum('ijk, ...j -> ...ik', points, bcs)


def homo_entity_barycenter(entity: Array, node: Array):
    r"""Entity barycenter in homogeneous meshes."""
    return jnp.mean(node[entity, :], axis=1)


# Triangle Mesh & Tetrahedron Mesh
# ================================

def simplex_ldof(p: int, iptype: int) -> int:
    r"""Number of local DoFs of a simplex."""
    if iptype == 0:
        return 1
    return comb(p + iptype, iptype)


def simplex_gdof(p: int, mesh) -> int:
    r"""Number of global DoFs of a mesh with simplex cells."""
    coef = 1
    count = mesh.node.shape[0]

    for i in range(1, mesh.TD + 1):
        coef = (coef * (p-i)) // i
        count += coef * mesh.entity(i).shape[0]
    return count


# Quadrangle Mesh & Hexahedron Mesh
# =================================
