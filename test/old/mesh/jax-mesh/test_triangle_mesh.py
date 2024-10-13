
import jax.numpy as jnp
import matplotlib.pyplot as plt
import ipdb
import pytest
from fealpy.mesh.jax import TriangleMesh  


def test_triangle_mesh_init():
    node = jnp.array([[0, 0], [1, 0], [0, 1]], dtype=jnp.float64)
    cell = jnp.array([[0, 1, 2]], dtype=jnp.uint64)

    mesh = TriangleMesh(node, cell)

    assert mesh.node.shape == (3, 2)

if __name__ == "__main__":
    test_triangle_mesh_init()
