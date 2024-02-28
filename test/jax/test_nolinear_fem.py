
import pytest

import jax.numpy as jnp

from fealpy.mesh import TriangleMesh as Mesh
from fealpy.jax.mesh import TriangleMesh as TriangleMesh
from fealpy.jax.functionspace import LagrangeFESpace
from fealpy.jax import logger


class PDE():
    def solution(self, p):
        # 真解函数
        pi = jnp.pi
        x = jnp[..., 0]
        y = jnp[..., 1]
        return jnp.cos(pi*x)*jnp.cos(pi*y)

    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = jnp.pi
        val = np.vstack([-pi*np.sin(pi*x)*np.cos(pi*y), -pi*np.cos(pi*x)*np.sin(pi*y)]).T
        return val #val.shape ==p.shape

    def source(self, p):
        # 源项
        x = p[..., 0]
        y = p[..., 1]
        pi = jnp.pi
        val = 2*pi**2*(3*jnp.cos(pi*x)**2*jnp.cos(pi*y)**2-jnp.cos(pi*x)**2-jnp.cos(pi*y)**2+1)*jnp.cos(pi*x)*jnp.cos(pi*y)
        return val


def test_nolinear_fem():

    mesh = Mesh.from_box(nx=1, ny=1)
    node = jnp.array(mesh.entity('node'), dtype=jnp.float64)
    cell = jnp.array(mesh.entity('cell'), dtype=jnp.int32)

    mesh = TriangleMesh(node, cell)
    NN = mesh.number_of_nodes()
    NC = mesh.number_of_cells()
    space = LagrangeFESpace(mesh, p=1)

    uh = jnp.zeros(NN, dtype=jnp.float64)

    qf = mesh.integrator(2)
    bcs, ws = qf.get_quadrature_points_and_weights()
    print(bcs)
    print(ws)

    phi = space.basis(bcs)
    gphi = space.grad_basis(bcs)

    print('phi:\n', phi.shape)
    print('gphi:\n', gphi)


if __name__ == "__main__":
    test_nolinear_fem()
