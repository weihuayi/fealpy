import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh import UniformMesh2d

#bm.set_backend('numpy')
#bm.set_backend('pytorch')
bm.set_backend('jax')

def test_uniform_mesh_2d_init():
    nelx, nely = 2, 2
    domain = [0, 2, 0, 2]
    hx = (domain[1] - domain[0]) / nelx
    hy = (domain[3] - domain[2]) / nely
    mesh = UniformMesh2d(extent=(0, nelx, 0, nely), h=(hx, hy), origin=(domain[0], domain[2]))

    GD = 2
    assert mesh.node.shape == (nelx+1, nely+1, GD)
    assert mesh.edge.shape == (12, GD)
    assert mesh.face.shape == (12, GD)
    assert mesh.cell.shape == (nelx*nely, 4)

    assert mesh.number_of_nodes() == (nelx+1) * (nely+1)
    assert mesh.number_of_edges() == 12
    assert mesh.number_of_faces() == 12
    assert mesh.number_of_cells() == nelx * nely