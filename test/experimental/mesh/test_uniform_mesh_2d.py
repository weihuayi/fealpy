import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.mesh import UniformMesh2d

bm.set_backend('numpy')
#bm.set_backend('pytorch')
#bm.set_backend('jax')

def test_uniform_mesh_2d_init():
    nelx, nely = 2, 2
    domain = [0, 2, 0, 2]
    hx = (domain[1] - domain[0]) / nelx
    hy = (domain[3] - domain[2]) / nely
    mesh = UniformMesh2d(extent=(0, nelx, 0, nely), h=(hx, hy), origin=(domain[0], domain[2]))

    GD = 2
    assert mesh.node.shape == (nelx+1, nely+1, GD)
    assert mesh.edge.shape == (nelx*(nely+1) + nely*(nelx+1), GD)
    assert mesh.face.shape == (nelx*(nely+1) + nely*(nelx+1), GD)
    assert mesh.cell.shape == (nelx*nely, 4)

    assert mesh.number_of_nodes() == (nelx+1) * (nely+1)
    assert mesh.number_of_edges() == nelx*(nely+1) + nely*(nelx+1)
    assert mesh.number_of_faces() == nelx*(nely+1) + nely*(nelx+1)
    assert mesh.number_of_cells() == nelx * nely

    assert mesh.entity_measure('edge') == (hx, hy)
    assert mesh.entity_measure('cell') == hx * hy

def test_uniform_mesh_2d_uniform_refine():
    nelx, nely = 2, 2
    domain = [0, 2, 0, 2]
    hx = (domain[1] - domain[0]) / nelx
    hy = (domain[3] - domain[2]) / nely
    mesh = UniformMesh2d(extent=(0, nelx, 0, nely), h=(hx, hy), origin=(domain[0], domain[2]))
    mesh.uniform_refine(n=1)

    GD = 2
    assert mesh.node.shape == (2*nelx+1, 2*nely+1, GD)
    assert mesh.edge.shape == (2*nelx*(2*nely+1) + 2*nely*(2*nelx+1), GD)
    assert mesh.face.shape == (2*nelx*(2*nely+1) + 2*nely*(2*nelx+1), GD)
    assert mesh.cell.shape == (2*nelx*2*nely, 4)

    assert mesh.number_of_nodes() == (2*nelx+1) * (2*nely+1)
    assert mesh.number_of_edges() == 2*nelx*(2*nely+1) + 2*nely*(2*nelx+1)
    assert mesh.number_of_faces() == 2*nelx*(2*nely+1) + 2*nely*(2*nelx+1)
    assert mesh.number_of_cells() == 2*nelx * 2*nely

if __name__ == "__main__":
    test_uniform_mesh_2d_init()
    test_uniform_mesh_2d_uniform_refine()