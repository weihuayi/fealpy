import numpy as np
import ipdb
import pytest
import matplotlib.pyplot as plt

from fealpy.mesh.hexahedron_mesh import HexahedronMesh

def test_hexahedrom_mesh_measure():
    mesh = HexahedronMesh.from_one_tetrahedron()
    mesh.uniform_refine(2)
    cell = mesh.entity('cell')
    face = mesh.entity('face')
    node = mesh.entity('node')
    edge = mesh.entity('edge')
    vol = mesh.entity_measure('cell')
    assert np.abs(sum(vol) - np.sqrt(2)/12) < 1e-13

    area = mesh.entity_measure('face')
    isBdFace = mesh.ds.boundary_face_flag()

    assert np.abs(np.sum(area[isBdFace]) - np.sqrt(3)) < 1e-13

def test_hexahedrom_mesh_interpolation(p):
    from fealpy.decorator import cartesian, barycentric
    mesh = HexahedronMesh.from_box([0, 1, 0, 1, 0, 1], nx=2, ny=2, nz=2)
    cell = mesh.entity('cell')
    node = mesh.entity('node')

    @cartesian
    def u(ps):
        x = ps[..., 0]
        y = ps[..., 1]
        z = ps[..., 2]
        return x*y*z

    ips = mesh.interpolation_points(p)
    cell2dof = mesh.cell_to_ipoint(p)
    uI = u(ips)

    @barycentric
    def uh(bcs):
        phi = mesh.shape_function(bcs, p=p)
        val = np.einsum('qi, ci->qc', phi, uI[cell2dof])
        return val

    e = mesh.error(u, uh, q=3)
    print(e)


if __name__ == "__main__":
    test_hexahedrom_mesh_interpolation(1)
