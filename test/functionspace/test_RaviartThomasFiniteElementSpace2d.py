
import numpy as np
import pytest

from fealpy.functionspace.RaviartThomasFiniteElementSpace2d import RTDof2d
from fealpy.functionspace.RaviartThomasFiniteElementSpace2d import RTFiniteElementSpace2d
from fealpy.backend import backend_manager as bm
from fealpy.mesh.triangle_mesh import TriangleMesh

from RaviartThomasFiniteElementSpace2d_data import *

class TestFirstNedelecDof2d:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_cell_to_dof(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  3
        a = RTDof2d(mesh,p)
        assert a.number_of_local_dofs("edge") == meshdata["edof"]
        assert a.number_of_local_dofs("cell") == meshdata["cdof"]
        assert a.number_of_global_dofs() == meshdata["gdof"]
        
        cell2dof = a.cell2dof
        np.testing.assert_array_equal(bm.to_numpy(cell2dof), meshdata["cell2dof"])

class TestRTFiniteElementSpace2d:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_basis(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  2
        a = RTFiniteElementSpace2d(mesh,p)
        bc = bm.array([[0.1,0.2,0.7],
                      [0.5,0.3,0.2]],dtype=bm.float64)
        basis = a.basis(bc)
        np.testing.assert_allclose(bm.to_numpy(basis), meshdata["basis"],1e-5)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_div_basis(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  2
        a = RTFiniteElementSpace2d(mesh,p)
        bc = bm.array([[0.1,0.2,0.7],
                      [0.5,0.3,0.2]],dtype=bm.float64)
        div_basis = a.div_basis(bc)
        np.testing.assert_allclose(bm.to_numpy(div_basis), meshdata["div_basis"],1e-15)


    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_edge_basis(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  2
        a = RTFiniteElementSpace2d(mesh,p)
        bc = bm.array([[0.1,0.2,0.7],
                      [0.5,0.3,0.2]],dtype=bm.float64)
        edge_basis = a.edge_basis(bc)
        np.testing.assert_allclose(bm.to_numpy(edge_basis), meshdata["edge_basis"],1e-15)


    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_edge_basis(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  2
        a = RTFiniteElementSpace2d(mesh,p)
        bc = bm.array([[0.1,0.2,0.7],
                      [0.5,0.3,0.2]],dtype=bm.float64)
        mass_matrix = a.mass_matrix()
        np.testing.assert_allclose(bm.to_numpy(mass_matrix), meshdata["edge_basis"],1e-15)

if __name__ == "__main__":
    # test = TestFirstNedelecDof2d()
    # test.test_cell_to_dof(init_data[0],'pytorch')
    test = TestRTFiniteElementSpace2d()
    # test.test_basis(init_data[0],'pytorch')
    # test.test_div_basis(init_data[0],'pytorch')
    test.test_edge_basis(init_data[0],'numpy')
