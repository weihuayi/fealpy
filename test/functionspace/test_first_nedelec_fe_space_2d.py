
import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.functionspace.first_nedelec_fe_space_2d import FirstNedelecDof2d
from fealpy.functionspace.first_nedelec_fe_space_2d import FirstNedelecFiniteElementSpace2d

from first_nedelec_fe_space_2d_data import *


def assert_absolute_error(a,b,atol=1e-10):
    diff = np.abs(a-b)
    if np.any(diff>atol):
        raise AssertionError


class TestFirstNedelecDof2d:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_cell_to_dof(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  4
        a = FirstNedelecDof2d(mesh,p)
        assert a.number_of_local_dofs("edge") == meshdata["edof"]
        assert a.number_of_local_dofs("cell") == meshdata["cdof"]
        assert a.number_of_global_dofs() == meshdata["gdof"]
        
        cell2dof = a.cell2dof
        np.testing.assert_array_equal(bm.to_numpy(cell2dof), meshdata["cell2dof"])

class TestFirstNedelecFiniteElementSpace2d:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_basis(self,meshdata,backend):
        bm.set_backend(backend)
        
        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  1
        a = FirstNedelecFiniteElementSpace2d(mesh=mesh,p=p)
        bcs = bm.tensor([[0.5,0.2,0.3],
                [0.7,0.2,0.1]],dtype=mesh.ftype)
        basis = a.basis(bcs=bcs)
        np.testing.assert_allclose(bm.to_numpy(basis), meshdata["basis"].swapaxes(0,1),1e-15)
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_curl_basis(self,meshdata,backend):
        bm.set_backend(backend)
        
        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  1
        a = FirstNedelecFiniteElementSpace2d(mesh=mesh,p=p)
        bcs = bm.tensor([[0.5,0.2,0.3],
                [0.7,0.2,0.1]],dtype=mesh.ftype)
        curl_basis = a.curl_basis(bcs=bcs)
        np.testing.assert_allclose(bm.to_numpy(curl_basis), meshdata["curl_basis"].swapaxes(0,1),1e-15)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_value(self,meshdata,backend):
        bm.set_backend(backend)
        
        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  1
        a = FirstNedelecFiniteElementSpace2d(mesh=mesh,p=p)
        bcs = bm.tensor([[0.5,0.2,0.3],
                [0.7,0.2,0.1]],dtype=mesh.ftype)
        uh = bm.arange(14,dtype=mesh.ftype)
        value = a.value(uh=uh,bcs=bcs)
        np.testing.assert_allclose(bm.to_numpy(value), meshdata["value"].swapaxes(0,1),1e-14)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_curl_value(self,meshdata,backend):
        bm.set_backend(backend)
        
        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  1
        a = FirstNedelecFiniteElementSpace2d(mesh=mesh,p=p)
        bcs = bm.tensor([[0.5,0.2,0.3],
                [0.7,0.2,0.1]],dtype=mesh.ftype)
        uh = bm.arange(14,dtype=mesh.ftype)
        curl_value = a.curl_value(uh=uh,bcs=bcs)
        np.testing.assert_allclose(bm.to_numpy(curl_value), meshdata["curl_value"].swapaxes(0,1),1e-14)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_mass_matrix(self,meshdata,backend):
        bm.set_backend(backend)
        
        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  1
        a = FirstNedelecFiniteElementSpace2d(mesh=mesh,p=p)
        mass_matrix = a.mass_matrix()
        # np.testing.assert_allclose((mass_matrix).toarray(), meshdata["mass_matrix"],1e-8)
        assert_absolute_error((mass_matrix).toarray(),meshdata["mass_matrix"])
           
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_curl_matrix(self,meshdata,backend):
        bm.set_backend(backend)
        
        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  1
        a = FirstNedelecFiniteElementSpace2d(mesh=mesh,p=p)
        curl_matrix = a.curl_matrix()
        # np.testing.assert_allclose(bm.to_numpy(curl_matrix).toarray(), meshdata["curl_matrix"],1e-8)
        assert_absolute_error((curl_matrix).toarray(),meshdata["curl_matrix"])
           

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_source_vector(self,meshdata,backend):
        bm.set_backend(backend)
        
        mesh = TriangleMesh.from_box(nx = 1,ny =1)
        p =  1
        def fun(k):
            x=k[...,0]
            y=k[...,1]
            E1 = 2*x+y
            E2 = 5*x+3
            array =bm.zeros_like(k)
            array[...,0] = E1
            array[...,1] = E2
            return array

        a = FirstNedelecFiniteElementSpace2d(mesh=mesh,p=p)
        source_vector = a.source_vector(f=fun)
        np.testing.assert_allclose(bm.to_numpy(source_vector), meshdata["source_vector"],1e-7)


if __name__ == "__main__":
    test = TestFirstNedelecDof2d()
    test.test_cell_to_dof(init_data[0],'numpy')
    #test = TestFirstNedelecFiniteElementSpace2d()
    #test.test_basis(init_data[0],"numpy")
    # test.test_curl_basis(init_data[0],"numpy")
    # test.test_value(init_data[0],"pytorch")
    # test.test_curl_value(init_data[0],"pytorch")
    # test.test_mass_matrix(init_data[0],"numpy")
    # test.test_curl_matrix(init_data[0],"pytorch")
    # test.test_source_vector(init_data[0],"pytorch")

