import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh.tetrahedron_mesh import TetrahedronMesh
from fealpy.functionspace.first_nedelec_fe_space_3d import FirstNedelecDof3d
from fealpy.functionspace.first_nedelec_fe_space_3d import FirstNedelecFiniteElementSpace3d

from first_nedelec_fe_space_3d_data import *


def assert_absolute_error(a,b,atol=1e-10):
    diff = np.abs(a-b)
    if np.any(diff>atol):
        raise AssertionError

class TestFirstNedelecDof3d:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_cell_to_dof(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TetrahedronMesh.from_box(nx = 1,ny =1,nz=1)
        p =  3
        a = FirstNedelecDof3d(mesh,p)
        assert a.number_of_local_dofs("edge") == meshdata["edof"]
        assert a.number_of_local_dofs("cell") == meshdata["cdof"]
        assert a.number_of_global_dofs() == meshdata["gdof"]
        
        cell2dof = a.cell2dof
        np.testing.assert_array_equal(bm.to_numpy(cell2dof), meshdata["cell2dof"])

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_face_to_dof(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TetrahedronMesh.from_box(nx = 1,ny =1,nz=1)
        p =  1
        a = FirstNedelecDof3d(mesh,p)
        
        face2dof = a.face_to_dof()
        np.testing.assert_array_equal(bm.to_numpy(face2dof), meshdata["face2dof"])

class TestFirstNedelecFiniteElementSpace3d:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_basis(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TetrahedronMesh.from_box(nx = 1,ny =1,nz=1)
        p =  1
        a = FirstNedelecFiniteElementSpace3d(mesh,p)
        bcs = bm.tensor([[0.1,0.2,0.1,0.6],
                         [0.5,0.3,0.1,0.1]],dtype=mesh.ftype)

        basis = a.basis(bcs=bcs)
        np.testing.assert_allclose(bm.to_numpy(basis), meshdata["basis"],1e-15)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_curl_basis(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TetrahedronMesh.from_box(nx = 1,ny =1,nz=1)
        p =  1
        a = FirstNedelecFiniteElementSpace3d(mesh,p)
        bcs = bm.tensor([[0.1,0.2,0.1,0.6],
                         [0.5,0.3,0.1,0.1]],dtype=mesh.ftype)

        curl_basis = a.curl_basis(bcs=bcs)
        np.testing.assert_allclose(bm.to_numpy(curl_basis), meshdata["curl_basis"],1e-15)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_value(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TetrahedronMesh.from_box(nx = 1,ny =1,nz=1)
        p =  1
        a = FirstNedelecFiniteElementSpace3d(mesh,p)
        bcs = bm.tensor([[0.1,0.2,0.1,0.6],
                         [0.5,0.3,0.1,0.1]],dtype=mesh.ftype)
        uh = bm.arange(74,dtype=mesh.ftype)
        value = a.value(uh=uh,bcs=bcs)
        np.testing.assert_allclose(bm.to_numpy(value), meshdata["value"],1e-14)
     
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_curl_value(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TetrahedronMesh.from_box(nx = 1,ny =1,nz=1)
        p =  1
        a = FirstNedelecFiniteElementSpace3d(mesh,p)
        bcs = bm.tensor([[0.1,0.2,0.1,0.6],
                         [0.5,0.3,0.1,0.1]],dtype=mesh.ftype)
        uh = bm.arange(74,dtype=mesh.ftype)
        curl_value = a.curl_value(uh=uh,bcs=bcs)
        np.testing.assert_allclose(bm.to_numpy(curl_value), meshdata["curl_value"],1e-14)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_mass_matrix(self,meshdata,backend):
        bm.set_backend(backend)
        
        mesh = TetrahedronMesh.from_box(nx = 1,ny =1,nz=1)
        p =  1
        a = FirstNedelecFiniteElementSpace3d(mesh=mesh,p=p)
        mass_matrix = a.mass_matrix()
        # np.testing.assert_allclose((mass_matrix).toarray(), meshdata["mass_matrix"],1e-8)
        assert_absolute_error((mass_matrix).toarray(),meshdata["mass_matrix"])

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_curl_matrix(self,meshdata,backend):
        bm.set_backend(backend)
        
        mesh = TetrahedronMesh.from_box(nx = 1,ny =1,nz=1)
        p =  1
        a = FirstNedelecFiniteElementSpace3d(mesh=mesh,p=p)
        curl_matrix = a.curl_matrix()
        # np.testing.assert_allclose((mass_matrix).toarray(), meshdata["mass_matrix"],1e-8)
        assert_absolute_error((curl_matrix).toarray(),meshdata["curl_matrix"])
    
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("meshdata", init_data)
    def test_vector(self,meshdata,backend):
        bm.set_backend(backend)

        mesh = TetrahedronMesh.from_box(nx = 1,ny =1,nz=1)
        p =  1
        a = FirstNedelecFiniteElementSpace3d(mesh,p)
        def func(k):
            x = k[...,0]
            y = k[...,1]
            z = k[...,2]
            E1= 4*x+3*z
            E2= y +12
            E3= x*y +2*z
            array = bm.zeros_like(k)
            array[...,0] = E1
            array[...,1] = E2
            array[...,2] = E3
            return array
        vector = a.source_vector(f = func)
        np.testing.assert_allclose(bm.to_numpy(vector), meshdata["vector"],1e-5)


if __name__ == "__main__":
    # test = TestFirstNedelecDof3d()
    # test.test_cell_to_dof(init_data[0],'pytorch')
    # test.test_face_to_dof(init_data[0],"numpy")
    test = TestFirstNedelecFiniteElementSpace3d()
    # test.test_basis(init_data[0],"numpy")
    test.test_curl_basis(init_data[0],"numpy")
    # test.test_value(init_data[0],"pytorch")
    # test.test_curl_value(init_data[0],"pytorch")
    # test.test_mass_matrix(init_data[0],"numpy")
    # test.test_curl_matrix(init_data[0],"numpy")
    # test.test_vector(init_data[0],"pytorch")


