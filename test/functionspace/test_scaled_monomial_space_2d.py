import numpy as np
import matplotlib.pyplot as plt
import pytest
from scipy.sparse import csr_matrix
from fealpy.backend import backend_manager as bm

from fealpy.backend import backend_manager as bm
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.mesh import PolygonMesh
from fealpy.functionspace import ScaledMonomialSpace2d

from scaled_monomial_space_2d_data import *

class TestScaledMonomialSpace2d():
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data",multi_index_matrix)
    def test_multi_index_matrix(self, backend, data): 
        bm.set_backend(backend)
        mesh = PolygonMesh.from_box([0,1,0,1],3,2)
        space = ScaledMonomialSpace2d(mesh, p=5)
        multi_index_matrix = space.multi_index_matrix(p=4)
        np.testing.assert_equal(multi_index_matrix, data["multi_index_matrix"])

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data",cell_to_dof)
    def test_cell_to_dof(self, backend, data): 
        bm.set_backend(backend)
        mesh = PolygonMesh.from_box([0,1,0,1],3,2)
        space = ScaledMonomialSpace2d(mesh, p=5)
        cell2dof = space.cell_to_dof(p=4)
        np.testing.assert_equal(cell2dof, data["cell2dof"])
        ldof = space.number_of_local_dofs(p=4)
        gdof = space.number_of_global_dofs(p=4)
        assert ldof == data["ldof"]
        assert gdof == data["gdof"]

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data",index)
    def test_index(self, backend, data): 
        bm.set_backend(backend)
        mesh = PolygonMesh.from_box([0,1,0,1],3,2)
        space = ScaledMonomialSpace2d(mesh, p=3)
        diff_index_1 = space.diff_index_1() 
        diff_index_2 = space.diff_index_2()
        edge_index_1 = space.edge_index_1()
        face_index_1 = space.edge_index_1()
        a = diff_index_1.values()
        for x,y in zip(a, data["index_1"].values()):
            np.testing.assert_equal(x, y)
        a = diff_index_2.values()
        for x,y in zip(a, data["index_2"].values()):
            np.testing.assert_equal(x, y)
        a = edge_index_1.values()
        for x,y in zip(a, data["edge_index_1"].values()):
            np.testing.assert_equal(x, y)
        a = face_index_1.values()
        for x,y in zip(a, data["face_index_1"].values()):
            np.testing.assert_equal(x, y)

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", basis)
    def test_basis(self, backend, data): 
        bm.set_backend(backend)
        mesh = PolygonMesh.from_box([0,1,0,1],1,1)
        space = ScaledMonomialSpace2d(mesh, p=3)
        bcs = mesh.multi_index_matrix(3,1,dtype=mesh.ftype)
        point = mesh.edge_bc_to_point(bcs)
        edge_basis = space.edge_basis(point) # (NE, NQ, ldof)
        np.testing.assert_allclose(edge_basis.swapaxes(1,0), data["edge_basis"], atol=1e-8)
        edge_basis_with_bcs = space.edge_basis_with_barycentric(bcs)
        np.testing.assert_allclose(edge_basis_with_bcs,
                                   data["edge_basis_with_bcs"], atol=1e-8)
        bcs = mesh.multi_index_matrix(3,2)
        point = mesh.entity_barycenter('cell')
        node = mesh.entity('node')
        cell2node = mesh.cell_to_node(return_sparse=True)
        point = bm.array(cell2node.toarray(),dtype=bm.float64)@node #(NC,GD)

        point = point[:,None,:]
        point = bm.tile(point,(1,2,1)) # NC, NQ, 2
        basis = space.basis(point)
        gbasis = space.grad_basis(point)
        lbasis = space.laplace_basis(point)
        hbasis = space.hessian_basis(point)
        gmbasis = space.grad_m_basis(3,point)
        np.testing.assert_allclose(basis.swapaxes(1,0), data["basis"], atol=1e-8)
        np.testing.assert_allclose(gbasis.swapaxes(1,0), data["gbasis"], atol=1e-8)
        np.testing.assert_allclose(lbasis.swapaxes(1,0), data["lbasis"], atol=1e-8)
        np.testing.assert_allclose(hbasis.swapaxes(1,0), data["hbasis"], atol=1e-8)
        np.testing.assert_allclose(gmbasis.swapaxes(1,0), data["gmbasis"], atol=1e-8)
        pe = space.partial_matrix_on_edge()
            
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data",value)
    def test_value(self, backend, data): 
        bm.set_backend(backend)
        mesh = PolygonMesh.from_box([0,1,0,1],1,1)
        space = ScaledMonomialSpace2d(mesh, p=3)
        gdof = space.number_of_global_dofs(p=3)
        cell2node = mesh.cell_to_node(return_sparse=True)                        
        node = mesh.entity('node')
        #point = cell2node@node # (NC=4, 2)
        point = bm.array(cell2node.toarray(),dtype=bm.float64)@node #(NC,GD)
        #print(point.shape)
        #point = point[None,:,:]
        #point = np.tile(point,(2,1,1))
        uh = space.function()
        uh[:] = bm.ones(gdof)
        value = space.value(uh, point) 
        np.testing.assert_allclose(value, data["value"], atol=1e-8)
        gvalue = space.grad_value(uh,point)
        np.testing.assert_allclose(gvalue, data["gvalue"], atol=1e-8)
        lvalue = space.laplace_value(uh,point)
        np.testing.assert_allclose(lvalue, data["lvalue"], atol=1e-8)
        hvalue = space.hessian_value(uh,point)
        np.testing.assert_allclose(hvalue, data["hvalue"], atol=1e-8)
        g3value = space.grad_3_value(uh,point)
        np.testing.assert_allclose(g3value, data["g3value"], atol=1e-8)
 
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data",matrix)
    def test_matrix(self, backend, data): 
        bm.set_backend(backend)
        mesh = PolygonMesh.from_box([0,1,0,1],1,1)
        space = ScaledMonomialSpace2d(mesh, p=3)
        cell_mass_matrix = space.cell_mass_matrix(p=3)
        edge_mass_matrix = space.edge_mass_matrix(p=3)
        edge_mass_matrix_1 = space.edge_mass_matrix_1(p=3)
        edge_cell_mass_matrix0, edge_cell_mass_matrix1 = space.edge_cell_mass_matrix(p=3)
        np.testing.assert_allclose(cell_mass_matrix, data["cell_mass_matrix"], atol=1e-8)
        np.testing.assert_allclose(edge_mass_matrix, data["edge_mass_matrix"], atol=1e-8)
        np.testing.assert_allclose(edge_mass_matrix_1, data["edge_mass_matrix_1"], atol=1e-8)
        np.testing.assert_allclose(edge_cell_mass_matrix0, data["edge_cell_mass_matrix[0]"], atol=1e-8)
        np.testing.assert_allclose(edge_cell_mass_matrix1, data["edge_cell_mass_matrix[1]"], atol=1e-8)
        cell_hessian_matrix = space.cell_hessian_matrix(p=3)
        cell_grad_m_matrix = space.cell_grad_m_matrix(m=3)
        np.testing.assert_allclose(cell_hessian_matrix, data["cell_hessian_matrix"], atol=1e-8)
        np.testing.assert_allclose(cell_grad_m_matrix, data["cell_grad_m_matrix"], atol=1e-8)
        
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data",matrix1)
    def test_matrix1(self, backend, data): 
        bm.set_backend(backend)
        mesh = PolygonMesh.from_box([0,1,0,1],1,1)
        space = ScaledMonomialSpace2d(mesh, p=2)
        stiff_matrix = space.stiff_matrix(p=2)
        mass_matrix = space.mass_matrix(p=2)
        penalty_matrix = space.penalty_matrix(p=2)
        flux_matrix = space.flux_matrix(p=2)
        normal_grad_penalty_matrix = space.normal_grad_penalty_matrix(p=2)
        np.testing.assert_allclose(stiff_matrix.toarray(), data["stiff_matrix"], atol=1e-8)
        np.testing.assert_allclose(mass_matrix.toarray(), data["mass_matrix"], atol=1e-8)
        np.testing.assert_allclose(penalty_matrix.toarray(), data['penalty_matrix'],atol=1e-10)
        np.testing.assert_allclose(flux_matrix.toarray(), data['flux_matrix'],atol=1e-10)
        np.testing.assert_allclose(normal_grad_penalty_matrix.toarray(), data['normal_grad_penalty_matrix'],atol=1e-10)
        
    #@pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    #@pytest.mark.parametrize("data", vector)
    #def test_vector(self, backend, data): 
    #    bm.set_backend(backend)
    #    mesh = PolygonMesh.from_box([0,1,0,1],1,1)
    #    space = ScaledMonomialSpace2d(mesh, p=3)
    #def f(p):                                                                   
    #    x = p[...,0]                                                            
    #    y = p[...,1]                                                            
    #    return np.sin(np.pi*x)*np.sin(np.pi*y)                                  
    #edge_normal_source_vector = space.edge_normal_source_vector(f)              
    #edge_source_vector = space.edge_source_vector(f)                            
    #source_vector0 = space.source_vector0(f)                                    

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", cellmatrix)
    def test_cellmatrix(self, backend, data): 
        bm.set_backend(backend)
        mesh = PolygonMesh.from_box([0,1,0,1],1,1)
        p = 3
        space = ScaledMonomialSpace2d(mesh, p=p)
        #from fealpy.fem import  ScalarMassIntegrator
        #integrator = ScalarMassIntegrator(q=p+1,method='homogeneous')
        #mass_matrix = integrator.homogeneous_assembly(space)
        #np.testing.assert_allclose(mass_matrix, data["mass_matrix"], atol=1e-8)
        #from fealpy.fem import  ScalarDiffusionIntegrator
        #integrator = ScalarDiffusionIntegrator(q=p+1,method='homogeneous')
        #stiff_matrix = integrator.homogeneous_assembly(space)
        stiff = space.cell_stiff_matrix()
        np.testing.assert_allclose(stiff, data["cell_stiff_matrix"], atol=1e-8)
        #cell_grad_m_matrix = space.cell_grad_m_matrix(m=1)
        #np.testing.assert_allclose(stiff_matrix, stiff, atol=1e-8)


        

    
 
if __name__ == '__main__':
    ts = TestScaledMonomialSpace2d()
    ts.test_cellmatrix('numpy',cellmatrix[0])
    ts.test_cellmatrix('pytorch',cellmatrix[0])
    #ts.test_multi_index_matrix('numpy', multi_index_matrix[0])
    #ts.test_multi_index_matrix('pytorch', multi_index_matrix[0])
    #ts.test_cell_to_dof('pytorch', cell_to_dof[0])
    #ts.test_basis('pytorch', basis[0])
    #ts.test_basis('numpy', basis[0])
    #ts.test_value('pytorch', value[0])
    #ts.test_value('numpy', value[0])
    #ts.test_matrix('pytorch', matrix[0])
    #ts.test_matrix('numpy', matrix[0])
    #ts.test_matrix1('pytorch', matrix1[0])
    #ts.test_matrix1('numpy', matrix1[0])



