import numpy as np
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.experimental.fem.linear_elasticity_integrator import LinearElasticityIntegrator
from fealpy.experimental.fem.bilinear_form import BilinearForm
from fealpy.experimental.mesh import UniformMesh2d
from fealpy.experimental.tests.fem.linear_elasticity_integrator_data import *

class TestLinearElasticityIntegratorInterfaces:

    @pytest.mark.parametrize("meshdata", element_stiffness_matrix_4node_p1_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_element_stiffness_matrix_4node_p1(self, meshdata, backend):
        bm.set_backend(backend)

        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh2d(extent, h, origin)

        space = LagrangeFESpace(mesh, p=1, ctype='C')
        tensor_space_node = TensorFunctionSpace(space, shape=(-1, 2))

        integrator_bi = LinearElasticityIntegrator(E=1.0, nu=0.3, 
                                           elasticity_type='stress', q=5)
        KK_node = integrator_bi.assembly(space=tensor_space_node)

        KK_node_ture = meshdata['yx_node']
        np.testing.assert_almost_equal(
            bm.to_numpy(KK_node[0]), KK_node_ture, decimal=7)
        
    @pytest.mark.parametrize("meshdata", element_stiffness_matrix_4node_p1_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_global_stiffness_matrix_4node_p1(self, meshdata, backend):
        bm.set_backend(backend)
        
        extent = meshdata['extent']
        h = meshdata['h']
        origin = meshdata['origin']
        mesh = UniformMesh2d(extent, h, origin)

        space = LagrangeFESpace(mesh, p=1, ctype='C')
        tensor_space_node = TensorFunctionSpace(space, shape=(-1, 2))
        tgdof = tensor_space_node.number_of_global_dofs()
        cell2tldof = tensor_space_node.cell_to_dof()

        integrator_bi = LinearElasticityIntegrator(E=1.0, nu=0.3, 
                                           elasticity_type='stress', q=5)
        KK_node = integrator_bi.assembly(space=tensor_space_node)
        bform = BilinearForm(tensor_space_node)
        bform.add_integrator(integrator_bi)
        K_node = bform.assembly().to_dense()

        assert K_node.shape == (tgdof, tgdof), \
            "Global stiffness matrix shape error!"

        