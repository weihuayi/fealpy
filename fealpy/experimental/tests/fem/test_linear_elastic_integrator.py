import pytest

from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.tests.fem.linear_elastic_integrator_data import *
from fealpy.experimental.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.experimental.material.elastic_material import LinearElasticMaterial
from app.stopt.soptx.cases.material_properties import MaterialProperties
from fealpy.experimental.mesh import UniformMesh2d, TriangleMesh
from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace

class TestLinearElasticIntegratorInterfaces:

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("elasticdata", stress_KE_uniform_mesh_2d_p1_data)
    def test_assembly(self, elasticdata, backend):
        bm.set_backend(backend)

        extent, h, origin = elasticdata['extent'], elasticdata['h'], elasticdata['origin']
        mesh = UniformMesh2d(extent, h, origin)
        NC = mesh.number_of_cells()

        p = 1
        space = LagrangeFESpace(mesh, p=p, ctype='C')
        tensor_space = TensorFunctionSpace(space, shape=(-1, 2))
        tldof = tensor_space.number_of_local_dofs()
        uh = tensor_space.function()

        linear_elastic_material_stress = LinearElasticMaterial(
                                                name='LinearlElasticStressMaterial', 
                                                elastic_modulus=1, poisson_ratio=0.3, 
                                                hypo='plane_stress')
        rho = bm.tensor([0.5]*NC, dtype=bm.float64)
        top_isotropic_material_stress = MaterialProperties(
                                                E0=1, Emin=1e-9, nu=0.3, penal=3, 
                                                hypo='plane_stress', rho=rho)
        linear_elastic_material_strain = LinearElasticMaterial(
                                                name='LinearlElasticStrainMaterial', 
                                                elastic_modulus=1, poisson_ratio=0.3, 
                                                hypo='plane_strain')
        
        integrator_linear_elastic_material_stress = LinearElasticIntegrator(
                                    material=linear_elastic_material_stress, q=p+3)
        integrator_top_isotropic_material_stress = LinearElasticIntegrator(
                                    material=top_isotropic_material_stress, q=p+3)
        integrator_linear_elastic_material_strain = LinearElasticIntegrator(
                                    material=linear_elastic_material_strain, q=p+3)

        KK_linear_elastic_material_stress = integrator_linear_elastic_material_stress.assembly(
                                                                             space=tensor_space)
        KK_top_isotropic_material_stress = integrator_top_isotropic_material_stress.assembly(
                                                                            space=tensor_space)
        KK_linear_elastic_material_strain = integrator_linear_elastic_material_strain.assembly(
                                                                            space=tensor_space)
        
        assert KK_linear_elastic_material_stress.shape == (NC, tldof, tldof), (
            f"Shape error in KK_linear_elastic_material_stress: expected {(NC, tldof, tldof)}, "
            f"got {KK_linear_elastic_material_stress.shape}"
        )
        assert KK_top_isotropic_material_stress.shape == (NC, tldof, tldof), (
            f"Shape error in KK_top_isotropic_material_stress: expected {(NC, tldof, tldof)}, "
            f"got {KK_top_isotropic_material_stress.shape}"
        )
        assert KK_linear_elastic_material_strain.shape == (NC, tldof, tldof), (
            f"Shape error in KK_linear_elastic_material_strain: expected {(NC, tldof, tldof)}, "
            f"got {KK_linear_elastic_material_strain.shape}"
        )

        KK_linear_elastic_material_stress_true = elasticdata['yx_gd_priority']
        KK_top_isotropic_material_stress_true = elasticdata['yx_gd_priority'] * 0.5**3

        np.testing.assert_almost_equal(
            KK_linear_elastic_material_stress[0], 
            KK_linear_elastic_material_stress_true, decimal=7,
            err_msg="Mismatch in numerical values for KK_linear_elastic_material_stress[0]: "
                    "The calculated stiffness matrix does not match the expected values."
        )
        np.testing.assert_almost_equal(
            KK_top_isotropic_material_stress[0],
            KK_top_isotropic_material_stress_true, decimal=7,
            err_msg="Mismatch in numerical values for KK_top_isotropic_material_stress[0]: "
                    "The calculated stiffness matrix with SIMP scaling does not match the expected values."
        )