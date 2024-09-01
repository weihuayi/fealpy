import pytest

from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.tests.fem.linear_elastic_integrator_data import *
from fealpy.experimental.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.experimental.material.elastic_material import LinearElasticMaterial
from app.stopt.soptx.cases.material_properties import MaterialProperties
from fealpy.experimental.mesh import UniformMesh2d, TriangleMesh, UniformMesh3d, TetrahedronMesh
from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace

class TestLinearElasticIntegratorInterfaces:

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("elasticdata", stress_KE_uniform_mesh_2d_p1_data)
    def test_assembly_2d(self, elasticdata, backend):
        bm.set_backend(backend)

        extent, h, origin = elasticdata['extent'], elasticdata['h'], elasticdata['origin']
        mesh = UniformMesh2d(extent, h, origin)
        NC = mesh.number_of_cells()

        p = 1
        space = LagrangeFESpace(mesh, p=p, ctype='C')
        tensor_space = TensorFunctionSpace(space, shape=(-1, 2))
        tldof = tensor_space.number_of_local_dofs()

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
    

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("elasticdata", KE_uniform_mesh_3d_p1_data)
    def test_assembly_3d(self, elasticdata, backend):
        bm.set_backend(backend)

        extent, h, origin = elasticdata['extent'], elasticdata['h'], elasticdata['origin']
        mesh = UniformMesh3d(extent, h, origin)
        NC = mesh.number_of_cells()

        p = 1
        space = LagrangeFESpace(mesh, p=p, ctype='C')
        tensor_space = TensorFunctionSpace(space, shape=(-1, 3))
        tldof = tensor_space.number_of_local_dofs()

        linear_elastic_material = LinearElasticMaterial(
                                                name='LinearlElasticStressMaterial', 
                                                elastic_modulus=1, poisson_ratio=0.3, 
                                                hypo='3D')
        rho = bm.tensor([0.5]*NC, dtype=bm.float64)
        top_isotropic_material = MaterialProperties(
                                                E0=1, Emin=1e-9, nu=0.3, penal=3, 
                                                hypo='3D', rho=rho)

        integrator_linear_elastic_material = LinearElasticIntegrator(
                                    material=linear_elastic_material, q=p+3)
        integrator_top_isotropic_material = LinearElasticIntegrator(
                                    material=top_isotropic_material, q=p+3)
 

        KK_linear_elastic_material = integrator_linear_elastic_material.assembly(
                                                                            space=tensor_space)
        KK_top_isotropic_material = integrator_top_isotropic_material.assembly(
                                                                            space=tensor_space)
        
        assert KK_linear_elastic_material.shape == (NC, tldof, tldof), (
            f"Shape error in KK_linear_elastic_material: expected {(NC, tldof, tldof)}, "
            f"got {KK_linear_elastic_material.shape}"
        )
        assert KK_top_isotropic_material.shape == (NC, tldof, tldof), (
            f"Shape error in KK_top_isotropic_material: expected {(NC, tldof, tldof)}, "
            f"got {KK_top_isotropic_material.shape}"
        )   
    

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("elasticdata", strain_KE_triangle_p1_data)
    def test_fast_assembly_strain(self, elasticdata, backend):
        bm.set_backend(backend)

        box, nx, ny = elasticdata['box'], elasticdata['nx'], elasticdata['ny']
        mesh = TriangleMesh.from_box(box=box, nx=nx, ny=ny)
        NC = mesh.number_of_cells()

        p = 1
        space = LagrangeFESpace(mesh, p=p, ctype='C')

        tensor_space_gd_priority = TensorFunctionSpace(space, shape=(-1, 2))
        tensor_space_dof_priority = TensorFunctionSpace(space, shape=(2, -1))

        tldof = tensor_space_gd_priority.number_of_local_dofs()

        linear_elastic_material_strain = LinearElasticMaterial(
                                                name='LinearlElasticStressMaterial', 
                                                elastic_modulus=1, poisson_ratio=0.3, 
                                                hypo='plane_strain')

        integrator_linear_elastic_material_strain = LinearElasticIntegrator(
                                    material=linear_elastic_material_strain, q=p+3, method='fast_strain')

        KK_linear_elastic_material_strain_gd_priority = integrator_linear_elastic_material_strain.fast_assembly_strain(
                                                                            space=tensor_space_gd_priority)
        KK_linear_elastic_material_strain_dof_priority = integrator_linear_elastic_material_strain.fast_assembly_strain(
                                                                            space=tensor_space_dof_priority)
        
        assert KK_linear_elastic_material_strain_gd_priority.shape == (NC, tldof, tldof), (
            f"Shape error in KK_linear_elastic_material_strain_gd_priority: expected {(NC, tldof, tldof)}, "
            f"got {KK_linear_elastic_material_strain_gd_priority.shape}"
        )
        assert KK_linear_elastic_material_strain_dof_priority.shape == (NC, tldof, tldof), (
            f"Shape error in KK_linear_elastic_material_strain_dof_priority: expected {(NC, tldof, tldof)}, "
            f"got {KK_linear_elastic_material_strain_dof_priority.shape}"
        )

        KK_linear_elastic_material_strain_gd_priority_true = elasticdata['yx_gd_priority']
        KK_linear_elastic_material_strain_dof_priority_true = elasticdata['yx_dof_priority']

        np.testing.assert_almost_equal(
            bm.to_numpy(KK_linear_elastic_material_strain_gd_priority[0]),
            KK_linear_elastic_material_strain_gd_priority_true, decimal=7,
            err_msg="Mismatch in numerical values for KK_linear_elastic_material_strain_gd_priority[0]: "
                    "The calculated stiffness matrix with SIMP scaling does not match the expected values."
        )
        np.testing.assert_almost_equal(
            bm.to_numpy(KK_linear_elastic_material_strain_dof_priority[0]),
            KK_linear_elastic_material_strain_dof_priority_true, decimal=7,
            err_msg="Mismatch in numerical values for KK_linear_elastic_material_strain_dof_priority[0]: "
                    "The calculated stiffness matrix with SIMP scaling does not match the expected values."
        )


    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("elasticdata", stress_KE_triangle_p1_data)
    def test_fast_assembly_stress(self, elasticdata, backend):
        bm.set_backend(backend)

        box, nx, ny = elasticdata['box'], elasticdata['nx'], elasticdata['ny']
        mesh = TriangleMesh.from_box(box=box, nx=nx, ny=ny)
        NC = mesh.number_of_cells()

        p = 1
        space = LagrangeFESpace(mesh, p=p, ctype='C')

        tensor_space_gd_priority = TensorFunctionSpace(space, shape=(-1, 2))
        tensor_space_dof_priority = TensorFunctionSpace(space, shape=(2, -1))
        
        tldof = tensor_space_gd_priority.number_of_local_dofs()

        
        linear_elastic_material_stress = LinearElasticMaterial(
                                                name='LinearlElasticStressMaterial', 
                                                elastic_modulus=1, poisson_ratio=0.3, 
                                                hypo='plane_stress')
        rho = bm.tensor([0.5]*NC, dtype=bm.float64)
        top_isotropic_material_stress = MaterialProperties(
                                                E0=1, Emin=1e-9, nu=0.3, penal=3, 
                                                hypo='plane_stress', rho=rho)

        integrator_linear_elastic_material_stress = LinearElasticIntegrator(
                                    material=linear_elastic_material_stress, q=p+3, method='fast_stress')
        integrator_top_isotropic_material_stress = LinearElasticIntegrator(
                                    material=top_isotropic_material_stress, q=p+3, method='fast_stress')

        KK_linear_elastic_material_stress_gd_priority = integrator_linear_elastic_material_stress.fast_assembly_stress(
                                                                             space=tensor_space_gd_priority)
        KK_linear_elastic_material_stress_dof_priority = integrator_linear_elastic_material_stress.fast_assembly_stress(
                                                                                space=tensor_space_dof_priority)
        KK_top_isotropic_material_stress_gd_priority = integrator_top_isotropic_material_stress.fast_assembly_stress(
                                                                            space=tensor_space_gd_priority)
        KK_top_isotropic_material_stress_dof_priority = integrator_top_isotropic_material_stress.fast_assembly_stress(
                                                                            space=tensor_space_dof_priority)            
        
        assert KK_linear_elastic_material_stress_gd_priority.shape == (NC, tldof, tldof), (
            f"Shape error in KK_linear_elastic_material_stress_gd_priority: expected {(NC, tldof, tldof)}, "
            f"got {KK_linear_elastic_material_stress_gd_priority.shape}"
        )
        assert KK_linear_elastic_material_stress_dof_priority.shape == (NC, tldof, tldof), (
            f"Shape error in KK_linear_elastic_material_stress_dof_priority: expected {(NC, tldof, tldof)}, "
            f"got {KK_linear_elastic_material_stress_dof_priority.shape}"
        )
        assert KK_top_isotropic_material_stress_gd_priority.shape == (NC, tldof, tldof), (
            f"Shape error in KK_top_isotropic_material_stress_gd_priority: expected {(NC, tldof, tldof)}, "
            f"got {KK_top_isotropic_material_stress_gd_priority.shape}"
        )
        assert KK_top_isotropic_material_stress_dof_priority.shape == (NC, tldof, tldof), (
            f"Shape error in KK_top_isotropic_material_stress_dof_priority: expected {(NC, tldof, tldof)}, "
            f"got {KK_top_isotropic_material_stress_dof_priority.shape}"
        )

        KK_linear_elastic_material_stress_gd_priority_true = elasticdata['yx_gd_priority']
        KK_linear_elastic_material_stress_dof_priority_true = elasticdata['yx_dof_priority']

        np.testing.assert_almost_equal(
            KK_linear_elastic_material_stress_gd_priority[0],
            KK_linear_elastic_material_stress_gd_priority_true, decimal=7,
            err_msg="Mismatch in numerical values for KK_linear_elastic_material_stress_gd_priority[0]: "
                    "The calculated stiffness matrix with SIMP scaling does not match the expected values."
        )
        np.testing.assert_almost_equal(
            KK_linear_elastic_material_stress_dof_priority[0],
            KK_linear_elastic_material_stress_dof_priority_true, decimal=7,
            err_msg="Mismatch in numerical values for KK_linear_elastic_material_stress_dof_priority[0]: "
                    "The calculated stiffness matrix with SIMP scaling does not match the expected values."
        )
        np.testing.assert_almost_equal(
            bm.to_numpy(KK_top_isotropic_material_stress_gd_priority[0]),
            bm.to_numpy(KK_linear_elastic_material_stress_gd_priority[0]) * 0.5**3, decimal=7,
            err_msg="Mismatch in numerical values for KK_top_isotropic_material_stress_gd_priority[0]: "
                    "The calculated stiffness matrix with SIMP scaling does not match the expected values."
        )
        np.testing.assert_almost_equal(
            bm.to_numpy(KK_top_isotropic_material_stress_dof_priority[0]),
            bm.to_numpy(KK_linear_elastic_material_stress_dof_priority[0]) * 0.5**3, decimal=7,
            err_msg="Mismatch in numerical values for KK_top_isotropic_material_stress_dof_priority[0]: "
                    "The calculated stiffness matrix with SIMP scaling does not match the expected values."
        )


    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("elasticdata", KE_tetrahedron_p1_data)
    def test_fast_assembly(self, elasticdata, backend):
        bm.set_backend(backend)

        box, nx, ny, nz = elasticdata['box'], elasticdata['nx'], elasticdata['ny'], elasticdata['nz']
        mesh = TetrahedronMesh.from_box(box=box, nx=nx, ny=ny, nz=nz)
        NC = mesh.number_of_cells()

        p = 1
        space = LagrangeFESpace(mesh, p=p, ctype='C')

        tensor_space_gd_prioriy = TensorFunctionSpace(space, shape=(-1, 3))
        tensor_space_dof_prioriy = TensorFunctionSpace(space, shape=(3, -1))

        tldof = tensor_space_gd_prioriy.number_of_local_dofs()

        linear_elastic_material = LinearElasticMaterial(
                                                name='LinearlElasticStressMaterial', 
                                                elastic_modulus=1, poisson_ratio=0.3, 
                                                hypo='3D')
        rho = bm.tensor([0.5]*NC, dtype=bm.float64)
        top_isotropic_material = MaterialProperties(
                                                E0=1, Emin=1e-9, nu=0.3, penal=3, 
                                                hypo='3D', rho=rho)

        integrator_linear_elastic_material = LinearElasticIntegrator(
                                    material=linear_elastic_material, q=p+3, method='fast_3d')
        integrator_top_isotropic_material = LinearElasticIntegrator(
                                    material=top_isotropic_material, q=p+3, method='fast_3d')

        KK_linear_elastic_material_gd_priority = integrator_linear_elastic_material.fast_assembly(
                                                                    space=tensor_space_gd_prioriy)
        KK_linear_elastic_material_dof_priority = integrator_linear_elastic_material.fast_assembly(
                                                                    space=tensor_space_dof_prioriy)
        KK_top_isotropic_material_gd_priority = integrator_top_isotropic_material.fast_assembly(
                                                                    space=tensor_space_gd_prioriy)
        KK_top_isotropic_material_dof_priority = integrator_top_isotropic_material.fast_assembly(
                                                                    space=tensor_space_dof_prioriy)
        
        assert KK_linear_elastic_material_gd_priority.shape == (NC, tldof, tldof), (
            f"Shape error in KK_linear_elastic_material_gd_priority: expected {(NC, tldof, tldof)}, "
            f"got {KK_linear_elastic_material_gd_priority.shape}"
        )
        assert KK_linear_elastic_material_dof_priority.shape == (NC, tldof, tldof), (
            f"Shape error in KK_linear_elastic_material_dof_priority: expected {(NC, tldof, tldof)}, "
            f"got {KK_linear_elastic_material_dof_priority.shape}"
        )   
        assert KK_top_isotropic_material_gd_priority.shape == (NC, tldof, tldof), (
            f"Shape error in KK_top_isotropic_material_gd_priority: expected {(NC, tldof, tldof)}, "
            f"got {KK_top_isotropic_material_gd_priority.shape}"
        )
        assert KK_top_isotropic_material_dof_priority.shape == (NC, tldof, tldof), (
            f"Shape error in KK_top_isotropic_material_dof_priority: expected {(NC, tldof, tldof)}, "
            f"got {KK_top_isotropic_material_dof_priority.shape}"
        )

        np.testing.assert_almost_equal(
            bm.to_numpy(KK_top_isotropic_material_gd_priority[0]),
            bm.to_numpy(KK_linear_elastic_material_gd_priority[0]) * 0.5**3, decimal=7,
            err_msg="Mismatch in numerical values for KK_top_isotropic_material_gd_priority[0]: "
                    "The calculated stiffness matrix with SIMP scaling does not match the expected values."
        )
        np.testing.assert_almost_equal(
            bm.to_numpy(KK_top_isotropic_material_dof_priority[0]),
            bm.to_numpy(KK_linear_elastic_material_dof_priority[0]) * 0.5**3, decimal=7,
            err_msg="Mismatch in numerical values for KK_top_isotropic_material_dof_priority[0]: "
                    "The calculated stiffness matrix with SIMP scaling does not match the expected values."
        )

