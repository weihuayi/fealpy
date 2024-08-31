import pytest

from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.typing import TensorLike
from fealpy.experimental.tests.fem.linear_elastic_integrator_data import *
from fealpy.experimental.fem.linear_elastic_integrator import LinearElasticIntegrator
from fealpy.experimental.material.elastic_material import LinearElasticMaterial
from fealpy.experimental.mesh import UniformMesh2d, TriangleMesh
from fealpy.experimental.functionspace import LagrangeFESpace, TensorFunctionSpace

class TestLinearElasticIntegratorInterfaces:

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("elasticdata", stress_KE_uniform_mesh_2d_p1_data)
    def test_assembly(self, elasticdata, backend):
        bm.set_backend(backend)

        extent = [0, 2, 0, 2]
        extent, h, origin = elasticdata['extent'], elasticdata['h'], elasticdata['origin']
        mesh = UniformMesh2d(extent, h, origin)
        NC = mesh.number_of_cells()

        p = 1
        space = LagrangeFESpace(mesh, p=p, ctype='C')
        tensor_space = TensorFunctionSpace(space, shape=(2, -1))
        tldof = tensor_space.number_of_local_dofs()
        uh = tensor_space.function()

        material = LinearElasticMaterial(name='LinearlIsotropicMaterial')
        integrator_strain = LinearElasticIntegrator(material=material, elastic_type='strain', q=p+3)
        integrator_stress = LinearElasticIntegrator(material=material, elastic_type='stress', q=p+3)

        KK_strain = integrator_strain.assembly(space=tensor_space)
        KK_stress = integrator_stress.assembly(space=tensor_space)

        assert KK_strain.shape == (NC, tldof, tldof), "Shape error!"
        assert KK_stress.shape == (NC, tldof, tldof), "Shape error!"
