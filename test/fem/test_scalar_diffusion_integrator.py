import ipdb
import numpy as np
import pytest

from fealpy.backend import backend_manager as bm
from fealpy.mesh.triangle_mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem.scalar_diffusion_integrator import ScalarDiffusionIntegrator

from scalar_diffusion_integrator_data import *

class TestScalarDiffusionIntegrator:

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", triangle_mesh_one_box)
    def test_scalar_diffusion_integrator(self, backend, data): 
        bm.set_backend(backend)
        
        mesh = TriangleMesh.from_box([0,1,0,1],1,1)
        space = LagrangeFESpace(mesh, 2)
        integrator = ScalarDiffusionIntegrator(1, 3)
        assembly_cell_matrix = integrator.assembly(space)
        np.testing.assert_array_almost_equal(assembly_cell_matrix ,data["assembly_cell_matrix"], 
                                     err_msg=f" `assembly_cell_matrix` function is not equal to real result in backend {backend}")
if __name__ == "__main__":
    #pytest.main(['test_lagrange_fe_space.py', "-q", "-k","test_basis", "-s"])
    pytest.main(['test_scalar_diffusion_integrator.py', "-q"])   