import numpy as np
import pytest
from fealpy.experimental.backend import backend_manager as bm
from fealpy.experimental.functionspace import LagrangeFESpace
from fealpy.experimental.fem.linear_elasticity_integrator import LinearElasticityIntegrator
from fealpy.experimental.tests.fem.linear_elasticity_integrator_data import *

class TestUniformMesh2dInterfaces:

    @pytest.mark.parametrize("meshdata", element_stiffness_matrix_4node_p1_data)
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch', 'jax'])
    def test_element_stiffness_matrix_4node_p1(self, meshdata, backend):
        pass