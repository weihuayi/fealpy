import numpy as np
import pytest
from fealpy.experimental.backend import backend_manager as bm

from fealpy.experimental.functionspace import LagrangeFESpace
from fealpy.experimental.fem import (
        BilinearForm, ScalarDiffusionIntegrator
    )


class TestBilinearFormInterface:
    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", mesh_data)
    @pytest.mark.parametrize("p", range(1, 5))
    def test_matmul(backend, data):
        mesh = data["mesh"]
        node = mesh.node
        space = LagrangeFESpace(mesh, p)
        gdof = space.number_of_global_dofs()
        kwargs = bm.context(node)
        x = bm.ones(gdof, **kwargs)
        bform = BilinearForm(space)
        bform.add_integrator(ScalarDiffusionIntegrator())
        y = bform @ x
        np.testing.assert_array_equal(bm.to_numpy(y), data["y"])


if __name__ == "__main__":
    pytest.main(['./test_bilinear_form', '-k', 'test_matmul'])
