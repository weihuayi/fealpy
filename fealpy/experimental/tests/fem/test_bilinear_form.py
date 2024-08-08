import numpy as np
import pytest
from fealpy.experimental.backend import backend_manager as bm

from fealpy.experimental.mesh import TriangleMesh
from fealpy.experimental.functionspace import LagrangeFESpace
from fealpy.experimental.fem import (
        BilinearForm, ScalarDiffusionIntegrator
    )

from fealpy.experimental.tests.fem.bilinear_form_data import *

mesh_map = {
        "TriangleMesh": TriangleMesh,
        }

class TestBilinearFormInterface:

    @pytest.mark.parametrize("backend", ['numpy', 'pytorch'])
    @pytest.mark.parametrize("data", mesh_data)
    @pytest.mark.parametrize("p", range(1, 5))
    def test_matmul(self, backend, data, p):
        bm.set_backend(backend)

        Mesh = mesh_map[data["class"]]
        node = bm.from_numpy(data['node'])
        cell = bm.from_numpy(data['cell'])
        mesh = Mesh(node, cell) 
        space = LagrangeFESpace(mesh, p)
        gdof = space.number_of_global_dofs()

        kwargs = bm.context(mesh.node)
        x = bm.ones(gdof, **kwargs)
        bform = BilinearForm(space)
        bform.add_integrator(ScalarDiffusionIntegrator())
        y = bform @ x # 只组装单元刚度矩阵
        bform.assembly() # 组装整体矩阵
        z = bform @ x
        np.testing.assert_array_equal(bm.to_numpy(y), bm.to_numpy(z))


if __name__ == "__main__":
    pytest.main(['./test_bilinear_form.py', '-k', 'test_matmul'])
