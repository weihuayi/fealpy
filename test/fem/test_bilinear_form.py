import numpy as np
import pytest
from fealpy.backend import backend_manager as bm

from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import (
        BilinearForm, ScalarDiffusionIntegrator
    )

from bilinear_form_data import *

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

        if backend == "pytorch":
            kwargs = bm.context(mesh.node)
            x = bm.random.rand(gdof, **kwargs)
        else:
            x = bm.random.rand(gdof)
        bform = BilinearForm(space)
        bform.add_integrator(ScalarDiffusionIntegrator())
        y = bm.to_numpy(bform @ x) # 只组装单元刚度矩阵
        assert bform._M is None
        bform.assembly() # 组装整体矩阵
        assert bform._M is not None
        z = bm.to_numpy(bform @ x)
        assert np.linalg.norm(y-z) < 1e-12 


if __name__ == "__main__":
    pytest.main(['./test_bilinear_form.py', '-k', 'test_matmul'])
