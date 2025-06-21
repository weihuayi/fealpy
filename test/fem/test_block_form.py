#!/usr/bin/python3
import pytest

from fealpy.fem.form import Form
from fealpy.fem import BlockForm
from fealpy.fem import BilinearForm
from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import ScalarDiffusionIntegrator
from fealpy.backend import backend_manager as bm

from fealpy.sparse import COOTensor, CSRTensor
from block_form_data import *

class TestBlockForm:
    @pytest.mark.parametrize("backend", ['numpy','pytorch'])
    @pytest.mark.parametrize("data", diag_diffusion)
    def test_diag_diffusion(self, backend, data): 
        bm.set_backend(backend)
        
        mesh = TriangleMesh.from_box(*data["args"])
        space = LagrangeFESpace(mesh, p=1)
        space1 = LagrangeFESpace(mesh, p=2)

        bform0 = BilinearForm(space)
        bform0.add_integrator(ScalarDiffusionIntegrator())
        bform1 = BilinearForm(space1)
        bform1.add_integrator(ScalarDiffusionIntegrator())
        
        blockform = BlockForm([[bform0,None],[None,bform1]])
        
        a = bm.arange(blockform.shape[1], dtype=mesh.ftype)
        vector = blockform @ a

        matrix = blockform.assembly() 
        true_matrix = COOTensor(bm.array(data["indices"],dtype=mesh.itype), bm.array(data["values"], dtype=mesh.ftype), blockform.shape) 

        true_vector = true_matrix.toarray() @ a

        np.testing.assert_array_almost_equal(matrix.toarray(), true_matrix.toarray(), 
                                     err_msg=f" `blockform` function is not equal to real result in backend {backend}")
        np.testing.assert_array_almost_equal(vector, true_vector, 
                                     err_msg=f" `blockform __mult__` function is not equal to real result in backend {backend}")

if __name__ == "__main__":
    pytest.main(['./test_block_form.py', '-sk', 'test_diag_diffusion'])


