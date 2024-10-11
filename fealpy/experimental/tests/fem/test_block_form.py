#!/usr/bin/python3
import pytest

from fealpy.experimental.fem.form import Form
from fealpy.experimental.fem import BlockForm
from fealpy.experimental.fem import BilinearForm
from fealpy.experimental.mesh import TriangleMesh
from fealpy.experimental.functionspace import LagrangeFESpace
from fealpy.experimental.fem import ScalarDiffusionIntegrator
from fealpy.experimental.backend import backend_manager as bm

from fealpy.experimental.sparse import COOTensor, CSRTensor
from fealpy.experimental.tests.fem.block_form_data import *

class TestBlockForm:
    @pytest.mark.parametrize("backend", ['pytorch', 'numpy'])
    @pytest.mark.parametrize("data", triangle_mesh_one_box)
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
        
        matrix = blockform.assembly() 
        true_matrix = COOTensor(bm.array(data["indices"]), bm.array(data["values"]), blockform.shape)
        
        np.testing.assert_array_almost_equal(matrix.toarray(), true_matrix.toarray(), 
                                     err_msg=f" `blockform` function is not equal to real result in backend {backend}")

if __name__ == "__main__":
    pytest.main(['./test_block_form.py', '-k', 'test_diag_diffusion'])


'''
bm.set_backend('numpy')
#bm.set_backend('pytorch')

mesh = TriangleMesh.from_box(box=[0, 1, 0, 1], nx=1, ny=1)
space = LagrangeFESpace(mesh, p=1)
space1 = LagrangeFESpace(mesh, p=2)

bform0 = BilinearForm(space)
bform0.add_integrator(ScalarDiffusionIntegrator())
A1 = bform0.assembly()
bform1 = BilinearForm(space1)
bform1.add_integrator(ScalarDiffusionIntegrator())
A2 = bform1.assembly()

from scipy.sparse import coo_array, bmat
def coo(A):
    data = A._values
    indices = A._indices
    return coo_array((data, indices))

A = bmat([[coo(A1), None],
        [None, coo(A2)]], format='coo')
A = COOTensor(bm.stack([A.row,A.col],axis=0), A.data, spshape=A.shape)
print(A.indices())
print(A.values())
print(A)
'''
