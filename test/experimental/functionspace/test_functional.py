import numpy as np

from fealpy.experimental.backend import backend_manager as bm

from fealpy.common.tensor import symmetry_index as symmetry_index0
from fealpy.common.tensor import symmetry_span_array as symmetry_span_array0

from fealpy.experimental.functionspace.functional import  symmetry_index as symmetry_index1
from fealpy.experimental.functionspace.functional import  symmetry_span_array as symmetry_span_array1

from fealpy.mesh import TetrahedronMesh

bm.set_backend('pytorch')

def test_symmetry_index():
    a    = symmetry_index0(3, 4)
    b, _ = symmetry_index1(3, 4)
    np.testing.assert_array_equal(bm.to_numpy(b), a)

def test_symmetry_span_array():

    t = np.random.rand(3, 3, 3)
    alpha = np.array([2, 2, 1])
    a = symmetry_span_array0(t, alpha)

    t = bm.tensor(t)
    alpha = bm.tensor(alpha)
    b = symmetry_span_array1(t, alpha)

    np.testing.assert_allclose(bm.to_numpy(b), a, 1e-14)

if __name__ == '__main__':
    test_symmetry_index()
    test_symmetry_span_array()
































