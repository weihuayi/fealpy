import numpy as np

from fealpy.backend import backend_manager as bm

from fealpy.common.tensor import symmetry_index as symmetry_index0
from fealpy.common.tensor import symmetry_span_array as symmetry_span_array0

from fealpy.functionspace.functional import  symmetry_index as symmetry_index1
from fealpy.functionspace.functional import  symmetry_span_array as symmetry_span_array1

from fealpy.mesh import TetrahedronMesh

#bm.set_backend('pytorch')
symmetry_span_array_data = [
        {'t': 0,
         'alpha': [2, 1, 3],
         'symt': 0,
         'd': 3
        },
        {'t': 0,
         'alpha': [2, 1, 2, 1],
         'symt': 0,
         'd': 2
        },
        {'t': 0,
         'alpha': [2],
         'symt': 0,
         'd': 3
        },
        {'t': 0,
         'alpha': [3, 2],
         'symt': 0,
         'd': 2
        },
]


symmetry_index_data = [
        { 'd' : 2,
          'r' : 5,
          'symidx' : 0,
          'num' : 0
        },
        { 'd' : 3,
          'r' : 4,
          'symidx' : 0,
          'num' : 0
        },
        { 'd' : 4,
          'r' : 6,
          'symidx' : 0,
          'num' : 0
        }
        ]
def test_symmetry_index():
    a    = symmetry_index0(3, 8)
    b, _ = symmetry_index1(3, 8)
    np.testing.assert_array_equal(bm.to_numpy(b), a)

def test_symmetry_span_array():

    t = np.random.rand(3, 3, 3)
    alpha = np.array([2, 2, 1])
    a = symmetry_span_array0(t, alpha)

    t = bm.tensor(t)
    alpha = bm.tensor(alpha)
    b = symmetry_span_array1(t, alpha)

    np.testing.assert_allclose(bm.to_numpy(b), a, 1e-14)

def generate_data():
    np.set_printoptions(precision=16)
    for data in symmetry_index_data:
        d = data['d']
        r = data['r']
        a = symmetry_index0(d, r)
        _, num = symmetry_index1(d, r)
        data['symidx'] = a
        data['num'] = num
    print(symmetry_index_data)

    for data in symmetry_span_array_data:
        d = data['d']
        alpha = data['alpha']
        l = len(alpha)
        t = np.random.rand(3, l, d)
        symt = symmetry_span_array0(t, alpha)
        data['symt'] = symt
        data['t'] = t
    print(symmetry_span_array_data)


if __name__ == '__main__':
    test_symmetry_index()
    test_symmetry_span_array()
    generate_data()





