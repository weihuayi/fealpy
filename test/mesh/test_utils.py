
import pytest

import numpy as np
from fealpy.backend import backend_manager as bm
from fealpy.mesh.utils import inverse_relation

inverse_relation_with_index_data = [
    {
        "entity": np.array(
            [[0, 3, 4], [0, 2, 1], [1, 4, 5], [1, 5, 2],
             [3, 6, 7], [3, 7, 4], [4, 7, 8], [4, 8, 5]],
            dtype=np.int32),
        "size": 9,
        "index": np.array([0, 1, 2, 3, 4, 5]),
        "row": np.array([0, 3, 4, 0, 2, 1, 1, 4, 5, 1, 5, 2, 3, 3, 4, 4, 4, 5], dtype=np.int32),
        "col": np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 5, 6, 7, 7], dtype=np.int32),
        "spshape": (9, 8)
    }
]


@pytest.mark.parametrize('data', inverse_relation_with_index_data)
@pytest.mark.parametrize('backend', ['numpy', 'pytorch', 'jax'])
def test_inverse_relation_with_index(data, backend):
    bm.set_backend(backend)

    entity = bm.from_numpy(data['entity'])
    size = data['size']
    index = bm.from_numpy(data['index'])
    row, col, spshape = inverse_relation(entity, size, index)

    assert bm.all(bm.equal(row, bm.from_numpy(data['row'])))
    assert bm.all(bm.equal(col, bm.from_numpy(data['col'])))
    assert spshape == data['spshape']


inverse_relation_with_flag_data = [
    {
        "entity": np.array(
            [[0, 3, 4], [0, 2, 1], [1, 4, 5], [1, 5, 2],
             [3, 6, 7], [3, 7, 4], [4, 7, 8], [4, 8, 5]],
            dtype=np.int32),
        "size": 9,
        "flag": np.array([True, True, False, True, True, False, True, True, False], dtype=np.bool),
        "row": np.array([0, 3, 4, 0, 1, 1, 4, 1, 3, 6, 7, 3, 7, 4, 4, 7, 4], dtype=np.int32),
        "col": np.array([0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7], dtype=np.int32),
        "spshape": (9, 8)
    }
]

@pytest.mark.parametrize('data', inverse_relation_with_flag_data)
@pytest.mark.parametrize('backend', ['numpy', 'pytorch', 'jax'])
def test_inverse_relation_with_flag(data, backend):
    bm.set_backend(backend)

    entity = bm.from_numpy(data['entity'])
    size = data['size']
    index = bm.from_numpy(data['flag'])
    row, col, spshape = inverse_relation(entity, size, index)

    assert bm.all(bm.equal(row, bm.from_numpy(data['row'])))
    assert bm.all(bm.equal(col, bm.from_numpy(data['col'])))
    assert spshape == data['spshape']
