import numpy as np
import taichi as ti
import pytest
import fealpy.ti.functional as F

ti.init(arch=ti.cpu)

def test_from_numpy():
    np_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    ti_field = F.from_numpy(np_array)

    assert ti_field.shape == np_array.shape
    assert ti_field.dtype == ti.i32

    np.testing.assert_array_equal(ti_field.to_numpy(), np_array)

def test_zeros():
    shape = (3, 4)
    dtype = ti.f32
    ti_field = F.zeros(shape, dtype)

    assert ti_field.shape == shape
    assert ti_field.dtype == dtype

    np_field = ti_field.to_numpy()
    assert np.all(np_field == 0)

def test_ones():
    shape = (3, 4)
    dtype = ti.f32
    ti_field = F.ones(shape, dtype)

    assert ti_field.shape == shape
    assert ti_field.dtype == dtype

    np_field = ti_field.to_numpy()
    assert np.all(np_field == 1)

def test_arange():
    start, stop, step = 0, 5, 1
    ti_field = F.arange(start, stop, step, ti.i32)

    expected = np.arange(start, stop, step, dtype=np.int32)

    assert ti_field.shape == (len(expected),)
    assert ti_field.dtype == ti.i32

    np.testing.assert_array_equal(ti_field.to_numpy(), expected)

    start, stop, step = 1, 10, 2
    ti_field = F.arange(start, stop, step, ti.f32)

    expected = np.arange(start, stop, step, dtype=np.float32)

    assert ti_field.shape == (len(expected),)
    assert ti_field.dtype == ti.f32

    np.testing.assert_array_equal(ti_field.to_numpy(), expected)

if __name__ == "__main__":
    test_arange()
    pytest.main([__file__])

