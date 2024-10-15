
import pytest

import mindspore as ms
import mindspore.numpy as mnp

from fealpy.backend import backend_manager as bm

bm.set_backend('mindspore')

node = bm.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]], dtype=bm.float_) # (NN, 2)

edge = bm.array([
     [1,0],
     [0,2],
     [3,0],
     [3,1],
     [2,3]], dtype=bm.int_)

cell = bm.array([
     [2,3,0],
     [1,0,3]], dtype=bm.int_)

def test_multi_index_matrix():
    m = bm.multi_index_matrix(3, 2)
    print(m)

