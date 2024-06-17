import pytest
import ipdb

import numpy as np

from fealpy.mesh import TriangleMesh as Mesh

from fealpy.np import logger
from fealpy.np.mesh.utils import *
from fealpy.np.mesh.functional import *


mesh = Mesh.from_box(nx=2, ny=2)

def test_multi_index_matrix():

    TD = mesh.top_dimension()
    p = 2

    multiIndex = multi_index_matrix(p, TD)

    assert(multiIndex == )

def test_simplex_shape_function():
    pass

def test_grad_simplex_shape_sunction():
    pass

if __name__ == "__main__":
    test_multi_index_matrix()
    test_simplex_shape_function()
    test_grad_simplex_shape_sunction()
