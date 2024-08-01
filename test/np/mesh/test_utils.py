import pytest
import ipdb

import numpy as np
from scipy.sparse import csr_matrix

from fealpy.mesh import TriangleMesh as Mesh

from fealpy.np import logger
from fealpy.np.mesh.utils import *


def test_estr2dim():

    mesh = Mesh.from_box(nx=1, ny=1)

    node = mesh.entity('node')
    cell = mesh.entity('cell')
    edge = mesh.entity('edge')
    
    estr1 = estr2dim(mesh, 'node')
    estr2 = estr2dim(mesh, 'cell')
    estr3 = estr2dim(mesh, 'edge')

    assert(estr1 == 0)
    assert(estr2 == 2)
    assert(estr3 == 1)

def test_arr_to_csr():

    arr = np.array([[0, 1, 3], [2, 3, 1]])

    row = np.array([0, 0, 0, 1, 1, 1])
    col = np.array([0, 1, 3, 1, 2, 3])
    data = np.array([1, 1, 1, 1, 1, 1])
    csr_m = csr_matrix((data, (row, col)), shape=(2, 4))

    matrix = arr_to_csr(arr).astype(np.int_)

    err0 = np.abs(csr_m - matrix)
    err = np.max(err0)

    assert(err == 0)

if __name__ == "__main__":
    test_estr2dim()
    test_arr_to_csr()
