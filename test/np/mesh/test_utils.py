import pytest
import ipdb

import numpy as np

from fealpy.mesh import TriangleMesh as Mesh

from fealpy.np import logger
from fealpy.np.mesh.utils import *


def test_estr2dim():

    mesh = Mesh.from_box(nx=2, ny=2)

    node = mesh.entity('node')
    cell = mesh.entity('cell')
    edge = mesh.entity('edge')

    estr1 = estr2dim(mesh, 'node')
    estr2 = estr2dim(mesh, 'cell')
    estr3 = estr2dim(mesh, 'edge')

    print('edim1:', estr1)
    print('edim2:', estr2)
    print('edim3:', estr3)

def test_arr_to_csr():
    pass

if __name__ == "__main__":
    test_estr2dim()
