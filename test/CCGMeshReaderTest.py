

import sys
import numpy as np

from fealpy.mesh import CCGMeshReader


def CCGMeshReader_test(fname):
    reader = CCGMeshReader(fname)
    reader.read()

fname = sys.argv[1]

CCGMeshReader_test(fname)



