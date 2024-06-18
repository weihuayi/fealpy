import numpy as np
import taichi as ti
import pytest

from fealpy.ti.mesh import MeshDS

ti.init(arch=ti.cuda)

mds = MeshDS(10, 2)
