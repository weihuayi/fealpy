import numpy as np

import taichi as ti

from fealpy.ti.mesh import TriangleMesh

ti.init(arch=ti.cuda)

mesh = TriangleMesh.from_box()

mesh.ds.print_data()




