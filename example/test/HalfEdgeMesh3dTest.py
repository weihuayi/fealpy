#!/usr/bin/env python3
# 

import numpy as np
import matplotlib.pyplot as plt


from fealpy.mesh import MeshFactory
from fealpy.mesh import HalfEdgeMesh3d


class HalfEdgeMesh3dTest:

    def __init__(self):
        self.meshfactory = MeshFactory()

    def one_tetrahedron_mesh_test(self):
        tmesh = self.meshfactory.one_tetrahedron_mesh(ttype='equ')
        mesh = HalfEdgeMesh3d.from_mesh(tmesh)
        mesh.print()



test = HalfEdgeMesh3dTest()

test.one_tetrahedron_mesh_test()

