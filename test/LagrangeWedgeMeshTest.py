#!/usr/bin/env python3
# 

import sys

import numpy as np

import matplotlib.pyplot as plt

from fealpy.mesh import LagrangeWedgeMesh, MeshFactory
from fealpy.mesh.vtk_extent import vtk_cell_index


class LagrangeWedgeMeshTest():

    def __init__(self):
        pass

    def data_structure(self):

        node = np.array([
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 0.0),
                (1.0, 0.0, 1.0),
                (0.0, 1.0, 0.0),
                (0.0, 1.0, 1.0)], dtype=np.float64)
        cell = np.array([(0, 1, 2, 3, 4, 5)], dtype=np.int_)

        mesh = LagrangeWedgeMesh(node, cell, (2, 2))
        index = vtk_cell_index((2, 2), 73)
        print(2*mesh.node[0, index])




test = LagrangeWedgeMeshTest()

test.data_structure()

