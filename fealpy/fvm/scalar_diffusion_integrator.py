#!/usr/bin/python3
import numpy as np
class ScalarDiffusionIntegrator:
    def __init__(self, mesh, c=None):
        self.c = c
        self.mesh = mesh

    def cell_center_matrix(self, bf):
        c = self.c
        mesh = self.mesh
        edge = mesh.entity('edge')
        A = 1
        b = 2
        return A,b

    def node_center_matrix(self):
        pass
