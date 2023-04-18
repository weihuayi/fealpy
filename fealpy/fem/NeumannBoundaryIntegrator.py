import numpy as np


class NeumannBoundaryIntegrator:

    def __init__(self, space, gN, threshold=None):
        self.space = space
        self.gN = gN
        self.threshold = threshold


    def apply(self, f, A=None):
        pass
