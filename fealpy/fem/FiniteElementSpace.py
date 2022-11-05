import numpy as np


class FiniteElementSpace():
    def __init__(self, mesh):
        self.mesh = mesh
        self.cell_order = None

