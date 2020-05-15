import numpy as np
from timeit import default_timer as timer

class MatlabShow:
    def __init__(self, matlab=None):
        self.matlab = matlab

    def show_solution(self, mesh, uh, fname='test.fig'):
        NC = mesh.number_of_cells()
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        if mesh.meshtype in {'polygon', 'hepolygon'}:
            cell, cellLocation = cell
            cell = [cell[cellLocation[i]:cellLocation[i+1]]+1 for i in range(NC)]
        else:
            cell = list(cell+1)
        self.matlab._call('show_psolution', [node, cell, uh, fname])
