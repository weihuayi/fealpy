"""
Provide plotting supports for the mesh module, based on matplotlib.
"""

from .classic import Plotable, MeshPloter
from .classic import AddPlot1d, AddPlot2dHomo, AddPlot2dPoly, AddPlot3d
from .artist import show_index, scatter, line, poly, poly_
