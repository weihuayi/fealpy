"""
Provide plotting supports for the mesh module, based on matplotlib.
"""

from .classic import MeshPloter, get_ploter
from .classic import AddPlot1d, AddPlot2dHomo, AddPlot2dPoly, AddPlot3dHomo
from .artist import show_index, show_multi_index, scatter, line, poly, poly_
