import numpy as np
from .GaussLobattoQuadrature import GaussLobattoQuadrature
from .GaussLegendreQuadrature import GaussLegendreQuadrature

class PolyhedronMeshIntegralAlg():
    def __init__(self, mesh, q, cellmeasure=None, cellbarycenter=None):
        self.mesh = mesh
        self.cellmeasure = cellmeasure if cellmeasure is not None \
                else mesh.entity_measure('cell')
        self.cellbarycenter = cellbarycenter if cellbarycenter is not None \
                else mesh.entity_barycenter('cell')
        self.cellintegrator = pmesh.integrator(q)

        self.facemeasure = mesh.entity_measure('face')
        self.facebarycenter = mesh.entity_barycenter('face')

        self.edgemeasure = mesh.entity_measure('edge')
        self.edgebarycenter = mesh.entity_barycenter('edge')
