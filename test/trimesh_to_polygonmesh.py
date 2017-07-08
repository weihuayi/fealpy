import sys

import numpy as np

from fealpy.mesh.TriangleMesh import TriangleMeshWithInfinityPoint
from fealpy.mesh import rectangledomainmesh
from fealpy.mesh.PolygonMesh import PolygonMesh

import matplotlib.pyplot as plt

n = 2
box = [0, 1, 0, 1]
mesh = rectangledomainmesh(box, nx=n, ny=n, meshtype='tri') 
nmesh = TriangleMeshWithInfinityPoint(mesh)
ppoint, pcell, pcellLocation =  nmesh.to_polygonmesh()
pmesh = PolygonMesh(ppoint, pcell, pcellLocation)
