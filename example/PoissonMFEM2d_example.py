#!/usr/bin/env python3
#

import sys

import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

from fealpy.pde.poisson_2d import CosCosData
from fealpy.mesh.simple_mesh_generator import triangle
from fealpy.mesh import PolygonMesh
from fealpy.functionspace import ConformingVirtualElementSpace2d
from fealpy.boundarycondition import BoundaryCondition
