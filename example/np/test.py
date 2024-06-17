import numpy as np

CONTEXT = 'numpy'

from fealpy.mesh import TriangleMesh as TMD
from fealpy.utils import timer
from fealpy.decorator import cartesian

from fealpy.np.mesh import TriangleMesh
from fealpy.torch.mesh import TriangleMesh as tri
from fealpy.np.functionspace import LagrangeFESpace
from fealpy.np.fem import (
    BilinearForm, LinearForm,
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    DirichletBC
)
from scipy.sparse.linalg import spsolve

from matplotlib import pyplot as plt

from typing import Sequence

NX, NY = 64, 64

PI = np.pi

mesh = TriangleMesh.from_box(nx=NX, ny=NY)
mesh_ = tri.from_box(nx=NX, ny=NY)
node = mesh.entity('node')
node_ = mesh_.entity('node')
print(node_.shape)


