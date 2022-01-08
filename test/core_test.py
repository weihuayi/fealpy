import pytest
import taichi as ti
import numpy as np
from opt_einsum import contract

from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.ti import lagrange_cell_stiff_matrix_0
from fealpy.ti import lagrange_cell_stiff_matrix_1

from timeit import default_timer as dtimer 

