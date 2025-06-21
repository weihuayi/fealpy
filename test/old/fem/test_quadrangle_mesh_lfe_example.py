import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import pytest
import ipdb

from fealpy.functionspace import LagrangeFESpace as Space

from fealpy.fem import ScalarDiffusionIntegrator
from fealpy.fem import ScalarSourceIntegrator

from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm
from fealpy.fem import DirichletBC
from fealpy.fem import ScalarNeumannBCIntegrator

@pytest.mark.parametrize("p, n, maxit", 
        [(1, 8, 4), (2, 6, 4), (3, 4, 4), (4, 4, 4)])
def test_dirichlet_bc_on_quadrangele_mesh(p, n, maxit):
    pass

@pytest.mark.parametrize("p, n, maxit", 
        [(1, 8, 4), (2, 6, 4), (3, 4, 4), (4, 4, 4)])
def test_neumann_bc_on_quadrangle_mesh(p, n, maxit):
    pass

@pytest.mark.parametrize("p, n, maxit", 
        [(1, 8, 4), (2, 6, 4), (3, 4, 4), (4, 4, 4)])
def test_robin_bc_on_quadrangle_mesh(p, n, maxit):
    pass

@pytest.mark.parametrize("p, n, maxit", 
        [(1, 8, 4), (2, 6, 4), (3, 4, 4), (4, 4, 4)])
def test_mixed_bc_on_quadrangle_mesh(p, n, maxit):
    pass

