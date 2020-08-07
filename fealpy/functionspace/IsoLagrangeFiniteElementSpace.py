import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix, spdiags, bmat
from scipy.sparse.linalg import spsolve

from ..decorator import barycentric

from .Function import Function

from .femdof import multi_index_matrix1d
from .femdof import multi_index_matrix2d
from .femdof import multi_index_matrix3d

from .femdof import CPLFEMDof1d, CPLFEMDof2d, CPLFEMDof3d
from .femdof import DPLFEMDof1d, DPLFEMDof2d, DPLFEMDof3d

from ..quadrature import FEMeshIntegralAlg
from ..decorator import timer
