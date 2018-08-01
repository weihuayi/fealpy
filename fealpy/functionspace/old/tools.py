import numpy as np

from .lagrange_fem_space import LagrangeFiniteElementSpace2d 
from .lagrange_fem_space import LagrangeFiniteElementSpace3d
from .lagrange_fem_space import VectorLagrangeFiniteElementSpace2d 
from .bi_fem_space import QuadrangleFiniteElementSpace 
from .bi_fem_space import  HexahedronFiniteElementSpace
from .function import FiniteElementFunction

def function_space(mesh, femtype, p, dtype=np.float):
    if femtype is 'Lagrange':
        if mesh.meshtype is 'tri':
            return LagrangeFiniteElementSpace2d(mesh, p, dtype=dtype)
        if mesh.meshtype is 'tet':
            return LagrangeFiniteElementSpace3d(mesh, p, dtype=dtype)
    elif femtype is "Lagrange_2":
        if mesh.meshtype is 'tri':
            return VectorLagrangeFiniteElementSpace2d(mesh, p, dtype=dtype)
    elif femtype is "Q":
        if mesh.meshtype is 'quad':
            return QuadrangleFiniteElementSpace(mesh, p, dtype=dtype)
        if mesh.meshtype is 'hex':
            return HexahedronFiniteElementSpace(mesh, p, dtype=dtype) 
