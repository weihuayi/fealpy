"""
Provide ABCs for topology data class of mesh
"""

from .mesh_ds import (
    MeshDataStructure, HomogeneousMeshDS, StructureMeshDS,
    ArrRedirector
)
from .mesh1d_ds import Mesh1dDataStructure, StructureMesh1dDataStructure
from .mesh2d_ds import Mesh2dDataStructure, StructureMesh2dDataStructure
from .mesh3d_ds import Mesh3dDataStructure, StructureMesh3dDataStructure
