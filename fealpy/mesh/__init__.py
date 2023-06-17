'''
mesh
====

This module provide mesh

'''

# 结构化网格
from .interval_mesh import IntervalMesh
from .polygon_mesh import PolygonMesh
from .triangle_mesh import TriangleMesh, TriangleMeshWithInfinityNode
from .quadrangle_mesh import QuadrangleMesh
from .tetrahedron_mesh import TetrahedronMesh
from .hexahedron_mesh import HexahedronMesh
from .edge_mesh import EdgeMesh

from .UniformMesh1d import UniformMesh1d
from .UniformMesh2d import UniformMesh2d
from .UniformMesh3d import UniformMesh3d
from .StructureQuadMesh import StructureQuadMesh
from .StructureHexMesh import StructureHexMesh



from .LagrangeTriangleMesh import LagrangeTriangleMesh
from .LagrangeQuadrangleMesh import LagrangeQuadrangleMesh
from .LagrangeHexahedronMesh import LagrangeHexahedronMesh
from .LagrangeWedgeMesh import LagrangeWedgeMesh

#from .Tritree import Tritree
#from .Quadtree import Quadtree
#from .Octree import Octree
#from .QuadtreeForest import QuadtreeMesh, QuadtreeForest

from .HalfEdgeDomain import HalfEdgeDomain
from .HalfEdgeMesh2d import HalfEdgeMesh2d
from .DartMesh3d import DartMesh3d

from .PolyFileReader import PolyFileReader
from .InpFileReader import InpFileReader
from .CCGMeshReader import CCGMeshReader
from .FABFileReader import FABFileReader

from .meshio import load_mat_mesh

# Mesher
#from .DistMesher2d import DistMesher2d
#from .DistMesher3d import DistMesher3d
#from .CVTPMesher import CVTPMesher, VoroAlgorithm
