# Developing Standard 

## General Python Coding Rule

* ClassName
* keyname
* memberdata
* function_name
* member_function_name
* thisFunctionVarable


* ClassName.py
* module_name_file.py


## FEALPy Naming Rules 

### Mesh

The kinds of mesh :

* IntervalMesh
* TriangleMesh
* QuadrangleMesh
* TetrahedronMesh
* HexahedronMesh
* PolygonMesh
* PolyhedronMesh
* StructureQuadMesh
* StructureHexMesh
* BlockStructureQuadMesh
* BlockStructureHexMesh
* QuadtreeMesh
* OctreeMesh
* TritreeMesh
* BitreeMesh

* MeshNameDataStructure

The basic entity array in a mesh:
*  node 
*  cell

The addition entity array in a mesh
*  edge: appear in 2d and 3d mesh
*  face: appear in 3d mesh

The toplogy relationship array or sparse matrix in mesh:
*  cell2node
*  cell2edge
*  cell2face
*  cell2cell
*  face2node
*  face2edge
*  face2face
*  face2cell
*  edge2node
*  edge2edge
*  edge2face
*  edge2cell
*  node2node
*  node2edge
*  node2face
*  node2cell

other data:
* nodedata = {}
* celldata = {}
* edgedata = {}
* facedata = {}

The number of entity:

* N: the number of nodes 
* NC: the number of cells 
* NE: the number of edges 
* NF: the number of faces
* NQ: the number of quadrature points

The general member function of a mesh

* geo_dimension(self)
* top_dimension(self)
* number_of_nodes(self)
* number_of_cells(self)
* number_of_edges(self)
* number_of_faces(self)
* number_of_entities(self, dim)
* entity(self, dim)
* entity_measure(self, dim)
* entity_barycenter(self, dim) : 
* integrator(self, index) : get the quadrature formula on this kind of mesh
* bc_to_points(self, bc) : transform quadrature points in barycentric coordinates to points in pysical cells

## Function Space

### degrees of freedom management 

* ldof: the number of local dof on each cell
* gdof: the number of global dof on the whole mesh
* dim: the space dimension 

(NQ, NC, ldof, m, n, ...)

The member function in a `FunctionSpace`:

* cell_to_dof(self)
* boundary_dof(self)
* basis(self, ...)
* function(self)
* number_of_global_dofs(self)
* number_of_local_dofs(self)
* function(self)
* interpolation(self, u):
* projection(self, u): 
* array(self, dim=None)
* value(self, u, bc, cellidx=None)
* grad_value(self, u, bc, cellidx=None)
* div_value(self, u, bc, cellidx=None)
* geo_dimension(self)
* top_dimension(self)
