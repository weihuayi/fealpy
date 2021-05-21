# Developing Standard 

## 发布新版本步骤
1. 修改如下文件中的 tag 版本号：
.bumpversion.cfg
fealpy/__init__.py
2. add and commit
3. 创建相应版本号的 tag
4. git push github --tags
5. github 页面 realease




## General Python Coding Rule

* ClassName
* keyname
* memberdata
* function_name
* member_function_name
* thisIsFunctionVarable


* ClassName.py
* module_name_file.py


## FEALPy Naming Rules 

### Mesh

The kinds of mesh :

* IntervalMesh: 区间网格
* TriangleMesh: 三角形网格
* QuadrangleMesh: 四边形网格
* TetrahedronMesh: 四面体网格
* HexahedronMesh: 六面体网格
* PolygonMesh: 多边形网格
* PolyhedronMesh: 多面体网格
* StructureQuadMesh: 结构四边形网格
* StructureHexMesh: 结构六面体体网格

* BlockStructureQuadMesh: 
* BlockStructureHexMesh

* Quadtree
* Octree

* Tritree
* BitreeMesh

The name of basic mesh entity:
* node
* cell

* edge
* face

The name rules of the number of entity:

* NN: the Number of Nodes 
* NE: the Number of Edges 
* NF: the Number of Faces
* NC: the Number of Cells 
* NQ: the Number of Quadrature points
* GD: the Geometry Dimension
* TD: the Toplogy Dimension 
* NFV: the Number of Face Vertices 
* NCV: the Number of Cell Vertices 

# c: 单元指标
# f: 面指标
# e: 边指标
# v: 顶点个数指标
# i, j, k, d: 自由度或基函数指标
# q: 积分点或重心坐标点指标
# m, n: 空间或拓扑维数指标


The basic member data in a mesh:
* node: a numpy array with shape `(NN, GD)`
* ds : the toplogy data structure, which have the following data member
    + NN: the number of nodes 
    + NC: the number of cells
    + edge: a numpy array with shape `(NE, 2)`
    + face: a numpy array with shape `(NF, NFV)`
    + cell: a numpy array iwth shape `(NC, NCV)`

other member datas:
* nodedata = {}
* celldata = {}

* edgedata = {}
* facedata = {}


The general member function of a mesh

* geo_dimension()
* top_dimension()

* number_of_nodes()
* number_of_cells()
* number_of_edges()
* number_of_faces()
* number_of_entities(etype)

* entity(etype)
* entity_measure(etype)
* entity_barycenter(etype) 

* integrator(index) : get the quadrature formula on this kind of mesh
* bc_to_points(bc) : transform quadrature points in barycentric coordinates to points in pysical cells

## the toplogy data structure class

For every mesh, there is toplogy data member `ds`, which contain the toplogy
data of mesh. 

The toplogy relationship function in `ds`:
*  cell_to_node(...): return `cell2node` 
*  cell_to_edge(...): return `cell2edge` 
*  cell_to_face(...): return `cell2face` 
*  cell_to_cell(...): return `cell2cell` 
*  face_to_node(...): return `face2node` 
*  face_to_edge(...): return `face2edge`
*  face_to_face(...): return `face2face`
*  face_to_cell(...): return `face2cell`
*  edge_to_node(...): return `edge2node`
*  edge_to_edge(...): return `edge2edge`
*  edge_to_face(...): return `edge2face`
*  edge_to_cell(...): return `edge2cell`
*  node_to_node(...): return `node2node`
*  node_to_edge(...): return `node2edge`
*  node_to_face(...): return `node2face`
*  node_to_cell(...): return `node2cell`

## Function Space

### degrees of freedom management 

* ldof: the number of local dof on each cell
* gdof: the number of global dof on the whole mesh

(NQ, NC, ldof, m, n, ...)

The member function in a `FunctionSpace`:

* number_of_global_dofs()
* number_of_local_dofs()
* cell_to_dof()
* boundary_dof()
* basis(bc)
* gradbasis(bc)
* function()

* interpolation(u):
* projection(u): 

* array(dim=None)

* value(u, bc, cellidx=None)
* grad_value(u, bc, cellidx=None)
* div_value(u, bc, cellidx=None)

## 指标循环约定 

 c: 单元指标
 f: 面指标
 e: 边指标
 v: 顶点个数指标
 i, j, k, d: 自由度或基函数指标
 q: 积分点或重心坐标点指标
 m, n: 空间或拓扑维数指标
