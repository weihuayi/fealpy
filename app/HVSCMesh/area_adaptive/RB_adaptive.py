import numpy as np
import gmsh

#from fealpy.mesh import TriangleMesh
from fealpy.old.mesh import TriangleMesh

class Mesh:
    def __init__(self):
        self.vtags, vxyz, _ = gmsh.model.mesh.getNodes()
        self.vxyz = vxyz.reshape((-1, 3))
        vmap = dict({j: i for i, j in enumerate(self.vtags)})
        self.triangles_tags, evtags = gmsh.model.mesh.getElementsByType(2)
        evid = np.array([vmap[j] for j in evtags])
        self.triangles = evid.reshape((self.triangles_tags.shape[-1], -1))

gmsh.initialize()
gmsh.model.add('RB_IGCT')
gmsh.merge('case_4_RB_IGCT.msh')

ntags, vxyz, _ = gmsh.model.mesh.getNodes()
node = vxyz.reshape((-1,3))
node = node[:,:2]
vmap = dict({j:i for i,j in enumerate(ntags)})
tris_tags,evtags = gmsh.model.mesh.getElementsByType(2)
evid = np.array([vmap[j] for j in evtags])
cell = evid.reshape((tris_tags.shape[-1],-1))

gmesh = Mesh()
mesh = TriangleMesh(node,cell)
NN = mesh.number_of_nodes()
cell = mesh.entity('cell')

area = mesh.entity_measure('cell')

element_size = np.sqrt(4*area/np.sqrt(3))
mesh.celldata["fist_element_size"] = element_size
nodesize = np.zeros(NN)

node2cell = mesh.ds.node_to_cell()
#node2cell = mesh.node_to_cell()
node2cell = node2cell.astype(float)

nodesize += node2cell.dot(element_size)

Nn2cs = np.array(np.sum(mesh.ds.node_to_cell(),axis=1)).reshape(-1)
nodesize = nodesize/Nn2cs
mesh.nodedata["first_nodesize"] = nodesize

for i in range(5):
    elementsize = (nodesize[cell[:,0]]+nodesize[cell[:,1]]+nodesize[cell[:,2]])/3
    nodesize = np.zeros(NN)
    nodesize += node2cell.dot(elementsize)
    nodesize = nodesize/Nn2cs

bgview = gmsh.view.add("bg")

gmsh.view.addModelData(bgview, 0, "RB_IGCT", "NodeData", gmesh.vtags,
        nodesize[:,None])
mesh.nodedata["nodesize"] = nodesize
#mesh.celldata["element_size"] = elementsize
#mesh.to_vtk(fname='test_RB.vtu')
gmsh.model.add("RB_IGCT")
gmsh.model.geo.addPoint(0,0,0,tag=1)
gmsh.model.geo.addPoint(92,0,0,tag=2)
gmsh.model.geo.addPoint(110,18,0,tag=3)
gmsh.model.geo.addPoint(250,18,0,tag=4)
gmsh.model.geo.addPoint(250,1500,0,tag=5)
gmsh.model.geo.addPoint(0,1500,0,tag=6)
gmsh.model.geo.addPoint(110,0,0,tag=7)

gmsh.model.geo.addLine(1,2,tag=1)
gmsh.model.geo.addCircleArc(2,7,3,tag=2)
gmsh.model.geo.addLine(3,4,tag=3)
gmsh.model.geo.addLine(4,5,tag=4)
gmsh.model.geo.addLine(5,6,tag=5)
gmsh.model.geo.addLine(6,1,tag=6)

gmsh.model.geo.addCurveLoop([1,2,3,4,5,6],tag=1)
gmsh.model.geo.addPlaneSurface([1],tag=1)

gmsh.model.geo.synchronize()
bg_field = gmsh.model.mesh.field.add("PostView")                                
gmsh.model.mesh.field.setNumber(bg_field, "ViewTag", bgview)                    
gmsh.model.mesh.field.setAsBackgroundMesh(bg_field)                             
#gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)                            
#gmsh.option.setNumber("Mesh.MeshSizeMax", 20)
gmsh.model.mesh.optimize('', True)
gmsh.model.mesh.generate(2)
gmsh.fltk.run()

ntags, vxyz, _ = gmsh.model.mesh.getNodes()
node = vxyz.reshape((-1,3))
node = node[:,:2]
vmap = dict({j:i for i,j in enumerate(ntags)})
tris_tags,evtags = gmsh.model.mesh.getElementsByType(2)
evid = np.array([vmap[j] for j in evtags])
cell = evid.reshape((tris_tags.shape[-1],-1))

mesh = TriangleMesh(node,cell)
angle = mesh.angle()
max_angle = np.max(angle,axis=1)
angles = max_angle*(180/np.pi)
m90 = np.sum(angles>90)
mesh.to_vtk(fname='area_RB_adaptive.vtu')
gmsh.finalize()


