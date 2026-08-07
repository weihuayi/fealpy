import gmsh

from fealpy.backend import backend_manager as bm
from fealpy.mesh import TetrahedronMesh
from fealpy.mesh import TriangleMesh

def to_TetrahedronMesh():
    ntags, vxyz, _ = gmsh.model.mesh.getNodes()
    node = vxyz.reshape((-1,3))
    vmap = dict({j:i for i,j in enumerate(ntags)})
    tets_tags,evtags = gmsh.model.mesh.getElementsByType(4)
    evid = bm.array([vmap[j] for j in evtags])
    cell = evid.reshape((tets_tags.shape[-1],-1))
    return TetrahedronMesh(node,cell)

def to_TriangleMesh():
    ntags, vxyz, _ = gmsh.model.mesh.getNodes()
    node = vxyz.reshape((-1,3))
    node = node[:,:2]
    vmap = dict({j:i for i,j in enumerate(ntags)})
    tris_tags,evtags = gmsh.model.mesh.getElementsByType(2)
    evid = bm.array([vmap[j] for j in evtags])
    cell = evid.reshape((tris_tags.shape[-1],-1))
    return TriangleMesh(node,cell)
    
def to_TriangleMesh_Surface():
    ntags, vxyz, _ = gmsh.model.mesh.getNodes()
    node = vxyz.reshape((-1,3))
    vmap = dict({j:i for i,j in enumerate(ntags)})
    tris_tags,evtags = gmsh.model.mesh.getElementsByType(2)
    evid = bm.array([vmap[j] for j in evtags])
    cell = evid.reshape((tris_tags.shape[-1],-1))
    return TriangleMesh(node,cell)
    
def unit_sphere_surface(h=0.1):
    gmsh.initialize()
    gmsh.model.occ.addSphere(0.0,0.0,0.0,1,1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0),h)
    gmsh.option.setNumber("Mesh.Optimize",0)
    gmsh.model.mesh.generate(2)

    mesh = to_TriangleMesh_Surface()
    gmsh.finalize()
    return mesh

def unit_sphere(h=0.1):
    gmsh.initialize()
    gmsh.model.occ.addSphere(0.0,0.0,0.0,1,1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0),h)
    gmsh.option.setNumber("Mesh.Optimize",0)
    gmsh.model.mesh.generate(3)

    mesh = to_TetrahedronMesh()
    gmsh.finalize()
    return mesh
    
def LShape(h=0.05):
    gmsh.initialize()
    gmsh.model.occ.addBox(0,0,0,1,1,1,1)
    gmsh.model.occ.addBox(0,0,0,0.5,0.5,0.5,2)
    gmsh.model.occ.cut([(3,1)],[(3,2)],3)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0),h)
    gmsh.option.setNumber("Mesh.Optimize",0)
    gmsh.model.mesh.generate(3)
    mesh = to_TetrahedronMesh()
    return mesh

def intersect_spheres(h=0.1):
    gmsh.initialize()
    gmsh.model.occ.addSphere(1.0,0.0,0.0,0.7,1)
    gmsh.model.occ.addSphere(-1.0,0.0,0.0,0.7,2)
    gmsh.model.occ.addSphere(0.5, 0.866025403784439,0.0,0.7,3)
    gmsh.model.occ.addSphere(-0.5,0.866025403784439,0.0,0.7,4)
    gmsh.model.occ.addSphere(0.5,-0.866025403784439,0.0,0.7,5)
    gmsh.model.occ.addSphere(-0.5,-0.866025403784439,0.0,0.7,6)
    gmsh.model.occ.addSphere(2.0, 0.0,0.0,0.7,7)
    gmsh.model.occ.addSphere(-2.0,0.0,0.0,0.7,8)
    gmsh.model.occ.addSphere(1.0,1.73205080756888,0.0,0.7,9)
    gmsh.model.occ.addSphere(-1.0,1.73205080756888,0.0,0.7,10)
    gmsh.model.occ.addSphere(-1.0,-1.73205080756888,0.0,0.7,11)
    gmsh.model.occ.addSphere(1.0,-1.73205080756888,0.0,0.7,12)
    gmsh.model.occ.fuse([(3,1),(3,2),(3,3),(3,4),(3,5),(3,6),(3,7)],[(3,8),(3,9),(3,10),(3,11),(3,12)],13)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0),h)
    gmsh.option.setNumber("Mesh.Optimize",0)
    gmsh.model.mesh.generate(3)
    mesh = to_TetrahedronMesh()
    return mesh

def square_hole(h=0.05):
    gmsh.initialize()
    lc = h

    gmsh.model.geo.addPoint(0, 0, 0, lc, 1)
    gmsh.model.geo.addPoint(1, 0, 0, lc, 2)
    gmsh.model.geo.addPoint(1, 1, 0, lc, 3)
    gmsh.model.geo.addPoint(0, 1, 0, lc, 4)

    gmsh.model.geo.addLine(1, 2, 1)
    gmsh.model.geo.addLine(3, 2, 2)
    gmsh.model.geo.addLine(3, 4, 3)
    gmsh.model.geo.addLine(4, 1, 4)

    gmsh.model.geo.addCurveLoop([4, 1, -2, 3], 1)

    gmsh.model.geo.addPoint(0.5,0.5,0,lc,5)
    gmsh.model.geo.addPoint(0.3,0.5,0,lc,6)
    gmsh.model.geo.addPoint(0.7,0.5,0,lc,7)

    gmsh.model.geo.addCircleArc(6,5,7,tag=5)
    gmsh.model.geo.addCircleArc(7,5,6,tag=6)

    gmsh.model.geo.addCurveLoop([5,6],2)

    gmsh.model.geo.addPlaneSurface([1,2], 1)

    gmsh.model.geo.synchronize() 
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0),h)

    gmsh.model.mesh.generate(2)
    ntags, vxyz, _ = gmsh.model.mesh.getNodes()
    node = vxyz.reshape((-1,3))
    node = node[:,:2]
    deletetag = bm.zeros((node.shape[0],),dtype=bm.bool)
    deletetag[4] = True
    node = node[~deletetag]
    vmap = dict({j:i for i,j in enumerate(ntags)})
    tris_tags,evtags = gmsh.model.mesh.getElementsByType(2)
    evid = bm.array([vmap[j] for j in evtags])
    cell = evid.reshape((tris_tags.shape[-1],-1))
    cell[cell>4] = cell[cell>4]-1
    mesh = TriangleMesh(node,cell)
    gmsh.finalize()
    return mesh

def triangle_domain():
    node = bm.array([[0.0,0.0],[2.0,0.0],[1,bm.sqrt(3)]],dtype=bm.float64)
    cell = bm.array([[0,1,2]],dtype=bm.int32)
    mesh = TriangleMesh(node,cell)
    mesh.uniform_refine(2)
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    node[cell[-1,0]] = node[cell[-1,0]]+[-0.15,0.05]
    node[cell[-1,1]] = node[cell[-1,1]]+[-0.1,0.15]
    node[cell[-1,2]] = node[cell[-1,2]]+[0,-0.15]
    mesh.uniform_refine(3)
    return mesh
