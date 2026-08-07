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

def unit_circle(h=0.05):
    gmsh.initialize()
    gmsh.model.occ.addDisk(0.0,0.0,0.0,1,1,1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0),h)
    gmsh.option.setNumber("Mesh.Optimize",0)
    gmsh.model.mesh.generate(2)
    mesh = to_TriangleMesh()
    gmsh.finalize()
    return mesh

def unit_square(h=0.05):
    gmsh.initialize()
    gmsh.model.occ.addRectangle(0.0,0.0,0.0,1,1,1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0),h)
    gmsh.option.setNumber("Mesh.Optimize",0)
    gmsh.model.mesh.generate(2)

    mesh = to_TriangleMesh()
    gmsh.finalize()
    return mesh

def unit_sphere(h=0.1):
    gmsh.initialize()
    gmsh.model.occ.addSphere(0.0,0.0,0.0,1,1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0),h)
    gmsh.option.setNumber("Mesh.Optimize",1)
    gmsh.model.mesh.generate(3)

    mesh = to_TetrahedronMesh()
    gmsh.finalize()
    return mesh

def unit_square3d(h=0.1):
    gmsh.initialize()
    gmsh.model.occ.addBox(0.0,0.0,0.0,1,1,1,1)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0),h)
    gmsh.option.setNumber("Mesh.Optimize",1)
    gmsh.model.mesh.generate(3)

    mesh = to_TetrahedronMesh()
    gmsh.finalize()
    return mesh

