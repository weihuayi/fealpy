import gmsh
import numpy as np
from fealpy.mesh import TriangleMesh

def to_TriangleMesh():
    ntags, vxyz, _ = gmsh.model.mesh.getNodes()
    node = vxyz.reshape((-1,3))
    node = node[:,:2]
    vmap = dict({j:i for i,j in enumerate(ntags)})
    tris_tags,evtags = gmsh.model.mesh.getElementsByType(2)
    evid = np.array([vmap[j] for j in evtags])
    cell = evid.reshape((tris_tags.shape[-1],-1))
    return TriangleMesh(node,cell)

def triangle_domain():
    node = np.array([[0.0,0.0],[2.0,0.0],[1,np.sqrt(3)]],dtype=np.float64)
    cell = np.array([[0,1,2]],dtype=np.int_)
    mesh = TriangleMesh(node,cell)
    mesh.uniform_refine(2)
    node = mesh.entity('node')
    cell = mesh.entity('cell')
    node[cell[-1,0]] = (node[cell[-1,0]]+node[cell[-1,2]])/2
    node[cell[-1,1]] = (node[cell[-1,1]]+node[cell[-1,2]])/2    
    node[cell[-1,2],1] = 0.65
    mesh.uniform_refine(3)
    return mesh

