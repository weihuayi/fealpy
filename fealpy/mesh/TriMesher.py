import numpy as np
from .TriangleMesh import TriangleMesh

import meshpy.triangle as triangle



def distmesh2d(hmin, fd, fh, bbox, pfix=None, maxit=200, showanimation=False):
    from .distmesh import DistMesh2d
    from ..geometry import DistDomain2d

    domain = DistDomain2d(fd, fh, bbox, pfix)
    distmesh2d = DistMesh2d(domain, hmin)
    if showanimation:
        distmesh2d.show_animation(frames=maxit)
    else:
        distmesh2d.run()
    return distmesh2d.mesh

def meshpy2d(points, facets, h, hole_points=None, refine_function=None):
    from meshpy.triangle import MeshInfo, build

    mesh_info = triangle.MeshInfo()
    mesh_info.set_points(points)
    mesh_info.set_facets(facets)
    
    if hole_points is not None:
        mesh_info.set_holes(hole_points)
    mesh = triangle.build(mesh_info,max_volume=h**2)
    
    node = np.array(mesh.points, dtype=np.float64)
    cell = np.array(mesh.elements, dtype=np.int_)

    return TriangleMesh(node, cell)

def gmsh_to_TriangleMesh():
    import gmsh
    ntags, vxyz, _ = gmsh.model.mesh.getNodes()
    node = vxyz.reshape((-1,3))
    node = node[:,:2]
    vmap = dict({j:i for i,j in enumerate(ntags)})
    tris_tags,evtags = gmsh.model.mesh.getElementsByType(2)
    evid = np.array([vmap[j] for j in evtags])
    cell = evid.reshape((tris_tags.shape[-1],-1))
    return TriangleMesh(node,cell)

def smoothing(self, mesh, stype='laplace'):
    pass

def laplace_smoothing(self):
    pass

def cpt_smoothing(self):
    pass

def cvt_smoothing(self):
    pass

def global_smoothing(self):
    pass
