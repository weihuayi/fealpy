
import numpy as np




def distmesh(self, h, fd, fh, bbox, pfix=None):
    from .distmesh import DistMesh2d
    domain = DistDomain2d(fd, fh, bbox, pfix)
    distmesh2d = DistMesh2d(domain, h)
    distmesh2d.run()
    return distmesh2d.mesh

def meshpy(self):
    pass

def gmsh(self):
    pass

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
