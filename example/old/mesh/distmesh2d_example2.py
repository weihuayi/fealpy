import numpy as np
import matplotlib.pyplot as plt
import argparse

from fealpy.mesh.simple_mesh_generator import distmesh2d,unitcircledomainmesh
from fealpy.geometry import huniform
from fealpy.geometry import dcircle,drectangle,ddiff,dmin
from fealpy.geometry import DistDomain2d

from fealpy.mesh import DistMesh2d
from fealpy.mesh import PolygonMesh
from fealpy.mesh import TriangleMeshWithInfinityNode

parser = argparse.ArgumentParser(description='复杂二维区域distmesh网格生成示例')
parser.add_argument('--domain', 
        default='circle', type=str, 
        help = """
        区域类型, 默认是 circle, 还可以选择:
        circle_h, 
        square_h, 
        adaptive_geo, 
        superellipse""")

parser.add_argument('--h0', default=0.1, type=float, help='网格尺寸, 默认为0.1')

args = parser.parse_args()
domain = args.domain
h0 = args.h0

if domain == 'circle':
    fd = lambda p: dcircle(p,[0,0],1)
    fh = huniform
    bbox = [-1,1,-1,1]
    domain = DistDomain2d(fd, fh, bbox)
    distmesh2d = DistMesh2d(domain,h0)
    distmesh2d.run()

if domain == 'circle_h':
    fd = lambda p:ddiff(dcircle(p,cxy=[0,0],r=1),dcircle(p,cxy=[0,0],r=0.4))
    fh = huniform
    bbox = [-1,1,-1,1]
    domain = DistDomain2d(fd, fh, bbox)
    distmesh2d = DistMesh2d(domain,h0)
    distmesh2d.run()

if domain =='square_h':
    fd = lambda p: ddiff(drectangle(p,[-1.0,1.0,-1.0,1.0]),dcircle(p,[0,0],0.4))
    #fh = huniform
    def fh(p):
        h = 4*np.sqrt(p[:,0]*p[:,0]+p[:,1]*p[:,1])-1
        h[h>2] = 2
        return h
    bbox = [-1,1,-1,1]
    pfix = np.array([(-1.0,-1.0),(1.0,-1.0),(1.0,-1.0),(1.0,1.0)],dtype=np.float)
    domain = DistDomain2d(fd,fh,bbox,pfix)
    distmesh2d = DistMesh2d(domain,h0)
    distmesh2d.run()

if domain =='pologon':
    pass

if domain =='adaptive_geo':
    fd1 = lambda p: dcircle(p,[0,0],1)
    fd2 = lambda p: dcircle(p,[-0.4,0],0.55)
    fd = lambda p: ddiff(ddiff(fd1(p),fd2(p)),p[:,1])
    fh1 = lambda p: 0.15-0.2*fd1(p)
    fh2 = lambda p: 0.06+0.2*fd2(p)
    fh3 = lambda p: (fd2(p)-fd1(p))/3
    fh = lambda p: dmin(dmin(fh1(p),fh2(p)),fh3(p))
    bbox = [-1,1,0,1]
    pfix = np.array([(-1.0,0.0),(-0.95,0.0),(0.15,0.0),(1.0,0.0)],dtype = np.float)
    domain = DistDomain2d(fd,fh,bbox,pfix)
    distmesh2d = DistMesh2d(domain,h0)
    distmesh2d.run()

if domain =='superellipse':
    fd1 = lambda p: (p[:,0]**4+p[:,1]**4)**0.25-1
    fd2 = lambda p: (p[:,0]**4+p[:,1]**4)**0.25-0.5
    fd = lambda p: ddiff(fd1(p),fd2(p))
    fh = huniform
    bbox = [-1,1,-1,1]
    domain = DistDomain2d(fd,fh,bbox)
    distmesh2d = DistMesh2d(domain,h0)
    distmesh2d.run()


fig = plt.figure()
axes = fig.gca()
distmesh2d.mesh.add_plot(axes)
plt.show()
