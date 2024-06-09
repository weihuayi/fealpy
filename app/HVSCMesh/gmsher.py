import numpy as np
import gmsh

from fealpy.mesh import TriangleMesh
from domain import Rectangle_BJT_Domain,RB_IGCT_Domain

class RB_IGCT_gmshing():
    def __init__(self):
        domain = RB_IGCT_Domain()
        self.facets = domain.facets
         
    def model(self):
        facets = self.facets
        gmsh.initialize()
        vertices = facets[0]
        circenter = facets[10]
        lines = facets[1]
        circleArc = facets[2]
        circenter = facets[10]
        
        # 节点
        for i in range(vertices.shape[0]):
            gmsh.model.geo.addPoint(vertices[i,0], vertices[i,1], 0,tag=i+1)
        # 圆心
        for i in range(circenter.shape[0]):
            gmsh.model.geo.addPoint(circenter[i,0], circenter[i,1], 0)
        # 直线
        for i in range(lines.shape[0]):
            gmsh.model.geo.addLine(int(lines[i,0]+1),int(lines[i,1]+1),tag=int(lines[i,0]+1))
        # 圆弧
        for i in range(circleArc.shape[0]):
            gmsh.model.geo.addCircleArc(int(circleArc[i,0]+1),int(vertices.shape[0]+i+1),int(circleArc[i,1]+1),tag=int(circleArc[i,0]+1))
        
        curvelooplist = list(range(1,lines.shape[0]+circleArc.shape[0]+1))
        gmsh.model.geo.addCurveLoop(curvelooplist)
        # add plane surface
        gmsh.model.geo.addPlaneSurface([1])
        # add physical group
        gmsh.model.geo.synchronize()
    def meshing(self):
        self.model()
        gmsh.model.mesh.generate(2)
        gmsh.fltk().run()
        ntags, vxyz, _ = gmsh.model.mesh.getNodes()
        node = vxyz.reshape((-1,3))
        node = node[:,:2]
        vmap = dict({j:i for i,j in enumerate(ntags)})
        tris_tags,evtags = gmsh.model.mesh.getElementsByType(2)
        evid = np.array([vmap[j] for j in evtags])
        cell = evid.reshape((tris_tags.shape[-1],-1))
        return TriangleMesh(node,cell)
