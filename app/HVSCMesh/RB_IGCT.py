import numpy as np
import gmsh
import math
from fealpy.mesh import TriangleMesh
import matplotlib.pyplot as plt
#from Doping import TotalDoping
#import cdoping

gmsh.initialize()

gmsh.model.add("RB IGCT")

Ltotal = 245        #总宽度
Lcathode  = 70      #阴极电极的宽度
Lgate= 75           #门极电极的宽度
Lnemitter = 110     #没切圆角的长度

Hslot = 18
Htotal = 1500

scale = 1       #可以更改为 1500，来获得无量纲化后的 [0, 1] 的网格


# 深度为 J3 结矩形网格区域的底部
nemitterdepth2 = 42
# 深度为 J3 结矩形网格区域的顶部
nemitterdepth1 = math.sqrt(2) * Hslot

pplusbasedepth = 70
pbasedepth = 120

# 阳极侧的 p base 型掺杂要用在两侧
pbasedepth_anode_side = 160
pemitterthick = 15

refine_mesh = 1

# 边界点gmsh.model.geo
gmsh.model.geo.addPoint(0,0,0,1)
gmsh.model.geo.addPoint((Lnemitter-Hslot)/scale,0,0,2)
gmsh.model.geo.addPoint(Lnemitter/scale, Hslot/scale,0,3)
gmsh.model.geo.addPoint(Ltotal/scale,Hslot/scale,0,4)
gmsh.model.geo.addPoint(Ltotal/scale,Htotal/scale,0,5)
gmsh.model.geo.addPoint(0,Htotal/scale,0,6)

# 对于圆的点
# 圆心
gmsh.model.geo.addPoint(Lnemitter/scale,0,0,7)

# 边界
gmsh.model.geo.addLine(1,2,1)
gmsh.model.geo.addCircleArc(2,7,3,2)
gmsh.model.geo.addLine(3,4,3)
gmsh.model.geo.addLine(4,5,4)
gmsh.model.geo.addLine(5,6,5)
gmsh.model.geo.addLine(6,1,6)

gmsh.model.geo.addCurveLoop([1,2,3,4,5,6],1)

gmsh.model.geo.addPlaneSurface([1],1)
gmsh.model.geo.synchronize()

gmsh.model.mesh.setSize(gmsh.model.getEntities(0), 20)
gmsh.model.mesh.generate(2)

ntags, vxyz, _ = gmsh.model.mesh.getNodes()
node = vxyz.reshape((-1,3))
node = node[:,:2]
vmap = dict({j:i for i,j in enumerate(ntags)})
tris_tags,evtags = gmsh.model.mesh.getElementsByType(2)
evid = np.array([vmap[j] for j in evtags])
cell = evid.reshape((tris_tags.shape[-1],-1))
mesh = TriangleMesh(node,cell)
gmsh.finalize()
angle = mesh.angle()
max_angle = np.max(angle,axis=1)
fig,axes1= plt.subplots()
mesh.show_angle(axes1,max_angle)
plt.show()
