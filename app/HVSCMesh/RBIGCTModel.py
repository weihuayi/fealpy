import numpy as np
import gmsh
import math
from fealpy.mesh import TriangleMesh
import matplotlib.pyplot as plt
from Doping import TotalDoping

def RB_IGCT():
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

    # 边界点
    factory = gmsh.model.geo
    p_1 = factory.addPoint(0, 0, 0)
    p_2 = factory.addPoint(0, nemitterdepth2/scale, 0 )
    p_3 = factory.addPoint( 0, pplusbasedepth / scale , 0 )
    p_4 = factory.addPoint( 0, pbasedepth / scale , 0 )
    p_5 = factory.addPoint( 0, (Htotal - pbasedepth_anode_side) / scale , 0 )
    p_6 = factory.addPoint( 0, (Htotal - pemitterthick) / scale , 0 )
    p_7 = factory.addPoint( 0, Htotal / scale, 0 )
    p_8 = factory.addPoint( Ltotal / scale , Htotal / scale , 0 )
    p_9 = factory.addPoint( Ltotal / scale , (Htotal - pemitterthick) / scale , 0 )
    p_10 = factory.addPoint( Ltotal / scale , (Htotal - pbasedepth_anode_side) / scale , 0 )
    p_11 = factory.addPoint( Ltotal / scale , pbasedepth / scale , 0 )
    p_12 = factory.addPoint( Ltotal / scale , pplusbasedepth / scale , 0)
    p_13 = factory.addPoint(Ltotal / scale, Hslot / scale, 0)
    p_14 = factory.addPoint((Ltotal - Lgate) / scale, Hslot / scale, 0)
    p_15 = factory.addPoint( Lnemitter / scale, Hslot / scale, 0 )
    p_16 = factory.addPoint((Lnemitter - Hslot) / scale, 0, 0 )
    p_17 = factory.addPoint( Lcathode / scale, 0, 0 )

    # 内点
    p_18 = factory.addPoint( (Lnemitter - Hslot) / scale, nemitterdepth2 / scale, 0)

    # 对于圆的点
    # 圆心
    p_19 = factory.addPoint( Lnemitter / scale , 0 , 0 )

    p_20 = factory.addPoint((Lnemitter - Hslot + math.sqrt(nemitterdepth2 * nemitterdepth2 - Hslot * Hslot)) / scale, Hslot / scale, 0)

    # 1004 添加的点
    p_21 = factory.addPoint( 0, nemitterdepth1 / scale, 0 )
    p_22 = factory.addPoint( (Lnemitter - Hslot) / scale, nemitterdepth1 / scale, 0 )

    # 2023 11 16 新添加的点
    p_23 = factory.addPoint( Lcathode / scale, nemitterdepth1 / scale, 0 )
    p_24 = factory.addPoint( Lcathode / scale, nemitterdepth2 / scale, 0 )


    # 线
    L_1 = factory.addLine( 1, 21 ,1)
    L_25 = factory.addLine( 21 ,2 ,25)
    L_2 = factory.addLine( 2 , 3 , 2)
    L_3 = factory.addLine( 3 , 4 , 3)
    L_4 = factory.addLine( 4 , 5 , 4)
    L_5 = factory.addLine( 5 , 6 , 5)
    L_6 = factory.addLine( 6 , 7 , 6)
    L_7 = factory.addLine( 7 , 8 , 7)
    L_8 = factory.addLine( 8 , 9 , 8)
    L_9 = factory.addLine( 9 , 10 , 9)
    L_10 = factory.addLine( 10 , 11 , 10)
    L_11 = factory.addLine( 11 , 12 , 11)
    L_12 = factory.addLine( 12 , 13 , 12)
    L_13 = factory.addLine( 13 , 14 , 13)
    L_14 = factory.addLine( 14 , 20 , 14)
    L_15 = factory.addLine( 20 , 15 , 15)
    L_16 = factory.addLine( 16 , 17 , 16)
    L_17 = factory.addLine( 17 , 1 , 17)
    L_18 = factory.addLine( 2 , 24 , 18)
    L_19 = factory.addLine( 3 , 12 , 19)
    L_20 = factory.addLine( 4 , 11 , 20)
    L_21 = factory.addLine( 5 , 10 , 21)
    L_22 = factory.addLine( 6 , 9 , 22)

    # 圆
    C_23 = factory.addCircleArc( 15, 19 , 16 , 23)
    C_24 = factory.addCircleArc( 18, 16 , 20 , 24)

    C_26 = factory.addCircleArc( 22, 16 , 15 , 26)

    L_27 = factory.addLine( 21 , 23 , 27)
    L_28 = factory.addLine( 22 , 18 , 28)

    # Add lines on 2023 11 16 to better suit J3 junction
    L_29 = factory.addLine( 23 , 22 , 29)
    L_30 = factory.addLine( 22 , 16 , 30)
    L_31 = factory.addLine( 23 , 17 , 31)
    L_32 = factory.addLine( 24 , 18 , 32)
    L_33 = factory.addLine( 24 , 23 , 33)

    # 划分区域
    # n+ emitter
    CL_1 = factory.addCurveLoop([1 , 27 , 31 , 17] , 1)
    CL_9 = factory.addCurveLoop([-31 , 29 , 30 ,16] , 9)
    CL_10 = factory.addCurveLoop([-30 , 26, 23] , 10 )
    PS_1 = factory.addPlaneSurface([1] , 1)
    PS_9 = factory.addPlaneSurface([9] , 9)
    PS_10 = factory.addPlaneSurface([10] , 10)

    # J3 junction
    CL_7 = factory.addCurveLoop([25 , 18 , 33 , -27] , 7)
    CL_11 = factory.addCurveLoop([-33 , 32 , -28 , -29] , 11)
    PS_7 = factory.addPlaneSurface([7] , 7)
    PS_11 = factory.addPlaneSurface([11] , 11)

    CL_8 = factory.addCurveLoop([28 , 24 , 15 , -26] , 8)
    PS_8 = factory.addPlaneSurface([8] , 8)

    # p+ base
    CL_2 = factory.addCurveLoop([2 , 19 , 12 , 13 , 14 , -24 , -32 , -18] , 2)
    PS_2 = factory.addPlaneSurface([2] , 2)

    # p base
    CL_3 = factory.addCurveLoop([3, 20 , 11 , -19], 3)
    PS_3 = factory.addPlaneSurface([3] , 3)

    # n base
    CL_4 = factory.addCurveLoop([4 , 21 , 10, -20], 4)
    PS_4 = factory.addPlaneSurface([4] , 4)

    # n buffer
    CL_5 = factory.addCurveLoop([5 , 22 , 9 , -21] , 5)
    PS_5 = factory.addPlaneSurface([5] , 5)

    # p+ emitter
    CL_6 = factory.addCurveLoop([6, 7, 8 , -22] , 6)
    PS_6 = factory.addPlaneSurface([6] , 6)
    
    # 阳极
    gmsh.model.addPhysicalGroup(1, [7], 0)

    # 门极
    gmsh.model.addPhysicalGroup(1, [13], 1)

    # 阴极
    gmsh.model.addPhysicalGroup(1, [17], 2)

    factory.synchronize()

def struct_mesh():
    RB_IGCT()
    refine_mesh = 1
    factory = gmsh.model.geo
    factory.mesh.setTransfiniteCurve(1,28*refine_mesh,"Progression",1)
    factory.mesh.setTransfiniteCurve(31,28*refine_mesh,"Progression",-1)
    factory.mesh.setTransfiniteCurve(30,28*refine_mesh,"Progression",-1)
    factory.mesh.setTransfiniteCurve(23,32*refine_mesh,"Progression",-1)
    factory.mesh.setTransfiniteCurve(16,12*refine_mesh,"Progression",1)
    factory.mesh.setTransfiniteCurve(29,12*refine_mesh,"Progression",1)
    factory.mesh.setTransfiniteCurve(32,12*refine_mesh,"Progression",1)

    # J3
    factory.mesh.setTransfiniteCurve(17,24*refine_mesh,"Progression",-1)
    factory.mesh.setTransfiniteCurve(18,24*refine_mesh,"Progression",1)
    factory.mesh.setTransfiniteCurve(27,24*refine_mesh,"Progression",1)
    factory.mesh.setTransfiniteCurve(24,24*refine_mesh,"Progression",1.0)
    factory.mesh.setTransfiniteCurve(26,24*refine_mesh,"Progression",1.0)
    factory.mesh.setTransfiniteCurve(25,12*refine_mesh,"Progression",1.0)
    factory.mesh.setTransfiniteCurve(33,12*refine_mesh,"Progression",1.0)
    factory.mesh.setTransfiniteCurve(28,12*refine_mesh,"Progression",1.0)
    factory.mesh.setTransfiniteCurve(15,12*refine_mesh,"Progression",1.0)
    factory.mesh.setTransfiniteSurface(1)
    factory.mesh.setTransfiniteSurface(7)
    factory.mesh.setTransfiniteSurface(8)
    factory.mesh.setTransfiniteSurface(9)
    factory.mesh.setTransfiniteSurface(11)

    # p+ base
    factory.mesh.setTransfiniteCurve(2,8*refine_mesh,"Progression",-0.92)
    factory.mesh.setTransfiniteCurve(19,42*refine_mesh,"Progression",1.0)
    factory.mesh.setTransfiniteCurve(12,16*refine_mesh,"Bump",0.9)
    factory.mesh.setTransfiniteCurve(13,20*refine_mesh,"Progression",0.98)
    factory.mesh.setTransfiniteCurve(14,12*refine_mesh,"Bump",0.86)

    # p base
    factory.mesh.setTransfiniteCurve(19,32*refine_mesh,"Progression",1)
    factory.mesh.setTransfiniteCurve(20,32*refine_mesh,"Progression",1)
    factory.mesh.setTransfiniteCurve(3,14*refine_mesh,"Bump",0.48)
    factory.mesh.setTransfiniteCurve(11,14*refine_mesh,"Bump",0.48)
    factory.mesh.setTransfiniteSurface(3)

    # n base
    factory.mesh.setTransfiniteCurve(4,80*refine_mesh,"Bump",0.24)
    factory.mesh.setTransfiniteCurve(10,80*refine_mesh,"Bump",0.24)
    factory.mesh.setTransfiniteCurve(21,32*refine_mesh,"Progression",1)
    factory.mesh.setTransfiniteCurve(20,32*refine_mesh,"Progression",1)
    factory.mesh.setTransfiniteSurface(4)

    # 原本的 n buffer 区域，现在没有 buffer 了
    factory.mesh.setTransfiniteCurve(5,50*refine_mesh,"Progression",1)
    factory.mesh.setTransfiniteCurve(9,50*refine_mesh,"Progression",-1)

    # p+ emitter
    factory.mesh.setTransfiniteCurve(7,32*refine_mesh,"Progression",1)
    factory.mesh.setTransfiniteCurve(22,32*refine_mesh,"Progression",1)
    factory.mesh.setTransfiniteCurve(6,15*refine_mesh,"Progression",-1)
    factory.mesh.setTransfiniteCurve(8,15*refine_mesh,"Progression",1)
    factory.mesh.setTransfiniteSurface(6)
    factory.mesh.setTransfiniteSurface(5)

    # 必须集合出来 Physical Surface 才能被 dealii 以二维网格读入
    gmsh.model.addPhysicalGroup(2,[1,2,3,4,5,6,7,8,9,10,11])

    # Neumann 边界
    gmsh.model.addPhysicalGroup(1, [1,25,2,3,4,5,6,8,9,10,11,12,14,15,23,16],10001)

    factory.synchronize()
    gmsh.model.mesh.generate(2)

    ntags, vxyz, _ = gmsh.model.mesh.getNodes()
    node = vxyz.reshape((-1,3))
    node = node[:,:2]
    vmap = dict({j:i for i,j in enumerate(ntags)})
    tris_tags,evtags = gmsh.model.mesh.getElementsByType(2)
    evid = np.array([vmap[j] for j in evtags])
    cell = evid.reshape((tris_tags.shape[-1],-1))
    
    gmsh.finalize()
    return TriangleMesh(node,cell)

def unstruct_mesh():
    RB_IGCT()
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    def Constant_field(in_size,out_size,SurfacesList,tag):
        gmsh.model.mesh.field.add("Constant",tag)
        gmsh.model.mesh.field.setNumbers(tag,"SurfacesList",SurfacesList)
        gmsh.model.mesh.field.setNumber(tag,"VIn",in_size)
        gmsh.model.mesh.field.setNumber(tag,"VOut",out_size)
        gmsh.model.mesh.field.setNumber(tag,"IncludeBoundary",1)
        gmsh.model.mesh.field.setAsBackgroundMesh(tag)

    SurfacesList1 = [6]
    SurfacesList2 = [3,4,5]
    SurfacesList3 = [2]
    SurfacesList4 = [7,11,8]
    SurfacesList5 = [1,9]
    SurfacesList6 = [10]
    Constant_field(5,10,SurfacesList1,1)
    Constant_field(10,10,SurfacesList2,2)
    Constant_field(5,10,SurfacesList3,3)
    Constant_field(2,10,SurfacesList4,4)
    Constant_field(1.5,10,SurfacesList5,5)
    Constant_field(1,10,SurfacesList6,6)

    gmsh.model.mesh.field.add("Min",7)
    gmsh.model.mesh.field.setNumbers(7,"FieldsList",[1,2,3,4,5,6])
    gmsh.model.mesh.field.setAsBackgroundMesh(7)
    gmsh.model.mesh.generate(2)

    ntags, vxyz, _ = gmsh.model.mesh.getNodes()
    node = vxyz.reshape((-1,3))
    node = node[:,:2]
    vmap = dict({j:i for i,j in enumerate(ntags)})
    tris_tags,evtags = gmsh.model.mesh.getElementsByType(2)
    evid = np.array([vmap[j] for j in evtags])
    cell = evid.reshape((tris_tags.shape[-1],-1))

    gmsh.finalize()
    return TriangleMesh(node,cell)

