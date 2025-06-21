from dataclasses import dataclass, field
from typing import Callable, Any, Tuple

import numpy as np
import cv2
import os
import pickle
import glob
import matplotlib.pyplot as plt

from fealpy.mesh import DistMesher2d
from ...geometry.domain import Domain
from fealpy.geometry import dintersection,drectangle,dcircle

@dataclass
class OCAMModel:
    location: np.ndarray
    axes: np.ndarray 
    center: np.ndarray 
    height: float
    width: float
    ss: np.ndarray
    pol : np.ndarray
    affine: np.ndarray
    fname: str
    flip: str
    chessboardpath: str
    icenter: tuple
    radius : float
    mark_board: np.ndarray
    camera_points: list
    viewpoint : tuple
    data_path: str
    name: str

    def __post_init__(self):
        fname = os.path.expanduser(self.data_path+"DIM_K_D_{}.pkl".format(self.name))
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                self.DIM, self.K, self.D = pickle.load(f)
        else:
            self.DIM, self.K, self.D = self.get_K_and_D((4, 6), self.chessboardpath)
            with open(fname, 'wb') as f:
                pickle.dump((self.DIM, self.K, self.D), f)

        self.icenter, self.radius = self.get_center_and_radius(self.fname)
        cps_all = []
        for cps in self.camera_points:
            val = self.world_to_image(cps)
            val *= np.array([[self.width, self.height]])
            val = val[self.signed_dist_function(val)<0] # 去掉不在区域内的点
            val[:, 1] = self.height-val[:, 1]
            cps_all.append(val)
        self.camera_points = cps_all

        # 判断是否存在网格文件
        fname = os.path.expanduser(self.data_path+"ocam_mesh_{}.pkl".format(self.name))
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                self.imagemesh = pickle.load(f)
        else:
            self.imagemesh = self.gmshing_new()
            # 保存 cps:
            with open(fname, 'wb') as f:
                pickle.dump(self.imagemesh, f)

    def set_location(self, location):
        self.location = location

    def set_axes(self, axes):
        self.axes = axes

    def __call__(self, u):
        icenter = self.icenter
        r = self.radius
        return dintersection(drectangle(u,[0,1920,0,1080]),dcircle(u,cxy=icenter,r=r))

    def signed_dist_function(self, u):
        return self(u)

    def gmshing_new(self):
        import gmsh
        from fealpy.mesh import TriangleMesh

        gmsh.initialize()
        occ = gmsh.model.occ
        gmsh.option.setNumber("Geometry.Tolerance", 1e-6)  # 设置容差值
        #gmsh.option.setNumber("Mesh.MeshSizeMax", 40)  # 最大网格尺寸
        #gmsh.option.setNumber("Mesh.MeshSizeMin", 10)    # 最小网格尺寸

        def add_rectangle(p0, p1, p2, p3):
            l0 = occ.addLine(p0, p1)
            l1 = occ.addLine(p1, p2)
            l2 = occ.addLine(p2, p3)
            l3 = occ.addLine(p3, p0)
            return l0, l1, l2, l3

        # 获得分割线
        cps = self.camera_points
        lines = []
        for cp in cps:
            curves = [] 
            for p in cp:
                curves.append(occ.addPoint(p[0], p[1], 0))
            lines.append(occ.addSpline(curves))

        # 获取标记板
        mb = self.mark_board
        for i in range(6):
            p0 = occ.addPoint(mb[4*i+0, 0], mb[4*i+0, 1], 0.0)
            p1 = occ.addPoint(mb[4*i+1, 0], mb[4*i+1, 1], 0.0)
            p2 = occ.addPoint(mb[4*i+2, 0], mb[4*i+2, 1], 0.0)
            p3 = occ.addPoint(mb[4*i+3, 0], mb[4*i+3, 1], 0.0)
            ret = add_rectangle(p0, p1, p2, p3)
            [lines.append(r) for r in ret]

        # 生成区域
        rec  = occ.addRectangle(0, 0, 0, 1920, 1080)
        circ = occ.addDisk(self.icenter[0], self.height-self.icenter[1], 0, self.radius, self.radius)

        ## 保留 rec 和 circ 的交集
        dom = occ.intersect([(2, rec)], [(2, circ)])[0]

        ## 分割线和区域的交集
        occ.fragment([(1, l) for l in lines], dom)

        occ.synchronize()

        # 定义网格尺寸场函数
        def f(dim, tag, x, y, z, lc): 
            m = self.mesh_to_image(np.array([[x, y]]))
            l = np.linalg.norm(m[0]-self.icenter)
            #return 40*(self.radius-l)/self.radius + 2
            return 80*(self.radius-l)/self.radius + 4
        gmsh.model.mesh.setSizeCallback(f)

        ## 生成网格
        gmsh.model.mesh.generate(2)
        #gmsh.fltk.run()


        # 转化为 fealpy 的网格
        node = gmsh.model.mesh.get_nodes()[1].reshape(-1, 3)[:, :2]
        NN = node.shape[0]

        ## 节点编号到标签的映射
        nid2tag = gmsh.model.mesh.get_nodes()[0]
        tag2nid = np.zeros(NN*2, dtype = np.int_)
        tag2nid[nid2tag] = np.arange(NN)

        ## 单元
        cell = gmsh.model.mesh.get_elements(2, -1)[2][0]
        cell = tag2nid[cell].reshape(-1, 3)

        gmsh.finalize()
        return TriangleMesh(node, cell)

    def gmshing(self):
        import gmsh
        from fealpy.mesh import TriangleMesh
        icenter = self.icenter
        r = self.radius
        mark_board=self.mark_board        
        camera_points = self.camera_points
        #print(camera_points)

        x1 = np.sqrt(r*r-icenter[...,1]*icenter[...,1])
        x2 = np.sqrt(r*r-(self.height-icenter[...,1])**2)
        gmsh.initialize()

        # 边界区域
        gmsh.model.geo.addPoint(icenter[...,0],icenter[...,1],0,tag=1)
        gmsh.model.geo.addPoint(icenter[...,0]-x1,0,0,tag=2)
        gmsh.model.geo.addPoint(icenter[...,0]+x1,0,0,tag=3)
        gmsh.model.geo.addPoint(icenter[...,0]+x2,1080,0,tag=4)
        gmsh.model.geo.addPoint(icenter[...,0]-x2,1080,0,tag=5)
 
        gmsh.model.geo.addLine(2,3,tag=1)
        gmsh.model.geo.addCircleArc(3,1,4,tag=2)
        gmsh.model.geo.addLine(4,5,tag=3)
        gmsh.model.geo.addCircleArc(5,1,2,tag=4)

        gmsh.model.geo.addCurveLoop([1,2,3,4],1)
        
        
        # 标记板区域
        #for i in range(24):
        #    gmsh.model.geo.addPoint(mark_board[i,0],mark_board[i,1],0,tag=6+i)

        #gmsh.model.geo.addLine(6,7,tag=5)
        #gmsh.model.geo.addLine(7,8,tag=6)
        #gmsh.model.geo.addLine(8,9,tag=7)
        #gmsh.model.geo.addLine(9,6,tag=8)
        #gmsh.model.geo.addCurveLoop([5,6,7,8],2)

        #gmsh.model.geo.addLine(10,11,tag=9)
        #gmsh.model.geo.addLine(11,12,tag=10)
        #gmsh.model.geo.addLine(12,13,tag=11)
        #gmsh.model.geo.addLine(13,10,tag=12)
        #gmsh.model.geo.addCurveLoop([9,10,11,12],3)

        #gmsh.model.geo.addLine(14,15,tag=13)
        #gmsh.model.geo.addLine(15,16,tag=14)
        #gmsh.model.geo.addLine(16,17,tag=15)
        #gmsh.model.geo.addLine(17,14,tag=16)
        #gmsh.model.geo.addCurveLoop([13,14,15,16],4)

        #gmsh.model.geo.addLine(18,19,tag=17)
        #gmsh.model.geo.addLine(19,20,tag=18)
        #gmsh.model.geo.addLine(20,21,tag=19)
        #gmsh.model.geo.addLine(21,18,tag=20)
        #gmsh.model.geo.addCurveLoop([17,18,19,20],5)

        #gmsh.model.geo.addLine(22,23,tag=21)
        #gmsh.model.geo.addLine(23,24,tag=22)
        #gmsh.model.geo.addLine(24,25,tag=23)
        #gmsh.model.geo.addLine(25,22,tag=24)
        #gmsh.model.geo.addCurveLoop([21,22,23,24],6)

        #gmsh.model.geo.addLine(26,27,tag=25)
        #gmsh.model.geo.addLine(27,28,tag=26)
        #gmsh.model.geo.addLine(28,29,tag=27)
        #gmsh.model.geo.addLine(29,26,tag=28)
        #gmsh.model.geo.addCurveLoop([25,26,27,28],7)
        
        # 分割线
        t0 = False # 判断点是否是同一点
        t1 = False
        if np.sum((camera_points[0][0][0]-camera_points[1][0][0])**2)<1e-3:
            t0=True
        if np.sum((camera_points[2][0][0]-camera_points[3][0][0])**2)<1e-3:
            t1=True

        # 左侧分割线
        camera_points_tag1 = []
        for j in range(camera_points[0][0].shape[0]):
            p=gmsh.model.geo.addPoint(camera_points[0][0][j,0],camera_points[0][0][j,1],0)
            camera_points_tag1.append(p)
        gmsh.model.geo.addSpline(camera_points_tag1,29)
        camera_points_tag2 = camera_points_tag1[-1:]
        for j in range(1,camera_points[0][1].shape[0]):
            p=gmsh.model.geo.addPoint(camera_points[0][1][j,0],camera_points[0][1][j,1],0)
            camera_points_tag2.append(p)
        gmsh.model.geo.addSpline(camera_points_tag2,30)

        if t0 == 1:
            camera_points_tag1=camera_points_tag1[:1]
        else: 
            p=gmsh.model.geo.addPoint(camera_points[1][0][0,0],camera_points[1][0][0,1],0)
            camera_points_tag1 = [p]
        for j in range(1,camera_points[1][0].shape[0]):
            p=gmsh.model.geo.addPoint(camera_points[1][0][j,0],camera_points[1][0][j,1],0)
            camera_points_tag1.append(p)
        gmsh.model.geo.addSpline(camera_points_tag1,31)
        camera_points_tag2 = camera_points_tag1[-1:]
        for j in range(1,camera_points[1][1].shape[0]):
            p=gmsh.model.geo.addPoint(camera_points[1][1][j,0],camera_points[1][1][j,1],0)
            camera_points_tag2.append(p)
        gmsh.model.geo.addSpline(camera_points_tag2,32)
        if t0 is False:
            gmsh.model.geo.addCurveLoop([29,30,-30,-29])
            gmsh.model.geo.addCurveLoop([31,32,-32,-31])
        else:
            gmsh.model.geo.addCurveLoop([-30,-29,31,32,-32,-31,29,30])
        
        # 右侧分割线
        camera_points_tag1 = []
        for j in range(camera_points[2][0].shape[0]):
            p=gmsh.model.geo.addPoint(camera_points[2][0][j,0],camera_points[2][0][j,1],0)
            camera_points_tag1.append(p)
        gmsh.model.geo.addSpline(camera_points_tag1,33)
        camera_points_tag2 = camera_points_tag1[-1:]
        for j in range(1,camera_points[2][1].shape[0]):
            p=gmsh.model.geo.addPoint(camera_points[2][1][j,0],camera_points[2][1][j,1],0)
            camera_points_tag2.append(p)
        gmsh.model.geo.addSpline(camera_points_tag2,34)

        if t1 == 1:
            camera_points_tag1=camera_points_tag1[:1]
        else: 
            p=gmsh.model.geo.addPoint(camera_points[3][0][0,0],camera_points[3][0][0,1],0)
            camera_points_tag1 = [p]
        for j in range(1,camera_points[3][0].shape[0]):
            p=gmsh.model.geo.addPoint(camera_points[3][0][j,0],camera_points[3][0][j,1],0)
            camera_points_tag1.append(p)
        gmsh.model.geo.addSpline(camera_points_tag1,35)
        camera_points_tag2 = camera_points_tag1[-1:]
        for j in range(1,camera_points[3][1].shape[0]):
            p=gmsh.model.geo.addPoint(camera_points[3][1][j,0],camera_points[3][1][j,1],0)
            camera_points_tag2.append(p)
        gmsh.model.geo.addSpline(camera_points_tag2,36)
        if t1 is False:
            gmsh.model.geo.addCurveLoop([33,34,-34,-33])
            gmsh.model.geo.addCurveLoop([35,36,-36,-35])
        else:
            gmsh.model.geo.addCurveLoop([-34,-33,35,36,-36,-35,33,34])
   
        # 生成面
        """
        if t0+t1==2:
            gmsh.model.geo.addPlaneSurface([1,2,5,8,9],1)
        else:
            gmsh.model.geo.addPlaneSurface([1,2,5,8,9,10],1)
        gmsh.model.geo.addPlaneSurface([2,3],2)
        gmsh.model.geo.addPlaneSurface([3,4],3)
        gmsh.model.geo.addPlaneSurface([4],4)
        gmsh.model.geo.addPlaneSurface([5,6],5)
        gmsh.model.geo.addPlaneSurface([6,7],6)
        gmsh.model.geo.addPlaneSurface([7],7)
        """

        # 如果有标记板就把这一段删掉, 上面的取消注释
        if t0+t1==2:
            gmsh.model.geo.addPlaneSurface([1, 2, 3],1)
        else:
            gmsh.model.geo.addPlaneSurface([1, 2, 3, 4],1)

        
        gmsh.model.geo.synchronize()
        #gmsh.fltk().run()
        #gmsh.option.setNumber("Mesh.Algorithm",6) 
        gmsh.model.mesh.field.add("Distance",1)
        gmsh.model.mesh.field.setNumbers(1,"CurvesList",[2,4])
        gmsh.model.mesh.field.setNumber(1,"Sampling",100)
        lc1 = 50
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", lc1)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", 3*lc1)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 200)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 800)
         
        gmsh.model.mesh.field.add("Distance",3)
        if t0==1:
            gmsh.model.mesh.field.setNumbers(3,"CurvesList",[29,30,31,32])
            lc2 = 15
        if t1==1:
            gmsh.model.mesh.field.setNumbers(3,"CurvesList",[33,34,35,36])
            lc2 = 15
        if t0+t1==2:
            gmsh.model.mesh.field.setNumbers(3,"CurvesList",[29,30,31,32,33,34,35,36])
            lc2 = 30
        gmsh.model.mesh.field.setNumber(3,"Sampling",100)
        gmsh.model.mesh.field.add("Threshold", 4)
        gmsh.model.mesh.field.setNumber(4, "InField", 3)
        gmsh.model.mesh.field.setNumber(4, "SizeMin", lc2)
        gmsh.model.mesh.field.setNumber(4, "SizeMax", 3*lc1)
        gmsh.model.mesh.field.setNumber(4, "DistMin", 20)
        gmsh.model.mesh.field.setNumber(4, "DistMax", 150)
        
        gmsh.model.mesh.field.add("Min",5)
        gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2,4])
        
        gmsh.model.mesh.field.setAsBackgroundMesh(5)
        
        
        gmsh.model.mesh.generate(2)
        #gmsh.fltk().run()

        ntags, vxyz, _ = gmsh.model.mesh.getNodes()
        node = vxyz.reshape((-1,3))
        node = node[:,:2]
        vmap = dict({j:i for i,j in enumerate(ntags)})
        tris_tags,evtags = gmsh.model.mesh.getElementsByType(2)
        evid = np.array([vmap[j] for j in evtags])
        cell = evid.reshape((tris_tags.shape[-1],-1))
        
        gmsh.finalize()
        return TriangleMesh(node,cell)
        
    def distmeshing(self,hmin=50,fh=None):
        domain=OCAMDomain(icenter=self.icenter,radius=self.radius,fh=fh)
        hmin=50
        mesher=DistMesher2d(domain,hmin)
        mesh = mesher.meshing(maxit=100)
        return mesh

    def world_to_image(self, node):
        """
        @brief 把世界坐标系转化为归一化的圈像坐标系
        """
        node = self.world_to_cam(node)
        uv = self.cam_to_image(node)
        return uv

    def world_to_cam(self, node):
        """
        @brief 把世界坐标系中的点转换到相机坐标系下
        """
        node = np.einsum('ij, kj->ik', node-self.location, self.axes)
        return node

    def project_to_cam_sphere(self, node):
        """
        @brief 将球面上的点投影到相机单位球面上
        """
        node = node-self.location
        r = np.sqrt(np.sum(node**2, axis=1))
        node /= r[:, None]
        return node + self.location

    def mesh_to_image(self, node):
        node[:, 1] = self.height - node[:, 1]
        return node

    def cam_to_image(self, node, ptype='L'):
        """
        @brief 把相机坐标系中的点投影到归一化的图像 uv 坐标系
        """

        NN = len(node)

        fx = self.K[0, 0] 
        fy = self.K[1, 1]
        u0 = self.K[0, 2]
        v0 = self.K[1, 2]

        """
        w = self.width
        h = self.height
        f = np.sqrt((h/2)**2 + (w/2)**2)
        fx = f
        fy = f
        u0 = self.center[0]
        v0 = self.center[1]
        """

        r = np.sqrt(np.sum(node**2, axis=1))
        theta = np.arccos(node[:, 2]/r)
        phi = np.arctan2(node[:, 1], node[:, 0])
        phi = phi % (2 * np.pi)

        uv = np.zeros((NN, 2), dtype=np.float64)

        if ptype == 'L': # 等距投影
            uv[:, 0] = fx * theta * np.cos(phi) + u0 
            uv[:, 1] = fy * theta * np.sin(phi) + v0 
        elif ptype == 'O': # 正交投影
            uv[:, 0] = fx * np.sin(theta) * np.cos(phi) + u0 
            uv[:, 1] = fy * np.sin(theta) * np.sin(phi) + v0 
        elif ptype == 'A': # 等积投影
            uv[:, 0] = 2 * fx * np.sin(theta/2) * np.cos(phi) + u0 
            uv[:, 1] = 2 * fy * np.sin(theta/2) * np.sin(phi) + v0 
        elif ptype == 'S': # 体视投影, Stereographic Projection
            uv[:, 0] = 2 * fx * np.tan(theta/2) * np.cos(phi) + u0 
            uv[:, 1] = 2 * fy * np.tan(theta/2) * np.sin(phi) + v0 
        else:
            raise ValueError(f"投影类型{ptype}错误!")


        # 标准化
        #uv[:, 0] = (uv[:, 0] - np.min(uv[:, 0]))/(np.max(uv[:, 0])-np.min(uv[:, 0]))
        #uv[:, 1] = (uv[:, 1] - np.min(uv[:, 1]))/(np.max(uv[:, 1])-np.min(uv[:, 1]))

        #uv[:, 0] = (uv[:, 0] - np.min(uv[:, 0]))/self.width
        #uv[:, 1] = (uv[:, 1] - np.min(uv[:, 1]))/self.height
        uv[:, 0] = uv[:, 0]/self.width
        uv[:, 1] = uv[:, 1]/self.height

        return uv

    def cam_to_image_fast(self, node):
        """
        @brief 利用 matlab 工具箱 中的算法来处理
        """
        theta = np.zeros(len(node), dtype=np.float64)

        norm = np.sqrt(node[:, 0]**2 + node[:, 1]**2)
        flag = (norm == 0)
        norm[flag] = np.finfo(float).eps
        theta = np.arctan(node[:, 2]/norm)

        rho = np.polyval(self.pol, theta)
        ps = node[:, 0:2]/norm[:, None]*rho[:, None]
        uv = np.zeros_like(ps)
        c, d, e = self.affine
        xc, yc = self.center
        uv[:, 0] = ps[:, 0] * c + ps[:, 1] * d + xc
        uv[:, 1] = ps[:, 0] * e + ps[:, 1]     + yc

        # 标准化
        uv[:, 0] = (uv[:, 0] - np.min(uv[:, 0]))/(np.max(uv[:, 0])-np.min(uv[:, 0]))
        uv[:, 1] = (uv[:, 1] - np.min(uv[:, 1]))/(np.max(uv[:, 1])-np.min(uv[:, 1]))
        return uv

    def world_to_image_fast(self, node):
        """
        @brief 利用 matlab 工具箱 中的算法来处理
        """
        node = self.world_to_cam(node)
        theta = np.zeros(len(node), dtype=np.float64)

        norm = np.sqrt(node[:, 0]**2 + node[:, 1]**2)
        flag = (norm == 0)
        norm[flag] = np.finfo(float).eps
        theta = np.arctan(node[:, 2]/norm)

        rho = np.polyval(self.pol, theta)
        ps = node[:, 0:2]/norm[:, None]*rho[:, None]
        uv = np.zeros_like(ps)
        c, d, e = self.affine
        xc, yc = self.center
        uv[:, 0] = ps[:, 0] * c + ps[:, 1] * d + xc
        uv[:, 1] = ps[:, 0] * e + ps[:, 1]     + yc

        # 标准化
        uv[:, 0] = (uv[:, 0] - np.min(uv[:, 0]))/(np.max(uv[:, 0])-np.min(uv[:, 0]))
        uv[:, 1] = (uv[:, 1] - np.min(uv[:, 1]))/(np.max(uv[:, 1])-np.min(uv[:, 1]))
        return uv

    def camera_to_world(self, node):
        """
        @brief 把相机坐标系中的点转换到世界坐标系下
        """
        A = np.linalg.inv(self.axes.T)
        node = np.einsum('ij, jk->ik', node, A)
        node += self.location
        return node
    
    def image_to_camera_sphere(self, uv,ptype='L'):
        NN = len(uv)

        fx = self.K[0, 0] 
        fy = self.K[1, 1]
        u0 = self.K[0, 2]
        v0 = self.K[1, 2]
        node = np.zeros((NN,3),dtype=np.float64)

        node[:,0] = uv[:,0]-u0
        node[:,1] = uv[:,1]-v0

        #phi = np.arctan(fx*node[:,1]/(fy*node[:,0]))
        phi = np.arctan2(fx*node[:,1], (fy*node[:,0]))
        phi[phi<0] = phi[phi<0]+np.pi

        idx = np.abs(fx*np.cos(phi))>1e-10
        rho = np.zeros_like(phi)
        rho[idx] = node[idx,0]/(fx*np.cos(phi[idx]))
        rho[~idx] = node[~idx, 1]/(fy*np.sin(phi[~idx]))

        if ptype=='L':
            theta=rho

        node[:,0] = np.sin(theta)*np.cos(phi)
        node[:,1] = np.sin(theta)*np.sin(phi)
        node[:,2] = np.cos(theta)
        return self.camera_to_world(node)

    def sphere_project_to_implict_surface(self, nodes, Fun): 
        """
        @brief 将球面上的点投影到隐式曲面上
        """
        from scipy.optimize import fsolve
        ret = np.zeros_like(nodes)
        for i, node in enumerate(nodes):
            g = lambda t : Fun(self.location + t*(node-self.location))
            t = fsolve(g, 1000)
            ret[i] = self.location + t*(node-self.location)
        return ret
        
    def equirectangular_projection(self, fovd=195):
        """
        @brief 使用等矩形投影将鱼眼图像转换为平面图像。
        @return: 转换后的平面图像
        """
        # 读取输入鱼眼图像
        src_img = cv2.imread(self.fname)
        hs, ws = src_img.shape[:2]
        u0, v0 = ws // 2, hs // 2

        # 计算目标图像尺寸
        wd = int(ws * 360/fovd)
        hd = hs
        u1, v1 = wd // 2, hd // 2

        # 使用数组化计算
        y_indices, x_indices = np.indices((hd, wd))
        xd = x_indices - u1 
        yd = y_indices - v1 

        # 使用矢量化运算
        phi = 2 * np.pi * xd / wd
        theta = -2 * np.pi * yd / wd + np.pi / 2

        flag = theta < 0
        phi[flag] += np.pi
        theta[flag] *= -1

        flag = theta > np.pi
        phi[flag] += np.pi
        theta[flag] = 2*np.pi - theta[flag]

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        r = np.sqrt(y ** 2 + z ** 2)
        f = wd * np.arctan2(r, x)/ 2.0 / np.pi

        map_x = np.array(f * y / r + u0, dtype=np.float32)
        map_y = np.array(f * z / r + v0, dtype=np.float32) 

        # 使用映射表将鱼眼图像转换为等矩形投影图像
        dst_img = cv2.remap(src_img, map_x, map_y, cv2.INTER_LINEAR)
        return dst_img

    def undistort(self, image, fc=5, width=640, height=480):
        """
        Undistort the given image using the camera model parameters.

        Parameters:
        - image: Input image as a NumPy array of shape (height, width, 3).
        - fc: Factor proportional to the distance of the camera to the plane.
        - width, height: Dimensions of the output image.
        - display: If True, display the undistorted image.

        Returns:
        - Undistorted image as a NumPy array.
        """
        # Parameters of the new image
        Nxc = height / 2
        Nyc = width / 2
        Nz = -width / fc
        
        # Generate grid for the new image
        i, j = np.meshgrid(np.arange(height), np.arange(width))
        M = np.zeros((len(i.flat), 3), dtype=np.float64)
        M[:, 0] = (i - Nxc).flat
        M[:, 1] = (j - Nyc).flat
        M[:, 2] = Nz
        
        # Map points using world2cam_fast and adjust for NumPy indexing
        m = self.world2cam_fast(M)
        
        # Get color from original image points
        r, g, b = self.get_color_from_imagepoints(image, m)
        
        # Construct the undistorted image
        Nimg = np.zeros((height, width, 3), dtype=np.uint8)
        Nimg[i.flat, j.flat] = np.column_stack((r, g, b))
        
        return Nimg

    def get_color_from_imagepoints(self, img, points):
        """
        Extract color values from an image at given points.
        
        Parameters:
        - img: Input image as a NumPy array of shape (height, width, 3).
        - points: NumPy array of shape (N, 2) containing N points with (row, column) format.
        
        Returns:
        - A tuple of three NumPy arrays (r, g, b), each containing the color values for the points.
        """
        # Ensure points are rounded to the nearest integer and correct points outside image borders
        points = np.round(points).astype(int)
        points[:, 0] = np.clip(points[:, 0], 0, img.shape[0] - 1)
        points[:, 1] = np.clip(points[:, 1], 0, img.shape[1] - 1)
        
        # Extract color values
        r = img[points[:, 0], points[:, 1], 0]
        g = img[points[:, 0], points[:, 1], 1]
        b = img[points[:, 0], points[:, 1], 2]
        
        return r, g, b

    def world2cam(self, node):
        """
        """
        ps = self.omni3d2pixel(node)
        uv = np.zeros_like(ps)
        uv[:, 0] = ps[:, 0] * self.c + ps[:, 1] * self.d + self.xc
        uv[:, 1] = ps[:, 0] * self.e + ps[:, 1]          + self.yc
        return uv 

    def omni3d2pixel(self, node):
        """
        """
        pcoef = np.flip(self.ss)
        # 解决 node = [0,0,+-1] 的情况
        flag = (node[:, 0] == 0) & (node[:, 1] == 0)
        node[flag, :] = np.finfo(float).eps

        l = np.sqrt(node[:, 0]**2 + node[:, 1]**2)
        m = node[:, 2] / l
        rho = np.zeros(m.shape)
        pcoef_tmp = np.copy(pcoef)
        for i in range(len(m)):
            pcoef_tmp[-2] = pcoef[-2] - m[i]
            r = np.roots(pcoef_tmp)
            flag = (np.imag(r) == 0) & (r > 0)
            res = r[flag]
            if len(res) == 0:
                rho[i] = np.nan
            else:
                rho[i] = np.min(res)

        ps = node[:, 0:2]/l.reshape(-1, 1)*rho.reshape(-1, 1)
        return ps 

    def get_K_and_D(self, checkerboard, imgsPath):
        CHECKERBOARD = checkerboard
        subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
        objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
        objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
        _img_shape = None
        objpoints = []
        imgpoints = []
        images = glob.glob(imgsPath + '/*.jpg')
        for fname in images:
            img = cv2.imread(fname)
            if _img_shape == None:
                _img_shape = img.shape[:2]
            else:
                assert _img_shape == img.shape[:2], "All images must share the same size."

            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, checkerboard,
                flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

            #ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret == True:
                objpoints.append(objp)
                cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
                imgpoints.append(corners)
        N_OK = len(objpoints)
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        rms, _, _, _, _ = cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
        DIM = _img_shape[::-1]
        return DIM, K, D

    def show_camera_image_and_mesh(self, imgname=None, outname=None):
        from PIL import Image
        mesh = self.imagemesh
        mesh.add_plot(plt, alpha=0.2)
        if imgname is None:
            imgname = self.fname
        img = cv2.imread(imgname)
        plt.imshow(img, extent=[0, 1920, 0, 1080])

        points = self.camera_points
        for i in range(len(points)):
            # 绘制线
            plt.plot(points[i][:, 0], points[i][:, 1], 'r')
            # 标记是第i条线, 字体大小为30
            j = len(points[i]) // 2
            plt.text(points[i][j, 0], points[i][j, 1], str(i), color='r',
                     fontsize=30)

        if outname is not None:
            plt.savefig(outname)
        plt.show()

    def undistort_chess(self, imgname, scale=0.6):
        img = cv2.imread(imgname)
        K, D, DIM = self.K, self.D, self.DIM
        dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
        assert dim1[0]/dim1[1] == DIM[0]/DIM[1] #Image to undistort needs to have same aspect ratio as the ones used in calibration
        if dim1[0]!=DIM[0]:
            img = cv2.resize(img,DIM,interpolation=cv2.INTER_AREA)
        Knew = K.copy()
        if scale: #change fov
            Knew[(0,1), (0,1)] = scale * Knew[(0,1), (0,1)]
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, DIM, cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return undistorted_img

    def perspective(self, img):
        # 定义原始图像中的四个角点坐标
        original_points = np.array([[430.479, 233.444],
                                    [1072.281, 238.995],
                                    [1096.582, 38.359],
                                    [252.12, 23.290]], dtype=np.float32)[::-1]
        # 定义目标图像中对应的四个角点坐标
        target_points = np.array([[430.479, 233.444],
                                  [1072.281, 233.444],
                                  [1080.281, 38.359],
                                  [350.479, 38.359]], dtype=np.float32)[::-1]

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(original_points, target_points)

        # 进行透视矫正
        result = cv2.warpPerspective(img, M, (1920, 1080))
        return result

    def get_center_and_radius(self, image_path):
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image0 = cv2.imread(image_path)
        if image is None:
            print("Error: Unable to load image.")
            return None, None

        # 对图像进行阈值处理，保留非黑色区域
        _, thresholded = cv2.threshold(image, 70, 255, cv2.THRESH_BINARY)

        # 找到非黑色区域的轮廓
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 寻找最大轮廓
        max_contour = max(contours, key=cv2.contourArea)

        # 使用最小外接圆找到中心和半径
        center, radius = cv2.minEnclosingCircle(max_contour)

        # 绘制最小外接圆
        #circle_image = image0
        #cv2.circle(circle_image, center, radius, (0, 255, 0), 2)  # 绘制圆
        #cv2.circle(circle_image, center, 5, (0, 0, 255), -1)  # 绘制中心点

        # 绘制最大轮廓
        #contour_image = np.zeros_like(image)
        #cv2.drawContours(contour_image, [max_contour], 0, (255, 255, 255), 2)
        #print(f"Center: {center}, Radius: {radius}")

        ## 显示结果
        #plt.figure(figsize=(8, 6))
        #plt.imshow(circle_image)
        #plt.title('Fisheye Center and Radius')
        #plt.axis('off')
        #plt.show()
        return center, radius

    def mesh_to_ground(self, points, ground_location = -3.0):
        """
        @brief 将图像上的点投影到地面
        @param points: 图像上的点 (...， 2) 的数组
        """
        f2 = lambda x : x[..., 2] - ground_location
        points = self.mesh_to_image(points)
        points = self.image_to_camera_sphere(points)
        retp = self.sphere_project_to_implict_surface(points, f2)
        return retp

    def harmonic_map(self):
        """
        @brief 调和映射
        """
        pass






class OCAMDomain(Domain):
    def __init__(self,icenter,radius,hmin=10,hmax=20,fh=None):
        super().__init__(hmin=hmin, hmax=hmax, GD=2)
        if fh is not None:
            self.fh = fh
        self.box = [0,2020,0,1180]
        self.icenter=icenter
        self.radius=radius
        vertices = np.array([
            (icenter[...,0]-np.sqrt(radius*radius-icenter[...,1]*icenter[...,1]),0),
            (icenter[...,0]+np.sqrt(radius*radius-icenter[...,1]*icenter[...,1]),0),
            (icenter[...,0]-np.sqrt(radius*radius-(1080-icenter[...,1])*(1080-icenter[...,1])),1080),
            (icenter[...,0]+np.sqrt(radius*radius-(1080-icenter[...,1])*(1080-icenter[...,1])),1080)])
        curves = np.array([[0,1],[2,3]])
        self.facets = {0:vertices,1:curves}

    def __call__(self,u):
        icenter = self.icenter
        r = self.radius
        return dintersection(drectangle(u,[0,1920,0,1080]),dcircle(u,cxy=icenter,r=r))

    def signed_dist_function(self,u):
        return self(u)

    def sizing_function(self,p):
        return self.fh(p,self)
    def facet(self,dim):
        return self.facets[0]
    import numpy as np

def rotation_matrix_from_euler_angles(theta, gamma, beta):
    """
    Compute the rotation matrix from Euler angles.

    Parameters:
    theta (float): Rotation angle around the x-axis in radians.
    gamma (float): Rotation angle around the y-axis in radians.
    beta (float): Rotation angle around the z-axis in radians.

    Returns:
    np.ndarray: The resulting rotation matrix.
    """
    # Rotation matrix around x-axis
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

    # Rotation matrix around y-axis
    R_y = np.array([
        [np.cos(gamma), 0, np.sin(gamma)],
        [0, 1, 0],
        [-np.sin(gamma), 0, np.cos(gamma)]
    ])

    # Rotation matrix around z-axis
    R_z = np.array([
        [np.cos(beta), -np.sin(beta), 0],
        [np.sin(beta), np.cos(beta), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix
    R = R_z @ R_y @ R_x
    return R


    

