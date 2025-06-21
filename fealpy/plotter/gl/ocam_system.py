import numpy as np
import gmsh
import os
import cv2
import matplotlib.pyplot as plt

from dataclasses import dataclass, field
from .ocam_model import OCAMModel
from fealpy.mesh import TriangleMesh
import pickle
from app.svads3d.harmonic_map import *

@dataclass
class GroundMarkPoint:
    """
    @brief 地面标记点
    @param points0: 点在相机 cam0 的图像中的坐标
    @param points1: 点在相机 cam1 的图像中的坐标
    @param points2: 点的真实坐标
    """
    cam0:    int
    cam1:    int
    points0: np.ndarray
    points1: np.ndarray
    points2: np.ndarray

class OCAMSystem:
    def __init__(self, data):

        self.size=data['size'] # 椭球面的长宽高
        self.center_height = data['center_height'] # 椭球面的中心高度
        self.scale_ratio = data['scale_ratio'] # 椭球面的缩放比例
        self.viewpoint = np.array(data['viewpoint']) # 视点
        self.data_path = data['data_path']
        
        self.cams = []

        # 判断是否已经生成了splite point 文件
        fname = os.path.expanduser(self.data_path+"cps.pkl")
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                cps = pickle.load(f)
        else:
            #cps = self.get_split_point0(scheme=0)
            cps = self.get_split_point()
            # 保存 cps:
            with open(fname, 'wb') as f:
                pickle.dump(cps, f)
        self.camera_points = cps
        
        for i in range(data['nc']):
            axes = np.zeros((3, 3), dtype=np.float64)
            axes[0, :] = data['axes'][0][i]
            axes[1, :] = data['axes'][1][i]
            axes[2, :] = data['axes'][2][i]
            self.cams.append(OCAMModel(
                location = data['location'][i],
                axes = axes,
                center = data['center'][i],
                height = data['height'],
                width = data['width'],
                ss = np.array(data['ss'][i]),
                pol = np.array(data['pol'][i]),
                affine = data['affine'][i],
                fname = data['fname'][i],
                flip = data['flip'][i],
                chessboardpath=data['chessboardpath'][i],
                icenter=data['icenter'][i],
                radius=data['radius'][i],
                mark_board=data['mark_board'][i],
                camera_points = cps[i],
                viewpoint = self.viewpoint,
                data_path = data['data_path'],
                name = "cam{}".format(i),
            ))
        self.gmp = self.ground_mark_board()

        # 判断是否已经生成了 groundmesh 文件
        fname = os.path.expanduser(self.data_path+"groundmesh.pkl")
        self.groundmesh, self.didx, self.dval = self.get_ground_mesh(only_ground=False)
        #if os.path.exists(fname):
        #    with open(fname, 'rb') as f:
        #        self.groundmesh, self.didx, self.dval = pickle.load(f) 
        #else:
        #    self.groundmesh, self.didx, self.dval = self.get_ground_mesh(only_ground=False)
        #    # 保存 cps:
        #    with open(fname, 'wb') as f:
        #        pickle.dump((self.groundmesh, self.didx, self.dval), f)


    # 获取特征点
    def ground_mark_board(self):
        """
        @brief 获取地面标记点
        """
        gmp = []
        for i in range(6):
            ri = (i+1)%6 # 右边相邻相机
            ps0 = self.cams[i].mark_board[12:]
            ps1 = self.cams[ri].mark_board[:12]
            ps2 = 0.5*self.cams[i].mesh_to_ground(ps0, -self.center_height)
            ps2+= 0.5*self.cams[ri].mesh_to_ground(ps1, -self.center_height)
            for j in range(len(ps0)):
                gmp.append(GroundMarkPoint(i, ri, ps0[j], ps1[j], ps2[j]))
        return gmp

    def set_parameters(self, data):
        """
        @brief 设置参数
        """
        loc = data[:6]
        cx  = data[6:12]
        cy  = data[12:18]
        cz  = np.cross(cx, cy)

        for i in range(6):
            axes = np.zeros((3, 3), dtype=np.float64)
            axes[0, :] = cx[i]
            axes[1, :] = cy[i]
            axes[2, :] = cz[i]

            self.cams[i].set_location(loc[i])
            self.cams[i].set_axes(axes)

    def get_implict_surface_function(self):
        """
        @brief 获取隐式表面函数
        """
        a, b, c = self.size
        a *= self.scale_ratio[0]
        b *= self.scale_ratio[1]
        c *= self.scale_ratio[2]
        def f0(p):
            x = p[..., 0]
            y = p[..., 1]
            z = p[..., 2]
            return x**2/a**2 + y**2/b**2 + z**2/c**2 - 1.0

        z0 = self.center_height
        def f1(p):
            z = p[..., 2]
            return z + z0
        return f0, f1

    def get_ground_mesh(self, theta = np.pi/6, only_ground=True):
        gmsh.initialize()

        gmsh.option.setNumber("Mesh.MeshSizeMax", 1)  # 最大网格尺寸
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.5)    # 最小网格尺寸

        def add_rectangle(p0, p1, p2, p3):
            # 添加线
            l1 = gmsh.model.occ.addLine(p0, p1)
            l2 = gmsh.model.occ.addLine(p1, p2)
            l3 = gmsh.model.occ.addLine(p2, p3)
            l4 = gmsh.model.occ.addLine(p3, p0)
            # 添加 loop
            curve = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
            return gmsh.model.occ.addPlaneSurface([curve])


        # 构造椭球面
        l, w, h = self.size
        a = l * self.scale_ratio[0]
        b = w * self.scale_ratio[1]
        c = h * self.scale_ratio[2]
        z0 = -self.center_height

        phi = np.arcsin(z0 / c)

        # 构造单位球面
        r = 1.0
        ball = gmsh.model.occ.addSphere(0, 0, 0, 1, 1, phi, 0)
        gmsh.model.occ.dilate([(3, ball)],0, 0, 0, a, b, c)
        gmsh.model.occ.remove([(3, ball), (2, 2)])

        # 车辆区域
        vehicle = gmsh.model.occ.addRectangle(-l/2, -w/2, z0, l, w)
        ground = gmsh.model.occ.cut([(2, 1), (2, 3)], [(2, vehicle)])[0]

        v = 30*np.array([[-np.cos(theta), -np.sin(theta)], 
                      [np.cos(theta), -np.sin(theta)], 
                      [np.cos(theta), np.sin(theta)], 
                      [-np.cos(theta), np.sin(theta)]])
        point = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]],
                         dtype=np.float64)
        ps = [3, 4, 5, 6]
        planes  = []
        for i in range(4):
            pp1 = gmsh.model.occ.addPoint(point[i, 0]+v[i, 0], point[i, 1]+v[i, 1], z0)
            pp2 = gmsh.model.occ.addPoint(point[i, 0]+v[i, 0], point[i, 1]+v[i, 1], 0.0)
            pp3 = gmsh.model.occ.addPoint(point[i, 0], point[i, 1], 0.0)
            planes.append(add_rectangle(ps[i], pp1, pp2, pp3))

        point = np.array([[0, -w/2], [0, w/2]], dtype=np.float64)
        v = 30*np.array([[0, -1], [0, 1]], dtype=np.float64)
        for i in range(2):
            pp0 = gmsh.model.occ.addPoint(point[i, 0], point[i, 1], z0)
            pp1 = gmsh.model.occ.addPoint(point[i, 0]+v[i, 0], point[i, 1]+v[i, 1], z0)
            pp2 = gmsh.model.occ.addPoint(point[i, 0]+v[i, 0], point[i, 1]+v[i, 1], 0.0)
            pp3 = gmsh.model.occ.addPoint(point[i, 0], point[i, 1], 0.0)
            planes.append(add_rectangle(pp0, pp1, pp2, pp3))

        # 地面特征点
        gmp = self.gmp
        gmps = [gmsh.model.occ.addPoint(p.points2[0], p.points2[1], p.points2[2]) for p in gmp]
        gmsh.model.occ.fragment(ground, [(0, p) for p in gmps])

        frag = gmsh.model.occ.fragment([(2, 1), (2, 3)], [(2, plane) for plane in planes])
        for i in range(len(frag[1]))[2:]:
            gmsh.model.occ.remove(frag[1][i], recursive=True)

        if only_ground:
            gmsh.model.occ.remove([(2, i+1) for i in range(7)], recursive=True)

        gmsh.model.occ.synchronize()
        #gmsh.fltk.run()

        node = gmsh.model.mesh.getNodes()

        gmsh.model.mesh.generate(2)

        def creat_part_mesh(node, cell):
            idx = np.unique(cell)
            nidxmap = np.zeros(node.shape[0], dtype = np.int_)
            nidxmap[idx] = np.arange(idx.shape[0])
            cell = nidxmap[cell]

            mesh = TriangleMesh(node[idx], cell)
            return mesh, nidxmap

        # 转化为 fealpy 的网格
        node = gmsh.model.mesh.get_nodes()[1].reshape(-1, 3)
        NN = node.shape[0]

        ## 节点编号到标签的映射
        nid2tag = gmsh.model.mesh.get_nodes()[0]
        tag2nid = np.zeros(NN*2, dtype = np.int_)
        tag2nid[nid2tag] = np.arange(NN)

        ## 获取地面特征点在网格中的编号
        gmpidx = np.array([gmsh.model.mesh.get_nodes(0, g)[0] for g in gmps]).reshape(-1)

        ## 网格分块
        didx = [] # 狄利克雷边界条件的节点编号
        dval = [] # 狄利克雷边界条件的节点值
        partmesh = []
        if only_ground:
            idxs = [[10], [12], [13], [11], [9], [8]]
        else:
            idxs = [[10, 4], [12, 6], [13, 7], [11, 5], [9, 3], [8, 1, 2]]
        for i, idx in enumerate(idxs):
            # 获取第 i 块网格的单元
            cell = []
            for j in idx:
                cell0 = gmsh.model.mesh.get_elements(2, j)[2][0].reshape(-1, 3)
                cell.append(tag2nid[cell0])
            cell = np.concatenate(cell, axis=0)

            f0 = lambda x : self.cams[i].project_to_cam_sphere(x)-self.cams[i].location
            f1 = lambda x : self.cams[i].image_to_camera_sphere(x)-self.cams[i].location

            # 获取第 i 块网格的屏幕边界特征点
            geoedge = []
            for j in idx:
                geoedge += gmsh.model.getBoundary([(2, j)])
            geoedge = np.unique([abs(ge[1]) for ge in geoedge])

            lists = [gmsh.model.mesh.get_nodes(1, ge) for ge in geoedge]
            didx_s = [tag2nid[ge[0]] for ge in lists]
            dval_s = [f0(ge[1].reshape(-1, 3)) for ge in lists]

            # 获取第 i 块网格的地面特征点
            ismeshi = np.zeros(NN, dtype = np.bool_)
            ismeshi[cell] = True # 第 i 块网格中的点

            flag = ismeshi[gmpidx]
            flag0 = np.array([(g.cam0 == i) | (g.cam1 == i) for g in gmp])
            flag = flag & flag0

            didx_g = [[gmpidx[j]] for j in range(len(flag)) if flag[j]]
            dval_g = [f1(g.points0[None, :]) if g.cam0 == i else
                      f1(g.points1[None, :]) for j, g in enumerate(gmp) if flag[j]]

            pmesh, nidxmap = creat_part_mesh(node, cell)
            partmesh.append(pmesh)
            didx_i = nidxmap[np.concatenate(didx_s+didx_g, dtype=np.int_)]
            dval_i = np.concatenate(dval_s+dval_g, axis=0)
            didx_i = nidxmap[np.concatenate(didx_s, dtype=np.int_)]
            dval_i = np.concatenate(dval_s, axis=0)

            didx_i, uniqueidx = np.unique(didx_i, return_index=True)
            dval_i = dval_i[uniqueidx]
            didx.append(didx_i)
            dval.append(dval_i)

        gmsh.finalize()
        return partmesh, didx, dval 

    def undistort_cv(self):
        for i, cam in enumerate(self.cams):
            outname = cam.fname[:-4]
            # 进行透视矫正
            result = cam.undistort_chess(cam.fname)
            result = cam.perspective(result)
            cv2.imwrite(outname+'_c.jpg', result)

    def show_ground_mesh(self, plotter):
        for i, mesh in enumerate(self.groundmesh):
            node = mesh.entity('node')
            cell = mesh.entity('cell')
            vertices = np.array(node[cell].reshape(-1, 3), dtype=np.float64)
            uv = self.cams[i].world_to_image(vertices)
            uv[:, 0] = 1-uv[:, 0]
            no = np.concatenate((vertices, uv), axis=-1, dtype=np.float32)
            mesh.to_vtk(fname = 'ground_mesh_'+str(i)+'.vtu')

            plotter.add_mesh(no, cell=None, texture_path = self.cams[i].fname)

    def show_split_lines(self):
        """
        @brief 显示分割线
        """
        # 三维的分割线的绘制
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')
        axes.set_box_aspect([30,10,1.5])
        for i in range(6):
            if i != 3:
                continue
            points = self.camera_points[i]
            for point in points:
                axes.plot(point[:, 0], point[:, 1], point[:, 2])
        plt.show()

    def show_sphere_lines(self):
        """
        @brief 显示分割线
        """
        # 三维的分割线的绘制
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')
        axes.set_box_aspect([30,10,1.5])
        for i in range(6):
            if i != 3:
                continue
            points = self.dval
            for point in points:
                axes.plot(point[:, 0], point[:, 1], point[:, 2])
        plt.show()
            
    def show_parameters(self):
        for i, cam in enumerate(self.cams):
            print(i, "-th camara:")
            print("DIM:\n", cam.DIM)
            print("K:\n", cam.K)
            print("D:\n", cam.D)

    def show_images(self):
        import matplotlib.pyplot as plt
        from PIL import Image
        images = []
        for cam in self.cams:
            images.append(Image.open(cam.fname))

        # 设置显示窗口
        fig, axs = plt.subplots(3, 2, figsize=(10, 15))

        # 使用索引去除空的维度，确保每个 subplot 都用图片填充
        for i, ax in enumerate(axs.flat):
            ax.imshow(np.asarray(images[i]))
            ax.axis('off')  # 不显示坐标轴

        #plt.tight_layout()
        plt.show()


    def show_screen_mesh(self, plotter):
        z0 = self.center_height
        f1, f2 = self.get_implict_surface_function()
        for i in range(6):
            mesh = self.cams[i].imagemesh
            node = mesh.entity('node')
            mesh.to_vtk(fname = 'image_mesh_'+str(i)+'.vtu')

            node = self.cams[i].mesh_to_image(node)

            uv = np.zeros_like(node)
            uv[:, 0] = node[:, 0]/self.cams[i].width
            #if i==1:
            #    uv[:, 0] = (node[:, 0]-40)/self.cams[i].width
            uv[:, 1] = node[:, 1]/self.cams[i].height
            #uv[:, 0] = 1-uv[:, 0]

            node = self.cams[i].image_to_camera_sphere(node)
            mesh.node = node
            mesh.to_vtk(fname = 'sphere_mesh_'+str(i)+'.vtu')

            inode = self.cams[i].sphere_project_to_implict_surface(node, f1)
            outflag = inode[:, 2] <-z0
            inode[outflag] = self.cams[i].sphere_project_to_implict_surface(node[outflag], f2)

            mesh.node = inode
            mesh.to_vtk(fname = 'screen_mesh_'+str(i)+'.vtu')

            # 相机坐标系下的点
            node = mesh.entity('node')
            cell = mesh.entity('cell')
            vertices = np.array(node[cell].reshape(-1, 3), dtype=np.float64)
            uv       = np.array(uv[cell].reshape(-1, 2), dtype=np.float64)

            no = np.concatenate((vertices, uv), axis=-1, dtype=np.float32)
            plotter.add_mesh(no, cell=None, texture_path=self.cams[i].fname)

    def sphere_mesh(self, plotter):
        """
        @brief 在世界坐标系的相机位置处生成半球网格
        """
        mesh = TriangleMesh.from_unit_sphere_surface(refine=4)
        node = mesh.entity('node')
        cell = mesh.entity('cell')
        bc = mesh.entity_barycenter('cell')
        cell = cell[bc[:, 2] > 0]
        # 相机坐标系下的点
        vertices = np.array(node[cell].reshape(-1, 3), dtype=np.float64)

        for i, cam in enumerate(self.cams):
            # 相机坐标系下的坐标转换为世界坐标系的坐标
            no = np.einsum('ik, kj->ij', vertices, cam.axes) + cam.location
            uv = cam.cam_to_image(vertices)
            no = np.concatenate((no, uv), axis=-1, dtype=np.float32)
            plotter.add_mesh(no, cell=None, texture_path=cam.fname, flip=cam.flip)

    def ellipsoid_mesh(self, plotter):
        """
        @brief 把椭球面环视网格加入 plotter
        """
        mesh= TriangleMesh.from_section_ellipsoid(
            size=self.size,
            center_height=self.center_height,
            scale_ratio=self.scale_ratio,
            density=0.1,
            top_section=np.pi / 2,
            return_edge=False)

        node = mesh.entity('node')
        cell = mesh.entity('cell')
        domain = mesh.celldata['domain']

        """
        cd = domain.copy()
        cd[(domain == 11) | (domain == 12)] = domain[(domain == 51) | (domain == 52)]
        cd[(domain == 21) | (domain == 22)] = domain[(domain == 41) | (domain == 42)]
        cd[(domain == 41) | (domain == 42)] = domain[(domain == 21) | (domain == 22)]
        cd[(domain == 51) | (domain == 52)] = domain[(domain == 11) | (domain == 12)]
        domain = cd
        """

        i0, i1 = 11, 12
        for i, cam in enumerate(self.cams):
            ce = cell[(domain == i0) | (domain == i1)]
            no = node[ce].reshape(-1, node.shape[-1])
            uv = cam.world_to_image(no)
            no = np.concatenate((no, uv), axis=-1, dtype=np.float32)
            plotter.add_mesh(no, cell=None, texture_path=cam.fname, flip=cam.flip)
            i0 += 10
            i1 += 10

        # 卡车区域的贴图
        ce = cell[domain == 0]
        no = np.array(node[ce].reshape(-1, node.shape[-1]), dtype=np.float32)
        plotter.add_mesh(no, cell=None, texture_path=None)

    def test_plain_domain(self, plotter, z=10, icam=-1):
        """
        @brief 测试半球外一平面网格
        """
        mesh = TriangleMesh.from_box([-10, 10, -10, 10], nx=100, ny=100)

        NN = mesh.number_of_nodes()
        node = np.zeros((NN, 3), dtype=np.float64)
        node[:, 0:2] = mesh.entity('node')
        node[:, -1] = z
        cell = mesh.entity('cell')

        #投影到球面上
        snode = node / np.sqrt(np.sum(node**2, axis=-1, keepdims=True))
        snode = snode[cell].reshape(-1, node.shape[-1])

        uv = self.cams[icam].cam_to_image(snode)
        pnode = node[cell].reshape(-1, node.shape[-1])
        pnode = np.concatenate((pnode, uv), axis=-1, dtype=np.float32)
        snode = np.concatenate((snode, uv), axis=-1, dtype=np.float32)

        #添加平面网格
        plotter.add_mesh(pnode, cell=None, texture_path=self.cams[icam].fname)

        #添加球面网格
        plotter.add_mesh(snode, cell=None, texture_path=self.cams[icam].fname)

    def test_half_sphere_surface(self, plotter, r=1.0, icam=-1):
        """
        @brief 在单位半球(z > 0)上的三角形网格上进行帖图
        """
        mesh = TriangleMesh.from_unit_sphere_surface(refine=4)
        node = r*mesh.entity('node')
        cell = mesh.entity('cell')
        bc = mesh.entity_barycenter('cell')
        cell = cell[bc[:, 2] > 0]
        # 相机坐标系下的点
        vertices = np.array(node[cell].reshape(-1, 3), dtype=np.float64)

        uv = self.cams[icam].cam_to_image(vertices)
        no = np.concatenate((vertices, uv), axis=-1, dtype=np.float32)
        plotter.add_mesh(no, cell=None, texture_path=self.cams[icam].fname)
        return mesh, uv
    
    def test_half_sphere_surface_with_cutting(self, plotter, theta=np.pi*35/180, h=0.1, icam=-1, ptype='O'):
        """
        """
        mesh = TriangleMesh.from_half_sphere_surface_with_cutting(theta=theta, h=h)
        node = mesh.entity('node')
        cell = mesh.entity('cell')


        # 相机坐标系下的点
        vertices = np.array(node[cell].reshape(-1, 3), dtype=np.float64)
        uv = self.cams[icam].cam_to_image(vertices, ptype=ptype)


        """
        r = np.sin(theta) # 圆柱面半径
        phi = np.arctan2(node[:, 0], node[:, 2])
        phi = phi % (2 * np.pi)
        node[:, 2] = r * np.cos(phi)
        node[:, 0] = r * np.sin(phi)
        vertices = np.array(node[cell].reshape(-1, 3), dtype=np.float64)
        """

        no = np.concatenate((vertices, uv), axis=-1, dtype=np.float32)
        plotter.add_mesh(no, cell=None, texture_path=self.cams[icam].fname)
        return mesh, uv

    def get_split_point0(self, densty=0.02, v=0.5, theta0=np.pi/6, theta1=0, 
                         scheme=1):
        size = self.size
        scale_ratio = self.scale_ratio
        center_height = self.center_height

        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.02)  # 最大网格尺寸
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.01)    # 最小网格尺寸

        def add_rectangle(p0, p1, p2, p3):
            # 添加线
            l1 = gmsh.model.occ.addLine(p0, p1)
            l2 = gmsh.model.occ.addLine(p1, p2)
            l3 = gmsh.model.occ.addLine(p2, p3)
            l4 = gmsh.model.occ.addLine(p3, p0)
            # 添加 loop
            curve = gmsh.model.occ.addCurveLoop([l1, l2, l3, l4])
            return gmsh.model.occ.addPlaneSurface([curve])


        # 构造椭球面
        l, w, h = size
        a = l * scale_ratio[0]
        b = w * scale_ratio[1]
        c = h * scale_ratio[2]

        phi = np.arcsin(-center_height / c)

        # 构造单位球面
        r = 1.0
        ball = gmsh.model.occ.addSphere(0, 0, 0, 1, 1, phi, 0)
        gmsh.model.occ.dilate([(3, ball)],0, 0, 0, a, b, c)
        gmsh.model.occ.remove([(3, ball), (2, 2)])

        # 车辆区域
        vehicle = gmsh.model.occ.addRectangle(-l/2, -w/2, -center_height, l, w)
        gmsh.model.occ.cut([(2, 3)], [(2, vehicle)])

        # 分割线对应的固定点
        planes  = []
        vpoint = np.array([[-l/2, -w/2], [l/2, -w/2], 
                                 [l/2, w/2], [-l/2, w/2]], dtype=np.float64)
        z0 = -center_height
        # 角点分界线 1
        v = 30*np.array([[-np.cos(theta0), -np.sin(theta0)], 
                      [np.cos(theta0), -np.sin(theta0)], 
                      [np.cos(theta0), np.sin(theta0)], 
                      [-np.cos(theta0), np.sin(theta0)]])
        ps = [3, 4, 5, 6]
        for i in range(4):
            pp1 = gmsh.model.occ.addPoint(vpoint[i, 0]+v[i, 0], vpoint[i, 1]+v[i, 1], z0)
            pp2 = gmsh.model.occ.addPoint(vpoint[i, 0]+v[i, 0], vpoint[i, 1]+v[i, 1], 0)
            pp3 = gmsh.model.occ.addPoint(vpoint[i, 0], vpoint[i, 1], 0)
            planes.append(add_rectangle(ps[i], pp1, pp2, pp3))

        if scheme == 1:
            # 角点分界线 2 
            v = 30*np.array([[-np.sin(theta0), -np.cos(theta0)], 
                          [np.sin(theta0), -np.cos(theta0)], 
                          [np.sin(theta0), np.cos(theta0)], 
                          [-np.sin(theta0), np.cos(theta0)]])
            for i in range(4):
                pp1 = gmsh.model.occ.addPoint(vpoint[i, 0]+v[i, 0], vpoint[i, 1]+v[i, 1], z0)
                pp2 = gmsh.model.occ.addPoint(vpoint[i, 0]+v[i, 0], vpoint[i, 1]+v[i, 1], 0)
                pp3 = gmsh.model.occ.addPoint(vpoint[i, 0], vpoint[i, 1], 0)
                planes.append(add_rectangle(ps[i], pp1, pp2, pp3))

        # 中点分界线
        mpoint = np.array([[0, -w/2], [0, w/2]], dtype=np.float64)
        v = 30*np.array([[0, -1], [0, 1]], dtype=np.float64)
        for i in range(2):
            pp0 = gmsh.model.occ.addPoint(mpoint[i, 0], mpoint[i, 1], z0)
            pp1 = gmsh.model.occ.addPoint(mpoint[i, 0]+v[i, 0], mpoint[i, 1]+v[i, 1], z0)
            pp2 = gmsh.model.occ.addPoint(mpoint[i, 0]+v[i, 0], mpoint[i, 1]+v[i, 1], 1.0)
            pp3 = gmsh.model.occ.addPoint(mpoint[i, 0], mpoint[i, 1], 1.0)
            planes.append(add_rectangle(pp0, pp1, pp2, pp3))

        frag = gmsh.model.occ.fragment([(2, 1), (2, 3)], [(2, plane) for plane in planes])
        for i in range(len(frag[1]))[2:]:
            gmsh.model.occ.remove(frag[1][i], recursive=True)

        dimtags = gmsh.model.occ.getEntities(2)
        gmsh.model.occ.remove(dimtags)
        gmsh.model.occ.remove([(1, 1)])

        gmsh.model.occ.synchronize()
        #gmsh.fltk.run()

        gmsh.model.mesh.generate(1)

        lines = []
        if scheme == 0:
            parttag = [[28, 11, 13, 22, 27, 3, 12], 
                       [32, 17, 19, 31, 27, 18, 12],
                       [33, 21, 20, 30, 31, 15, 18],
                       [29, 16, 14, 26, 30, 9, 15],
                       [25, 10, 8, 24, 26, 6, 9],
                       [23, 7, 5, 2, 4, 24, 22, 6, 3]]
        elif scheme == 1:
            parttag = [[41, 38, 34, 18, 12 ,3, 42, 17, 19, 11, 13],
                       [49, 45, 41, 30, 24, 18, 46, 23, 25, 29, 31],
                       [44, 47, 49, 45, 21, 27, 30, 24, 28, 26, 48, 33, 32, 29, 31],
                       [40, 44, 47, 15, 21, 27, 43, 22, 20, 28, 26],
                       [36, 37, 40, 6, 9, 15, 39, 16, 14, 10, 8],
                       [38, 34, 36, 37, 12, 3, 6, 9, 11, 13, 35, 2, 7, 4, 5, 10, 8]]

        for tags in parttag:
            l = []
            for tag in tags:
                node = gmsh.model.mesh.getNodes(1, tag)[1].reshape(-1, 3)
                l.append(node)
            lines.append(l)
        gmsh.finalize()
        return lines

    def get_split_point(self,
                        densty=0.02,
                        v=0.5,
                        theta1=0,
                        theta2=0):
        '''
        获取分割线
        @param densty: 节点密度
        @param v: 两侧摄像头分割线相对位置
        @param theta1: 主分割线偏转角
        @param theta2: 两侧分割线
        @return: 分割线上点的笛卡尔坐标
        '''
        size = self.size
        scale_ratio = self.scale_ratio
        center_height = self.center_height

        import gmsh
        pi = np.pi
        gmsh.initialize()
        l, w, h = size

        # 构造椭球面
        a = l * scale_ratio[0]
        b = w * scale_ratio[1]
        c = h * scale_ratio[2]

        bottom = center_height / c

        # 构造单位球面
        r = 1.0
        ball = gmsh.model.occ.addSphere(0, 0, 0, r, 1, -pi / 2, 0)

        # 底面截取
        box = gmsh.model.occ.addBox(-1, -1, -1, 2, 2, 1 - bottom)
        half_ball = gmsh.model.occ.cut([(3, ball)], [(3, box)])[0]

        # 底部矩形附着
        rec = gmsh.model.occ.addRectangle(-0.5 / scale_ratio[0], -0.5 / scale_ratio[1], -bottom, 1 / scale_ratio[0],
                                          1 / scale_ratio[1])
        # gmsh.model.occ.synchronize()
        # gmsh.model.mesh.generate(2)
        # gmsh.fltk.run()

        # 分割线对应的固定点
        fixed_point1 = [0.5 * r / scale_ratio[0], -0.5 * r / scale_ratio[1], -bottom - 0.5 * r]
        fixed_point2 = [0.5 * r * v / scale_ratio[0], -0.5 * r / scale_ratio[1], -bottom - 0.5 * r]
        fixed_point3 = [-0.5 * r * v / scale_ratio[0], -0.5 * r / scale_ratio[1], -bottom - 0.5 * r]
        fixed_point4 = [-0.5 * r / scale_ratio[0], -0.5 * r / scale_ratio[1], -bottom - 0.5 * r]
        fixed_point5 = [-0.5 * r / scale_ratio[0], 0.5 * r / scale_ratio[1], -bottom - 0.5 * r]
        fixed_point6 = [-0.5 * r * v / scale_ratio[0], 0.5 * r / scale_ratio[1], -bottom - 0.5 * r]
        fixed_point7 = [0.5 * r * v / scale_ratio[0], 0.5 * r / scale_ratio[1], -bottom - 0.5 * r]
        fixed_point8 = [0.5 * r / scale_ratio[0], 0.5 * r / scale_ratio[1], -bottom - 0.5 * r]

        def get_plane(fixed_point, phi):
            p1 = gmsh.model.occ.addPoint(fixed_point[0], fixed_point[1], fixed_point[2])
            p2 = gmsh.model.occ.addPoint(fixed_point[0] + r * np.cos(phi), fixed_point[1] + r * np.sin(phi),
                                         fixed_point[2])
            p3 = gmsh.model.occ.addPoint(fixed_point[0] + r * np.cos(phi), fixed_point[1] + r * np.sin(phi),
                                         fixed_point[2] + 1.5 * r)
            p4 = gmsh.model.occ.addPoint(fixed_point[0], fixed_point[1], fixed_point[2] + 1.5 * r)
            line1 = gmsh.model.occ.addLine(p1, p2)
            line2 = gmsh.model.occ.addLine(p2, p3)
            line3 = gmsh.model.occ.addLine(p3, p4)
            line4 = gmsh.model.occ.addLine(p4, p1)
            curve = gmsh.model.occ.addCurveLoop([line1, line2, line3, line4])
            plane = gmsh.model.occ.addPlaneSurface([curve])

            return plane

        # 分割平面
        plane1 = get_plane(fixed_point1, -theta1)
        plane2 = get_plane(fixed_point1, -pi / 2 + theta1)
        plane3 = get_plane(fixed_point2, -pi / 2 - theta2)
        plane4 = get_plane(fixed_point3, -pi / 2 + theta2)
        plane5 = get_plane(fixed_point4, -pi / 2 - theta1)
        plane6 = get_plane(fixed_point4, -pi + theta1)
        plane7 = get_plane(fixed_point5, pi - theta1)
        plane8 = get_plane(fixed_point5, pi / 2 + theta1)
        plane9 = get_plane(fixed_point6, pi / 2 - theta2)
        plane10 = get_plane(fixed_point7, pi / 2 + theta2)
        plane11 = get_plane(fixed_point8, pi / 2 - theta1)
        plane12 = get_plane(fixed_point8, theta1)

        ov = gmsh.model.occ.fragment(half_ball, [(2, rec)])[0]

        # 分割面与主体求交，得到分割线
        intersection = gmsh.model.occ.intersect(ov,
                                                [(2, plane1), (2, plane2), (2, plane3), (2, plane4), (2, plane5),
                                                 (2, plane6),
                                                 (2, plane7), (2, plane8), (2, plane9), (2, plane10), (2, plane11),
                                                 (2, plane12)], removeObject=False)

        gmsh.model.occ.synchronize()
        # 调整网格密度
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), densty)
        gmsh.model.mesh.generate(2)

        # 获取分割线节点
        total_vects = []
        start_surface_tag = 5
        for i in range(12):
            surface_tag = start_surface_tag + i
            curves = gmsh.model.getBoundary([(2, surface_tag)], combined=False, oriented=True)[1:3]
            curve_vects = []
            for curve in curves:
                tag = abs(curve[1])
                node_pre = gmsh.model.mesh.getNodes(1, tag, includeBoundary=True)[1].reshape(-1, 3)
                nodes = np.zeros_like(node_pre)
                nodes[0, :] = node_pre[-2, :]
                nodes[-1, :] = node_pre[-1, :]
                nodes[1:-1, :] = node_pre[0:-2, :]
                # 由球面映射到椭球面
                nodes[:, 0] *= a
                nodes[:, 1] *= b
                nodes[:, 2] *= c
                curve_vects.append(nodes)
            total_vects.append(curve_vects)

        # 显示
        # gmsh.fltk.run()
        camera_points = []
        for i in range(6):
            camera_points.append([])
            camera_points[i].append(total_vects[(2 * i) % 12][0])
            camera_points[i].append(total_vects[(2 * i) % 12][1])
            camera_points[i].append(total_vects[(2 * i + 1) % 12][0])
            camera_points[i].append(total_vects[(2 * i + 1) % 12][1])
            camera_points[i].append(total_vects[(2 * i + 2) % 12][0])
            camera_points[i].append(total_vects[(2 * i + 2) % 12][1])
            camera_points[i].append(total_vects[(2 * i + 3) % 12][0])
            camera_points[i].append(total_vects[(2 * i + 3) % 12][1])
        gmsh.finalize()

        # 构建横向分割线
        intersection_points_list = np.zeros((12, 3, 3))
        for i in range(12):
            intersection_points_list[i, 0] = total_vects[i][1][-1]
            intersection_points_list[i, 1] = total_vects[i][1][0]
            intersection_points_list[i, 2] = total_vects[i][0][0]
        # 恢复到单位球面
        intersection_points_list[..., 0] /= a
        intersection_points_list[..., 1] /= b
        intersection_points_list[..., 2] /= c
        # 拟合横向分割线
        gmsh.initialize()
        intersection_line_tag = []
        intersection_line_tag.append([])
        intersection_line_tag.append([])
        intersection_line_tag.append([])
        # 分割线 1、2
        o1 = gmsh.model.occ.addPoint(0, 0, 0)
        o2 = gmsh.model.occ.addPoint(0, 0, -bottom)
        for i in range(12):
            p11 = intersection_points_list[i, 0]
            p12 = intersection_points_list[(i + 1) % 12, 0]
            p21 = intersection_points_list[i, 1]
            p22 = intersection_points_list[(i + 1) % 12, 1]
            p11_tag = gmsh.model.occ.addPoint(p11[0], p11[1], p11[2])
            p12_tag = gmsh.model.occ.addPoint(p12[0], p12[1], p12[2])
            p21_tag = gmsh.model.occ.addPoint(p21[0], p21[1], p21[2])
            p22_tag = gmsh.model.occ.addPoint(p22[0], p22[1], p22[2])
            int_tag1 = gmsh.model.occ.addCircleArc(p11_tag, o1, p12_tag)
            intersection_line_tag[0].append(int_tag1)
            int_tag2 = gmsh.model.occ.addCircleArc(p21_tag, o2, p22_tag)
            intersection_line_tag[1].append(int_tag2)
        # 分割线 3
        for i in range(8):
            if i in [0, 1, 2]:
                j = i + 1
            if i == 3:
                j = i + 2
            if i in [4, 5, 6]:
                j = i + 3
            if i == 7:
                j = i + 4
            p1 = intersection_points_list[j, 2]
            p2 = intersection_points_list[(j + 1) % 12, 2]
            p1_tag = gmsh.model.occ.addPoint(p1[0], p1[1], p1[2])
            p2_tag = gmsh.model.occ.addPoint(p2[0], p2[1], p2[2])
            int_tag = gmsh.model.occ.addLine(p1_tag, p2_tag)
            intersection_line_tag[2].append(int_tag)

        gmsh.model.occ.synchronize()
        gmsh.model.mesh.setSize(gmsh.model.getEntities(0), densty)
        gmsh.model.mesh.generate(1)

        horizontal_split_points = []
        for i in range(len(intersection_line_tag)):
            horizontal_split_points.append([])
            for j in range(len(intersection_line_tag[i])):
                node_pre = gmsh.model.mesh.getNodes(1, intersection_line_tag[i][j], includeBoundary=True)[1].reshape(-1,
                                                                                                                     3)
                nodes = np.zeros_like(node_pre)
                nodes[0, :] = node_pre[-2, :]
                nodes[-1, :] = node_pre[-1, :]
                nodes[1:-1, :] = node_pre[0:-2, :]
                # 由球面映射到椭球面
                nodes[:, 0] *= a
                nodes[:, 1] *= b
                nodes[:, 2] *= c
                horizontal_split_points[i].append(nodes)

        gmsh.finalize()
        # 装配相机分割线
        for i in range(6):
            for j in range(3):
                camera_points[i].append(horizontal_split_points[0][(2 * i + j) % 12])
                camera_points[i].append(horizontal_split_points[1][(2 * i + j) % 12])
            if i in [0, 1, 3, 4]:
                if i in [0, 1]:
                    j = i
                else:
                    j = i + 1
                camera_points[i].append(horizontal_split_points[2][j])
                camera_points[i].append(horizontal_split_points[2][j + 1])
            else:
                if i == 2:
                    j = i + 1
                else:
                    j = i + 2
                camera_points[i].append(horizontal_split_points[2][j])

        return camera_points

    def screen_to_viewpoint(self, points):
        """
        @brief 将屏幕上的点映射到视点单位球
        """
        vp = self.viewpoint
        v = points-vp[None, :]
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
        return v+vp[None, :]

    def screen_to_image(self):
        """
        @brief 从视点到相机球面坐标
        """
        uv = []
        for i, cam in enumerate(self.cams):
            mesh   = self.groundmesh[i]
            node_s = mesh.entity('node').copy()
            node   = self.screen_to_viewpoint(node_s)
            mesh.node = node

            data = HarmonicMapData(mesh, self.didx[i], self.dval[i])
            node = sphere_harmonic_map(data).reshape(-1, 3)
            node += cam.location
            uvi = cam.world_to_image(node)
            uvi[:, 0] = 1-uvi[:, 0]

            uv.append(uvi)
            mesh.node = node_s
        return uv

    def show_ground_mesh_with_view_point(self, plotter):
        """
        @brief 显示地面网格和视点
        """
        mesh = TriangleMesh.from_box([-10, 10, -10, 10], nx=100, ny=100)
        node = np.zeros((mesh.number_of_nodes(), 3), dtype=np.float64)
        node[:, 0:2] = mesh.entity('node')
        cell = mesh.entity('cell')
        # 投影到单位球面
        uv = self.screen_to_image()
        for i, cam in enumerate(self.cams):
            mesh = self.groundmesh[i]
            node = mesh.entity('node')
            cell = mesh.entity('cell')
            no = np.concatenate((node[cell].reshape(-1, 3), uv[i][cell].reshape(-1, 2)), axis=-1, dtype=np.float32)
            plotter.add_mesh(no, cell=None, texture_path=cam.fname)

    def undistort_cv(self):
        import cv2
        images = []
        for cam in self.cams:
            h = cam.height
            w = cam.width


    @classmethod
    def from_data(cls, data_path: str='~'):
        """
        @brief 测试数据

        @param[in] h : 世界坐标原点到地面的高度
        """
        h = 3
        location = np.array([ # 相机在世界坐标系中的位置
            [ 8.35/2.0, -3.47/2.0, 1.515], # 右前
            [-8.35/2.0, -3.47/2.0, 1.505], # 右后
            [-17.5/2.0,       0.0, 1.295], # 后
            [-8.35/2.0,  3.47/2.0, 1.495], # 左后
            [ 8.35/2.0,  3.47/2.0, 1.495], # 左前
            [ 17.5/2.0,       0.0, 1.345]  # 前
            ], dtype=np.float64)
        location[:, 2] -= h

        t = np.sqrt(2.0)/2.0
        cz = np.array([ # 相机 z 轴在世界坐标系中的指向
            [0.0,  -t, -t],  # 右前
            [0.0,  -t, -t],  # 右后
            [ -t, 0.0, -t],  # 后 
            [0.0,   t, -t],  # 左后
            [0.0,   t, -t],  # 左前
            [  t, 0.0, -t]   # 前
            ], dtype=np.float64)
        cx = np.array([ # 相机 x 轴在世界坐标系中的指向
            [-1.0,  0.0, 0.0],  # 右前
            [-1.0,  0.0, 0.0],  # 右后
            [ 0.0,  1.0, 0.0],  # 后 
            [ 1.0,  0.0, 0.0],  # 左后
            [ 1.0,  0.0, 0.0],  # 左前
            [ 0.0, -1.0, 0.0]   # 前
            ], dtype=np.float64)
        cy = np.cross(cz, cx) # 相机 y 轴在世界坐标系中的指向

        #polynomial coefficients for the DIRECT mapping function (ocam_model.ss in MATLAB). These are used by cam2world
        ss = [
        [-5.763797e+02, 0.000000e+00, 7.185556e-04, -3.399070e-07, 5.242219e-10],
        [-5.757232e+02, 0.000000e+00, 7.587041e-04, -3.740247e-07, 5.173472e-10],
        [-5.769944e+02, 0.000000e+00, 6.960907e-04, -2.129561e-07, 3.806627e-10],
        [-5.757232e+02, 0.000000e+00, 7.587041e-04, -3.740247e-07, 5.173472e-10],
        [-5.763797e+02, 0.000000e+00, 7.185556e-04, -3.399070e-07, 5.242219e-10],
        [-5.751298e+02, 0.000000e+00, 7.332358e-04, -3.633660e-07, 5.286731e-10]
        ]

        #polynomial coefficients for the inverse mapping function (ocam_model.invpol in MATLAB). These are used by world2cam
        pol = [
            [845.644875, 482.093504, -4.074978, 71.443521, 34.750033, 3.348958, 19.469493, 10.236789, -11.771018, -10.331102, -2.154892],
            [842.618702, 489.883562, 3.551579, 68.390516, 35.533898, -0.486649, 12.653096, 21.865068, 9.894399, 1.351086],
            [853.690706, 511.122043, 22.215504, 72.273914, 36.289875, 7.590651, 14.715128, 16.256317, 6.272230, 0.752288],
            [842.618702, 489.883562, 3.551579, 68.390516, 35.533898, -0.486649, 12.653096, 21.865068, 9.894399, 1.351086],
            [845.644875, 482.093504, -4.074978, 71.443521, 34.750033, 3.348958, 19.469493, 10.236789, -11.771018, -10.331102, -2.154892],
            [845.738193, 486.117526, -3.075807, 69.772397, 36.084962, 2.499655, 17.305930, 12.154529, -8.322921, -8.780900, -1.922651],
            ]

        #center: "row" and "column", starting from 0 (C convention)
        center = np.array([
            [559.875074, 992.836922],
            [575.297515, 987.142409],
            [533.159817, 992.262661],
            [595.297515, 987.142409],
            [559.875074, 982.836922],
            [539.106804, 939.819626],
            ], dtype=np.float64)

        # 仿射系数
        affine = np.array([
            [1.000938,  0.000132, -0.000096],
            [1.000004, -0.000176, -0.000151],
            [1.000921,  0.000077,  0.000329],
            [1.000004, -0.000176, -0.000151],
            [1.000938,  0.000132, -0.000096],
            [1.000375,  0.000070,  0.000432],
            ], dtype=np.float64)

        # 默认文件目录位置
        fname = [
            os.path.expanduser(data_path+'src_1.jpg'),
            os.path.expanduser(data_path+'src_2.jpg'),
            os.path.expanduser(data_path+'src_3.jpg'),
            os.path.expanduser(data_path+'src_4.jpg'),
            os.path.expanduser(data_path+'src_5.jpg'),
            os.path.expanduser(data_path+'src_6.jpg'),
            ]

        fname = [
            os.path.expanduser(data_path+'camera_inputs/src_1.jpg'),
            os.path.expanduser(data_path+'camera_inputs/src_2.jpg'),
            os.path.expanduser(data_path+'camera_inputs/src_3.jpg'),
            os.path.expanduser(data_path+'camera_inputs/src_4.jpg'),
            os.path.expanduser(data_path+'camera_inputs/src_5.jpg'),
            os.path.expanduser(data_path+'camera_inputs/src_6.jpg'),
            ]

        flip = [
            None, None, None, None, None, None 
        ]

        chessboardpath = [
            os.path.expanduser(data_path+'camera_models/chessboard_1'),
            os.path.expanduser(data_path+'camera_models/chessboard_2'),
            os.path.expanduser(data_path+'camera_models/chessboard_3'),
            os.path.expanduser(data_path+'camera_models/chessboard_4'),
            os.path.expanduser(data_path+'camera_models/chessboard_5'),
            os.path.expanduser(data_path+'camera_models/chessboard_6'),
            ]

        icenter = np.array([
            [992.559,526.134],
            [986.667,513.157],
            [978.480,535.08],
            [985.73,559.472],
            [961.628,551.019],
            [940.2435,541.3385],
            ], dtype=np.float64)
        icenter[:, 1] *= -1
        icenter[:, 1] += 1080
        #icenter = center[:, ::-1]

        radius = np.array([877.5,882.056,886.9275,884.204,883.616,884.5365],dtype=np.float64)
        mark_board = np.array(
        [[(240.946,702.130),(265.611,726.620),(326.989,605.794),(291.346,599.545),
         (248.493,694.023),(268.282,711.997),(314.241,614.862),(288.833,609.140),
         (265.521,668.427),(275.295,673.864),(291.393,641.555),(280.467,638.472),
         (1673.001,772.285),(1711.515,733.990),(1654.236,612.171),(1600.493,622.823),
         (1671.702,754.323),(1702.732,725.841),(1654.219,624.930),(1616.665,634.889),
         (1666.478,705.236),(1680.064,697.104),(1661.974,660.024),(1647.485,665.864)],
         [(268.593,717.706),(308.979,755.603),(384.439,608.193),(329.074,598.007),
         (279.318,709.072),(309.456,737.338),(369.147,619.301),(329.805,611.696),
         (301.373,681.322),(315.767,690.052,),(339.370,650.618),(320.949,644.135),
         (1704.463,768.055),(1732.520,732.221),(1681.072,626.701),(1646.601,638.913),
         (1703.475,749.778),(1722.529,726.180),(1682.977,638.310),(1657.391,648.564),
         (1696.748,710.659),(1706.177,701.772),(1690.590,669.869),(1680.413,675.212)],
         [(336.722,761.334),(459.065,843.159),(561.643,572.304),(421.546,567.267),
         (356.073,747.818),(451.158,806.274),(534.843,593.878),(428.283,584.402),
         (400.842,700.596),(436.129,717.193),(467.284,647.240),(427.526,638.794),
         (1532.010,805.481),(1654.638,732.667),(1560.331,543.526),(1423.584,549.188),
         (1540.329,771.869),(1635.315,721.673),(1557.946,560.666),(1450.886,568.042),
         (1552.074,688.311),(1588.403,675.260),(1557.017,613.734),(1522.558,621.695)],
         [(233.503,674.422),(259.214,702.625),(316.859,578.079),(279.466,569.207),
         (243.090,665.626),(260.053,685.466),(305.689,587.776),(281.988,581.209),
         (258.467,642.556),(268.377,648.475),(284.372,614.911),(272.615,611.244),
         (1666.714,711.499),(1703.773,672.993),(1643.246,554.714),(1594.974,564.733),
         (1665.166,695.445),(1695.455,665.853),(1643.087,566.604),(1606.476,577.471),
         (1658.640,646.255),(1670.754,637.334),(1652.185,600.063),(1637.795,607.301)],
         [(260.645,674.616),(302.728,708.959),(378.877,560.912),(320.521,553.519),
         (270.127,666.156),(303.906,692.382),(364.535,571.969),(321.852,565.919),
         (295.149,637.813),(309.516,644.059),(331.469,605.169),(312.906,599.970),
         (1690.343,688.457),(1716.687,664.436),(1665.597,557.933),(1629.563,564.493),
         (1688.991,673.511),(1708.099,655.327),(1667.512,571.222),(1642.266,576.273),
         (1681.099,637.213),(1691.977,630.243),(1675.108,598.143),(1663.869,602.714)],
         [(266.546,761.228),(368.820,838.300),(471.901,599.210),(350.959,587.009),
         (282.813,752.674),(364.447,809.124),(447.787,614.515),(352.791,602.933),
         (323.830,712.295),(356.378,726.475),(383.816,661.568),(349.424,653.671),
         (1460.643,835.675),(1581.636,748.868),(1483.032,566.232),(1344.212,587.207),
         (1468.790,802.020),(1562.534,736.320),(1480.875,583.683),(1372.462,602.471),
         (1480.763,713.575),(1514.355,696.954),(1483.021,638.530),(1446.418,650.523)]
         ],dtype=np.float64)

        mark_board[...,1] = 1080-mark_board[...,1] 
        data = {
            "nc" : 6,
            "location" : location, # 相机位置
            "axes" : (cx, cy, cz), # 相机坐标系旋转矩阵
            "center" : center,     # 相机中心
            "ss" : ss,             # 鱼眼相机映射的多项式系数
            "pol" : pol,           # 鱼眼相机逆映射多项式系数
            "affine" : affine,     # 仿射变换系数
            "fname" : fname,       # 图片文件名
            "chessboardpath": chessboardpath, # 棋盘格文件名
            "width" : 1920,        # 图片宽度
            "height" : 1080,       # 图片高度
            "vfield" : (110, 180), # 水平视场角，垂直视场角
            'flip' : flip,         # 是否翻转
            'icenter': icenter,    # 图片中心
            'radius' : radius,     # 图片半径
            'mark_board': mark_board, # 地面标记的点
            'center_height' : h,      # 世界坐标原点到地面的高度
            'size' : (17.5, 3.47, 3), # 小车长宽高
            'scale_ratio' : (1.618, 3.618, 1.618), # 三个主轴的伸缩比例
            'viewpoint' : (0, 0, 0),  # 视点
            'data_path': data_path,   # 数据文件路径
        }

        return cls(data)

