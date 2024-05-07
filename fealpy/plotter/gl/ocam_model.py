from dataclasses import dataclass, field
from typing import Callable, Any, Tuple
import numpy as np
import cv2
import glob
from fealpy.mesh import DistMesher2d
from ...geometry.domain import Domain

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

    def __post_init__(self):
        self.DIM, self.K, self.D = self.get_K_and_D((4, 6), self.chessboardpath)

    def __call__(self, u):
        icenter = self.icenter
        r = self.radius
        d = np.zeros(u.shape[0])
        y1 = icenter[...,1]-np.sqrt(r*r-icenter[...,0]*icenter[...,0])
        y2 = icenter[...,1]+np.sqrt(r*r-icenter[...,0]*icenter[...,0])
        flag1 = np.zeros(u.shape[0],dtype=np.bool_)
        flag1[u[...,1]<y1]=True
        flag1[u[...,1]>y2]=True
        u1 = u[flag1]
        u2 = u[~flag1]
        
        d1 = np.sqrt(np.sum((u1-icenter)**2,axis=-1))-r
        
        v1 = np.array([-icenter[...,0],y2-icenter[...,1]])
        v2 = np.array([-icenter[...,0],y1-icenter[...,1]])
        v3 = np.array([1080-icenter[...,0],y1-icenter[...,1]])
        v4 = np.array([1080-icenter[...,0],y2-icenter[...,1]])
        v = u2-icenter
        
        c1 = np.cross(v1,v)
        c2 = np.cross(v,v2)
        a1 = c1>0
        a2 = c2>0
        c3 = np.cross(v3,v)
        c4 = np.cross(v,v4)
        a3 = c3>0
        a4 = c4>0
        flag2 = np.zeros(u2.shape[0],dtype=np.int_)
        flag2[a1 & a2] = 1
        flag2[a3 & a4] = 2
        d2 = -u2[flag2==1,0] 
        d3 = u2[flag2==2,0]-1080
        
        u3 = u2[flag2==0]
        flag3 = u3[...,0]<icenter[...,0]
        d4 = np.zeros((len(u3[flag3]),2),dtype=np.float64)
        d4[:,0] = -u3[flag3,0]
        d4[:,1] = np.sqrt(np.sum((u3[flag3]-icenter)**2,axis=-1))-r
        d4 = np.min(d4,axis=1)

        d5 = np.zeros((len(u3[~flag3]),2),dtype=np.float64)
        d5[:,0] = u3[~flag3,0]-1080
        d5[:,1] = np.sqrt(np.sum((u3[~flag3]-icenter)**2,axis=-1))-r
        d5 = np.min(d5,axis=1)
        
        d[flag1]=d1
        dd = d[~flag1]
        dd[flag2==1]=d2
        dd[flag2==2]=d3
        ddd=dd[flag2==0]
        ddd[flag3]=d4
        ddd[~flag3]=d5
        dd[flag2==0]=ddd
        d[~flag1]=dd
        return d

    def signed_dist_function(self, u):
        return self(u)

    def gmeshing(self):
        import gmsh
        from fealpy.mesh import TriangleMesh
        icenter = self.icenter
        r = self.radius
        y1 = icenter[...,1]-np.sqrt(r*r-icenter[...,0]*icenter[...,0])
        y2 = icenter[...,1]+np.sqrt(r*r-icenter[...,0]*icenter[...,0])
        gmsh.initialize()

        gmsh.model.geo.addPoint(icenter[...,0],icenter[...,1],0,tag=1)
        gmsh.model.geo.addPoint(0,y1,0,tag=2)
        gmsh.model.geo.addPoint(0,y2,0,tag=3)
        gmsh.model.geo.addPoint(1080,y1,0,tag=4)
        gmsh.model.geo.addPoint(1080,y2,0,tag=5)

        gmsh.model.geo.addCircleArc(2,1,4,tag=1)
        gmsh.model.geo.addLine(4,5,tag=2)
        gmsh.model.geo.addCircleArc(5,1,3,tag=3)
        gmsh.model.geo.addLine(3,2,tag=4)

        gmsh.model.geo.addCurveLoop([1,2,3,4],1)
        gmsh.model.geo.addPlaneSurface([1],1)

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.field.add("Distance",1)
        gmsh.model.mesh.field.setNumbers(1,"CurvesList",[1,3])
        gmsh.model.mesh.field.setNumber(1,"Sampling",100)
        lc = 50
        gmsh.model.mesh.field.add("Threshold", 2)
        gmsh.model.mesh.field.setNumber(2, "InField", 1)
        gmsh.model.mesh.field.setNumber(2, "SizeMin", lc)
        gmsh.model.mesh.field.setNumber(2, "SizeMax", 3*lc)
        gmsh.model.mesh.field.setNumber(2, "DistMin", 200)
        gmsh.model.mesh.field.setNumber(2, "DistMax", 800)

        gmsh.model.mesh.field.setAsBackgroundMesh(2)
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

    def distmeshing(self,fh=None):
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

    def cam_to_image(self, node, ptype='O'):
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

class OCAMDomain(Domain):
    def __init__(self,icenter,radius,hmin=10,hmax=20,fh=None):
        super().__init__(hmin=hmin, hmax=hmax, GD=2)
        if fh is not None:
            self.fh = fh
        self.box = [0,1180,0,2020]
        self.icenter=icenter
        self.radius=radius
        vertices = np.array([
            (0,icenter[...,1]-np.sqrt(radius*radius-icenter[...,0]*icenter[...,0])),
            (0,icenter[...,1]+np.sqrt(radius*radius-icenter[...,0]*icenter[...,0])),
            (1080,icenter[...,1]-np.sqrt(radius*radius-icenter[...,0]*icenter[...,0])),
            (1080,icenter[...,1]+np.sqrt(radius*radius-icenter[...,0]*icenter[...,0]))])
        curves = np.array([[0,1],[2,3]])
        self.facets = {0:vertices,1:curves}
    def __call__(self,u):
        icenter = self.icenter
        r = self.radius
        d = np.zeros(u.shape[0])
        y1 = icenter[...,1]-np.sqrt(r*r-icenter[...,0]*icenter[...,0])
        y2 = icenter[...,1]+np.sqrt(r*r-icenter[...,0]*icenter[...,0])
        flag1 = np.zeros(u.shape[0],dtype=np.bool_)
        flag1[u[...,1]<y1]=True
        flag1[u[...,1]>y2]=True
        u1 = u[flag1]
        u2 = u[~flag1]
        
        d1 = np.sqrt(np.sum((u1-icenter)**2,axis=-1))-r
        
        v1 = np.array([-icenter[...,0],y2-icenter[...,1]])
        v2 = np.array([-icenter[...,0],y1-icenter[...,1]])
        v3 = np.array([1080-icenter[...,0],y1-icenter[...,1]])
        v4 = np.array([1080-icenter[...,0],y2-icenter[...,1]])
        v = u2-icenter
        
        c1 = np.cross(v1,v)
        c2 = np.cross(v,v2)
        a1 = c1>0
        a2 = c2>0
        c3 = np.cross(v3,v)
        c4 = np.cross(v,v4)
        a3 = c3>0
        a4 = c4>0
        flag2 = np.zeros(u2.shape[0],dtype=np.int_)
        flag2[a1 & a2] = 1
        flag2[a3 & a4] = 2
        d2 = -u2[flag2==1,0] 
        d3 = u2[flag2==2,0]-1080
        
        u3 = u2[flag2==0]
        flag3 = u3[...,0]<icenter[...,0]
        d4 = np.zeros((len(u3[flag3]),2),dtype=np.float64)
        d4[:,0] = -u3[flag3,0]
        d4[:,1] = np.sqrt(np.sum((u3[flag3]-icenter)**2,axis=-1))-r
        d4 = np.min(d4,axis=1)

        d5 = np.zeros((len(u3[~flag3]),2),dtype=np.float64)
        d5[:,0] = u3[~flag3,0]-1080
        d5[:,1] = np.sqrt(np.sum((u3[~flag3]-icenter)**2,axis=-1))-r
        d5 = np.min(d5,axis=1)
        
        d[flag1]=d1
        dd = d[~flag1]
        dd[flag2==1]=d2
        dd[flag2==2]=d3
        ddd=dd[flag2==0]
        ddd[flag3]=d4
        ddd[~flag3]=d5
        dd[flag2==0]=ddd
        d[~flag1]=dd
        return d
    def signed_dist_function(self,u):
        return self(u)

    def sizing_function(self,p):
        return self.fh(p,self)
    def facet(self,dim):
        return self.facets[0]
    

