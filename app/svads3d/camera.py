import numpy as np
from typing import Union
import os
import cv2
import glob
from scipy.optimize import fsolve


class Camera():
    """
    相机对象，记录相机的位置与朝向，是构造相机系统的基础。
    """
    def __init__(self, picture, data_path, chessboard_dir, location, eular_angle):
        """
        @brief 构造函数。
            1. 获取图片到自身的特征点（地面特征点）

        @param picture: 相机对应的图像。
        @param data_path: 相机的数据路径。
        @param chessboard_dir: 棋盘格图片的路径。
        @param location: 相机的空间位置（世界坐标）。
        @param eular_angle: 相机的欧拉角。
        """
        print("相机初始化...")
        self.picture = picture
        self.picture.camera = self
        self.data_path = data_path
        self.chessboard_dir = chessboard_dir
        self.location = np.array(location)

        self.eular_angle = eular_angle
        self.axes = self.get_rot_matrix(eular_angle[0], eular_angle[1], eular_angle[2])

        print("计算内参矩阵...")
        self.DIM, self.K, self.D = self.get_K_and_D((4, 6), data_path + chessboard_dir)
        print("计算内参矩阵完成。")

        self.theta2rho = lambda theta: theta 
        self.picture.rho2theta = lambda rho: fsolve(lambda theta: self.theta2rho(theta)-rho, 0)[0]

        self.feature_points = {}
        self.feature_points['camera_sphere'] = self.picture_to_self(picture.feature_point['image'])
        self.feature_points['ground'] = picture.feature_point['ground']
        self.camera_system = None
        print("相机初始化完成。")

    def set_screen_frature_points(self, feature_point):
        """
        @brief 设置相机的屏幕特征点。
        @param feature_point: 屏幕特征点。
        @return:
        """
        if isinstance(feature_point, list):
            self.feature_point.extend(feature_point)
        else:
            self.feature_point.append(feature_point)

    def get_rot_matrix(self, theta, gamma, beta) -> np.ndarray:
        """
        @brief 从欧拉角计算旋转矩阵。
        @param theta: 绕x轴的旋转角度。
        @param gamma: 绕y轴的旋转角度。
        @param beta: 绕z轴的旋转角度。
        @return: 旋转矩阵。
        """
        # 绕 x 轴的旋转矩阵
        R_x = np.array([[1, 0, 0],
                        [0, np.cos(theta), -np.sin(theta)],
                        [0, np.sin(theta), np.cos(theta)]])

        # 绕 y 轴的旋转矩阵
        R_y = np.array([[np.cos(gamma), 0, np.sin(gamma)],
                        [0, 1, 0],
                        [-np.sin(gamma), 0, np.cos(gamma)]])

        # 绕 z 轴的旋转矩阵
        R_z = np.array([[np.cos(beta), -np.sin(beta), 0],
                        [np.sin(beta), np.cos(beta), 0],
                        [0, 0, 1]])
        R = R_z @ R_y @ R_x
        return R

    def to_camera_system(self, *args):
        """
        @brief 调和映射，将相机上的点或网格映射到相机系统（视点）。
        @param args: 相机上的点或网格。
        @return:
        """
        assert self.camear_system is not None, "当前相机所属的相机系统未初始化。"
        if type(args[0]) in [list[np.ndarray], np.ndarray, list]:
            pass
        else:
            raise NotImplemented

    def to_picture(self, points, normalizd=False, maptype="L"):
        """
        @brief 将相机上的点或网格映射到图像上。
        @param args: 相机上的点或网格
        @return:
        """
        node = self.world_to_camera(points)

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

        if maptype == 'L': # 等距投影
            rho = self.theta2rho(theta)
            uv[:, 0] = fx * rho * np.cos(phi) + u0
            uv[:, 1] = fy * rho * np.sin(phi) + v0
        elif maptype == 'O': # 正交投影
            uv[:, 0] = fx * np.sin(theta) * np.cos(phi) + u0
            uv[:, 1] = fy * np.sin(theta) * np.sin(phi) + v0
        elif maptype == 'A': # 等积投影
            uv[:, 0] = 2 * fx * np.sin(theta/2) * np.cos(phi) + u0
            uv[:, 1] = 2 * fy * np.sin(theta/2) * np.sin(phi) + v0
        elif maptype == 'S': # 体视投影, Stereographic Projection
            uv[:, 0] = 2 * fx * np.tan(theta/2) * np.cos(phi) + u0
            uv[:, 1] = 2 * fy * np.tan(theta/2) * np.sin(phi) + v0
        else:
            raise ValueError(f"投影类型{ptype}错误!")

        # 标准化
        if normalizd:
            uv = self.picture.normalizd_coordinate(uv)
        return uv


    def set_parameters(self, location, eular_angle, project_coefficient):
        """
        @brief 设置相机的位置与朝向。
        @param location: 相机的位置。
        @param eular_angle: 相机的欧拉角。
        @return:
        """
        init_loc = self.location.copy()
        self.location = np.array(location)
        self.eular_angle = eular_angle
        self.axes = self.get_rot_matrix(eular_angle[0], eular_angle[1], eular_angle[2])

        k1, k2, k3, k4 = project_coefficient
        self.theta2rho = lambda theta: k1*theta + k2*theta**2 + k3*theta**3 + k4*theta**4 
        self.picture.rho2theta = lambda rho: fsolve(lambda theta: self.theta2rho(theta)-rho, 0)[0]

        # 偏移量
        offset = self.location - init_loc
        self.feature_points['camera_sphere'] = self.picture_to_self(self.picture.feature_point['image'])
        #self.feature_points['ground'] = self.picture.feature_point['ground']+offset[:2]


    def projecte_to_self(self, points):
        """
        将点投影到相机球面上。
        @param points: 要投影的点。
        @return: 投影后的点。
        """
        v = points - self.location
        v = v/np.linalg.norm(v, axis=-1, keepdims=True)
        return v + self.location

    def picture_to_self(self, point):
        """
        """
        p = self.picture.to_camera(point, "L")
        return self.camera_to_world(p)

    def to_screen(self, points, on_ground=False):
        """
        将相机球面上的点投影到屏幕上。
        @param args: 相机球面上的点。
        @return:
        """
        screen = self.camera_system.screen
        ret = screen.sphere_to_self(points, self.location, 1.0, on_ground)
        return ret

    def world_to_camera(self, points):
        """
        @brief 把世界坐标系中的点转换到相机坐标系下
        """
        node = np.einsum('ij, kj->ik', points-self.location, self.axes)
        return node

    def camera_to_world(self, node):
        """
        @brief 把相机坐标系中的点转换到世界坐标系下
        """
        node = np.array(node)
        A = np.linalg.inv(self.axes.T)
        node = np.einsum('...j, jk->...k', node, A)
        node += self.location
        return node

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
