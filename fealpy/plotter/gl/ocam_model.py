from dataclasses import dataclass, field
from typing import Callable, Any, Tuple
import numpy as np

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
        print(self.location)
        print(self.axes)
        node = np.einsum('ij, kj->ik', node-self.location, self.axes)
        return node

    def cam_to_image(self, node):
        """
        @brief 把相机坐标系中的点投影到归一化的图像 uv 坐标系
        """

        NN = len(node)
        f = np.sqrt((self.height/2)**2 + (self.width/2)**2)
        r = np.sqrt(np.sum(node**2, axis=1))
        theta = np.arccos(node[:, 2]/r)
        phi = np.arctan2(node[:, 1], node[:, 0])
        phi = phi % (2 * np.pi)

        uv = np.zeros((NN, 2), dtype=np.float64)

        uv[:, 0] = f * theta * np.cos(phi) + self.center[0] 
        uv[:, 1] = f * theta * np.sin(phi) + self.center[1] 

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


