from dataclasses import dataclass, field
from typing import Callable, Any
import numpy as np


@dataclass
class OCAMModel:
    ss: np.ndarray = np.array([-576.3797, 0, 0.0007185556, -3.39907e-07, 5.242219e-10])
    pol : np.ndarray = np.array([845.644875, 482.093504, -4.074978,
        71.443521, 34.750033, 3.348958, 19.469493, 10.236789, -11.771018,
        -10.331102, -2.154892])
    xc: float = 559.875074
    yc: float = 992.836922
    c: float = 1.000938 
    d: float = 0.000132
    e: float = -0.000096


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



    def world2cam_fast(self, node):
        """
        """
        NN = len(node)
        theta = np.zeros(NN, dtype=np.float64)

        norm = np.sqrt(node[:, 0]**2 + node[:, 1]**2)
        flag = (norm == 0)
        norm[flag] = np.finfo(float).eps
        theta = np.arctan(node[:, 2]/norm)

        rho = np.polyval(self.pol, theta)
        ps = node[:, 0:2]/norm[:, None]*rho[:, None]
        uv = np.zeros_like(ps)
        uv[:, 0] = ps[:, 0] * self.c + ps[:, 1] * self.d + self.xc
        uv[:, 1] = ps[:, 0] * self.e + ps[:, 1]          + self.yc
        return uv

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


