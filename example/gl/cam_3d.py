
import numpy as np

from fealpy.mesh import TriangleMesh
from fealpy.plotter.gl import OpenGLPlotter, OCAMModel, OCAMSystem 


# 准备数据
h = 3.0 # 世界坐标原点到地面的高度
location = np.array([ # 相机在世界坐标系中的位置
    [ 8.35/2.0, -3.47/2.0, 1.515], # 右前
    [-8.25/2.0, -3.47/2.0, 1.505], # 右后
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
    '/home/why/data/src_1.jpg',
    '/home/why/data/src_2.jpg',
    '/home/why/data/src_3.jpg',
    '/home/why/data/src_4.jpg',
    '/home/why/data/src_5.jpg',
    '/home/why/data/src_6.jpg',
    ]

data = {
    "nc" : 6,
    "location" : location,
    "axes" : (cx, cy, cz),
    "center" : center,
    "ss" : ss,
    "pol" : pol,
    "affine" : affine,
    "fname" : fname,
    "height" : 1080,
    "width"  : 1920
}

mesh= TriangleMesh.from_section_ellipsoid()
node = mesh.entity('node')
cell = mesh.entity('cell')
domain = mesh.celldata['domain']

#csys = OCAMSystem(data)
#csys.show_images()

node = np.array(node, dtype=np.float32)
vertices = node[cell].reshape(-1, node.shape[1])
plotter = OpenGLPlotter()
plotter.add_mesh(vertices, cell=None, texture_path=None)
plotter.run()