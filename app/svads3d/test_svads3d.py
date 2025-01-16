import os
import numpy as np
import json
from typing import Union
from picture import Picture
from camera import Camera
from camera_system import CameraSystem
from screen import Screen
#from fealpy.plotter.gl import OpenGLPlotter
from opengl_plotter import OpenGLPlotter
from meshing_type import MeshingType
from partition_type import PartitionType
from os.path import expanduser

from eular_to_rotation_matrix import * 

if __name__ == '__main__':
    file_path = './data.json'
    with open(file_path, 'r') as file:
        data = json.load(file)

    data_path = expanduser("~") + '/data/'

    mtype = MeshingType.TRIANGLE
    ptype = PartitionType("overlap2", np.pi/4-np.pi/10, np.pi/4+np.pi/10, np.pi/2, 0.1)
    #ptype = PartitionType("nonoverlap", np.pi/6)

    # 计算相机的欧拉角
    eulars = []
    direction = data['direction']
    for di in direction: 
        di = np.array(di)
        eu = euler_angles_from_rotation_matrix(di)
        eulars.append(eu)
    eulars = np.array(eulars)

    feature_points = [data[name+"_feature_points"] for name in data['name']]
    pictures = [Picture(data_path, picture, fp, pic_folder) for picture, fp,
                pic_folder in 
                zip(data['test_pictures'], feature_points, data['pic_folder'])]

    cameras = [Camera(pic, data_path, chessboard_dir, loc, axes) 
               for pic, chessboard_dir, loc, axes in 
               zip(pictures, data['chessboard_dir'], data['locations'], eulars)]
    camear_sys = CameraSystem(cameras, data['view_point'])
    screen = Screen(camear_sys, data["car_size"], data["scale_factor"],
                    data["center_height"], ptype, mtype)

    plotter = OpenGLPlotter()

    print("Displaying the screen...")
    screen.display(plotter)
    plotter.run_pic()
    #plotter.run()










