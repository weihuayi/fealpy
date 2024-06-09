import os
import numpy as np
import json
from typing import Union
from .screen import Screen 


if __name__ == '__main__':
    file_path = './data.json'
    with open(file_path, 'r') as file:
        data = json.load(file)

    pictures = [Picture(picture, mb) for picture, mb in zip(data['pictures'], data["mark_board"])]
    cameras = [Camera(pic, chessboard_dir, loc, axes) for pic, calibration_dir,
               loc, axes in zip(pictures, data['chessboard_dir'], data['locations'], data['axes'])]
    camear_sys = CameraSystem(cameras, data['view_point'])
    screen = Screen(camear_sys, data["size"], data["axis_length"], data["scale_factor"])

    while True:
        screen.display()









