import numpy as np

class OCAMSystem:
    def __init__(self):
        self.cams = []

    def set_cam(self, cam):
        self.cams.append(cam)