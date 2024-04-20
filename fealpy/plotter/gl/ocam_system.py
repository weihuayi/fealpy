import numpy as np
from .ocam_model import OCAMModel

class OCAMSystem:
    def __init__(self, data):
        self.cams = []
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
                affine = data['affine'],
                fname = data['fname'][i],
            ))

        

