
import numpy as np
from .ocam_model import OCAMModel
from .ocam_system import OCAMSystem

class Calibration:
    def __init__(self, ocam_system: OCAMSystem):
        self.ocam_system = ocam_system
        lfront = ocam_systerm.cams[4]
        lrear = ocam_systerm.cams[3]
        rfront = ocam_systerm.cams[0]
        rrear = ocam_systerm.cams[1]
        front = ocam_systerm.cams[5]
        rear = ocam_systerm.cams[2]

        align_point = np.zeros([6, 2, 12, 2], dtype=np.float_)
        align_point[0, 0] = lfront.mark_board[12:]
        align_point[0, 1] =  front.mark_board[:12]
        align_point[1, 0] =  front.mark_board[12:]
        align_point[1, 1] = rfront.mark_board[:12]
        align_point[2, 0] = rfront.mark_board[12:]
        align_point[2, 1] = rrear.mark_board[:12]
        align_point[3, 0] = rrear.mark_board[12:]
        align_point[3, 1] =  rear.mark_board[:12]
        align_point[4, 0] =  rear.mark_board[12:]
        align_point[4, 1] = lrear.mark_board[:12]
        align_point[5, 0] = lrear.mark_board[12:]
        align_point[5, 1] = lfront.mark_board[:12]

        self.models = [[lfront, front], [front, rfront], [rfront, rrear], 
                       [rrear, rear], [rear, lrear], [lrear, lfront]]
        self.align_point = align_point



