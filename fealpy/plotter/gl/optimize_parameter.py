
import numpy as np

class OptimizeParameter:
    """
    @brief Optimize the parameters of the camera model.
    """
    def __init__(self, ocam_systerm):
        self.ocam_systerm = ocam_systerm
        lfront = ocam_systerm.model[0]
        lrear = ocam_systerm.model[1]
        rfront = ocam_systerm.model[2]
        rrear = ocam_systerm.model[3]
        front = ocam_systerm.model[4]
        rear = ocam_systerm.model[5]

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




    def object_function(self, x):
        """
        @brief The object function to be optimized.
        @param x The parameters to be optimized.
        """



    def optimize(self):
        """
        @brief Optimize the parameters of the camera model.
        """

