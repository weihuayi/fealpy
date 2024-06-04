
import itertools
import numpy as np
from .ocam_system import OCAMSystem

class OptimizeParameter:
    """
    @brief Optimize the parameters of the camera model.
    """
    def __init__(self, ocam_systerm : OCAMSystem) -> None:
        self.ocam_systerm = ocam_systerm
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



    def object_function(self, x):
        """
        @brief The object function to be optimized.
        @param x The parameters to be optimized.
        """
        osysterm = self.ocam_systerm
        models = self.models
        align_point = self.align_point

        osysterm.set_parameters(x)

        ## 要对齐的点在屏幕上的坐标
        align_point_screen = np.zeros([6, 2, 12, 3], dtype=np.float_)

        f1, f2 = osysterm.get_implict_surface_function()

        z0 = -osysterm.center_height
        for i, j in itertools.product(range(6), range(2)):
            mod = models[i][j]
            spoint = mod.mesh_to_image(align_point[i, j])
            spoint = align_point[i, j]
            spoint = mod.image_to_camera_sphere(spoint)
            inode = mod.sphere_project_to_implict_surface(spoint, f1)

            outflag = inode[:, 2] <-z0
            align_point_screen[i, j] = inode
        error = np.sum((align_point_screen[:, 0] - align_point_screen[:, 1])**2)
        return error


    def optimize(self):
        """
        @brief Optimize the parameters of the camera model.
        """
        init_x = np.zeros((18, 3), dtype=np.float_)
        for i in range(6):
            init_x[i]    = self.ocam_systerm.cams[i].location
            init_x[i+6]  = self.ocam_systerm.cams[i].axes[0]
            init_x[i+12] = self.ocam_systerm.cams[i].axes[1]

        error = self.object_function(init_x)












