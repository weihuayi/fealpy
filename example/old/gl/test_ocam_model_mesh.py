import numpy as np
from fealpy.plotter.gl import OCAMModel
from fealpy.plotter.gl import OCAMSystem
import matplotlib.pyplot as plt

from fealpy.mesh import TriangleMesh
from fealpy.plotter.gl import OpenGLPlotter, OCAMSystem

def euler_angles_from_rotation_matrix(R):
    """
    Compute the Euler angles from a rotation matrix.

    Parameters:
    R (np.ndarray): 3x3 rotation matrix.

    Returns:
    tuple: The Euler angles (alpha, beta, gamma) in radians.
    """
    if R.shape != (3, 3):
        raise ValueError("The rotation matrix must be 3x3.")

    # Calculate sy
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    # Check for singularity
    singular = sy < 1e-6

    if not singular:
        alpha = np.arctan2(R[2, 1], R[2, 2])
        beta = np.arctan2(-R[2, 0], sy)
        gamma = np.arctan2(R[1, 0], R[0, 0])
    else:
        alpha = np.arctan2(-R[1, 2], R[1, 1])
        beta = np.arctan2(-R[2, 0], sy)
        gamma = 0

    return alpha, beta, gamma

csys = OCAMSystem.from_data('~/data/')
plotter = OpenGLPlotter()
#csys.show_split_lines()

#csys.show_screen_mesh(plotter)
#csys.show_ground_mesh(plotter)
csys.show_ground_mesh_with_view_point(plotter)
plotter.run()
#csys.show_sphere_lines()
for i in range(6):
    model = csys.cams[i]
    print(euler_angles_from_rotation_matrix(model.axes))
#    model.show_camera_image_and_mesh(outname='cam%d.png' % i)
    #mesh = model.gmshing_new()

    #fig, axes = plt.subplots()
    #mesh.add_plot(axes)
    #plt.show()



















