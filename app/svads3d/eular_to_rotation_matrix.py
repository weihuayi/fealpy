import numpy as np

def get_rot_matrix(theta, gamma, beta):
    """
    @brief 从欧拉角计算旋转矩阵。
    @param theta: 绕x轴的旋转角度。
    @param gamma: 绕y轴的旋转角度。
    @param beta: 绕z轴的旋转角度。
    @return: 旋转矩阵。
    """
    # 绕 x 轴的旋转矩阵
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta), -np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])

    # 绕 y 轴的旋转矩阵
    R_y = np.array([[np.cos(gamma), 0, np.sin(gamma)],
                    [0, 1, 0],
                    [-np.sin(gamma), 0, np.cos(gamma)]])

    # 绕 z 轴的旋转矩阵
    R_z = np.array([[np.cos(beta), -np.sin(beta), 0],
                    [np.sin(beta), np.cos(beta), 0],
                    [0, 0, 1]])
    R = R_z @ R_y @ R_x
    return R


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






