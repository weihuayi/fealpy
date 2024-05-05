import numpy as np

def calculate_rotation_matrix(xoffset, yoffset, sensitivity=0.005):
    """
    根据偏移量计算旋转矩阵。
    :param xoffset: 鼠标在X轴方向的偏移量。
    :param yoffset: 鼠标在Y轴方向的偏移量。
    :param sensitivity: 控制旋转速度的灵敏度系数。
    :return: 4x4的旋转矩阵。
    """
    angle = np.sqrt(xoffset**2 + yoffset**2) * sensitivity
    if angle == 0:
        return np.identity(4)

    # 归一化旋转轴
    axis = np.array([xoffset, yoffset, 0.0])
    axis = axis / np.linalg.norm(axis)

    # 使用四元数计算旋转矩阵
    q = quaternion_from_axis_angle(axis, angle)
    rotation_matrix = quaternion_to_rotation_matrix(q)

    return rotation_matrix

def quaternion_from_axis_angle(axis, angle):
    """
    根据旋转轴和旋转角度创建四元数。
    """
    half_angle = angle / 2.0
    sin_half_angle = np.sin(half_angle)
    cos_half_angle = np.cos(half_angle)
    q = np.array([cos_half_angle, axis[0] * sin_half_angle, axis[1] * sin_half_angle, axis[2] * sin_half_angle])
    return q

def quaternion_to_rotation_matrix(q):
    """
    将四元数转换为旋转矩阵。
    """
    # 四元数的元素
    w, x, y, z = q
    # 计算旋转矩阵
    rot_matrix = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,       2*x*z + 2*y*w,     0],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w,     0],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y, 0],
        [0,                     0,                   0,                 1]
    ], dtype=np.float32)
    return rot_matrix

