from ..backend import backend_manager as bm
from ..backend import TensorLike


def apply_rotation(
        points: TensorLike, 
        centers: TensorLike, 
        rotation: TensorLike, GD: int) -> TensorLike:
    """Apply rotation to points relative to given centers.

    Parameters:
        points (TensorLike): Points to rotate, shape (NP, GD).
        centers (TensorLike): Centers of rotation, shape (NC, GD).
        rotation (TensorLike): Rotation angles in radians.
            - For 2D: shape (NC, 1).
            - For 3D: shape (NC, 3).
        GD (int): Geometric dimension (2 or 3).

    Returns:
        TensorLike: Rotated points.
    """
    if GD == 2:
        # For 2D, apply rotation around the center
        angle = rotation[:, 0]  # Rotation angle in radians
        cos_angle = bm.cos(angle)
        sin_angle = bm.sin(angle)

        translated_points = points - centers[:, None, :]
        rotated_points = bm.stack([
            cos_angle[:, None] * translated_points[..., 0] - sin_angle[:, None] * translated_points[..., 1],
            sin_angle[:, None] * translated_points[..., 0] + cos_angle[:, None] * translated_points[..., 1]
        ], axis=-1)
        return rotated_points + centers[:, None, :]

    elif GD == 3:
        # For 3D, apply rotation around each axis (assuming rotation order is x, y, z)
        angles = rotation
        translated_points = points - centers[:, None, :]

        # Rotation around x-axis
        cos_angle_x = bm.cos(angles[:, 0])[:, None]
        sin_angle_x = bm.sin(angles[:, 0])[:, None]
        rotated_points = bm.stack([
            translated_points[..., 0],
            cos_angle_x * translated_points[..., 1] - sin_angle_x * translated_points[..., 2],
            sin_angle_x * translated_points[..., 1] + cos_angle_x * translated_points[..., 2]
        ], axis=-1)

        # Rotation around y-axis
        cos_angle_y = bm.cos(angles[:, 1])[:, None]
        sin_angle_y = bm.sin(angles[:, 1])[:, None]
        rotated_points = bm.stack([
            cos_angle_y * rotated_points[..., 0] + sin_angle_y * rotated_points[..., 2],
            rotated_points[..., 1],
            -sin_angle_y * rotated_points[..., 0] + cos_angle_y * rotated_points[..., 2]
        ], axis=-1)

        # Rotation around z-axis
        cos_angle_z = bm.cos(angles[:, 2])[:, None]
        sin_angle_z = bm.sin(angles[:, 2])[:, None]
        rotated_points = bm.stack([
            cos_angle_z * rotated_points[..., 0] - sin_angle_z * rotated_points[..., 1],
            sin_angle_z * rotated_points[..., 0] + cos_angle_z * rotated_points[..., 1],
            rotated_points[..., 2]
        ], axis=-1)

        return rotated_points + centers[:, None, :]

