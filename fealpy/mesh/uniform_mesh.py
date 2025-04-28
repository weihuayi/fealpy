from typing import Union, Tuple, Type


class UniformMesh():
    """
    Factory for creating uniform mesh objects in 1D, 2D, or 3D, with strict type
    and shape checks.

    Parameters:
        extent (Tuple[int, ...]):
            Integer domain boundaries for each dimension, length must be 2*D:
            (min1, max1, ..., minD, maxD). All entries must be ints.
        h (Union[float, Tuple[float, ...]]):
            Uniform grid spacing. Can be a single float (applied to all D dims)
            or a tuple of D floats specifying spacing per dimension.
        origin (Union[float, Tuple[float, ...]]):
            Coordinates of the mesh origin. Can be a single float (applied to 
            all D dims) or a tuple of D floats.
        itype (Type):
            Integer data type for index arrays, e.g. numpy.int64, torch.int32.
        ftype (Type):
            Floating-point data type for coordinate and field arrays,
            e.g. numpy.float64, torch.float32.
        device (Any):
            Compute device identifier (e.g. 'cpu', 'cuda', torch.device).

    Returns:
        Instance of one of {UniformMesh1d, UniformMesh2d, UniformMesh3d},
        matching the dimension D = len(extent) // 2.

    Raises:
        ValueError:
            - If len(extent) is not divisible by 2, or contains non‐int.
            - If computed D is not 1, 2, or 3.
            - If `h` is a tuple whose length ≠ D, or contains non-float.
            - If `origin` is a tuple whose length ≠ D, or contains non-float.
    """
    def __new__(cls,
                extent: Tuple[int, ...] = (0, 1),
                h: Union[float, Tuple[float, ...]] = 1.0,
                origin: Union[float, Tuple[float, ...]] = 0.0,
                itype: Type = None,
                ftype: Type = None,
                device: any = None):

        # 1. 检查 extent 长度及整型
        if len(extent) % 2 != 0:
            raise ValueError(
                    f"`extent` length must be divisible by 2, got {len(extent)}")
        if any(not isinstance(x, int) for x in extent):
            raise ValueError(f"All entries in `extent` must be int, got {extent}")
        GD = len(extent) // 2

        # 2. 维度-类映射
        from .uniform_mesh_1d import UniformMesh1d
        from .uniform_mesh_2d import UniformMesh2d
        from .uniform_mesh_3d import UniformMesh3d
        dim2class = {1: UniformMesh1d, 2: UniformMesh2d, 3: UniformMesh3d}
        MeshClass = dim2class.get(GD)
        if MeshClass is None:
            raise ValueError(
                    f"Unsupported dimension: {GD}. Only 1, 2, or 3 are supported.")

        # 3. 统一处理 h：标量→元组，或校验元组长度与类型
        if isinstance(h, (int, float)):
            h_tuple = (float(h),) * GD
        else:
            if len(h) != GD or any(not isinstance(x, (int, float)) for x in h):
                raise ValueError(
                        f"`h` must be float or tuple of {GD} floats, got {h}")
            h_tuple = tuple(float(x) for x in h)

        # 4. 统一处理 origin：同上
        if isinstance(origin, (int, float)):
            origin_tuple = (float(origin),) * GD
        else:
            if len(origin) != GD or any(not isinstance(x, (int, float)) for x in origin):
                raise ValueError(
                        f"`origin` must be float or tuple of {GD} floats, got {origin}")
            origin_tuple = tuple(float(x) for x in origin)

        # 5. 调用对应维度的类
        return MeshClass(
            extent=extent,
            h=h_tuple,
            origin=origin_tuple,
            itype=itype,
            ftype=ftype,
            device=device
        )