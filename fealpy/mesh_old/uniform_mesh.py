from typing import Sequence, Tuple, Union, Type, Any


class UniformMesh():
    """
    Factory for creating uniform mesh objects in 1D, 2D, or 3D, with strict type
    and shape checks.

    Supports two calling conventions:
      1) domain + extent
      2) extent + h + origin

    Can be invoked with all positional args, or all keyword args.

    Parameters (positional or keyword):
        domain (Sequence[float], optional):
            Flat list [min1, max1, ..., minD, maxD] describing the physical domain.
            Required when using the domain+extent overload.
        extent (Sequence[int]):
            Integer index boundaries for each dimension, length must be 2*D:
            [i0_min, i0_max, ..., iD-1_min, iD-1_max].
        h (float or Sequence[float], optional):
            Uniform grid spacing. Either a scalar applied to all dimensions, or
            a sequence of length D. Required when using the extent+h+origin overload.
        origin (float or Sequence[float], optional):
            Coordinates of the mesh origin. Either a scalar applied to all
            dimensions, or a sequence of length D.
        itype (Type, optional):
            Integer data type for index arrays, e.g. numpy.int64, torch.int32.
        ftype (Type, optional):
            Floating-point data type for coordinate and field arrays,
            e.g. numpy.float64, torch.float32.
        device (Any, optional):
            Compute device identifier (e.g. 'cpu', 'cuda', torch.device).

    Returns:
        Instance of one of {UniformMesh1d, UniformMesh2d, UniformMesh3d},
        matching the dimension GD = len(extent) // 2.

    Raises:
        ValueError:
            - If len(domain) or len(extent) is invalid.
            - If required parameters for a chosen overload are missing.
            - If computed D is not 1, 2, or 3.
            - If `h` or `origin` lengths ≠ GD, or contain bad types.
    """
    def __new__(cls, *args, **kwargs):
        # Extract optional keyword parameters
        itype = kwargs.get('itype')
        ftype = kwargs.get('ftype')
        device = kwargs.get('device')

        # —————— Handle positional arguments ——————
        if args:
            if len(args) == 2:
                domain, extent = args
                h = origin = None
            elif len(args) == 3:
                extent, h, origin = args
                domain = None
            else:
                raise ValueError(f"Expected 2 or 3 positional args, got {len(args)}")
        else:
            # All-keyword invocation
            domain = kwargs.get('domain')
            extent = kwargs.get('extent')
            h = kwargs.get('h')
            origin = kwargs.get('origin')
            if extent is None:
                raise ValueError("`extent` must be provided.")

        # Determine which overload is used: domain+extent → compute h & origin
        if domain is not None and extent is not None and h is None and origin is None:
            # Validate domain and extent lengths
            if len(domain) % 2 != 0 or len(domain) != len(extent):
                raise ValueError("`domain` and `extent` must have equal even lengths (2*D).")
            GD = len(domain) // 2

            # Compute spacing and origin per dimension
            h = []
            origin = []
            for i in range(GD):
                dmin = float(domain[2*i])
                dmax = float(domain[2*i + 1])
                imin = int(extent[2*i])
                imax = int(extent[2*i + 1])
                ncell = imax - imin
                if ncell <= 0:
                    raise ValueError(f"Extent for dimension {i} yields non-positive cell count.")
                h.append((dmax - dmin) / ncell)
                origin.append(dmin)
            h = tuple(h)
            origin = tuple(origin)

        # Else require extent + h + origin overload
        elif extent is not None and h is not None and origin is not None and domain is None:
            # D will be inferred below
            pass

        else:
            raise ValueError(
                "Invalid arguments: use either (domain, extent) or (extent, h, origin),"
                " with positional or keyword form."
            )

        # —————— Shared factory logic ——————
        # 1) Validate extent
        if len(extent) % 2 != 0 or any(not isinstance(x, int) for x in extent):
            raise ValueError(f"`extent` must be an int sequence of length 2*D, got {extent}")
        GD = len(extent) // 2

        # 2) Dispatch to the appropriate dimension class
        from .uniform_mesh_1d import UniformMesh1d
        from .uniform_mesh_2d import UniformMesh2d
        from .uniform_mesh_3d import UniformMesh3d
        dim2class = {1: UniformMesh1d, 2: UniformMesh2d, 3: UniformMesh3d}
        MeshClass = dim2class.get(GD)
        if MeshClass is None:
            raise ValueError(f"Unsupported dimension: {GD}. Only 1, 2, or 3 are supported.")

        # 3) Normalize h to a tuple of floats
        if isinstance(h, (int, float)):
            h_tuple = (float(h),) * GD
        else:
            if len(h) != GD or any(not isinstance(x, (int, float)) for x in h):
                raise ValueError(f"`h` must be float or sequence of {GD} floats, got {h}")
            h_tuple = tuple(float(x) for x in h)

        # 4) Normalize origin to a tuple of floats
        if isinstance(origin, (int, float)):
            origin_tuple = (float(origin),) * GD
        else:
            if len(origin) != GD or any(not isinstance(x, (int, float)) for x in origin):
                raise ValueError(f"`origin` must be float or sequence of {GD} floats, got {origin}")
            origin_tuple = tuple(float(x) for x in origin)

        # 5) Create and return the subclass instance
        return MeshClass(
            extent=tuple(extent),
            h=h_tuple,
            origin=origin_tuple,
            itype=itype,
            ftype=ftype,
            device=device
        )

