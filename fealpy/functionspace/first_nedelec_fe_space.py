class FirstNedelecFESpace:
    """
    Factory for creating first-kind Nédélec finite element space objects in 2D or 3D,
    with automatic dimension detection and appropriate class selection.

    Parameters:
        mesh:
            The computational mesh object providing geometric information.
        p (int):
            The degree of the finite element space (typically 1 for first-kind Nédélec).

    Returns:
        Instance of either FirstNedelecFiniteElementSpace2d or FirstNedelecFiniteElementSpace3d,
        matching the geometric dimension of the input mesh.

    Raises:
        ValueError:
            - If geometric dimension is 1 (unsupported)
            - If geometric dimension is neither 2 nor 3
    """
    def __new__(cls, mesh, p : int):
        # Get geometric dimension from mesh
        GD = mesh.geo_dimension()
        
        # Validate dimension (Nédélec elements only supported in 2D/3D)
        if GD == 1:
            raise ValueError(
                f"Unsupported dimension: {GD}. Only 2, or 3 are supported.")
        
        from .first_nedelec_fe_space_2d import FirstNedelecFESpace2d
        from .first_nedelec_fe_space_3d import FirstNedelecFESpace3d

        dim2class = {
            2: FirstNedelecFESpace2d,
            3: FirstNedelecFESpace3d          
        }
        
        SpaceClass = dim2class.get(GD)
        return SpaceClass(mesh, p)