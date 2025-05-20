class RaviartThomasFESpace:
    """
    Factory for creating RavartThomasFiniteElementSpace finite element space objects in 2D or 3D,
    with automatic dimension detection and appropriate class selection.

    Parameters:
        mesh:
            The computational mesh object providing geometric information.
        p (int):
            The degree of the finite element space (typically 1 for RaviartThomasFiniteElementSpace).

    Returns:
        Instance of either RavartThomasFiniteElementSpace2d or RavartThomasFiniteElementSpace3d,
        matching the geometric dimension of the input mesh.

    Raises:
        ValueError:
            - If geometric dimension is 1 (unsupported)
            - If geometric dimension is neither 2 nor 3
    """
    def __new__(cls, mesh, p : int):
        # Get geometric dimension from mesh
        GD = mesh.geo_dimension()
        
        # Validate dimension (RavartThomasFiniteElementSpace elements only supported in 2D/3D)
        if GD == 1:
            raise ValueError(
                f"Unsupported dimension: {GD}. Only 2, or 3 are supported.")
        
        from .raviart_thomas_fe_space_2d import RaviartThomasFESpace2d
        from .raviart_thomas_fe_space_3d import RaviartThomasFESpace3d

        dim2class = {
            2: RaviartThomasFESpace2d,
            3: RaviartThomasFESpace3d
        }
        
        SpaceClass = dim2class.get(GD)
        return SpaceClass(mesh, p)