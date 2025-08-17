class ScaledMonomialSpace:
    """
    Factory for creating scaled monomial space objects in 2D or 3D,
    with automatic dimension detection and appropriate class selection.

    Parameters:
        mesh:
            The computational mesh object providing geometric information.
        p (int):
            The degree of the polynomial space.
        q (int, optional):
            The index of the quadrature formula.
        bc (TensorLike, optional):
            cell barycenter

    Returns:
        Instance of either ScaledMonomialSpace2d or ScaledMonomialSpace3d,
        matching the geometric dimension of the input mesh.

    Raises:
        ValueError:
            - If geometric dimension is 1 (unsupported)
            - If geometric dimension is neither 2 nor 3
    """
    def __new__(cls, mesh, p : int , q = None, bc = None):
        # Get geometric dimension from mesh
        GD = mesh.geo_dimension()

        # Validate dimension (scaled monomial space only supported in 2D/3D)
        if GD == 1:
            raise ValueError(
                f"Unsupported dimension: {GD}. Only 2, or 3 are supported.")
        
        from .scaled_monomial_space_2d import ScaledMonomialSpace2d
        from .scaled_monomial_space_3d import ScaledMonomialSpace3d

        dim2class = {
            2: ScaledMonomialSpace2d,
            3: ScaledMonomialSpace3d
        }
        
        SpaceClass = dim2class.get(GD)
        return SpaceClass(mesh, p, q, bc)