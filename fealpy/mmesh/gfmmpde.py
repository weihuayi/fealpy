from . import Config


class GFMMPDE:
    def __new__(cls,mesh,beta,space,config:Config):
        """
        The GFMMPDE class is a factory pattern implementation for 
        Gradient Flow Moving Mesh Partial Differential Equation solvers. 
        This class automatically selects the appropriate 2D or 3D GFMMPDE solver based on 
        the geometric dimension of the input mesh, 
        providing a unified interface for moving mesh methods across different dimensions.
        Automatically detects the geometric dimension of the input mesh using mesh.geo_dimension()
        Selects the appropriate solver implementation based on the detected dimension
        Parameters:
            mesh: The input mesh object, which should have a method geo_dimension() to determine its dimension.
            beta: A parameter for the GFMMPDE monitor.
            space: The finite element space associated with the mesh.
            config: Configuration settings for the GFMMPDE solver.
        Returns:
            An instance of the appropriate GFMMPDE solver class.
        """
        from . import GFMMPDE2d
        from . import GFMMPDE3d
        from . import GFMMPDELagrange2d
        
        GD = mesh.geo_dimension()
        
        if hasattr(mesh,'p'):
            p = 1
        else:
            p = 0
        
        if GD == 1:
            raise ValueError(
                f"Unsupported dimension: {GD}. Only 2, or 3 are supported.")

        dim2class = {
            (2,0): GFMMPDE2d,
            (3,0): GFMMPDE3d,
            (2,1): GFMMPDELagrange2d,
            # (3,1): GFMMPDELagrange3d,
        }

        MMClass = dim2class.get((GD,p))
        return MMClass(mesh,beta,space,config)