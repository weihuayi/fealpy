from ..utilfs.filter_parameters import compute_filter_2d, compute_filter_3d
from builtins import int, float

class FilterProperties:
    def __init__(self, nx: int, ny: int, rmin: float, ft: int, nz: int=None):
        """
        Initialize the filter properties.

        Args:
            nx (int): The number of elements in the x-direction of the mesh.
            ny (int): The number of elements in the y-direction of the mesh.
            rmin (float): The filter radius, which controls the minimum feature size.
            ft (int): The filter type, 0 for sensitivity filter, 1 for density filter.
            nz (int, optional): The number of elements in the z-direction of the mesh. 
                                If provided, a 3D filter is used.
        """
        self.ft = ft
        if nz is not None:
            self.H, self.Hs = compute_filter_3d(nx, ny, nz, rmin)
        else:
            self.H, self.Hs = compute_filter_2d(nx, ny, rmin)

    def __repr__(self):
        """
        Return a string representation of the FilterProperties object.

        This method provides a textual representation of the filter properties, 
        including the filter type (ft) and the computed filter matrix (H) and 
        scaling vector (Hs). 

        - ft: Indicates the type of filter used.
            0 - Sensitivity filter: Applies a filter to sensitivity values.
            1 - Density filter: Applies a filter to density values.

        - H: The filter matrix, which is used to apply the filter effect 
            to the design variables or sensitivities. This matrix has been computed 
            using the specified filter radius (rmin).

        - Hs: The scaling vector, which is used to normalize the effect 
            of the filter matrix (H) to ensure that the filtered values are properly 
            scaled.

        Returns:
            str: A string representation of the filter type, filter matrix, and scaling vector.
        """
        return (f"FilterProperties(ft={self.ft}, H={self.H}, Hs={self.Hs})")
 