from builtins import float, tuple, str

class GeometryProperties:
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float):
        """
        Initialize the region properties.

        Args:
            x_min (float): The minimum value in the x-direction.
            x_max (float): The maximum value in the x-direction.
            y_min (float): The minimum value in the y-direction.
            y_max (float): The maximum value in the y-direction.
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def get_dimensions(self) -> tuple:
        """
        Calculate and return the width and height of the region.

        Returns:
            tuple: A tuple containing the width and height of the region.
        """
        width = self.x_max - self.x_min 
        height = self.y_max - self.y_min

        return width, height
    
    def __repr__(self) -> str:
        """
        Return a string representation of the region properties, 
        which includes the minimum and maximum values in x and y directions.

        Returns:
            str: A string showing the region properties.
        """
        return (f"GeometryProperties(x_min={self.x_min}, x_max={self.x_max}, "
                f"y_min={self.y_min}, y_max={self.y_max})")
