import os
import imageio.v3 as iio
from ..backend import backend_manager as bm

class TerrainLoader:
    """
    A terrain data loader class specialized for reading and preprocessing TIFF format elevation data.

    This class provides class methods to load terrain data from GeoTIFF files, with built-in 
    preprocessing for negative elevation values. The loader automatically handles file path 
    resolution relative to the script location.

    Methods:
        load_terrain: Main interface for loading and preprocessing terrain data.
    """
    @classmethod
    def load_terrain(cls, filename: str):
        """Load and preprocess terrain elevation data from a TIFF file.

        The method performs the following processing pipeline:
        1. Resolves absolute file path relative to the script location
        2. Reads elevation data using imageio.v3 standard
        3. Clips negative values to 0 (sea level baseline)
        4. Returns a 2D numpy array of elevation values

        Parameters:
            filename : str
                The filename of the TIFF terrain data. 
                Both relative and absolute paths are supported.
                Expected to be in GeoTIFF format with elevation values.

        Returns:
            ndarray:
                A 2D numpy array containing elevation values in dtype float32.
                Negative values are clipped to 0.

        Raises:
            FileNotFoundError:
                When the specified terrain file cannot be located.
            RuntimeError: 
                For any file reading or data processing failures.
                Includes original error message for diagnostics.

        Notes:
            - The ibmut TIFF should contain single-channel elevation data
            - No georeferencing information is preserved in current version
            - For large datasets (>1GB), consider using memory-mapped loading

        Examples:
            >>> terrain = TerrainLoader.load_terrain('AlpsTerrain.tif')
            >>> terrain.shape
            (1200, 800)
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        tif_path = os.path.join(script_dir, filename)
        
        try:
            H = iio.imread(tif_path)
            H[H < 0] = 0  # Sea level normalization
            return H
        except FileNotFoundError:
            raise FileNotFoundError(f"Terrain file not found: {tif_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load terrain data: {str(e)}")
