from typing import Any, Dict, List, Tuple, Optional

from ..backend import bm
from ..typing import TensorLike
from ..decorator import variantmethod


class DLDMicrofluidicChipModeler:
    """
    A class for geometric modeling of microfluidic chips with deterministic lateral displacement (DLD) arrays.
    
    This class generates the geometric layout of DLD chips including pillar arrangements and boundary definitions.
    It supports various pillar shapes (circle, ellipse, droplet, triangle) through variant methods.
    
    Parameters:
        options (Optional[Dict[str, Any]]): Configuration dictionary controlling chip dimensions,
            stage assembly parameters, and lattice patterns. See :meth:`get_options` for the full list
            of available parameters and their default values.

    Attributes:
        circles (Optional[TensorLike]): Pillar array of shape (N, 3) where each row contains
            (x, y, radius) coordinates for circular pillars.
        boundary (Optional[TensorLike]): Boundary polyline points of shape (NE, 2).
        
    Examples:
        >>> modeler = DLDMicrofluidicChipModeler()
        >>> options = modeler.get_options()
        >>> modeler = DLDMicrofluidicChipModeler(options)
        >>> modeler.build()
        >>> modeler.show()
    """

    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the DLD microfluidic chip modeler with given or default options.
        
        Parameters:
            options: Configuration dictionary. If None, uses default options from get_options().
        """
        if not options:
            self.options = self.get_options()
        else:
            self.options = options
            auto_config = self.options.get('auto_config', True)
            if auto_config:
                self._apply_auto_config() 

        self.circles: Optional[TensorLike] = None
        self.boundary: Optional[TensorLike] = None

    @staticmethod
    def get_options() -> Dict[str, Any]:
        """
        Return the default option dictionary for the DLD geometry.

        Returns:
            Dict[str, Any]: Dictionary containing all default configuration options with descriptions:
            
            - init_point (Tuple[float, float]): Bottom-left corner coordinates of the chip (x, y)
            - chip_height (float): Total height of the chip (consistent units, e.g., micrometers)
            - inlet_length (float): Length of the inlet duct before the first stage
            - outlet_length (float): Length of the outlet duct after the last stage
            - start_center (Tuple[float, float]): Local origin offset (ox, oy) for lattice in a stage
            - radius (float): Radius of each pillar (half of pillar_diam)
            - n_rows (int): Number of pillar rows in the local y-direction within a stage
            - n_cols (int): Number of pillar columns in the local x-direction within a stage
            - pitch_x (float): Lattice spacing in the local x-direction between adjacent columns
            - pitch_y (float): Lattice spacing in the local y-direction between adjacent rows
            - tan_angle (float | List[float]): Tangent of rotation angle for pillars in each stage
            - n_stages (int): Number of stages in the chip
            - stage_gap (float): Vertical gap between two consecutive stages
            - auto_config (bool): If True, center lattice within stage rectangle before rotation
        """
        return {
            # chip-scale parameters
            'init_point': (0.0, 0.0),
            'chip_height': 5.0,
            'inlet_length': 1.0,
            'outlet_length': 1.0,

            # lattice parameters (local before rotation)
            'start_center': (0.30864197530864196, 0.5555555555555556),
            'radius': 0.1,
            'n_rows': 8,
            'n_cols': 6,
            'pitch_x': 0.9259259259259258,
            'pitch_y': 0.5555555555555556,
            'tan_angle': 0.1,  # 1/10
            
            # stage assembly parameters
            'n_stages': 7,
            'stage_gap': 0.2586419753086419,

            # auto-configuration
            'auto_config': True,
        }
            
    def _apply_auto_config(self) -> None:
        """
        Apply automatic configuration to optimize lattice parameters.
        
        This method calculates optimal pitch values and stage gap based on chip dimensions
        and pillar configuration when auto_config is enabled.
        """
        options: Dict[str, Any] = self.options
        chip_height: float = options['chip_height']
        n_rows: int = options['n_rows']
        n_cols: int = options['n_cols']
        radius: float = options['radius']
        tan: float = options['tan_angle']
        n: int = options['n_stages']

        if tan > 0:
            pitch_y: float = chip_height / n_rows
            pitch_x: float = pitch_y / (n_cols * tan)
            cx0: float = pitch_x / 3
            cy0: float = chip_height / (2 * n_rows)
            stage_length: float = n_cols * pitch_x - 0.2 * (pitch_x - cx0 - radius)
        else:
            cx0: float = 1 / (2 * n_cols)
            cy0: float = chip_height / (2 * n_rows)
            pitch_x: float = 1 / n_cols
            pitch_y: float = chip_height / n_rows
            stage_length: float = 1
            
        self.options['start_center'] = (cx0, cy0)
        self.options['pitch_x'] = pitch_x
        self.options['pitch_y'] = pitch_y
        self.options['stage_length'] = stage_length

    @variantmethod('circle')
    def build(self, gmsh: Any):
        """
        Build the microfluidic chip model with circular pillars.

        Parameters:
            gmsh: Optional; GMSH instance for mesh generation.
        """
        options: Dict[str, Any] = self.options
 
        x0, y0 = options['init_point']
        h: float = options['chip_height']
        h1: float = options['inlet_length']
        h2: float = options['outlet_length']

        cx0, cy0 = options['start_center']
        r: float = options['radius']
        m: int = options['n_rows']
        n: int = options['n_cols']
        l2: float = options['pitch_x']
        l1: float = options['pitch_y']
        tan: float = options['tan_angle']
        
        # stage assembly parameters
        N: int = options['n_stages']
        L: float = options['stage_length']

        # Create transformation vectors
        if tan > 0:
            l1 = tan * n * l2
        v_trans = bm.array([[n * l2, 0]], dtype=bm.float64)
        h_trans = bm.array([[0, h]], dtype=bm.float64)

        # Define boundary points
        p0 = bm.array([[x0, y0]], dtype=bm.float64)
        v0 = bm.array([[x0 + h1, y0], [x0 + h1 + L, y0 + tan * L]], dtype=bm.float64)
        all_v = v0[None, ...] + bm.arange(N)[:, None, None] * v_trans
        all_v = all_v.reshape(-1, 2)
        p_last = all_v[-1] + bm.array([[h2, 0]], dtype=bm.float64)

        if (h1 ==0) and (h2 ==0):
            self.boundary = bm.concat([all_v, all_v[::-1] + h_trans], axis=0)
            self.inlet_boundary = self.boundary[[0, 3], :]
            self.outlet_boundary = self.boundary[[1, 2], :]
            self.wall_boundary = self.boundary[[0, 1, 2, 3], :]
        else:
            right = bm.concat([p_last, p_last + h_trans], axis=0)
            self.boundary = bm.concat([p0, all_v, right, all_v[::-1] + h_trans, p0 + h_trans], axis=0)
            self.inlet_boundary = self.boundary[[0, -1], :]
            self.outlet_boundary = bm.concat([p_last, p_last + h_trans], axis=0)
            bottom = bm.concat([p0, bm.repeat(all_v, 2, axis=0)], axis=0)
            top = bm.concat([bm.repeat(all_v[::-1] + h_trans, 2, axis=0), p0 + h_trans], axis=0)
            self.wall_boundary = bm.concat([bottom, right, top], axis=0)

        # Generate pillar centers
        v_shift = bm.array([[l2, l2 * tan]], dtype=bm.float64)
        c0 = bm.array([[x0 + h1 + cx0, y0 + cy0]], dtype=bm.float64)

        row_centers = c0 + bm.arange(m)[:, None] * bm.array([[0, l1]], dtype=bm.float64)
        stage_centers = (row_centers[None, ...] + bm.arange(n)[:, None, None] * v_shift).reshape(-1, 2)
        centers = (stage_centers[None, ...] + bm.arange(N)[:, None, None] * v_trans).reshape(-1, 2)
        self.circles = bm.concat([centers, bm.full((len(centers), 1), r)], axis=1)

        if not hasattr(gmsh, "model") or not hasattr(gmsh.model, "occ"):
            raise ValueError("Unsupported geometry engine: only 'gmsh' is supported.")

        boundary_points = self.boundary.reshape(-1, 2)
        boundary_line_tags = []

        for i in range(len(boundary_points)):
            gmsh.model.occ.addPoint(boundary_points[i][0], boundary_points[i][1], 0, tag=i+1)

        for i in range(len(boundary_points)):
            gmsh.model.occ.addLine(i+1, (i+1) % len(boundary_points) + 1, tag=i+1)
            boundary_line_tags.append(i+1)

        boundary_tag = gmsh.model.occ.addCurveLoop(boundary_line_tags)
        gmsh.model.occ.addPlaneSurface([boundary_tag])
        gmsh.model.occ.synchronize()

        # Create the circle pillars (standing columns)
        circle_tags = []
        for (x, y, radius) in self.circles:
            circle = gmsh.model.occ.addDisk(x, y, 0, radius, radius)
            circle_tags.append(circle)
        
        gmsh.model.occ.cut(
            [(2, boundary_tag)], 
            [(2, tag) for tag in circle_tags], 
            removeObject=True, 
            removeTool=True
        )
        gmsh.model.occ.synchronize()
  
    @build.register('ellipse')
    def build(self, gmsh: Any):
        """Build chip model with elliptical pillars (not implemented)."""
        raise NotImplementedError

    @build.register('droplet')
    def build(self, gmsh: Any):
        """Build chip model with droplet-shaped pillars (not implemented)."""
        raise NotImplementedError

    @build.register('triangle')
    def build(self, gmsh: Any):
        """Build chip model with triangular pillars (not implemented)."""
        raise NotImplementedError  
        
    def add_plot(self, ax= None, **kwargs):
        """
        Plot the DLD geometry (boundary polygon and pillar circles) using Matplotlib.
        
        Parameters:
            ax: matplotlib.axes.Axes object to plot on. If None, creates new figure.
            **kwargs: Additional keyword arguments for matplotlib patches.
            
        Returns:
            matplotlib.axes.Axes: The axes containing the plotted geometry.
            
        Raises:
            RuntimeError: If geometry has not been built yet.
        """
        if self.circles is None or self.boundary is None:
            raise RuntimeError("Geometry not built yet. Call `build()` first.")

        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # plot boundary
        bnd = bm.concat((self.boundary, self.boundary[[0], :]), axis=0)
        ax.plot(
            bnd[:, 0],
            bnd[:, 1],
            color="blue",
            linewidth=1.5,
        )

        # plot circles
        from matplotlib.patches import Circle
        for (x, y, r) in self.circles:
            circ = Circle(
                (x, y),
                r,
                facecolor=kwargs.get("facecolor", "white"),
                edgecolor=kwargs.get("edgecolor", "black"),
                linewidth=0.5,
                alpha=kwargs.get("alpha", 1.0),
            )
            ax.add_patch(circ)

        ax.set_aspect("equal", "box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("DLD Geometry")
        return ax

    @variantmethod('matplotlib')
    def show(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is not installed. Please install it using 'pip install matplotlib'.")
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.add_plot(ax)
        plt.show()

    @show.register('gmsh')
    def show(self, gmsh: Any = None):
        gmsh.fltk.run()

       